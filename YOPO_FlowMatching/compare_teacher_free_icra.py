#!/usr/bin/env python3
"""Paired held-out comparison including teacher-free physical LatticeFlow."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compare_flow_spaces_icra import (
    paired_bootstrap,
    per_map,
    record_method,
)
from offline_eval_icra import append, perturb_inputs, summarize
from yopo_flow.checkpoint import load_policy_checkpoint
from yopo_flow.config import resolve_device
from yopo_flow.costs import TrajectoryCostEvaluator
from yopo_flow.dataset import configure_yopo_data_root, dataset_from_config
from yopo_flow.targets import YOPOTeacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--physical-checkpoint", required=True)
    parser.add_argument("--teacher-free-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def write_report(path: Path, stats: dict, maps: dict, comparisons: dict) -> None:
    methods = [
        ("yopo", "YOPO"),
        ("residual", "Residual-source + teacher"),
        ("physical", "Physical-anchor + teacher"),
        ("teacher_free", "Physical-anchor teacher-free"),
    ]
    lines = [
        "# Teacher-free Physical-anchor Flow Comparison",
        "",
        "All methods use the same deterministic held-out queries and perturbations from maps 8--9. The teacher-free model is randomly initialized and uses no YOPO model or pretrained backbone during training. Image-query intervals are descriptive because observations are nested within two maps.",
        "",
        "| Method | Selected cost ↓ | Oracle cost ↓ | Clearance (m) ↑ | Collision proxy ↓ | Perturb switch ↓ | Endpoint shift (m) ↓ | Selection regret ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in methods:
        lines.append(
            f"| {label} | {stats[f'{key}_selected_cost']['mean']:.6f} | "
            f"{stats[f'{key}_oracle_cost']['mean']:.6f} | "
            f"{stats[f'{key}_clearance_m']['mean']:.6f} | "
            f"{stats[f'{key}_collision_proxy']['mean']:.6f} | "
            f"{stats[f'{key}_perturb_switch']['mean']:.6f} | "
            f"{stats[f'{key}_endpoint_shift_m']['mean']:.6f} | "
            f"{stats[f'{key}_selection_regret']['mean']:.6f} |"
        )

    labels = {
        "teacher_free_minus_physical": "Teacher-free minus teacher-guided physical",
        "teacher_free_minus_residual": "Teacher-free minus teacher-guided residual",
        "teacher_free_minus_yopo": "Teacher-free minus YOPO",
    }
    for comparison_key, title in labels.items():
        lines.extend(["", f"## {title}", ""])
        for metric, result in comparisons[comparison_key].items():
            lines.append(
                f"- {metric}: {result['mean_first_minus_second']:.6f}, "
                f"descriptive 95% CI [{result['ci95_low']:.6f}, "
                f"{result['ci95_high']:.6f}]."
            )

    lines.extend(
        [
            "",
            "## Per-map means",
            "",
            "| Map | YOPO cost | Residual cost | Physical-teacher cost | Teacher-free cost | Teacher-free proxy | Teacher-free switch |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for map_id, row in maps.items():
        lines.append(
            f"| {map_id} | {row['yopo_selected_cost']:.6f} | "
            f"{row['residual_selected_cost']:.6f} | "
            f"{row['physical_selected_cost']:.6f} | "
            f"{row['teacher_free_selected_cost']:.6f} | "
            f"{row['teacher_free_collision_proxy']:.6f} | "
            f"{row['teacher_free_perturb_switch']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def main():
    args = parse_args()
    paths = {
        "residual": Path(args.residual_checkpoint).resolve(),
        "physical": Path(args.physical_checkpoint).resolve(),
        "teacher_free": Path(args.teacher_free_checkpoint).resolve(),
    }
    raw_checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    configs = {name: value["config"] for name, value in raw_checkpoints.items()}
    reference = configs["residual"]
    for name, config in configs.items():
        for key in ("train_maps", "valid_maps", "test_maps"):
            if config["data"][key] != reference["data"][key]:
                raise ValueError(f"Mismatched {key} for {name}")

    configure_yopo_data_root(reference["data"]["root"])
    device = resolve_device(str(reference["runtime"]["device"]))
    models = {
        name: load_policy_checkpoint(str(path), device)[0]
        for name, path in paths.items()
    }
    yopo = YOPOTeacher(reference["project"]["teacher_checkpoint"], device)
    evaluator = TrajectoryCostEvaluator(
        reference["depth_safety"],
        depth_safety_weight=float(reference["loss_weights"]["depth_safety"]),
    ).to(device)
    evaluator.requires_grad_(False)
    loader = DataLoader(
        dataset_from_config(reference, "test"),
        batch_size=int(reference["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    collision_distance = float(reference["evaluation"]["collision_distance"])
    generator = torch.Generator(device=device).manual_seed(
        int(reference["runtime"]["seed"]) + 991
    )
    values: dict[str, list[float]] = defaultdict(list)

    for batch_index, batch in enumerate(
        tqdm(loader, desc="Compare teacher-free flow", dynamic_ncols=True)
    ):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        depth, position, rotation, obs_body, map_id = (
            item.to(device, non_blocking=True) for item in batch
        )
        append(values, "map_id", map_id)
        noisy_depth, noisy_obs = perturb_inputs(
            depth, obs_body, reference, generator
        )

        for name, model in models.items():
            prepared = model.prepare_observation(obs_body)
            raw, score = model(depth, prepared)
            noisy_prepared = model.prepare_observation(noisy_obs)
            noisy_raw, noisy_score = model(noisy_depth, noisy_prepared)
            costs = evaluator(raw, depth, position, rotation, obs_body, map_id)
            record_method(
                values,
                name,
                raw,
                score,
                costs,
                model.state_transform,
                noisy_raw,
                noisy_score,
                collision_distance,
            )

        yopo_raw, yopo_score = yopo(depth, obs_body)
        noisy_yopo_raw, noisy_yopo_score = yopo(noisy_depth, noisy_obs)
        yopo_costs = evaluator(
            yopo_raw, depth, position, rotation, obs_body, map_id
        )
        record_method(
            values,
            "yopo",
            yopo_raw,
            yopo_score,
            yopo_costs,
            yopo.state_transform,
            noisy_yopo_raw,
            noisy_yopo_score,
            collision_distance,
        )

    stats = summarize(values)
    metrics = (
        "selected_cost",
        "oracle_cost",
        "clearance_m",
        "collision_proxy",
        "perturb_switch",
        "endpoint_shift_m",
        "score_mae",
        "selection_regret",
    )
    comparison_targets = {
        "teacher_free_minus_physical": "physical",
        "teacher_free_minus_residual": "residual",
        "teacher_free_minus_yopo": "yopo",
    }
    comparisons = {
        comparison_name: {
            metric: paired_bootstrap(
                values[f"teacher_free_{metric}"],
                values[f"{second}_{metric}"],
                args.bootstrap_samples,
                seed=1000 + target_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
        for target_index, (comparison_name, second) in enumerate(
            comparison_targets.items()
        )
    }
    methods = ["yopo", "residual", "physical", "teacher_free"]
    maps = per_map(values, methods)
    payload = {
        "checkpoints": {name: str(path) for name, path in paths.items()},
        "protocol": {
            "maps": reference["data"]["test_maps"],
            "query_count": stats["map_id"]["count"],
            "deterministic_observations": True,
            "identical_perturbations": True,
            "nested_query_intervals_are_descriptive": True,
        },
        "summary": stats,
        "per_map": maps,
        "paired_comparisons": comparisons,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(output_dir / "COMPARISON.md", stats, maps, comparisons)
    print(output_dir / "COMPARISON.md")


if __name__ == "__main__":
    main()
