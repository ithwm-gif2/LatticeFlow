#!/usr/bin/env python3
"""Paired held-out comparison of residual-space and physical-anchor flows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from offline_eval_icra import append, gather_endpoint, perturb_inputs, summarize
from yopo_flow.checkpoint import load_policy_checkpoint
from yopo_flow.config import resolve_device
from yopo_flow.costs import TrajectoryCostEvaluator, gather_lattice, robust_cost
from yopo_flow.dataset import configure_yopo_data_root, dataset_from_config
from yopo_flow.targets import YOPOTeacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--physical-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def paired_bootstrap(
    first: list[float],
    second: list[float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        count = min(100, samples - start)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        means[start : start + count] = delta[indices].mean(axis=1)
    return {
        "mean_first_minus_second": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def record_method(
    values: dict[str, list[float]],
    name: str,
    raw: torch.Tensor,
    score: torch.Tensor,
    costs,
    transform,
    noisy_raw: torch.Tensor,
    noisy_score: torch.Tensor,
    collision_distance: float,
) -> None:
    choice = score.flatten(1).argmin(dim=1)
    noisy_choice = noisy_score.flatten(1).argmin(dim=1)
    selected_cost = gather_lattice(costs.total, choice)
    selected_clearance = gather_lattice(costs.min_depth_clearance, choice)
    endpoint = gather_endpoint(raw, transform, choice)
    noisy_endpoint = gather_endpoint(noisy_raw, transform, noisy_choice)
    append(values, f"{name}_selected_cost", selected_cost)
    append(values, f"{name}_oracle_cost", costs.total.flatten(1).amin(dim=1))
    append(values, f"{name}_clearance_m", selected_clearance)
    append(
        values,
        f"{name}_collision_proxy",
        (selected_clearance < collision_distance).float(),
    )
    append(values, f"{name}_perturb_switch", (choice != noisy_choice).float())
    append(
        values,
        f"{name}_endpoint_shift_m",
        (endpoint[:, :3] - noisy_endpoint[:, :3]).norm(dim=1),
    )
    append(
        values,
        f"{name}_score_mae",
        F.l1_loss(score, robust_cost(costs.total), reduction="none").mean(
            dim=(1, 2)
        ),
    )
    append(
        values,
        f"{name}_selection_regret",
        selected_cost - costs.total.flatten(1).amin(dim=1),
    )


def per_map(values: dict[str, list[float]], methods: list[str]) -> dict:
    ids = np.asarray(values["map_id"], dtype=np.int64)
    result = {}
    for map_id in sorted(np.unique(ids).tolist()):
        mask = ids == map_id
        result[str(map_id)] = {}
        for method in methods:
            for metric in ("selected_cost", "collision_proxy", "perturb_switch"):
                key = f"{method}_{metric}"
                result[str(map_id)][key] = float(
                    np.asarray(values[key], dtype=np.float64)[mask].mean()
                )
    return result


def write_report(path: Path, stats: dict, maps: dict, comparisons: dict) -> None:
    methods = [
        ("yopo", "YOPO"),
        ("residual", "Residual-source"),
        ("physical", "Physical-anchor"),
    ]
    lines = [
        "# Residual-space vs Physical-anchor Flow",
        "",
        "All methods were evaluated on the same 20,000 deterministic queries from held-out maps 8--9. Image-query bootstrap intervals are descriptive because observations are nested within two maps.",
        "",
        "| Method | Selected cost ↓ | Oracle cost ↓ | Clearance (m) ↑ | Collision proxy ↓ | Perturb switch ↓ | Endpoint shift (m) ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in methods:
        lines.append(
            f"| {label} | {stats[f'{key}_selected_cost']['mean']:.6f} | "
            f"{stats[f'{key}_oracle_cost']['mean']:.6f} | "
            f"{stats[f'{key}_clearance_m']['mean']:.6f} | "
            f"{stats[f'{key}_collision_proxy']['mean']:.6f} | "
            f"{stats[f'{key}_perturb_switch']['mean']:.6f} | "
            f"{stats[f'{key}_endpoint_shift_m']['mean']:.6f} |"
        )
    lines.extend(["", "## Paired physical-anchor minus residual-space differences", ""])
    for metric, result in comparisons.items():
        lines.append(
            f"- {metric}: {result['mean_first_minus_second']:.6f}, descriptive 95% CI "
            f"[{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]."
        )
    lines.extend(
        [
            "",
            "## Per-map means",
            "",
            "| Map | Residual cost | Physical cost | Residual proxy | Physical proxy | Residual switch | Physical switch |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for map_id, row in maps.items():
        lines.append(
            f"| {map_id} | {row['residual_selected_cost']:.6f} | "
            f"{row['physical_selected_cost']:.6f} | "
            f"{row['residual_collision_proxy']:.6f} | "
            f"{row['physical_collision_proxy']:.6f} | "
            f"{row['residual_perturb_switch']:.6f} | "
            f"{row['physical_perturb_switch']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def main():
    args = parse_args()
    residual_path = Path(args.residual_checkpoint).resolve()
    physical_path = Path(args.physical_checkpoint).resolve()
    residual_raw = torch.load(residual_path, map_location="cpu", weights_only=False)
    physical_raw = torch.load(physical_path, map_location="cpu", weights_only=False)
    residual_config = residual_raw["config"]
    physical_config = physical_raw["config"]
    for key in ("train_maps", "valid_maps", "test_maps"):
        if residual_config["data"][key] != physical_config["data"][key]:
            raise ValueError(f"Mismatched data split for {key}")
    configure_yopo_data_root(residual_config["data"]["root"])
    device = resolve_device(str(residual_config["runtime"]["device"]))
    residual, _, _ = load_policy_checkpoint(str(residual_path), device)
    physical, _, _ = load_policy_checkpoint(str(physical_path), device)
    teacher = YOPOTeacher(residual_config["project"]["teacher_checkpoint"], device)
    evaluator = TrajectoryCostEvaluator(
        residual_config["depth_safety"],
        depth_safety_weight=float(residual_config["loss_weights"]["depth_safety"]),
    ).to(device)
    evaluator.requires_grad_(False)
    loader = DataLoader(
        dataset_from_config(residual_config, "test"),
        batch_size=int(residual_config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    collision_distance = float(residual_config["evaluation"]["collision_distance"])
    generator = torch.Generator(device=device).manual_seed(
        int(residual_config["runtime"]["seed"]) + 991
    )
    values: dict[str, list[float]] = defaultdict(list)

    for batch_index, batch in enumerate(
        tqdm(loader, desc="Compare flow spaces", dynamic_ncols=True)
    ):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        depth, position, rotation, obs_body, map_id = (
            item.to(device, non_blocking=True) for item in batch
        )
        append(values, "map_id", map_id)
        noisy_depth, noisy_obs = perturb_inputs(
            depth, obs_body, residual_config, generator
        )

        for name, model in (("residual", residual), ("physical", physical)):
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

        teacher_raw, teacher_score = teacher(depth, obs_body)
        noisy_teacher_raw, noisy_teacher_score = teacher(noisy_depth, noisy_obs)
        teacher_costs = evaluator(
            teacher_raw, depth, position, rotation, obs_body, map_id
        )
        record_method(
            values,
            "yopo",
            teacher_raw,
            teacher_score,
            teacher_costs,
            teacher.state_transform,
            noisy_teacher_raw,
            noisy_teacher_score,
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
    comparisons = {
        metric: paired_bootstrap(
            values[f"physical_{metric}"],
            values[f"residual_{metric}"],
            args.bootstrap_samples,
            seed=100 + index,
        )
        for index, metric in enumerate(metrics)
    }
    maps = per_map(values, ["yopo", "residual", "physical"])
    payload = {
        "residual_checkpoint": str(residual_path),
        "physical_checkpoint": str(physical_path),
        "protocol": {
            "maps": residual_config["data"]["test_maps"],
            "query_count": stats["map_id"]["count"],
            "deterministic_observations": True,
            "nested_query_intervals_are_descriptive": True,
        },
        "summary": stats,
        "per_map": maps,
        "paired_physical_minus_residual": comparisons,
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
