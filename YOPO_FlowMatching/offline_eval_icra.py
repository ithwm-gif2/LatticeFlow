#!/usr/bin/env python3
"""Reproducible held-out-map evaluation for the ICRA LatticeFlow study."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from yopo_flow.checkpoint import load_policy_checkpoint
from yopo_flow.config import resolve_device
from yopo_flow.costs import CostBundle, TrajectoryCostEvaluator, gather_lattice, robust_cost
from yopo_flow.dataset import configure_yopo_data_root, dataset_from_config
from yopo_flow.targets import YOPOTeacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--visualizations", type=int, default=None)
    return parser.parse_args()


def append(values: dict[str, list[float]], key: str, value) -> None:
    if torch.is_tensor(value):
        values[key].extend(value.detach().cpu().reshape(-1).tolist())
    elif isinstance(value, np.ndarray):
        values[key].extend(value.reshape(-1).tolist())
    else:
        values[key].append(float(value))


def selected_components(
    values: dict[str, list[float]],
    prefix: str,
    costs: CostBundle,
    choice: torch.Tensor,
) -> None:
    for name in (
        "total",
        "smooth",
        "safety",
        "guidance",
        "acceleration",
        "depth_safety",
        "min_depth_clearance",
    ):
        append(
            values,
            f"{prefix}_selected_{name}",
            gather_lattice(getattr(costs, name), choice),
        )
    append(values, f"{prefix}_oracle_total", costs.total.flatten(1).amin(dim=1))


def trace_metrics(trace: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack(trace, dim=0).float()
    velocity = stacked[1:] - stacked[:-1]
    path_length = velocity.square().sum(dim=2).sqrt().sum(dim=0)
    displacement = (stacked[-1] - stacked[0]).square().sum(dim=1).sqrt()
    efficiency = displacement / path_length.clamp_min(1e-6)
    if velocity.shape[0] > 1:
        curvature = (velocity[1:] - velocity[:-1]).square().mean(dim=(0, 2))
    else:
        curvature = torch.zeros_like(efficiency)
    return efficiency.mean(dim=(1, 2)), curvature.mean(dim=(1, 2))


def gather_endpoint(raw: torch.Tensor, transform, choice: torch.Tensor) -> torch.Tensor:
    endstate = transform.pred_to_endstate(raw)
    flat = endstate.permute(0, 2, 3, 1).reshape(raw.shape[0], -1, 9)
    index = choice[:, None, None].expand(-1, 1, 9)
    return flat.gather(1, index).squeeze(1)


def perturb_inputs(
    depth: torch.Tensor,
    obs_body: torch.Tensor,
    config: dict,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    continuity = config.get("continuity", {})
    depth_noise = torch.randn(
        depth.shape,
        generator=generator,
        device=depth.device,
        dtype=depth.dtype,
    ) * float(continuity.get("depth_noise_std", 0.005))
    obs_std = torch.as_tensor(
        continuity.get("observation_noise_std", [0.05] * 9),
        device=obs_body.device,
        dtype=obs_body.dtype,
    ).view(1, 9)
    obs_noise = torch.randn(
        obs_body.shape,
        generator=generator,
        device=obs_body.device,
        dtype=obs_body.dtype,
    ) * obs_std
    return (depth + depth_noise).clamp(0.0, 1.0), obs_body + obs_noise


def project_points(points: np.ndarray, config: dict, width: int, height: int):
    forward = np.maximum(points[:, 0], 1e-3)
    u = config["cx"] - config["fx"] * points[:, 1] / forward
    v = config["cy"] - config["fy"] * points[:, 2] / forward
    valid = (
        (points[:, 0] > 0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    return u[valid], v[valid]


def save_visualization(
    path: Path,
    depth: torch.Tensor,
    student_points: np.ndarray,
    teacher_points: np.ndarray,
    depth_config: dict,
    title: str,
) -> None:
    image = depth.squeeze().detach().cpu().numpy() * float(depth_config["max_depth"])
    height, width = image.shape
    student_u, student_v = project_points(student_points, depth_config, width, height)
    teacher_u, teacher_v = project_points(teacher_points, depth_config, width, height)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    view = axis.imshow(image, cmap="viridis", vmin=0, vmax=depth_config["max_depth"])
    axis.plot(teacher_u, teacher_v, color="white", linewidth=2, label="YOPO")
    axis.plot(student_u, student_v, color="#e41a1c", linewidth=2, label="LatticeFlow")
    axis.set_title(title)
    axis.set_axis_off()
    axis.legend(loc="lower right")
    fig.colorbar(view, ax=axis, fraction=0.025, label="Depth (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    result = {}
    for key, items in values.items():
        array = np.asarray(items, dtype=np.float64)
        result[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
            "count": int(array.size),
        }
    return result


def per_map_summary(values: dict[str, list[float]], metrics: list[str]) -> dict:
    map_ids = np.asarray(values["map_id"], dtype=np.int64)
    output = {}
    for map_id in sorted(np.unique(map_ids).tolist()):
        mask = map_ids == map_id
        output[str(map_id)] = {
            metric: float(np.asarray(values[metric], dtype=np.float64)[mask].mean())
            for metric in metrics
        }
    return output


def paired_bootstrap_ci(
    student: list[float], teacher: list[float], seed: int = 0, samples: int = 2000
) -> dict[str, float]:
    student_array = np.asarray(student, dtype=np.float64)
    teacher_array = np.asarray(teacher, dtype=np.float64)
    delta = student_array - teacher_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    boot = delta[indices].mean(axis=1)
    return {
        "mean_student_minus_teacher": float(delta.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def write_report(
    path: Path,
    checkpoint: str,
    split: str,
    stats: dict,
    per_map: dict,
    comparisons: dict,
    collision_distance: float,
) -> None:
    student_cost = stats["student_selected_total"]["mean"]
    teacher_cost = stats["teacher_selected_total"]["mean"]
    improvement = 100.0 * (teacher_cost - student_cost) / max(abs(teacher_cost), 1e-8)
    lines = [
        "# ICRA Held-out-map Offline Evaluation",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Split: `{split}`",
        f"- Samples: {stats['student_selected_total']['count']}",
        f"- Maps: {', '.join(per_map.keys())}",
        f"- Collision proxy threshold: {collision_distance:.3f} m",
        "",
        "## Main comparison",
        "",
        "| Metric | LatticeFlow | YOPO |",
        "|---|---:|---:|",
        f"| Selected trajectory cost ↓ | {student_cost:.6f} | {teacher_cost:.6f} |",
        f"| Oracle trajectory cost ↓ | {stats['student_oracle_total']['mean']:.6f} | {stats['teacher_oracle_total']['mean']:.6f} |",
        f"| Selected minimum clearance (m) ↑ | {stats['student_selected_min_depth_clearance']['mean']:.6f} | {stats['teacher_selected_min_depth_clearance']['mean']:.6f} |",
        f"| Collision proxy rate ↓ | {stats['student_collision_rate']['mean']:.6f} | {stats['teacher_collision_rate']['mean']:.6f} |",
        f"| Perturbation cell-switch rate ↓ | {stats['student_perturb_switch']['mean']:.6f} | {stats['teacher_perturb_switch']['mean']:.6f} |",
        f"| Perturbation endpoint shift (m) ↓ | {stats['student_perturb_endpoint_shift_m']['mean']:.6f} | {stats['teacher_perturb_endpoint_shift_m']['mean']:.6f} |",
        f"| Network inference (ms/batch) ↓ | {stats['student_inference_ms']['mean']:.3f} | {stats['teacher_inference_ms']['mean']:.3f} |",
        "",
        f"Selected-cost improvement over the fair YOPO baseline: **{improvement:.3f}%**.",
        "",
        "## Flow diagnostics",
        "",
        f"- Score MAE: {stats['student_score_mae']['mean']:.6f}",
        f"- Score selection regret: {stats['student_selection_regret']['mean']:.6f}",
        f"- Flow path efficiency: {stats['flow_path_efficiency']['mean']:.6f}",
        f"- Flow curvature: {stats['flow_curvature']['mean']:.6f}",
        "",
        "## Per-map means",
        "",
        "| Map | LatticeFlow cost ↓ | YOPO cost ↓ | LatticeFlow collision proxy ↓ | YOPO collision proxy ↓ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for map_id, row in per_map.items():
        lines.append(
            f"| {map_id} | {row['student_selected_total']:.6f} | "
            f"{row['teacher_selected_total']:.6f} | {row['student_collision_rate']:.6f} | "
            f"{row['teacher_collision_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired descriptive uncertainty",
            "",
            f"- Selected-cost difference (LatticeFlow − YOPO): {comparisons['selected_cost']['mean_student_minus_teacher']:.6f}, "
            f"image-level bootstrap 95% CI [{comparisons['selected_cost']['ci95_low']:.6f}, {comparisons['selected_cost']['ci95_high']:.6f}].",
            "",
            "The bootstrap treats held-out camera-pose queries as evaluation samples; per-map results are reported separately because queries are nested within maps. These offline differentiable-cost metrics do not replace independent ROS closed-loop trials.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = raw_checkpoint["config"]
    configure_yopo_data_root(config["data"]["root"])
    device = resolve_device(str(config["runtime"]["device"]))
    model, config, _ = load_policy_checkpoint(str(checkpoint_path), device)
    teacher = YOPOTeacher(config["project"]["teacher_checkpoint"], device)
    evaluator = TrajectoryCostEvaluator(
        config["depth_safety"],
        depth_safety_weight=float(config["loss_weights"]["depth_safety"]),
    ).to(device)
    evaluator.requires_grad_(False)

    split = args.split or config.get("evaluation", {}).get("split", "test")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent.parent / f"offline_{split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(exist_ok=True)
    loader = DataLoader(
        dataset_from_config(config, split),
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = config["evaluation"].get("max_batches")
    visualization_limit = args.visualizations
    if visualization_limit is None:
        visualization_limit = int(config["evaluation"]["save_visualizations"])
    collision_distance = float(config["evaluation"]["collision_distance"])
    generator = torch.Generator(device=device).manual_seed(int(config["runtime"]["seed"]) + 991)

    values: dict[str, list[float]] = defaultdict(list)
    visualized = 0
    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"Offline {split}", dynamic_ncols=True)
    ):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        depth, position, rotation, obs_body, map_id = (
            item.to(device, non_blocking=True) for item in batch
        )
        append(values, "map_id", map_id)
        prepared = model.prepare_observation(obs_body)
        features = model.encode_depth(depth)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        student_raw, trace = model.integrate_from_features(
            features, prepared, return_trace=True
        )
        student_score = model.score_from_features(features, prepared, student_raw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        append(values, "student_inference_ms", (time.perf_counter() - start) * 1000.0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        teacher_raw, teacher_score = teacher(depth, obs_body)
        if device.type == "cuda":
            torch.cuda.synchronize()
        append(values, "teacher_inference_ms", (time.perf_counter() - start) * 1000.0)

        student_costs = evaluator(student_raw, depth, position, rotation, obs_body, map_id)
        teacher_costs = evaluator(teacher_raw, depth, position, rotation, obs_body, map_id)
        student_choice = student_score.flatten(1).argmin(dim=1)
        teacher_choice = teacher_score.flatten(1).argmin(dim=1)
        selected_components(values, "student", student_costs, student_choice)
        selected_components(values, "teacher", teacher_costs, teacher_choice)
        student_clearance = gather_lattice(student_costs.min_depth_clearance, student_choice)
        teacher_clearance = gather_lattice(teacher_costs.min_depth_clearance, teacher_choice)
        append(values, "student_collision_rate", (student_clearance < collision_distance).float())
        append(values, "teacher_collision_rate", (teacher_clearance < collision_distance).float())
        append(
            values,
            "student_score_mae",
            F.l1_loss(student_score, robust_cost(student_costs.total), reduction="none").mean(
                dim=(1, 2)
            ),
        )
        append(
            values,
            "student_selection_regret",
            gather_lattice(student_costs.total, student_choice)
            - student_costs.total.flatten(1).amin(dim=1),
        )
        efficiency, curvature = trace_metrics(trace)
        append(values, "flow_path_efficiency", efficiency)
        append(values, "flow_curvature", curvature)

        noisy_depth, noisy_obs = perturb_inputs(depth, obs_body, config, generator)
        noisy_prepared = model.prepare_observation(noisy_obs)
        noisy_student_raw, noisy_student_score = model(noisy_depth, noisy_prepared)
        noisy_teacher_raw, noisy_teacher_score = teacher(noisy_depth, noisy_obs)
        noisy_student_choice = noisy_student_score.flatten(1).argmin(dim=1)
        noisy_teacher_choice = noisy_teacher_score.flatten(1).argmin(dim=1)
        append(values, "student_perturb_switch", (student_choice != noisy_student_choice).float())
        append(values, "teacher_perturb_switch", (teacher_choice != noisy_teacher_choice).float())
        student_endpoint = gather_endpoint(student_raw, model.state_transform, student_choice)
        noisy_student_endpoint = gather_endpoint(
            noisy_student_raw, model.state_transform, noisy_student_choice
        )
        teacher_endpoint = gather_endpoint(teacher_raw, teacher.state_transform, teacher_choice)
        noisy_teacher_endpoint = gather_endpoint(
            noisy_teacher_raw, teacher.state_transform, noisy_teacher_choice
        )
        append(
            values,
            "student_perturb_endpoint_shift_m",
            (student_endpoint[:, :3] - noisy_student_endpoint[:, :3]).norm(dim=1),
        )
        append(
            values,
            "teacher_perturb_endpoint_shift_m",
            (teacher_endpoint[:, :3] - noisy_teacher_endpoint[:, :3]).norm(dim=1),
        )

        if visualized < visualization_limit:
            student_points = evaluator._body_polynomial_points(student_raw, obs_body)
            teacher_points = evaluator._body_polynomial_points(teacher_raw, obs_body)
            for item in range(depth.shape[0]):
                if visualized >= visualization_limit:
                    break
                student_index = int(student_choice[item])
                teacher_index = int(teacher_choice[item])
                save_visualization(
                    visualization_dir / f"sample_{visualized:04d}.png",
                    depth[item],
                    student_points[item, student_index].cpu().numpy(),
                    teacher_points[item, teacher_index].cpu().numpy(),
                    config["depth_safety"],
                    title=f"map {int(map_id[item])}, query {visualized}",
                )
                visualized += 1

    stats = summarize(values)
    per_map_metrics = [
        "student_selected_total",
        "teacher_selected_total",
        "student_collision_rate",
        "teacher_collision_rate",
    ]
    maps = per_map_summary(values, per_map_metrics)
    comparisons = {
        "selected_cost": paired_bootstrap_ci(
            values["student_selected_total"], values["teacher_selected_total"]
        )
    }
    payload = {
        "summary": stats,
        "per_map": maps,
        "paired_comparisons": comparisons,
        "protocol": {
            "split": split,
            "maps": config["data"][f"{split}_maps"],
            "deterministic_observations": True,
            "collision_distance_m": collision_distance,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(
        output_dir / "OFFLINE_RESULTS.md",
        str(checkpoint_path),
        split,
        stats,
        maps,
        comparisons,
        collision_distance,
    )
    print(f"Offline report: {output_dir / 'OFFLINE_RESULTS.md'}")


if __name__ == "__main__":
    main()
