#!/usr/bin/env python3
"""Offline cost, clearance, flow and visualization evaluation."""

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

from yopo_flow.bootstrap import add_original_yopo_to_path
from yopo_flow.checkpoint import load_policy_checkpoint
from yopo_flow.config import resolve_device
from yopo_flow.costs import CostBundle, TrajectoryCostEvaluator, gather_lattice, robust_cost
from yopo_flow.targets import YOPOTeacher

add_original_yopo_to_path()

from policy.yopo_dataset import YOPODataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--visualizations", type=int, default=None)
    return parser.parse_args()


def append(metrics: dict[str, list[float]], key: str, value: torch.Tensor | float) -> None:
    if torch.is_tensor(value):
        metrics[key].extend(value.detach().cpu().reshape(-1).tolist())
    else:
        metrics[key].append(float(value))


def selected_components(
    metrics: dict[str, list[float]],
    prefix: str,
    costs: CostBundle,
    choice: torch.Tensor,
) -> None:
    append(metrics, f"{prefix}_selected_cost", gather_lattice(costs.total, choice))
    append(metrics, f"{prefix}_selected_smooth", gather_lattice(costs.smooth, choice))
    append(metrics, f"{prefix}_selected_safety", gather_lattice(costs.safety, choice))
    append(metrics, f"{prefix}_selected_guidance", gather_lattice(costs.guidance, choice))
    append(metrics, f"{prefix}_selected_acceleration", gather_lattice(costs.acceleration, choice))
    append(metrics, f"{prefix}_selected_depth_safety", gather_lattice(costs.depth_safety, choice))
    append(
        metrics,
        f"{prefix}_selected_min_depth_clearance",
        gather_lattice(costs.min_depth_clearance, choice),
    )
    append(metrics, f"{prefix}_oracle_cost", costs.total.flatten(1).amin(dim=1))


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
    axis.plot(teacher_u, teacher_v, color="white", linewidth=2, label="YOPO teacher")
    axis.plot(student_u, student_v, color="red", linewidth=2, label="Flow student")
    axis.set_title(title)
    axis.set_axis_off()
    axis.legend(loc="lower right")
    fig.colorbar(view, ax=axis, fraction=0.025, label="Depth (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    result = {}
    for key, items in values.items():
        array = np.asarray(items, dtype=np.float64)
        result[key] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "count": int(array.size),
        }
    return result


def write_report(path: Path, checkpoint: str, stats: dict, collision_distance: float) -> None:
    student = stats["student_selected_cost"]["mean"]
    teacher = stats["teacher_selected_cost"]["mean"]
    improvement = 100.0 * (teacher - student) / max(abs(teacher), 1e-8)
    lines = [
        "# Offline Evaluation Results",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Samples: {stats['student_selected_cost']['count']}",
        f"- Collision proxy threshold: {collision_distance:.3f} m",
        "",
        "## Main comparison",
        "",
        "| Metric | Flow student | YOPO teacher |",
        "|---|---:|---:|",
        f"| Selected total cost | {student:.6f} | {teacher:.6f} |",
        f"| Oracle total cost | {stats['student_oracle_cost']['mean']:.6f} | {stats['teacher_oracle_cost']['mean']:.6f} |",
        f"| Minimum depth clearance (m) | {stats['student_selected_min_depth_clearance']['mean']:.6f} | {stats['teacher_selected_min_depth_clearance']['mean']:.6f} |",
        f"| Collision proxy rate | {stats['student_collision_rate']['mean']:.6f} | {stats['teacher_collision_rate']['mean']:.6f} |",
        "",
        f"Selected-cost improvement over teacher: **{improvement:.3f}%**.",
        "",
        "## Student diagnostics",
        "",
        f"- Score MAE: {stats['student_score_mae']['mean']:.6f}",
        f"- Score selection regret: {stats['student_selection_regret']['mean']:.6f}",
        f"- Flow path efficiency: {stats['flow_path_efficiency']['mean']:.6f}",
        f"- Flow curvature: {stats['flow_curvature']['mean']:.6f}",
        f"- Mean inference time: {stats['student_inference_ms']['mean']:.3f} ms/batch",
        "",
        "## Interpretation",
        "",
        "These are offline differentiable-cost and current-depth visibility metrics. They do not replace the ROS closed-loop collision and goal-reaching evaluation; closed-loop results must be recorded separately.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = resolve_device(str(raw_checkpoint["config"]["runtime"]["device"]))
    model, config, _ = load_policy_checkpoint(str(checkpoint_path), device)
    teacher = YOPOTeacher(config["project"]["teacher_checkpoint"], device)
    evaluator = TrajectoryCostEvaluator(
        config["depth_safety"],
        depth_safety_weight=float(config["loss_weights"]["depth_safety"]),
    ).to(device)
    evaluator.requires_grad_(False)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent / "offline_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(exist_ok=True)

    dataset = YOPODataset(mode="valid")
    loader = DataLoader(
        dataset,
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

    values: dict[str, list[float]] = defaultdict(list)
    visualized = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="Offline evaluation", dynamic_ncols=True)):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        depth, position, rotation, obs_body, map_id = (
            item.to(device, non_blocking=True) for item in batch
        )
        prepared_obs = model.prepare_observation(obs_body)
        features = model.encode_depth(depth)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        student_raw, trace = model.integrate_from_features(
            features, prepared_obs, return_trace=True
        )
        student_score = model.score_from_features(features, prepared_obs, student_raw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        append(values, "student_inference_ms", (time.perf_counter() - start) * 1000.0)

        teacher_raw, teacher_score = teacher(depth, obs_body)
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
            F.l1_loss(student_score, robust_cost(student_costs.total)),
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

        if visualized < visualization_limit:
            student_points_all = evaluator._body_polynomial_points(student_raw, obs_body)
            teacher_points_all = evaluator._body_polynomial_points(teacher_raw, obs_body)
            for item in range(depth.shape[0]):
                if visualized >= visualization_limit:
                    break
                s_idx = int(student_choice[item])
                t_idx = int(teacher_choice[item])
                save_visualization(
                    visualization_dir / f"sample_{visualized:04d}.png",
                    depth[item],
                    student_points_all[item, s_idx].cpu().numpy(),
                    teacher_points_all[item, t_idx].cpu().numpy(),
                    config["depth_safety"],
                    title=f"sample {visualized}: student cell {s_idx}, teacher cell {t_idx}",
                )
                visualized += 1

    stats = summary(values)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    write_report(output_dir / "OFFLINE_RESULTS.md", str(checkpoint_path), stats, collision_distance)
    print(f"Offline report: {output_dir / 'OFFLINE_RESULTS.md'}")


if __name__ == "__main__":
    main()
