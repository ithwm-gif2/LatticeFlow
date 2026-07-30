#!/usr/bin/env python3
"""Evaluate LatticeFlow checkpoints on recorded real LiDAR depth inputs.

The NPZ is expected to contain ``raw_depth`` (real returns, normalized by the
maximum range) and ``depth`` (the deployed local-fill + virtual-ceiling input).
No ROS node is initialized and no control command is published.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from yopo_flow.bootstrap import add_original_yopo_to_path

add_original_yopo_to_path()

from config.config import cfg as yopo_cfg  # noqa: E402
from policy.primitive import LatticePrimitive  # noqa: E402
from yopo_flow.checkpoint import load_policy_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path; pass more than once to compare models.",
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path(
            "/home/hwm/CF_YOPO/Lattice_Flow/diagnostics/"
            "real_lidar_mask_bank_120.npz"
        ),
    )
    parser.add_argument("--runtime-velocity", type=float, default=4.0)
    parser.add_argument("--goal-distance", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def local_hole_fill(raw_depth: torch.Tensor) -> torch.Tensor:
    """Match the deployed 3x3, five-neighbour, one-iteration local fill."""

    kernel_size = 3
    padding = 1
    far_threshold = 0.999
    min_normalized = 0.05 / 20.0
    ones = torch.ones(
        1, 1, kernel_size, kernel_size, device=raw_depth.device, dtype=raw_depth.dtype
    )
    valid = (
        torch.isfinite(raw_depth)
        & (raw_depth > min_normalized)
        & (raw_depth < far_threshold)
    )
    neighbor_count = F.conv2d(valid.to(raw_depth.dtype), ones, padding=padding)
    padded = F.pad(
        raw_depth, (padding, padding, padding, padding), mode="constant", value=1.0
    )
    local_minimum = -F.max_pool2d(-padded, kernel_size=kernel_size, stride=1)
    fill = (
        (~valid)
        & (neighbor_count >= 5)
        & torch.isfinite(local_minimum)
        & (local_minimum > min_normalized)
        & (local_minimum < far_threshold)
    )
    return torch.where(fill, local_minimum, raw_depth)


def quintic_vertical_curve(
    endpoint: torch.Tensor,
    observation: torch.Tensor,
    duration: float,
    samples: int = 101,
) -> torch.Tensor:
    """Return body-frame z positions for selected quintic trajectories."""

    s = torch.linspace(
        0.0, 1.0, samples, device=endpoint.device, dtype=endpoint.dtype
    ).view(1, samples)
    s2, s3, s4, s5 = s.square(), s.pow(3), s.pow(4), s.pow(5)
    h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5
    h20 = 0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5)
    h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
    h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5
    h21 = 0.5 * (s3 - 2.0 * s4 + s5)
    velocity0 = observation[:, 2:3]
    acceleration0 = observation[:, 5:6]
    return (
        h10 * duration * velocity0
        + h20 * duration**2 * acceleration0
        + h01 * endpoint[:, 2:3]
        + h11 * duration * endpoint[:, 5:6]
        + h21 * duration**2 * endpoint[:, 8:9]
    )


def summarize(
    endpoints: np.ndarray,
    curves: np.ndarray,
    choices: np.ndarray,
    margins: np.ndarray,
) -> dict:
    curve_min = curves.min(axis=1)
    curve_max = curves.max(axis=1)
    envelope = np.maximum(np.abs(curve_min), np.abs(curve_max))
    counts = Counter(int(value) for value in choices.tolist())
    return {
        "frames": int(len(choices)),
        "selected_ids": [[key, counts[key]] for key in sorted(counts)],
        "terminal_z_v_a_mean": [
            float(endpoints[:, index].mean()) for index in (2, 5, 8)
        ],
        "terminal_z_v_a_std": [
            float(endpoints[:, index].std()) for index in (2, 5, 8)
        ],
        "curve_min_mean_worst": [float(curve_min.mean()), float(curve_min.min())],
        "curve_max_mean_worst": [float(curve_max.mean()), float(curve_max.max())],
        "envelope_mean": float(envelope.mean()),
        "envelope_p95": float(np.percentile(envelope, 95)),
        "flat_0.2_rate": float(np.mean(envelope <= 0.2)),
        "flat_0.3_rate": float(np.mean(envelope <= 0.3)),
        "up_0.3_rate": float(np.mean(curve_max > 0.3)),
        "down_0.3_rate": float(np.mean(curve_min < -0.3)),
        "score_margin_mean": float(margins.mean()),
    }


@torch.inference_mode()
def evaluate_model(
    checkpoint: Path,
    inputs: dict[str, torch.Tensor],
    observation: torch.Tensor,
    batch_size: int,
) -> dict:
    device = observation.device
    model, config, raw_checkpoint = load_policy_checkpoint(str(checkpoint), device)
    duration = float(model.state_transform.lattice_primitive.segment_time)
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "epoch": int(raw_checkpoint.get("epoch", -1)),
        "best_val_cost": float(raw_checkpoint.get("best_val_cost", float("nan"))),
        "runtime_velocity": float(yopo_cfg["velocity"]),
        "trajectory_duration": duration,
        "training_virtual_ceiling_enabled": bool(
            config.get("lidar_domain", {}).get("virtual_ceiling_enabled", False)
        ),
        "inputs": {},
    }
    choices_by_input: dict[str, np.ndarray] = {}
    for input_name, depth in inputs.items():
        endpoint_parts, curve_parts, choice_parts, margin_parts = [], [], [], []
        for start in range(0, depth.shape[0], batch_size):
            stop = min(start + batch_size, depth.shape[0])
            batch_depth = depth[start:stop]
            batch_observation = observation[start:stop]
            raw, score = model(batch_depth, model.prepare_observation(batch_observation))
            flat_score = score.flatten(1)
            choice = flat_score.argmin(dim=1)
            sorted_scores = flat_score.sort(dim=1).values
            margin = sorted_scores[:, 1] - sorted_scores[:, 0]
            endstate = model.state_transform.pred_to_endstate(raw)
            endpoint_grid = endstate.permute(0, 2, 3, 1).reshape(-1, 15, 9)
            endpoint = endpoint_grid[
                torch.arange(endpoint_grid.shape[0], device=device), choice
            ]
            curve = quintic_vertical_curve(
                endpoint, batch_observation, duration=duration
            )
            endpoint_parts.append(endpoint.cpu().numpy())
            curve_parts.append(curve.cpu().numpy())
            choice_parts.append(choice.cpu().numpy())
            margin_parts.append(margin.cpu().numpy())
        endpoints = np.concatenate(endpoint_parts)
        curves = np.concatenate(curve_parts)
        choices = np.concatenate(choice_parts)
        margins = np.concatenate(margin_parts)
        choices_by_input[input_name] = choices
        result["inputs"][input_name] = summarize(
            endpoints, curves, choices, margins
        )
    if "no_ceiling_local_fill" in choices_by_input and "ceiling_on" in choices_by_input:
        result["ceiling_effect"] = {
            "cell_switch_rate": float(
                np.mean(
                    choices_by_input["no_ceiling_local_fill"]
                    != choices_by_input["ceiling_on"]
                )
            )
        }
    return result


def main() -> None:
    args = parse_args()
    if args.runtime_velocity <= 0.0 or args.goal_distance <= 0.0:
        raise ValueError("Runtime velocity and goal distance must be positive")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    yopo_cfg["train"] = False
    yopo_cfg["velocity"] = float(args.runtime_velocity)
    # The singleton has not been instantiated yet in this standalone process.
    # Clearing it explicitly also keeps repeated invocations in embedded tests safe.
    LatticePrimitive._instance = None

    data = np.load(args.input_npz)
    raw_depth = torch.from_numpy(data["raw_depth"][:, None]).float()
    ceiling_depth = torch.from_numpy(data["depth"][:, None]).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_depth = raw_depth.to(device)
    ceiling_depth = ceiling_depth.to(device)
    no_ceiling_depth = local_hole_fill(raw_depth)
    inputs = {
        "no_ceiling_local_fill": no_ceiling_depth,
        "ceiling_on": ceiling_depth,
    }
    frame_count = raw_depth.shape[0]
    observation = torch.zeros(frame_count, 9, device=device)
    observation[:, 0] = float(args.runtime_velocity)
    observation[:, 6] = float(args.goal_distance)

    payload = {
        "input_npz": str(args.input_npz.resolve()),
        "far_ratio": {
            key: float((value >= 0.999).float().mean().item())
            for key, value in inputs.items()
        },
        "models": {},
    }
    for checkpoint_value in args.checkpoint:
        checkpoint = Path(checkpoint_value)
        key = checkpoint.parent.parent.name
        if key in payload["models"]:
            key = str(checkpoint.resolve())
        payload["models"][key] = evaluate_model(
            checkpoint, inputs, observation, args.batch_size
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
