#!/usr/bin/env python3
"""One-batch integration test for model, teacher, costs and backpropagation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from yopo_flow.bootstrap import add_original_yopo_to_path
from yopo_flow.config import load_config, resolve_device
from yopo_flow.costs import TrajectoryCostEvaluator, robust_cost
from yopo_flow.model import LatticeFlowPolicy
from yopo_flow.targets import TargetRefiner, YOPOTeacher

add_original_yopo_to_path()

from policy.yopo_dataset import YOPODataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["runtime"]["device"])
    dataset = YOPODataset(mode="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    depth, position, rotation, obs_body, map_id = next(iter(loader))
    depth, position, rotation, obs_body, map_id = (
        item.to(device) for item in (depth, position, rotation, obs_body, map_id)
    )

    model = LatticeFlowPolicy(**config["model"]).to(device)
    teacher = YOPOTeacher(config["project"]["teacher_checkpoint"], device)
    evaluator = TrajectoryCostEvaluator(
        config["depth_safety"], config["loss_weights"]["depth_safety"]
    ).to(device)
    evaluator.requires_grad_(False)
    refiner = TargetRefiner(evaluator, config["target_refinement"])

    teacher_raw, _ = teacher(depth, obs_body)
    prepared = model.prepare_observation(obs_body)
    features = model.encode_depth(depth)
    student_seed = model.integrate_from_features(features.detach(), prepared).detach()
    refined = refiner.refine(
        teacher_raw,
        depth,
        position,
        rotation,
        obs_body,
        map_id,
        student_seed,
    )
    source = model.canonical_source(depth.shape[0], device, depth.dtype)
    t = torch.rand(depth.shape[0], 1, 3, 5, device=device)
    x_t = (1.0 - t) * source + t * refined.raw_endstate
    velocity = model.velocity_from_features(features, prepared, x_t, t)
    predicted_raw = model.integrate_from_features(features, prepared)
    predicted_score = model.score_from_features(features, prepared, predicted_raw)
    costs = evaluator(predicted_raw, depth, position, rotation, obs_body, map_id)
    loss = (
        F.mse_loss(velocity, refined.raw_endstate - source)
        + 0.25 * F.mse_loss(predicted_raw, refined.raw_endstate)
        + 0.2 * robust_cost(costs.total).mean()
        + F.smooth_l1_loss(predicted_score, robust_cost(costs.total.detach()))
    )
    loss.backward()

    assert predicted_raw.shape == (args.batch_size, 9, 3, 5)
    assert predicted_score.shape == (args.batch_size, 3, 5)
    assert torch.isfinite(loss)
    assert torch.isfinite(costs.total).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
    print(
        "SMOKE TEST PASSED",
        {
            "loss": float(loss.detach()),
            "teacher_cost": float(refined.teacher_costs.total.mean()),
            "target_cost": float(refined.costs.total.mean()),
            "student_cost": float(costs.total.mean().detach()),
        },
    )


if __name__ == "__main__":
    main()
