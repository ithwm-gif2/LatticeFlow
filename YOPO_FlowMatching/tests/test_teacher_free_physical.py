#!/usr/bin/env python3
"""Unit tests for teacher-free target construction and random initialization."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from yopo_flow.config import load_config
from yopo_flow.costs import CostBundle
from yopo_flow.physical_model import PhysicalAnchorFlowPolicy
from yopo_flow.self_targets import SelfTargetRefiner
from yopo_flow.teacher_free_trainer import zero_initialize_flow_output


class ToyEvaluator:
    """Independent per-cell quadratic cost with optimum at raw=0.25."""

    def __call__(self, raw, depth, position, rotation, obs_body, map_id):
        del depth, position, rotation, obs_body, map_id
        total = (raw - 0.25).square().mean(dim=1)
        zeros = torch.zeros_like(total)
        return CostBundle(
            total=total,
            smooth=total,
            safety=zeros,
            guidance=zeros,
            acceleration=zeros,
            depth_safety=zeros,
            min_depth_clearance=-total,
        )


def main():
    torch.manual_seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicalAnchorFlowPolicy(integration_steps=2).to(device).eval()
    zero_initialize_flow_output(model)
    depth = torch.rand(2, 1, 96, 160, device=device)
    observation = torch.zeros(2, 9, device=device)
    observation[:, 6] = 1.0
    prepared = model.prepare_observation(observation)
    features = model.encode_depth(depth)
    initial_raw = model.integrate_from_features(features, prepared)
    assert initial_raw.abs().max().item() < 2e-5

    refiner = SelfTargetRefiner(
        ToyEvaluator(),
        {
            "enabled": True,
            "include_student_seed": True,
            "student_start_epoch": 0,
            "anchor_noise_candidates": 2,
            "student_noise_candidates": 1,
            "noise_std": 0.10,
            "gradient_steps": 2,
            "gradient_step_size": 0.04,
        },
    )
    dummy_vec = torch.zeros(2, 3, device=device)
    dummy_rot = torch.eye(3, device=device).repeat(2, 1, 1)
    dummy_map = torch.zeros(2, dtype=torch.long, device=device)
    target = refiner.refine(
        depth,
        dummy_vec,
        dummy_rot,
        observation,
        dummy_map,
        student_seed=initial_raw,
        epoch=0,
    )
    assert target.costs.total.mean() <= target.selected_seed_costs.total.mean() + 1e-8
    assert target.costs.total.mean() < target.anchor_costs.total.mean()
    assert abs(sum(target.source_fractions.values()) - 1.0) < 1e-6
    assert 0.0 <= target.gradient_accept_rate <= 1.0

    config = load_config(
        PROJECT_ROOT / "configs" / "icra2027_teacher_free_physical.yaml"
    )
    assert config["project"]["initialize_backbone_from_teacher"] is False
    assert config["project"]["training_regime"] == "teacher_free_cost_refined"
    print(
        "TEACHER-FREE PHYSICAL TEST PASSED "
        f"initial_raw={initial_raw.abs().max().item():.3e} "
        f"target_cost={target.costs.total.mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
