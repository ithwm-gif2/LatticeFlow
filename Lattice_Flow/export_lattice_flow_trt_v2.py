#!/usr/bin/env python3
"""TensorRT export for the physical-anchor LatticeFlow policy.

This variant keeps observation normalization inside a TensorRT-friendly wrapper
because the Jetson torch2trt build cannot convert the scalar clamp used by the
original Python StateTransform implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent


class ExportWrapper(nn.Module):
    def __init__(self, policy: nn.Module, nfe: int):
        super().__init__()
        self.policy = policy
        self.nfe = int(nfe)
        primitive = policy.state_transform.lattice_primitive
        self.vel_scale = float(primitive.vel_max)
        self.acc_scale = float(primitive.acc_max)
        self.goal_length = float(policy.state_transform.goal_length)
        self.register_buffer(
            "rotation_body_from_primitive",
            primitive.getRotation().flip(0).detach().clone(),
        )

    def prepare_observation(self, obs_body: torch.Tensor) -> torch.Tensor:
        velocity = obs_body[:, 0:3] / self.vel_scale
        acceleration = obs_body[:, 3:6] / self.acc_scale
        goal = obs_body[:, 6:9]
        goal_norm = torch.sqrt(torch.sum(goal * goal, dim=1, keepdim=True) + 1e-12)
        goal_floor = torch.ones_like(goal_norm) * self.goal_length
        denominator = torch.where(goal_norm > self.goal_length, goal_norm, goal_floor)
        normalized = torch.cat((velocity, acceleration, goal / denominator), dim=1)
        body_state = normalized.reshape(normalized.shape[0], 3, 3)
        rotation = self.rotation_body_from_primitive.to(
            device=body_state.device, dtype=body_state.dtype
        )
        transformed = torch.matmul(body_state[:, None], rotation[None])
        return transformed.reshape(normalized.shape[0], 15, 9).permute(0, 2, 1).reshape(
            normalized.shape[0], 9, 3, 5
        )

    def forward(self, depth: torch.Tensor, obs_body: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare_observation(obs_body)
        raw, score = self.policy(depth, prepared, num_steps=self.nfe)
        return torch.cat((raw.flatten(1), score.flatten(1)), dim=1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfe", type=int, default=6)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT conversion")
    if args.nfe < 1:
        raise ValueError("--nfe must be positive")

    from yopo_flow.checkpoint import load_policy_checkpoint

    device = torch.device("cuda")
    policy, config, _ = load_policy_checkpoint(str(args.checkpoint), device)
    if config.get("project", {}).get("policy_variant") != "physical_anchor":
        raise ValueError("Expected a physical_anchor checkpoint")
    policy.eval()
    wrapper = ExportWrapper(policy, args.nfe).to(device).eval()
    depth = torch.zeros((1, 1, 96, 160), dtype=torch.float32, device=device)
    obs = torch.zeros((1, 9), dtype=torch.float32, device=device)
    obs[:, 6] = 1.0
    with torch.inference_mode():
        reference = wrapper(depth, obs)

    from torch2trt import torch2trt

    print("Building fixed-shape LatticeFlow TensorRT engine...")
    engine = torch2trt(
        wrapper,
        [depth, obs],
        fp16_mode=not args.fp32,
        max_workspace_size=int(args.workspace_mib) * 1024 * 1024,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(engine.state_dict(), str(args.output))

    from torch2trt import TRTModule

    loaded = TRTModule()
    loaded.load_state_dict(torch.load(str(args.output), map_location=device))
    loaded = loaded.to(device).eval()
    with torch.inference_mode():
        converted = loaded(depth, obs)
        if isinstance(converted, (tuple, list)):
            converted = converted[0]
        converted = converted.reshape_as(reference)
    error = (converted - reference).abs()
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "engine": str(args.output),
        "precision": "fp32" if args.fp32 else "fp16",
        "nfe": args.nfe,
        "depth_shape": [1, 1, 96, 160],
        "obs_shape": [1, 9],
        "output_shape": [1, 150],
        "raw_shape": [1, 9, 3, 5],
        "score_shape": [1, 3, 5],
        "max_abs_error": float(error.max().cpu()),
        "mean_abs_error": float(error.mean().cpu()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
