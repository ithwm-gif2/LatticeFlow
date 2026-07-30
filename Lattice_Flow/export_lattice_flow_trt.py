#!/usr/bin/env python3
"""Convert the fixed-NFE physical-anchor LatticeFlow policy with torch2trt.

This exporter deliberately creates one fixed-shape engine.  The runtime inputs
are the same as deployment:

    depth: [1, 1, 96, 160] normalized to [0, 1]
    obs:   [1, 9] raw body-frame velocity, acceleration, and goal vector

The output is a flattened tensor containing 135 raw residual values followed
by 15 scores.  The ROS node reshapes these back to [9,3,5] and [3,5].
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints/best.pt"
DEFAULT_OUTPUT = ROOT / "engines/lattice_flow_nfe6_fp16.trt.pth"


class ExportWrapper(nn.Module):
    def __init__(self, policy: nn.Module, nfe: int):
        super().__init__()
        self.policy = policy
        self.nfe = int(nfe)

    def forward(self, depth: torch.Tensor, obs_body: torch.Tensor) -> torch.Tensor:
        prepared_obs = self.policy.prepare_observation(obs_body)
        raw, score = self.policy(depth, prepared_obs, num_steps=self.nfe)
        return torch.cat((raw.flatten(1), score.flatten(1)), dim=1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nfe", type=int, default=6)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this exporter on the Jetson CUDA environment")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.nfe < 1:
        raise ValueError("--nfe must be positive")

    # Import after the deployment environment has supplied cfyopo on PYTHONPATH.
    from yopo_flow.checkpoint import load_policy_checkpoint

    device = torch.device("cuda")
    policy, config, _ = load_policy_checkpoint(str(args.checkpoint), device)
    if config.get("project", {}).get("policy_variant") != "physical_anchor":
        raise ValueError("The TensorRT exporter expects a physical_anchor checkpoint")
    policy.eval()
    wrapper = ExportWrapper(policy, args.nfe).to(device).eval()
    depth = torch.zeros((1, 1, 96, 160), dtype=torch.float32, device=device)
    obs = torch.zeros((1, 9), dtype=torch.float32, device=device)
    obs[:, 6] = 1.0

    with torch.inference_mode():
        reference = wrapper(depth, obs)

    if args.benchmark:
        for _ in range(20):
            wrapper(depth, obs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            wrapper(depth, obs)
        end.record()
        torch.cuda.synchronize()
        print(f"PyTorch wrapper latency: {start.elapsed_time(end) / 100.0:.3f} ms")

    from torch2trt import torch2trt

    print("Building fixed-shape TensorRT engine...")
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
        "checkpoint_sha256": sha256(args.checkpoint),
        "engine": str(args.output),
        "backend": "torch2trt",
        "precision": "fp32" if args.fp32 else "fp16",
        "nfe": args.nfe,
        "depth_shape": [1, 1, 96, 160],
        "obs_shape": [1, 9],
        "output_shape": [1, 150],
        "raw_shape": [1, 9, 3, 5],
        "score_shape": [1, 3, 5],
        "max_abs_error_against_pytorch": float(error.max().cpu()),
        "mean_abs_error_against_pytorch": float(error.mean().cpu()),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
