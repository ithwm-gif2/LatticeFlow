#!/usr/bin/env python3
"""Deterministic checks for sparse LiDAR no-return domain adaptation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from yopo_flow.lidar_domain import LidarNoReturnAdapter


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = LidarNoReturnAdapter(
        {
            "enabled": True,
            "probability": 1.0,
            "target_far_ratio": [0.68, 0.80],
            "validation_far_ratio": 0.76,
            "noise_std": 0.0,
            "range_noise_std": 0.0,
            "quantization": 0.0,
        }
    )
    depth = torch.linspace(
        0.05, 0.95, 96 * 160, device=device, dtype=torch.float32
    ).reshape(1, 1, 96, 160).repeat(3, 1, 1, 1)
    depth[:, :, :4, :] = 1.0
    adapted_a, stats_a = adapter(depth, deterministic=True)
    adapted_b, stats_b = adapter(depth, deterministic=True)
    assert torch.equal(adapted_a, adapted_b)
    assert torch.isfinite(adapted_a).all()
    assert adapted_a.shape == depth.shape
    assert abs(stats_a.output_far_ratio.item() - 0.76) < 0.002
    assert abs(stats_b.output_far_ratio.item() - 0.76) < 0.002
    assert torch.all(adapted_a[(adapted_a < 0.999)] < 0.999)
    print(
        "LIDAR DOMAIN TEST PASSED",
        {
            "input_far": stats_a.input_far_ratio.item(),
            "output_far": stats_a.output_far_ratio.item(),
            "retained": stats_a.retained_nonfar_ratio.item(),
        },
    )


if __name__ == "__main__":
    main()
