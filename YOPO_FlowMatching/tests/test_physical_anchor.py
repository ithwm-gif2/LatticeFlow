#!/usr/bin/env python3
"""Representation and interface tests for physical-anchor flow."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from yopo_flow.physical_model import PhysicalAnchorFlowPolicy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicalAnchorFlowPolicy(integration_steps=2).to(device).eval()
    zeros = torch.zeros(2, 9, 3, 5, device=device)
    source = model.canonical_source(2, device, torch.float32)
    decoded_source = model.raw_to_flow_state(zeros)
    assert torch.allclose(source, decoded_source, atol=1e-6)
    assert torch.allclose(model.flow_state_to_raw(source), zeros, atol=2e-5)

    positions = source[0, :3].permute(1, 2, 0).reshape(15, 3)
    assert torch.unique(positions.round(decimals=6), dim=0).shape[0] == 15

    generator = torch.Generator(device=device).manual_seed(7)
    random_raw = 0.45 * (
        2.0 * torch.rand((2, 9, 3, 5), generator=generator, device=device) - 1.0
    )
    round_trip = model.flow_state_to_raw(model.raw_to_flow_state(random_raw))
    error = (round_trip - random_raw).abs().max().item()
    assert error < 5e-5, error

    depth = torch.rand((2, 1, 96, 160), generator=generator, device=device)
    observation = torch.zeros((2, 9), device=device)
    observation[:, 6] = 1.0
    prepared = model.prepare_observation(observation)
    raw, score = model(depth, prepared)
    assert raw.shape == (2, 9, 3, 5)
    assert score.shape == (2, 3, 5)
    assert torch.isfinite(raw).all() and torch.isfinite(score).all()
    print(f"PHYSICAL ANCHOR TEST PASSED max_round_trip_error={error:.3e}")


if __name__ == "__main__":
    main()
