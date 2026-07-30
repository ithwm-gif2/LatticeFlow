"""Checkpoint loading helpers shared by offline and ROS evaluation."""

from __future__ import annotations

import torch
from torch import nn

from .model import LatticeFlowPolicy
from .physical_model import PhysicalAnchorFlowPolicy


def policy_from_config(config: dict) -> nn.Module:
    variant = config.get("project", {}).get("policy_variant", "residual")
    if variant == "physical_anchor":
        return PhysicalAnchorFlowPolicy(**config["model"])
    if variant == "residual":
        return LatticeFlowPolicy(**config["model"])
    raise ValueError(f"Unknown policy_variant: {variant}")


def load_policy_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[nn.Module, dict, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "config" not in checkpoint or "model" not in checkpoint:
        raise KeyError(
            "Expected a FlowTrainer checkpoint with 'config' and 'model' entries"
        )
    config = checkpoint["config"]
    model = policy_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, checkpoint
