"""ICRA trainer for physical-anchor flow matching."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .icra_trainer import ICRAFlowTrainer
from .physical_model import PhysicalAnchorFlowPolicy


class PhysicalAnchorFlowTrainer(ICRAFlowTrainer):
    """Keep the ICRA protocol fixed while replacing the internal flow space."""

    def __init__(self, config: dict, run_dir: str | None = None):
        super().__init__(config, run_dir=run_dir)
        del self.model
        self.model = PhysicalAnchorFlowPolicy(**config["model"]).to(self.device)
        if bool(config["project"].get("initialize_backbone_from_teacher", True)):
            self.model.image_backbone.load_state_dict(
                self.teacher.policy.image_backbone.state_dict()
            )
        training = config["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )

    def _flow_loss(
        self,
        depth_features: torch.Tensor,
        prepared_obs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target_physical = self.model.raw_to_flow_state(target)
        source = self.model.canonical_source(
            target.shape[0], target.device, target.dtype
        )
        if self.source_noise_std > 0:
            source = self.model._project_flow_state(
                source + self.source_noise_std * torch.randn_like(source)
            )
        time = torch.rand(
            target.shape[0],
            1,
            target.shape[2],
            target.shape[3],
            device=target.device,
            dtype=target.dtype,
        )
        state_t = (1.0 - time) * source + time * target_physical
        target_velocity = target_physical - source
        predicted_velocity = self.model.velocity_from_features(
            depth_features, prepared_obs, state_t, time
        )
        return F.mse_loss(
            predicted_velocity.float(), target_velocity.float()
        )
