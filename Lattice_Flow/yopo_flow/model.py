"""Depth-and-state conditioned flow policy with the original YOPO interface."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F



from config.config import cfg as yopo_cfg  # noqa: E402
from policy.models.backbone import YopoBackbone  # noqa: E402
from policy.state_transform import StateTransform  # noqa: E402


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32, max_period: int = 10_000):
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed ``t`` with shape ``[B, 1, V, H]``."""

        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        angles = t * frequencies.view(1, half, 1, 1)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class PointwiseMLP(nn.Module):
    """A shared MLP applied to every lattice cell through 1x1 convolutions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 3):
        super().__init__()
        modules: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(max(layers - 1, 1)):
            modules.extend(
                [
                    nn.Conv2d(current_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                ]
            )
            current_dim = hidden_dim
        modules.append(nn.Conv2d(current_dim, output_dim, kernel_size=1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatticeFlowPolicy(nn.Module):
    """Flow from canonical lattice offsets to optimized YOPO end states.

    The external interface intentionally matches ``YopoNetwork``:

    ``forward(depth, prepared_obs) -> raw_endstate, score``

    ``raw_endstate`` has shape ``[B, 9, V, H]`` and remains in YOPO's
    normalized offset representation. ``score`` has shape ``[B, V, H]``.
    """

    def __init__(
        self,
        image_feature_dim: int = 64,
        observation_dim: int = 9,
        state_dim: int = 9,
        hidden_dim: int = 256,
        time_dim: int = 32,
        lattice_embed_dim: int = 16,
        flow_layers: int = 4,
        integration_steps: int = 6,
        clamp_state: bool = True,
    ):
        super().__init__()
        self.integration_steps = integration_steps
        self.clamp_state = clamp_state
        self.state_dim = state_dim
        self.vertical_num = int(yopo_cfg["vertical_num"])
        self.horizon_num = int(yopo_cfg["horizon_num"])
        self.state_transform = StateTransform()

        self.image_backbone = YopoBackbone(image_feature_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.lattice_embedding = nn.Parameter(
            torch.zeros(1, lattice_embed_dim, self.vertical_num, self.horizon_num)
        )
        nn.init.normal_(self.lattice_embedding, std=0.02)

        flow_input_dim = image_feature_dim + observation_dim + state_dim + time_dim + lattice_embed_dim
        self.flow_head = PointwiseMLP(flow_input_dim, hidden_dim, state_dim, layers=flow_layers)

        score_input_dim = image_feature_dim + observation_dim + state_dim + lattice_embed_dim
        self.score_head = PointwiseMLP(score_input_dim, hidden_dim, 1, layers=3)

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        features = self.image_backbone(depth)
        expected = (self.vertical_num, self.horizon_num)
        if features.shape[-2:] != expected:
            raise RuntimeError(
                f"Depth backbone produced {tuple(features.shape[-2:])}, expected lattice grid {expected}. "
                "Check image_height/image_width and the YOPO lattice configuration."
            )
        return features

    def canonical_source(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Raw offset zero means the center direction, radio_range endpoint and
        # zero terminal velocity/acceleration for each YOPO lattice cell.
        return torch.zeros(
            batch_size,
            self.state_dim,
            self.vertical_num,
            self.horizon_num,
            device=device,
            dtype=dtype,
        )

    def velocity_from_features(
        self,
        depth_features: torch.Tensor,
        prepared_obs: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None, None, None]
        if t.ndim != 4 or t.shape[1] != 1:
            raise ValueError(f"Expected t shaped [B] or [B,1,V,H], got {tuple(t.shape)}")
        if t.shape[-2:] == (1, 1):
            t = t.expand(-1, -1, self.vertical_num, self.horizon_num)
        time_features = self.time_embedding(t)
        lattice_features = self.lattice_embedding.expand(depth_features.shape[0], -1, -1, -1)
        flow_input = torch.cat(
            (depth_features, prepared_obs, x_t, time_features, lattice_features), dim=1
        )
        return self.flow_head(flow_input)

    def integrate_from_features(
        self,
        depth_features: torch.Tensor,
        prepared_obs: torch.Tensor,
        num_steps: int | None = None,
        return_trace: bool = False,
    ):
        steps = int(num_steps or self.integration_steps)
        if steps < 1:
            raise ValueError("integration steps must be positive")
        x = self.canonical_source(
            depth_features.shape[0], depth_features.device, depth_features.dtype
        )
        dt = 1.0 / steps
        trace = [x] if return_trace else None
        for step in range(steps):
            t = torch.full(
                (x.shape[0], 1, self.vertical_num, self.horizon_num),
                step / steps,
                device=x.device,
                dtype=x.dtype,
            )
            x = x + dt * self.velocity_from_features(depth_features, prepared_obs, x, t)
            if self.clamp_state:
                x = x.clamp(-1.0, 1.0)
            if trace is not None:
                trace.append(x)
        if return_trace:
            return x, trace
        return x

    def score_from_features(
        self,
        depth_features: torch.Tensor,
        prepared_obs: torch.Tensor,
        raw_endstate: torch.Tensor,
    ) -> torch.Tensor:
        lattice_features = self.lattice_embedding.expand(depth_features.shape[0], -1, -1, -1)
        score_input = torch.cat(
            (depth_features, prepared_obs, raw_endstate, lattice_features), dim=1
        )
        return F.softplus(self.score_head(score_input).squeeze(1))

    def forward(
        self,
        depth: torch.Tensor,
        prepared_obs: torch.Tensor,
        num_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depth_features = self.encode_depth(depth)
        raw_endstate = self.integrate_from_features(depth_features, prepared_obs, num_steps=num_steps)
        score = self.score_from_features(depth_features, prepared_obs, raw_endstate)
        return raw_endstate, score

    def prepare_observation(self, obs_body: torch.Tensor) -> torch.Tensor:
        normalized = self.state_transform.normalize_obs(obs_body.clone())
        return self.state_transform.prepare_input(normalized)

    def inference(
        self,
        depth: torch.Tensor,
        obs_body: torch.Tensor,
        num_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared_obs = self.prepare_observation(obs_body)
        raw_endstate, score = self.forward(depth, prepared_obs, num_steps=num_steps)
        return self.state_transform.pred_to_endstate(raw_endstate), score
