"""Flow policy whose ODE state is a normalized physical terminal state."""

from __future__ import annotations

import torch

from .model import LatticeFlowPolicy


from config.config import cfg as yopo_cfg  # noqa: E402
from policy.primitive import LatticePrimitive  # noqa: E402


class PhysicalAnchorFlowPolicy(LatticeFlowPolicy):
    """Transport explicit physical lattice anchors to physical terminal states.

    The internal flow coordinates are normalized body-frame terminal position,
    velocity, and acceleration.  The public interface remains YOPO-compatible:
    ``forward`` and ``integrate_from_features`` return normalized YOPO
    residuals shaped ``[B, 9, 3, 5]``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        primitive = LatticePrimitive.get_instance()
        self.position_scale = float(2.0 * primitive.radio_range)
        component_rotation_bound = 3.0**0.5
        # The ODE represents absolute physical terminal state and therefore
        # keeps training-time units fixed.  The inverse map separately divides
        # by the runtime primitive limits to retain YOPO speed scaling in ROS.
        self.velocity_scale = float(
            component_rotation_bound * yopo_cfg["vel_max_train"]
        )
        self.acceleration_scale = float(
            component_rotation_bound * yopo_cfg["acc_max_train"]
        )
        self.primitive_velocity_scale = float(primitive.vel_max)
        self.primitive_acceleration_scale = float(primitive.acc_max)
        self.radio_range = float(primitive.radio_range)
        self.yaw_diff = float(primitive.yaw_diff)
        self.pitch_diff = float(primitive.pitch_diff)

        primitive_device = primitive.getStateLattice().device
        zero_raw = torch.zeros(
            1,
            self.state_dim,
            self.vertical_num,
            self.horizon_num,
            device=primitive_device,
            dtype=torch.float32,
        )
        anchor_body = self.state_transform.pred_to_endstate(zero_raw)
        anchor_flow = self._normalize_physical(anchor_body)
        anchor_position = anchor_body[:, :3]
        anchor_radius = anchor_position.square().sum(dim=1).sqrt().clamp_min(1e-6)
        anchor_yaw = torch.atan2(anchor_position[:, 1], anchor_position[:, 0])
        anchor_pitch = torch.asin(
            (anchor_position[:, 2] / anchor_radius).clamp(-1.0, 1.0)
        )

        rotation = primitive.getRotation().flip(0).reshape(
            self.vertical_num, self.horizon_num, 3, 3
        )
        self.register_buffer("physical_anchor_source", anchor_flow.detach().clone())
        self.register_buffer("anchor_yaw", anchor_yaw.detach().clone())
        self.register_buffer("anchor_pitch", anchor_pitch.detach().clone())
        self.register_buffer(
            "rotation_body_from_primitive", rotation.detach().clone()
        )

    def _normalize_physical(self, endstate_body: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                endstate_body[:, 0:3] / self.position_scale,
                endstate_body[:, 3:6] / self.velocity_scale,
                endstate_body[:, 6:9] / self.acceleration_scale,
            ),
            dim=1,
        )

    def raw_to_flow_state(self, raw_endstate: torch.Tensor) -> torch.Tensor:
        """Convert normalized YOPO residuals to normalized physical state."""

        return self._normalize_physical(
            self.state_transform.pred_to_endstate(raw_endstate)
        )

    def flow_state_to_raw(self, flow_state: torch.Tensor) -> torch.Tensor:
        """Convert normalized physical state back to YOPO residuals."""

        position = flow_state[:, 0:3] * self.position_scale
        radius = position.square().sum(dim=1).sqrt().clamp_min(1e-6)
        yaw = torch.atan2(position[:, 1], position[:, 0])
        pitch = torch.asin((position[:, 2] / radius).clamp(-1.0, 1.0))
        yaw_delta = torch.atan2(
            torch.sin(yaw - self.anchor_yaw),
            torch.cos(yaw - self.anchor_yaw),
        )
        raw_yaw = yaw_delta / self.yaw_diff
        raw_pitch = (pitch - self.anchor_pitch) / self.pitch_diff
        raw_radius = radius / self.radio_range - 1.0

        rotation = self.rotation_body_from_primitive.to(
            device=flow_state.device, dtype=flow_state.dtype
        )
        rotation_t = rotation.transpose(-1, -2).unsqueeze(0)

        def body_to_primitive(vector: torch.Tensor) -> torch.Tensor:
            vector_grid = vector.permute(0, 2, 3, 1).unsqueeze(-1)
            primitive_grid = torch.matmul(rotation_t, vector_grid).squeeze(-1)
            return primitive_grid.permute(0, 3, 1, 2)

        raw_velocity = body_to_primitive(flow_state[:, 3:6]) * (
            self.velocity_scale / self.primitive_velocity_scale
        )
        raw_acceleration = body_to_primitive(flow_state[:, 6:9]) * (
            self.acceleration_scale / self.primitive_acceleration_scale
        )
        raw = torch.cat(
            (
                raw_yaw[:, None],
                raw_pitch[:, None],
                raw_radius[:, None],
                raw_velocity,
                raw_acceleration,
            ),
            dim=1,
        )
        return raw.clamp(-1.0, 1.0)

    @staticmethod
    def _project_flow_state(flow_state: torch.Tensor) -> torch.Tensor:
        position = flow_state[:, 0:3]
        norm = position.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        position = position / torch.maximum(norm, torch.ones_like(norm))
        velocity_acceleration = flow_state[:, 3:9].clamp(-1.0, 1.0)
        return torch.cat((position, velocity_acceleration), dim=1)

    def canonical_source(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.physical_anchor_source.to(device=device, dtype=dtype).expand(
            batch_size, -1, -1, -1
        )

    def integrate_flow_from_features(
        self,
        depth_features: torch.Tensor,
        prepared_obs: torch.Tensor,
        num_steps: int | None = None,
        return_trace: bool = False,
    ):
        steps = int(num_steps or self.integration_steps)
        if steps < 1:
            raise ValueError("integration steps must be positive")
        state = self.canonical_source(
            depth_features.shape[0], depth_features.device, depth_features.dtype
        )
        dt = 1.0 / steps
        trace = [state] if return_trace else None
        for step in range(steps):
            time = torch.full(
                (state.shape[0], 1, self.vertical_num, self.horizon_num),
                step / steps,
                device=state.device,
                dtype=state.dtype,
            )
            state = state + dt * self.velocity_from_features(
                depth_features, prepared_obs, state, time
            )
            if self.clamp_state:
                state = self._project_flow_state(state)
            if trace is not None:
                trace.append(state)
        if return_trace:
            return state, trace
        return state

    def integrate_from_features(
        self,
        depth_features: torch.Tensor,
        prepared_obs: torch.Tensor,
        num_steps: int | None = None,
        return_trace: bool = False,
    ):
        if return_trace:
            physical, trace = self.integrate_flow_from_features(
                depth_features,
                prepared_obs,
                num_steps=num_steps,
                return_trace=True,
            )
            return self.flow_state_to_raw(physical), trace
        physical = self.integrate_flow_from_features(
            depth_features, prepared_obs, num_steps=num_steps
        )
        return self.flow_state_to_raw(physical)
