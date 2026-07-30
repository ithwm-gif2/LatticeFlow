"""Differentiable YOPO trajectory costs plus a depth-image safety term."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .bootstrap import add_original_yopo_to_path

add_original_yopo_to_path()

from config.config import cfg as yopo_cfg  # noqa: E402
from loss.loss_function import YOPOLoss  # noqa: E402
from policy.state_transform import StateTransform, state_body2world  # noqa: E402


@dataclass
class CostBundle:
    total: torch.Tensor
    smooth: torch.Tensor
    safety: torch.Tensor
    guidance: torch.Tensor
    acceleration: torch.Tensor
    depth_safety: torch.Tensor
    min_depth_clearance: torch.Tensor

    def detached(self) -> "CostBundle":
        return CostBundle(**{name: value.detach() for name, value in self.__dict__.items()})


class TrajectoryCostEvaluator(nn.Module):
    """Evaluate every lattice trajectory in both global-map and image space."""

    def __init__(self, depth_config: dict, depth_safety_weight: float = 1.0):
        super().__init__()
        self.state_transform = StateTransform()
        self.yopo_loss = YOPOLoss()
        self.traj_num = int(yopo_cfg["traj_num"])
        self.vertical_num = int(yopo_cfg["vertical_num"])
        self.horizon_num = int(yopo_cfg["horizon_num"])
        self.depth_safety_weight = float(depth_safety_weight)

        self.max_depth = float(depth_config["max_depth"])
        self.fx = float(depth_config["fx"])
        self.fy = float(depth_config["fy"])
        self.cx = float(depth_config["cx"])
        self.cy = float(depth_config["cy"])
        self.safety_margin = float(depth_config["safety_margin"])
        self.temperature = float(depth_config["temperature"])
        self.trajectory_points = int(depth_config["trajectory_points"])
        camera_pitch = math.radians(float(depth_config.get("camera_pitch_deg", 0.0)))
        cosine = math.cos(camera_pitch)
        sine = math.sin(camera_pitch)
        # The dataset generator stores R_WC = R_WB R_BC with R_BC=Ry(pitch).
        # Project body-frame trajectory points with R_CB = R_BC^T.
        self.register_buffer(
            "rotation_camera_body",
            torch.tensor(
                [
                    [cosine, 0.0, -sine],
                    [0.0, 1.0, 0.0],
                    [sine, 0.0, cosine],
                ],
                dtype=torch.float32,
            ),
        )

    def _world_states(
        self,
        raw_endstate: torch.Tensor,
        position_world: torch.Tensor,
        rotation_world_body: torch.Tensor,
        obs_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = raw_endstate.shape[0]
        goal_world, start_vel_world, start_acc_world = state_body2world(
            position_world,
            rotation_world_body,
            obs_body[:, 6:9],
            obs_body[:, 0:3],
            obs_body[:, 3:6],
        )
        start_state_world = torch.stack(
            (position_world, start_vel_world, start_acc_world), dim=1
        )

        endstate_body = self.state_transform.pred_to_endstate(raw_endstate)
        endstate_flat = endstate_body.permute(0, 2, 3, 1).reshape(batch_size * self.traj_num, 9)
        position_expanded = position_world.repeat_interleave(self.traj_num, dim=0)
        rotation_expanded = rotation_world_body.repeat_interleave(self.traj_num, dim=0)
        end_pos_world, end_vel_world, end_acc_world = state_body2world(
            position_expanded,
            rotation_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9],
        )
        end_state_world = torch.stack(
            (end_pos_world, end_vel_world, end_acc_world), dim=1
        )
        return start_state_world, end_state_world, goal_world

    def _body_polynomial_points(
        self,
        raw_endstate: torch.Tensor,
        obs_body: torch.Tensor,
    ) -> torch.Tensor:
        """Return body-frame trajectory samples shaped ``[B, N, T, 3]``."""

        batch_size = raw_endstate.shape[0]
        endstate_body = self.state_transform.pred_to_endstate(raw_endstate)
        endstate_flat = endstate_body.permute(0, 2, 3, 1).reshape(batch_size * self.traj_num, 9)
        end_state = torch.stack(
            (endstate_flat[:, 0:3], endstate_flat[:, 3:6], endstate_flat[:, 6:9]),
            dim=1,
        )

        zero_position = torch.zeros_like(obs_body[:, 0:3])
        start_state = torch.stack(
            (zero_position, obs_body[:, 0:3], obs_body[:, 3:6]), dim=1
        ).repeat_interleave(self.traj_num, dim=0)

        fixed = start_state.permute(0, 2, 1)
        decision = end_state.permute(0, 2, 1)
        lattice_matrix = self.yopo_loss._L.unsqueeze(0).expand(decision.shape[0], -1, -1)
        coefficients = self.yopo_loss.safety_loss.get_coefficient_from_derivative(
            decision, fixed, lattice_matrix
        )
        dt = float(yopo_cfg["sgm_time"]) / self.trajectory_points
        times = torch.linspace(
            dt,
            float(yopo_cfg["sgm_time"]),
            self.trajectory_points,
            device=raw_endstate.device,
            dtype=raw_endstate.dtype,
        )
        times = times.view(1, -1, 1).expand(coefficients.shape[0], -1, -1)
        points = self.yopo_loss.safety_loss.get_position_from_coeff(coefficients, times)
        return points.reshape(batch_size, self.traj_num, self.trajectory_points, 3)

    def depth_image_cost(
        self,
        raw_endstate: torch.Tensor,
        depth: torch.Tensor,
        obs_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare projected trajectory depth with the observed depth image.

        YOPO's camera frame uses +x forward, +y left and +z up. The raycaster
        therefore projects with ``u = cx - fx*y/x`` and ``v = cy - fy*z/x``.
        """

        batch_size = raw_endstate.shape[0]
        height, width = depth.shape[-2:]
        points_body = self._body_polynomial_points(raw_endstate, obs_body)
        rotation_camera_body = self.rotation_camera_body.to(
            device=points_body.device, dtype=points_body.dtype
        )
        points_camera = torch.matmul(
            points_body, rotation_camera_body.transpose(0, 1)
        )
        points_flat = points_camera.reshape(
            batch_size * self.traj_num, self.trajectory_points, 3
        )

        forward = points_flat[..., 0]
        safe_forward = forward.clamp_min(1e-3)
        pixel_u = self.cx - self.fx * points_flat[..., 1] / safe_forward
        pixel_v = self.cy - self.fy * points_flat[..., 2] / safe_forward
        grid_u = 2.0 * pixel_u / max(width - 1, 1) - 1.0
        grid_v = 2.0 * pixel_v / max(height - 1, 1) - 1.0
        grid = torch.stack((grid_u, grid_v), dim=-1).view(
            batch_size * self.traj_num, 1, self.trajectory_points, 2
        )

        depth_expanded = depth.repeat_interleave(self.traj_num, dim=0)
        observed_depth = F.grid_sample(
            depth_expanded,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(batch_size * self.traj_num, self.trajectory_points)
        observed_depth = observed_depth * self.max_depth

        in_view = (
            (grid_u.abs() <= 1.0)
            & (grid_v.abs() <= 1.0)
            & (forward > 0.0)
        )
        clearance = observed_depth - forward
        penalty = F.softplus(
            (self.safety_margin - clearance) / self.temperature
        ) * self.temperature
        valid_count = in_view.sum(dim=1).clamp_min(1)
        depth_cost = (penalty * in_view).sum(dim=1) / valid_count

        infinity = torch.full_like(clearance, float("inf"))
        min_clearance = torch.where(in_view, clearance, infinity).amin(dim=1)
        min_clearance = torch.where(
            torch.isfinite(min_clearance), min_clearance, torch.full_like(min_clearance, self.max_depth)
        )
        shape = (batch_size, self.vertical_num, self.horizon_num)
        return depth_cost.reshape(shape), min_clearance.reshape(shape)

    def forward(
        self,
        raw_endstate: torch.Tensor,
        depth: torch.Tensor,
        position_world: torch.Tensor,
        rotation_world_body: torch.Tensor,
        obs_body: torch.Tensor,
        map_id: torch.Tensor,
    ) -> CostBundle:
        batch_size = raw_endstate.shape[0]
        start_state, end_state, goal_world = self._world_states(
            raw_endstate, position_world, rotation_world_body, obs_body
        )
        start_expanded = start_state.repeat_interleave(self.traj_num, dim=0)
        goal_expanded = goal_world.repeat_interleave(self.traj_num, dim=0)
        smooth, safety, guidance, acceleration = self.yopo_loss(
            start_expanded, end_state, goal_expanded, map_id
        )
        shape = (batch_size, self.vertical_num, self.horizon_num)
        smooth = smooth.reshape(shape)
        safety = safety.reshape(shape)
        guidance = guidance.reshape(shape)
        acceleration = acceleration.reshape(shape)
        depth_safety, min_depth_clearance = self.depth_image_cost(raw_endstate, depth, obs_body)
        total = (
            smooth
            + safety
            + guidance
            + acceleration
            + self.depth_safety_weight * depth_safety
        )
        return CostBundle(
            total=total,
            smooth=smooth,
            safety=safety,
            guidance=guidance,
            acceleration=acceleration,
            depth_safety=depth_safety,
            min_depth_clearance=min_depth_clearance,
        )


def gather_lattice(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather one flattened lattice value per batch item."""

    return values.reshape(values.shape[0], -1).gather(1, indices[:, None]).squeeze(1)


def robust_cost(values: torch.Tensor, maximum: float = 10_000.0) -> torch.Tensor:
    """Log-compress rare ESDF exponent outliers without changing cost ordering."""

    return torch.log1p(values.clamp(min=0.0, max=maximum))
