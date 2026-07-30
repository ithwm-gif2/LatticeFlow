"""Training-time adaptation from dense simulator depth to sparse LiDAR depth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class LidarDomainStats:
    applied_fraction: torch.Tensor
    input_far_ratio: torch.Tensor
    output_far_ratio: torch.Tensor
    retained_nonfar_ratio: torch.Tensor


class LidarNoReturnAdapter:
    """Simulate the post-projection no-return pattern seen on the MID-360.

    The adapter operates on normalized depth in ``[0, 1]`` and uses ``1`` as
    the 20 m no-return value.  A structured priority field selects a fixed
    number of valid pixels so every augmented sample reaches the configured
    far-pixel ratio.  Dense depth remains available to privileged trajectory
    costs; only the policy perception input is changed.
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.probability = float(config.get("probability", 1.0))
        far_range = config.get("target_far_ratio", [0.68, 0.80])
        self.far_min = float(far_range[0])
        self.far_max = float(far_range[1])
        self.validation_far_ratio = float(
            config.get("validation_far_ratio", 0.76)
        )
        self.far_threshold = float(config.get("far_threshold", 0.999))
        self.block_height = int(config.get("block_height", 8))
        self.block_width = int(config.get("block_width", 10))
        self.pixel_priority_weight = float(
            config.get("pixel_priority_weight", 0.35)
        )
        self.block_priority_weight = float(
            config.get("block_priority_weight", 0.35)
        )
        self.scanline_priority_weight = float(
            config.get("scanline_priority_weight", 0.20)
        )
        self.near_priority_weight = float(
            config.get("near_priority_weight", 0.10)
        )
        self.noise_std = float(config.get("noise_std", 0.003))
        self.range_noise_std = float(config.get("range_noise_std", 0.006))
        self.quantization = float(config.get("quantization", 1.0 / 1024.0))
        self.template_priority_weight = float(
            config.get("template_priority_weight", 0.0)
        )
        self.template_probability = float(config.get("template_probability", 1.0))
        self.mask_templates = self._load_mask_templates(
            config.get("mask_template_paths", []),
            str(config.get("mask_template_key", "depth")),
        )
        self.exact_template_pipeline = bool(
            config.get("exact_template_pipeline", False)
        )
        self.local_hole_kernel = int(config.get("local_hole_kernel", 3))
        self.local_hole_min_neighbors = float(
            config.get("local_hole_min_neighbors", 5)
        )
        self.local_hole_iterations = int(config.get("local_hole_iterations", 1))
        self.virtual_ceiling_enabled = bool(
            config.get("virtual_ceiling_enabled", False)
        )
        self.virtual_ceiling_world_z = float(
            config.get("virtual_ceiling_world_z", 2.0)
        )
        self.virtual_ceiling_stride = int(
            config.get("virtual_ceiling_stride", 2)
        )
        self.virtual_ceiling_min_forward = float(
            config.get("virtual_ceiling_min_forward", 0.4)
        )
        self.virtual_ceiling_max_forward = float(
            config.get("virtual_ceiling_max_forward", 0.0)
        )
        self.projection_pitch_deg = float(
            config.get("projection_pitch_deg", 22.5)
        )
        self.projection_horizontal_fov_deg = float(
            config.get("projection_horizontal_fov_deg", 90.0)
        )
        self.projection_vertical_fov_deg = float(
            config.get("projection_vertical_fov_deg", 59.0)
        )
        self.max_depth = float(config.get("max_depth", 20.0))
        self.min_depth = float(config.get("min_depth", 0.05))

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("lidar_domain.probability must be in [0, 1]")
        if not 0.0 <= self.far_min <= self.far_max < 1.0:
            raise ValueError("lidar_domain.target_far_ratio must lie in [0, 1)")
        if not 0.0 <= self.validation_far_ratio < 1.0:
            raise ValueError("lidar_domain.validation_far_ratio must lie in [0, 1)")
        if not 0.0 <= self.template_probability <= 1.0:
            raise ValueError("lidar_domain.template_probability must be in [0, 1]")
        if self.local_hole_kernel < 1 or self.local_hole_kernel % 2 == 0:
            raise ValueError("lidar_domain.local_hole_kernel must be positive and odd")
        if self.local_hole_iterations < 0:
            raise ValueError("lidar_domain.local_hole_iterations must be non-negative")
        if self.exact_template_pipeline and self.mask_templates is None:
            raise ValueError(
                "exact_template_pipeline requires at least one mask_template_path"
            )

    def _load_mask_templates(
        self,
        paths: list[str],
        key: str,
    ) -> torch.Tensor | None:
        templates = []
        for value in paths:
            path = Path(value).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"LiDAR mask template does not exist: {path}")
            with np.load(path) as archive:
                if key not in archive:
                    raise KeyError(f"{path} has no mask template key {key!r}")
                depth = np.asarray(archive[key], dtype=np.float32)
            if depth.ndim == 4:
                depth = depth[:, 0]
            elif depth.ndim == 2:
                depth = depth[None]
            if depth.ndim != 3:
                raise ValueError(
                    f"Expected LiDAR template [N,H,W] or [N,1,H,W], got {depth.shape}"
                )
            valid = np.isfinite(depth) & (depth > 0.0) & (
                depth < self.far_threshold
            )
            templates.append(torch.from_numpy(valid.astype(np.float32))[:, None])
        if not templates:
            return None
        spatial_shapes = {tuple(template.shape[-2:]) for template in templates}
        if len(spatial_shapes) != 1:
            raise ValueError(
                f"All LiDAR mask templates must share one resolution, got {spatial_shapes}"
            )
        return torch.cat(templates, dim=0).contiguous()

    @staticmethod
    def _deterministic_noise(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        phase: float,
    ) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        value = torch.sin(xx * 12.9898 + yy * 78.233 + phase) * 43758.5453
        return value - torch.floor(value)

    def _priority(
        self,
        depth: torch.Tensor,
        deterministic: bool,
    ) -> torch.Tensor:
        batch, _, height, width = depth.shape
        if deterministic:
            pixel_rows = [
                self._deterministic_noise(
                    height, width, depth.device, depth.dtype, 19.19 + index * 7.7
                )
                for index in range(batch)
            ]
            pixel = torch.stack(pixel_rows, dim=0)[:, None]
            low_rows = max(1, height // max(self.block_height, 1))
            low_cols = max(1, width // max(self.block_width, 1))
            block_rows = [
                self._deterministic_noise(
                    low_rows,
                    low_cols,
                    depth.device,
                    depth.dtype,
                    43.7 + index * 11.3,
                )
                for index in range(batch)
            ]
            block = torch.stack(block_rows, dim=0)[:, None]
        else:
            pixel = torch.rand_like(depth)
            block = torch.rand(
                batch,
                1,
                max(1, height // max(self.block_height, 1)),
                max(1, width // max(self.block_width, 1)),
                device=depth.device,
                dtype=depth.dtype,
            )
        block = F.interpolate(
            block, size=(height, width), mode="bilinear", align_corners=False
        )

        rows = torch.arange(height, device=depth.device, dtype=depth.dtype).view(
            1, 1, height, 1
        )
        if deterministic:
            phase = torch.arange(
                batch, device=depth.device, dtype=depth.dtype
            ).view(batch, 1, 1, 1)
            period = 3.5
        else:
            phase = torch.rand(
                batch, 1, 1, 1, device=depth.device, dtype=depth.dtype
            ) * 6.283185307179586
            period = 3.0 + 2.0 * torch.rand(
                batch, 1, 1, 1, device=depth.device, dtype=depth.dtype
            )
        scanline = 0.5 + 0.5 * torch.cos(
            6.283185307179586 * rows / period + phase
        )
        scanline = scanline.expand(batch, 1, height, width)
        near_priority = 1.0 - depth.clamp(0.0, 1.0)
        priority = (
            self.pixel_priority_weight * pixel
            + self.block_priority_weight * block
            + self.scanline_priority_weight * scanline
            + self.near_priority_weight * near_priority
        )
        if self.mask_templates is not None and self.template_priority_weight > 0.0:
            templates = self.mask_templates.to(device=depth.device, dtype=depth.dtype)
            if templates.shape[-2:] != (height, width):
                templates = F.interpolate(
                    templates, size=(height, width), mode="nearest"
                )
            if deterministic:
                indices = torch.arange(batch, device=depth.device) % templates.shape[0]
                apply_template = torch.ones(
                    batch, 1, 1, 1, device=depth.device, dtype=depth.dtype
                )
            else:
                indices = torch.randint(
                    templates.shape[0], (batch,), device=depth.device
                )
                apply_template = (
                    torch.rand(batch, 1, 1, 1, device=depth.device)
                    < self.template_probability
                ).to(depth.dtype)
            template = templates[indices]
            priority = priority + (
                self.template_priority_weight * apply_template * template
            )
        return priority

    def _template_batch(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        deterministic: bool,
    ) -> torch.Tensor:
        if self.mask_templates is None:
            raise RuntimeError("No LiDAR mask templates are loaded")
        templates = self.mask_templates.to(device=device, dtype=dtype)
        if templates.shape[-2:] != (height, width):
            templates = F.interpolate(
                templates, size=(height, width), mode="nearest"
            )
        if deterministic:
            indices = torch.arange(batch, device=device) % templates.shape[0]
        else:
            indices = torch.randint(templates.shape[0], (batch,), device=device)
        return templates[indices]

    def _local_hole_fill(self, depth: torch.Tensor) -> torch.Tensor:
        result = depth
        if self.local_hole_iterations <= 0:
            return result
        kernel_size = self.local_hole_kernel
        padding = kernel_size // 2
        ones = torch.ones(
            1,
            1,
            kernel_size,
            kernel_size,
            device=depth.device,
            dtype=depth.dtype,
        )
        min_normalized = self.min_depth / self.max_depth
        for _ in range(self.local_hole_iterations):
            valid = (
                torch.isfinite(result)
                & (result > min_normalized)
                & (result < self.far_threshold)
            )
            neighbor_count = F.conv2d(
                valid.to(result.dtype), ones, padding=padding
            )
            padded = F.pad(
                result,
                (padding, padding, padding, padding),
                mode="constant",
                value=1.0,
            )
            local_minimum = -F.max_pool2d(
                -padded, kernel_size=kernel_size, stride=1
            )
            fill = (
                (~valid)
                & (neighbor_count >= self.local_hole_min_neighbors)
                & torch.isfinite(local_minimum)
                & (local_minimum > min_normalized)
                & (local_minimum < self.far_threshold)
            )
            result = torch.where(fill, local_minimum, result)
        return result

    def _apply_virtual_ceiling(
        self,
        depth: torch.Tensor,
        position_world: torch.Tensor,
        rotation_world_body: torch.Tensor,
    ) -> torch.Tensor:
        if not self.virtual_ceiling_enabled:
            return depth
        batch, _, height, width = depth.shape
        dtype, device = depth.dtype, depth.device
        fx = (width / 2.0) / math.tan(
            math.radians(self.projection_horizontal_fov_deg) / 2.0
        )
        fy = (height / 2.0) / math.tan(
            math.radians(self.projection_vertical_fov_deg) / 2.0
        )
        cx, cy = width / 2.0, height / 2.0
        rows, cols = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        left = (cx - cols) / fx
        up = (cy - rows) / fy
        forward = torch.ones_like(left)
        pitch = math.radians(self.projection_pitch_deg)
        sine, cosine = math.sin(pitch), math.cos(pitch)
        ray_body = torch.stack(
            (
                cosine * forward - sine * up,
                left,
                sine * forward + cosine * up,
            ),
            dim=-1,
        )
        ray_world = torch.einsum(
            "bij,hwj->bhwi", rotation_world_body.to(dtype), ray_body
        )
        ray_vertical = ray_world[..., 2]
        plane_delta = (
            self.virtual_ceiling_world_z - position_world[:, 2]
        ).view(batch, 1, 1)
        positive = (ray_vertical > 1.0e-3) & (plane_delta > 0.0)
        infinity = torch.full_like(ray_vertical, float("inf"))
        scale = plane_delta / torch.where(positive, ray_vertical, infinity)
        x_body = scale * ray_body[..., 0]
        sample = torch.zeros(height, width, device=device, dtype=torch.bool)
        stride = max(1, self.virtual_ceiling_stride)
        sample[::stride, ::stride] = True
        max_forward = (
            self.max_depth
            if self.virtual_ceiling_max_forward <= 0.0
            else min(self.max_depth, self.virtual_ceiling_max_forward)
        )
        valid = (
            positive
            & sample[None]
            & (scale > self.min_depth)
            & (scale < self.max_depth)
            & (x_body > self.virtual_ceiling_min_forward)
            & (x_body < max_forward)
        )
        ceiling_depth = (scale / self.max_depth).clamp(0.0, 1.0)[:, None]
        return torch.where(valid[:, None], torch.minimum(depth, ceiling_depth), depth)

    def _exact_template_adaptation(
        self,
        depth: torch.Tensor,
        position_world: torch.Tensor | None,
        rotation_world_body: torch.Tensor | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, LidarDomainStats]:
        batch, _, height, width = depth.shape
        if self.virtual_ceiling_enabled and (
            position_world is None or rotation_world_body is None
        ):
            raise ValueError(
                "Exact LiDAR adaptation with a world ceiling requires position and rotation"
            )
        template = self._template_batch(
            batch, height, width, depth.device, depth.dtype, deterministic
        )
        if deterministic:
            apply = torch.ones(batch, dtype=torch.bool, device=depth.device)
        else:
            apply = (
                (torch.rand(batch, device=depth.device) < self.probability)
                & (
                    torch.rand(batch, device=depth.device)
                    < self.template_probability
                )
            )
        keep = template >= 0.5
        keep = torch.where(
            apply.view(batch, 1, 1, 1), keep, torch.ones_like(keep)
        )
        dense_valid = torch.isfinite(depth) & (depth > 0.0) & (
            depth < self.far_threshold
        )
        sparse = torch.where(keep & dense_valid, depth, torch.ones_like(depth))
        sparse = self._local_hole_fill(sparse)
        if self.virtual_ceiling_enabled:
            sparse = self._apply_virtual_ceiling(
                sparse, position_world, rotation_world_body
            )

        valid = sparse < self.far_threshold
        noisy = sparse
        if self.noise_std > 0.0 or self.range_noise_std > 0.0:
            if deterministic:
                unit_noise = self._deterministic_noise(
                    height, width, depth.device, depth.dtype, 97.3
                )[None, None].expand(batch, -1, -1, -1) * 2.0 - 1.0
            else:
                unit_noise = torch.randn_like(depth)
            scale = self.noise_std + self.range_noise_std * sparse
            noisy = sparse + unit_noise * scale
        if self.quantization > 0.0:
            noisy = torch.round(noisy / self.quantization) * self.quantization
        sparse = torch.where(valid, noisy.clamp(0.0, 1.0), torch.ones_like(noisy))

        input_far = depth >= self.far_threshold
        output_far = sparse >= self.far_threshold
        retained = (keep & dense_valid).float().sum() / dense_valid.float().sum().clamp_min(1.0)
        stats = LidarDomainStats(
            applied_fraction=apply.float().mean(),
            input_far_ratio=input_far.float().mean(),
            output_far_ratio=output_far.float().mean(),
            retained_nonfar_ratio=retained,
        )
        return sparse, stats

    def __call__(
        self,
        depth: torch.Tensor,
        *,
        position_world: torch.Tensor | None = None,
        rotation_world_body: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, LidarDomainStats]:
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(f"Expected depth [B,1,H,W], got {tuple(depth.shape)}")
        input_far = depth >= self.far_threshold
        if not self.enabled:
            ratio = input_far.float().mean()
            stats = LidarDomainStats(
                applied_fraction=depth.new_zeros(()),
                input_far_ratio=ratio,
                output_far_ratio=ratio,
                retained_nonfar_ratio=(~input_far).float().mean(),
            )
            return depth, stats

        if self.exact_template_pipeline:
            return self._exact_template_adaptation(
                depth,
                position_world,
                rotation_world_body,
                deterministic,
            )

        batch, _, height, width = depth.shape
        total_pixels = height * width
        if deterministic:
            apply = torch.ones(batch, dtype=torch.bool, device=depth.device)
            target_far = depth.new_full((batch,), self.validation_far_ratio)
        else:
            apply = torch.rand(batch, device=depth.device) < self.probability
            target_far = self.far_min + (self.far_max - self.far_min) * torch.rand(
                batch, device=depth.device, dtype=depth.dtype
            )

        priority = self._priority(depth, deterministic)
        valid = torch.isfinite(depth) & (depth > 0.0) & (~input_far)
        keep = torch.zeros_like(valid)
        flat_priority = priority.flatten(1)
        flat_valid = valid.flatten(1)
        flat_keep = keep.flatten(1)
        for index in range(batch):
            if not bool(apply[index]):
                flat_keep[index] = flat_valid[index]
                continue
            valid_indices = torch.nonzero(
                flat_valid[index], as_tuple=False
            ).squeeze(1)
            requested = int(
                round((1.0 - float(target_far[index])) * total_pixels)
            )
            count = min(max(requested, 1), int(valid_indices.numel()))
            if count <= 0:
                continue
            candidate_priority = flat_priority[index, valid_indices]
            selected = valid_indices[
                torch.topk(candidate_priority, k=count, largest=True).indices
            ]
            flat_keep[index, selected] = True
        keep = flat_keep.view_as(valid)

        adapted = torch.ones_like(depth)
        noisy = depth
        if self.noise_std > 0.0 or self.range_noise_std > 0.0:
            if deterministic:
                unit_noise = self._deterministic_noise(
                    height, width, depth.device, depth.dtype, 97.3
                )[None, None]
                unit_noise = unit_noise.expand(batch, -1, -1, -1) * 2.0 - 1.0
            else:
                unit_noise = torch.randn_like(depth)
            noise_scale = self.noise_std + self.range_noise_std * depth
            noisy = depth + unit_noise * noise_scale
        if self.quantization > 0.0:
            noisy = torch.round(noisy / self.quantization) * self.quantization
        noisy = noisy.clamp(0.0, 1.0)
        adapted = torch.where(keep, noisy, adapted)

        output_far = adapted >= self.far_threshold
        retained = keep.float().sum() / valid.float().sum().clamp_min(1.0)
        stats = LidarDomainStats(
            applied_fraction=apply.float().mean(),
            input_far_ratio=input_far.float().mean(),
            output_far_ratio=output_far.float().mean(),
            retained_nonfar_ratio=retained,
        )
        return adapted, stats
