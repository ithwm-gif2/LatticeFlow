"""Training loop for the lattice-conditioned flow policy."""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .bootstrap import add_original_yopo_to_path
from .config import configure_seed, resolve_device, save_config
from .costs import CostBundle, TrajectoryCostEvaluator, gather_lattice, robust_cost
from .model import LatticeFlowPolicy
from .targets import TargetRefiner, YOPOTeacher

add_original_yopo_to_path()

from policy.yopo_dataset import YOPODataset  # noqa: E402


class MeanMetrics:
    def __init__(self):
        self.values: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.values[key].append(float(value))

    def mean(self) -> dict[str, float]:
        return {key: float(np.mean(values)) for key, values in self.values.items() if values}


class FlowTrainer:
    def __init__(self, config: dict, run_dir: str | None = None):
        self.config = config
        configure_seed(int(config["runtime"]["seed"]))
        self.device = resolve_device(str(config["runtime"]["device"]))
        self.amp_enabled = bool(config["runtime"]["amp"]) and self.device.type == "cuda"

        if run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = str(Path(config["project"]["output_root"]) / f"flow_{stamp}")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        save_config(config, self.run_dir / "config.json")
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

        self.model = LatticeFlowPolicy(**config["model"]).to(self.device)
        self.teacher = YOPOTeacher(config["project"]["teacher_checkpoint"], self.device)
        if bool(config["project"].get("initialize_backbone_from_teacher", True)):
            self.model.image_backbone.load_state_dict(
                self.teacher.policy.image_backbone.state_dict()
            )
        self.cost_evaluator = TrajectoryCostEvaluator(
            config["depth_safety"],
            depth_safety_weight=float(config["loss_weights"]["depth_safety"]),
        ).to(self.device)
        self.cost_evaluator.requires_grad_(False)
        self.target_refiner = TargetRefiner(
            self.cost_evaluator, config["target_refinement"]
        )

        training_cfg = config["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.max_grad_norm = float(training_cfg["max_grad_norm"])
        self.loss_weights = config["loss_weights"]
        self.source_noise_std = float(training_cfg["source_noise_std"])
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_cost = float("inf")

        batch_size = int(training_cfg["batch_size"])
        workers = int(config["runtime"]["num_workers"])
        self.train_dataset = YOPODataset(mode="train")
        self.val_dataset = YOPODataset(mode="valid")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            drop_last=True,
        )

        print(f"Run directory: {self.run_dir}")
        print(f"Device: {self.device}; AMP: {self.amp_enabled}")
        print(f"Train batches: {len(self.train_loader)}; validation batches: {len(self.val_loader)}")

    def _move_batch(self, batch):
        return tuple(item.to(self.device, non_blocking=True) for item in batch)

    def _batch_limit_reached(self, batch_idx: int, key: str) -> bool:
        limit = self.config["training"].get(key)
        return limit is not None and batch_idx >= int(limit)

    def _flow_loss(
        self,
        depth_features: torch.Tensor,
        prepared_obs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        source = self.model.canonical_source(
            target.shape[0], target.device, target.dtype
        )
        if self.source_noise_std > 0:
            source = source + self.source_noise_std * torch.randn_like(source)
        t = torch.rand(
            target.shape[0],
            1,
            target.shape[2],
            target.shape[3],
            device=target.device,
            dtype=target.dtype,
        )
        x_t = (1.0 - t) * source + t * target
        target_velocity = target - source
        predicted_velocity = self.model.velocity_from_features(
            depth_features, prepared_obs, x_t, t
        )
        return F.mse_loss(predicted_velocity.float(), target_velocity.float())

    @staticmethod
    def _cost_means(costs: CostBundle, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/total_cost": costs.total.mean().item(),
            f"{prefix}/smooth_cost": costs.smooth.mean().item(),
            f"{prefix}/safety_cost": costs.safety.mean().item(),
            f"{prefix}/guidance_cost": costs.guidance.mean().item(),
            f"{prefix}/acceleration_cost": costs.acceleration.mean().item(),
            f"{prefix}/depth_safety_cost": costs.depth_safety.mean().item(),
            f"{prefix}/min_depth_clearance": costs.min_depth_clearance.mean().item(),
        }

    def _write_scalars(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def train_batch(self, batch, epoch: int) -> dict[str, float]:
        depth, position, rotation, obs_body, map_id = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_raw, _ = self.teacher(depth, obs_body)

        prepared_obs = self.model.prepare_observation(obs_body)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            depth_features = self.model.encode_depth(depth)
            student_seed = self.model.integrate_from_features(
                depth_features.detach(), prepared_obs, self.model.integration_steps
            )

        # Target construction is deliberately float32. The ESDF query and the
        # inner-loop gradients are less stable in fp16.
        refined = self.target_refiner.refine(
            teacher_raw.float(),
            depth.float(),
            position.float(),
            rotation.float(),
            obs_body.float(),
            map_id,
            student_seed=student_seed.float(),
        )
        target = refined.raw_endstate

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            flow_loss = self._flow_loss(depth_features, prepared_obs, target)
            predicted_raw = self.model.integrate_from_features(
                depth_features, prepared_obs, self.model.integration_steps
            )
            predicted_score = self.model.score_from_features(
                depth_features, prepared_obs, predicted_raw
            )

        predicted_costs = self.cost_evaluator(
            predicted_raw.float(),
            depth.float(),
            position.float(),
            rotation.float(),
            obs_body.float(),
            map_id,
        )
        endpoint_loss = F.mse_loss(predicted_raw.float(), target.float())
        trajectory_loss = robust_cost(predicted_costs.total).mean()
        score_loss = F.smooth_l1_loss(
            predicted_score.float(), robust_cost(predicted_costs.total.detach())
        )

        total_loss = (
            float(self.loss_weights["flow"]) * flow_loss
            + float(self.loss_weights["endpoint"]) * endpoint_loss
            + float(self.loss_weights["trajectory"]) * trajectory_loss
            + float(self.loss_weights["score"]) * score_loss
        )

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        metrics = {
            "train/total_loss": total_loss.item(),
            "train/flow_loss": flow_loss.item(),
            "train/endpoint_loss": endpoint_loss.item(),
            "train/trajectory_loss": trajectory_loss.item(),
            "train/score_loss": score_loss.item(),
            "train/gradient_norm": float(grad_norm),
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            "train/epoch": float(epoch),
            "target/improvement_vs_teacher": (
                refined.teacher_costs.total.mean() - refined.costs.total.mean()
            ).item(),
        }
        metrics.update(self._cost_means(predicted_costs, "student"))
        metrics.update(self._cost_means(refined.costs, "target"))
        metrics.update(self._cost_means(refined.teacher_costs, "teacher"))
        self._write_scalars(metrics, self.global_step)
        self.global_step += 1
        return metrics

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        accumulator = MeanMetrics()
        progress = tqdm(self.train_loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(progress):
            if self._batch_limit_reached(batch_idx, "max_train_batches"):
                break
            metrics = self.train_batch(batch, epoch)
            accumulator.update(metrics)
            progress.set_postfix(
                loss=f"{metrics['train/total_loss']:.3f}",
                flow=f"{metrics['train/flow_loss']:.3f}",
                cost=f"{metrics['student/total_cost']:.3f}",
            )
        return accumulator.mean()

    @torch.inference_mode()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        accumulator = MeanMetrics()
        progress = tqdm(self.val_loader, desc=f"Validate epoch {epoch}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(progress):
            if self._batch_limit_reached(batch_idx, "max_val_batches"):
                break
            depth, position, rotation, obs_body, map_id = self._move_batch(batch)
            teacher_raw, teacher_score = self.teacher(depth, obs_body)
            prepared_obs = self.model.prepare_observation(obs_body)
            predicted_raw, predicted_score = self.model(depth, prepared_obs)
            student_costs = self.cost_evaluator(
                predicted_raw.float(), depth, position, rotation, obs_body, map_id
            )
            teacher_costs = self.cost_evaluator(
                teacher_raw.float(), depth, position, rotation, obs_body, map_id
            )
            student_choice = predicted_score.reshape(depth.shape[0], -1).argmin(dim=1)
            teacher_choice = teacher_score.reshape(depth.shape[0], -1).argmin(dim=1)
            metrics = {
                "val/student_selected_cost": gather_lattice(
                    student_costs.total, student_choice
                ).mean().item(),
                "val/teacher_selected_cost": gather_lattice(
                    teacher_costs.total, teacher_choice
                ).mean().item(),
                "val/student_oracle_cost": student_costs.total.flatten(1).amin(dim=1).mean().item(),
                "val/teacher_oracle_cost": teacher_costs.total.flatten(1).amin(dim=1).mean().item(),
                "val/score_mae": F.l1_loss(
                    predicted_score, robust_cost(student_costs.total)
                ).item(),
                "val/endpoint_mse_to_teacher": F.mse_loss(predicted_raw, teacher_raw).item(),
            }
            metrics.update(self._cost_means(student_costs, "val_student"))
            metrics.update(self._cost_means(teacher_costs, "val_teacher"))
            accumulator.update(metrics)
            progress.set_postfix(cost=f"{metrics['val/student_selected_cost']:.3f}")

        mean_metrics = accumulator.mean()
        self._write_scalars(mean_metrics, self.global_step)
        return mean_metrics

    def save_checkpoint(self, epoch: int, name: str | None = None) -> Path:
        path = self.run_dir / "checkpoints" / (name or f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "best_val_cost": self.best_val_cost,
                "config": self.config,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str, resume_optimizer: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if resume_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_cost = float(checkpoint.get("best_val_cost", float("inf")))

    def fit(self) -> Path:
        training_cfg = self.config["training"]
        epochs = int(training_cfg["epochs"])
        last_path: Path | None = None
        for epoch in range(self.start_epoch, epochs):
            start = time.time()
            train_metrics = self.train_epoch(epoch)
            print(
                f"Epoch {epoch}: train loss={train_metrics.get('train/total_loss', math.nan):.4f}, "
                f"student cost={train_metrics.get('student/total_cost', math.nan):.4f}, "
                f"elapsed={time.time() - start:.1f}s"
            )

            if (epoch + 1) % int(training_cfg["validate_every"]) == 0:
                val_metrics = self.validate(epoch)
                selected_cost = val_metrics.get("val/student_selected_cost", float("inf"))
                print(
                    f"Epoch {epoch}: val student={selected_cost:.4f}, "
                    f"teacher={val_metrics.get('val/teacher_selected_cost', math.nan):.4f}"
                )
                if selected_cost < self.best_val_cost:
                    self.best_val_cost = selected_cost
                    last_path = self.save_checkpoint(epoch + 1, "best.pt")

            if (epoch + 1) % int(training_cfg["save_every"]) == 0 or epoch + 1 == epochs:
                last_path = self.save_checkpoint(epoch + 1)
            self.writer.flush()

        self.writer.close()
        if last_path is None:
            last_path = self.save_checkpoint(epochs, "last.pt")
        return last_path
