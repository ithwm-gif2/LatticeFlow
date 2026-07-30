"""Fair YOPO baseline training on the same map-level split as LatticeFlow."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .bootstrap import add_original_yopo_to_path
from .config import configure_seed, resolve_device, save_config
from .dataset import configure_yopo_data_root, dataset_from_config

add_original_yopo_to_path()

from config.config import cfg as yopo_cfg  # noqa: E402
from loss.loss_function import YOPOLoss  # noqa: E402
from policy.state_transform import state_body2world  # noqa: E402
from policy.yopo_network import YopoNetwork  # noqa: E402


class MetricMean:
    def __init__(self):
        self.values: dict[str, list[float]] = defaultdict(list)

    def update(self, values: dict[str, float]) -> None:
        for key, value in values.items():
            self.values[key].append(float(value))

    def mean(self) -> dict[str, float]:
        return {key: float(np.mean(value)) for key, value in self.values.items() if value}


class YOPOSplitTrainer:
    """Train the original one-stage YOPO policy without held-out-map leakage."""

    def __init__(self, config: dict, run_dir: str):
        self.config = config
        configure_seed(int(config["runtime"]["seed"]))
        configure_yopo_data_root(config["data"]["root"])
        self.device = resolve_device(str(config["runtime"]["device"]))
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.run_dir / "config.json")
        self.writer = SummaryWriter(str(self.run_dir / "tensorboard"))

        self.policy = YopoNetwork().to(self.device)
        self.loss = YOPOLoss()
        baseline = config["baseline"]
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=float(baseline["learning_rate"]),
            weight_decay=float(baseline["weight_decay"]),
        )
        self.max_grad_norm = float(baseline["max_grad_norm"])
        self.global_step = 0
        self.best_val_cost = float("inf")

        workers = int(config["runtime"]["num_workers"])
        batch_size = int(baseline["batch_size"])
        self.train_loader = DataLoader(
            dataset_from_config(config, "train"),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            dataset_from_config(config, "valid"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            drop_last=False,
        )
        print(
            f"YOPO baseline run={self.run_dir}; device={self.device}; "
            f"train_batches={len(self.train_loader)}; val_batches={len(self.val_loader)}"
        )

    def _limit(self, batch_index: int, key: str) -> bool:
        value = self.config["baseline"].get(key)
        return value is not None and batch_index >= int(value)

    def _forward_loss(self, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        depth, position, rotation, obs_body, map_id = (
            item.to(self.device, non_blocking=True) for item in batch
        )
        batch_size = depth.shape[0]
        goal_world, velocity_world, acceleration_world = state_body2world(
            position,
            rotation,
            obs_body[:, 6:9],
            obs_body[:, 0:3],
            obs_body[:, 3:6],
        )
        start_world = torch.stack(
            (position, velocity_world, acceleration_world), dim=1
        )
        endstate, score = self.policy.inference(depth, obs_body)
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(
            batch_size * int(yopo_cfg["traj_num"]), 9
        )
        score_flat = score.reshape(batch_size * int(yopo_cfg["traj_num"]))
        position_expanded = position.repeat_interleave(int(yopo_cfg["traj_num"]), dim=0)
        rotation_expanded = rotation.repeat_interleave(int(yopo_cfg["traj_num"]), dim=0)
        start_expanded = start_world.repeat_interleave(int(yopo_cfg["traj_num"]), dim=0)
        goal_expanded = goal_world.repeat_interleave(int(yopo_cfg["traj_num"]), dim=0)
        map_expanded = map_id
        end_position, end_velocity, end_acceleration = state_body2world(
            position_expanded,
            rotation_expanded,
            endstate_flat[:, :3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9],
        )
        end_world = torch.stack(
            (end_position, end_velocity, end_acceleration), dim=1
        )
        smooth, safety, guidance, acceleration = self.loss(
            start_expanded, end_world, goal_expanded, map_expanded
        )
        total_cost_flat = smooth + safety + guidance + acceleration
        trajectory_loss = total_cost_flat.mean()
        score_loss = F.smooth_l1_loss(score_flat, total_cost_flat.detach())
        total_loss = trajectory_loss + score_loss
        cost_grid = total_cost_flat.reshape(batch_size, -1)
        score_grid = score.reshape(batch_size, -1)
        selected = score_grid.argmin(dim=1)
        selected_cost = cost_grid.gather(1, selected[:, None]).mean()
        outputs = {
            "trajectory_loss": trajectory_loss,
            "score_loss": score_loss,
            "selected_cost": selected_cost,
            "oracle_cost": cost_grid.amin(dim=1).mean(),
            "smooth_cost": smooth.mean(),
            "safety_cost": safety.mean(),
            "guidance_cost": guidance.mean(),
            "acceleration_cost": acceleration.mean(),
        }
        return total_loss, outputs

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.policy.train()
        accumulator = MetricMean()
        progress = tqdm(self.train_loader, desc=f"YOPO train {epoch}", dynamic_ncols=True)
        for batch_index, batch in enumerate(progress):
            if self._limit(batch_index, "max_train_batches"):
                break
            self.optimizer.zero_grad(set_to_none=True)
            total_loss, values = self._forward_loss(batch)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            metrics = {
                "train/total_loss": float(total_loss.detach()),
                "train/gradient_norm": float(grad_norm),
                "train/epoch": float(epoch),
                **{f"train/{key}": float(value.detach()) for key, value in values.items()},
            }
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
            self.global_step += 1
            accumulator.update(metrics)
            progress.set_postfix(
                loss=f"{metrics['train/total_loss']:.3f}",
                cost=f"{metrics['train/selected_cost']:.3f}",
            )
        return accumulator.mean()

    @torch.inference_mode()
    def validate(self, epoch: int) -> dict[str, float]:
        self.policy.eval()
        accumulator = MetricMean()
        progress = tqdm(self.val_loader, desc=f"YOPO valid {epoch}", dynamic_ncols=True)
        for batch_index, batch in enumerate(progress):
            if self._limit(batch_index, "max_val_batches"):
                break
            total_loss, values = self._forward_loss(batch)
            metrics = {
                "val/total_loss": float(total_loss),
                **{f"val/{key}": float(value) for key, value in values.items()},
            }
            accumulator.update(metrics)
            progress.set_postfix(cost=f"{metrics['val/selected_cost']:.3f}")
        means = accumulator.mean()
        for key, value in means.items():
            self.writer.add_scalar(key, value, self.global_step)
        return means

    def save(self, epoch: int, name: str) -> Path:
        raw_path = self.checkpoint_dir / f"{name}.pth"
        torch.save(self.policy.state_dict(), raw_path)
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "best_val_cost": self.best_val_cost,
                "config": self.config,
            },
            self.checkpoint_dir / f"{name}.pt",
        )
        return raw_path

    def fit(self) -> Path:
        baseline = self.config["baseline"]
        epochs = int(baseline["epochs"])
        best_path: Path | None = None
        for epoch in range(epochs):
            start = time.time()
            train_metrics = self.train_epoch(epoch)
            print(
                f"YOPO epoch {epoch}: loss={train_metrics.get('train/total_loss', math.nan):.4f}; "
                f"elapsed={time.time() - start:.1f}s"
            )
            if (epoch + 1) % int(baseline["validate_every"]) == 0:
                val_metrics = self.validate(epoch)
                val_cost = val_metrics.get("val/selected_cost", float("inf"))
                print(f"YOPO epoch {epoch}: val selected cost={val_cost:.4f}")
                if val_cost < self.best_val_cost:
                    self.best_val_cost = val_cost
                    best_path = self.save(epoch + 1, "best")
            if (epoch + 1) % int(baseline["save_every"]) == 0 or epoch + 1 == epochs:
                self.save(epoch + 1, f"epoch_{epoch + 1:03d}")
            self.writer.flush()
        self.writer.close()
        if best_path is None:
            best_path = self.save(epochs, "best")
        return best_path
