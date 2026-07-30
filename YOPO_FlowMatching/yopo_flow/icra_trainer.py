"""ICRA training protocol for map-held-out, continuity-regularized LatticeFlow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import configure_seed, resolve_device, save_config
from .costs import TrajectoryCostEvaluator, robust_cost
from .dataset import configure_yopo_data_root, dataset_from_config
from .model import LatticeFlowPolicy
from .targets import TargetRefiner, YOPOTeacher
from .trainer import FlowTrainer


class ICRAFlowTrainer(FlowTrainer):
    """Flow trainer with held-out maps and local continuity regularization."""

    def __init__(self, config: dict, run_dir: str | None = None):
        self.config = config
        configure_seed(int(config["runtime"]["seed"]))
        configure_yopo_data_root(config["data"]["root"])
        self.device = resolve_device(str(config["runtime"]["device"]))
        self.amp_enabled = bool(config["runtime"]["amp"]) and self.device.type == "cuda"

        if run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = str(Path(config["project"]["output_root"]) / f"icra_flow_{stamp}")
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

        training = config["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.max_grad_norm = float(training["max_grad_norm"])
        self.loss_weights = config["loss_weights"]
        self.continuity_config = config.get("continuity", {"enabled": False})
        self.source_noise_std = float(training["source_noise_std"])
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_cost = float("inf")

        workers = int(config["runtime"]["num_workers"])
        batch_size = int(training["batch_size"])
        self.train_dataset = dataset_from_config(config, "train")
        self.val_dataset = dataset_from_config(config, "valid")
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
            drop_last=False,
        )
        print(f"Run directory: {self.run_dir}")
        print(f"Device: {self.device}; AMP: {self.amp_enabled}")
        print(
            f"Map split train={config['data']['train_maps']}, "
            f"valid={config['data']['valid_maps']}, test={config['data']['test_maps']}"
        )
        print(
            f"Train batches: {len(self.train_loader)}; "
            f"validation batches: {len(self.val_loader)}"
        )

    @staticmethod
    def lattice_curvature(raw_endstate: torch.Tensor) -> torch.Tensor:
        terms = []
        if raw_endstate.shape[-1] >= 3:
            terms.append(
                (
                    raw_endstate[..., 2:]
                    - 2.0 * raw_endstate[..., 1:-1]
                    + raw_endstate[..., :-2]
                ).square().mean()
            )
        if raw_endstate.shape[-2] >= 3:
            terms.append(
                (
                    raw_endstate[..., 2:, :]
                    - 2.0 * raw_endstate[..., 1:-1, :]
                    + raw_endstate[..., :-2, :]
                ).square().mean()
            )
        if not terms:
            return raw_endstate.new_zeros(())
        return torch.stack(terms).mean()

    def local_consistency_loss(
        self,
        depth: torch.Tensor,
        obs_body: torch.Tensor,
        reference_raw: torch.Tensor,
        reference_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Penalize output jumps under sensor-scale depth and state changes."""

        if not bool(self.continuity_config.get("enabled", False)):
            zero = reference_raw.new_zeros(())
            return zero, zero, zero
        noisy_depth = (
            depth
            + float(self.continuity_config.get("depth_noise_std", 0.0))
            * torch.randn_like(depth)
        ).clamp(0.0, 1.0)
        observation_std = torch.as_tensor(
            self.continuity_config.get("observation_noise_std", [0.0] * 9),
            device=obs_body.device,
            dtype=obs_body.dtype,
        ).view(1, 9)
        noisy_observation = obs_body + torch.randn_like(obs_body) * observation_std
        noisy_prepared = self.model.prepare_observation(noisy_observation)
        noisy_features = self.model.encode_depth(noisy_depth)
        noisy_raw = self.model.integrate_from_features(
            noisy_features, noisy_prepared, self.model.integration_steps
        )
        noisy_score = self.model.score_from_features(
            noisy_features, noisy_prepared, noisy_raw
        )
        endpoint_loss = F.smooth_l1_loss(
            noisy_raw.float(), reference_raw.detach().float()
        )
        noisy_centered = noisy_score - noisy_score.mean(dim=(1, 2), keepdim=True)
        reference_centered = reference_score.detach() - reference_score.detach().mean(
            dim=(1, 2), keepdim=True
        )
        score_loss = F.smooth_l1_loss(
            noisy_centered.float(), reference_centered.float()
        )
        total = (
            float(self.continuity_config.get("endpoint_weight", 1.0)) * endpoint_loss
            + float(self.continuity_config.get("score_weight", 0.25)) * score_loss
        )
        return total, endpoint_loss, score_loss

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
        consistency_loss, consistency_endpoint, consistency_score = (
            self.local_consistency_loss(
                depth, obs_body, predicted_raw, predicted_score
            )
        )
        curvature_loss = self.lattice_curvature(predicted_raw.float())
        total_loss = (
            float(self.loss_weights["flow"]) * flow_loss
            + float(self.loss_weights["endpoint"]) * endpoint_loss
            + float(self.loss_weights["trajectory"]) * trajectory_loss
            + float(self.loss_weights["score"]) * score_loss
            + float(self.loss_weights.get("local_consistency", 0.0)) * consistency_loss
            + float(self.loss_weights.get("lattice_curvature", 0.0)) * curvature_loss
        )

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        metrics = {
            "train/total_loss": total_loss.item(),
            "train/flow_loss": flow_loss.item(),
            "train/endpoint_loss": endpoint_loss.item(),
            "train/trajectory_loss": trajectory_loss.item(),
            "train/score_loss": score_loss.item(),
            "train/local_consistency_loss": consistency_loss.item(),
            "train/consistency_endpoint_loss": consistency_endpoint.item(),
            "train/consistency_score_loss": consistency_score.item(),
            "train/lattice_curvature_loss": curvature_loss.item(),
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
