"""Teacher-free physical-anchor LatticeFlow training protocol."""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import configure_seed, resolve_device, save_config
from .costs import TrajectoryCostEvaluator, gather_lattice, robust_cost
from .dataset import configure_yopo_data_root, dataset_from_config
from .lidar_domain import LidarNoReturnAdapter
from .physical_model import PhysicalAnchorFlowPolicy
from .physical_trainer import PhysicalAnchorFlowTrainer
from .self_targets import SelfTargetRefiner
from .trainer import MeanMetrics


def zero_initialize_flow_output(model: PhysicalAnchorFlowPolicy) -> None:
    """Make the initial neural ODE exactly preserve every physical anchor."""

    convolutions = [
        module for module in model.flow_head.modules() if isinstance(module, nn.Conv2d)
    ]
    if not convolutions:
        raise RuntimeError("Flow head has no Conv2d output layer")
    nn.init.zeros_(convolutions[-1].weight)
    if convolutions[-1].bias is not None:
        nn.init.zeros_(convolutions[-1].bias)


class TeacherFreePhysicalFlowTrainer(PhysicalAnchorFlowTrainer):
    """Train without a teacher using self-improved privileged targets."""

    def __init__(self, config: dict, run_dir: str | None = None):
        self.config = config
        if bool(config["project"].get("initialize_backbone_from_teacher", False)):
            raise ValueError("Teacher-free training forbids teacher backbone initialization")
        configure_seed(int(config["runtime"]["seed"]))
        configure_yopo_data_root(config["data"]["root"])
        self.device = resolve_device(str(config["runtime"]["device"]))
        self.amp_enabled = bool(config["runtime"]["amp"]) and self.device.type == "cuda"

        if run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = str(
                Path(config["project"]["output_root"])
                / f"teacher_free_physical_{stamp}"
            )
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        save_config(config, self.run_dir / "config.json")
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

        self.lidar_domain_config = config.get("lidar_domain", {"enabled": False})
        self.lidar_domain = LidarNoReturnAdapter(self.lidar_domain_config)

        self.model = PhysicalAnchorFlowPolicy(**config["model"]).to(self.device)
        initialization = config["project"].get("initialize_from_checkpoint")
        initialization_checkpoint = None
        if initialization:
            initialization_checkpoint = torch.load(
                str(initialization), map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(initialization_checkpoint["model"])
            print(f"Domain adaptation initialization: {initialization}")
        elif bool(config["project"].get("zero_initialize_flow_output", True)):
            zero_initialize_flow_output(self.model)

        # LiDAR adaptation is a perception-domain problem. A frozen copy of
        # the already teacher-free source policy supplies stable dense-depth
        # features and physical terminal states. This is not a YOPO teacher:
        # it is the same policy before adapting its perception backbone.
        self.domain_reference_model = None
        reference_path = self.lidar_domain_config.get(
            "reference_checkpoint", initialization
        )
        if self.lidar_domain.enabled and reference_path:
            self.domain_reference_model = PhysicalAnchorFlowPolicy(
                **config["model"]
            ).to(self.device)
            if initialization_checkpoint is not None and str(reference_path) == str(
                initialization
            ):
                reference_checkpoint = initialization_checkpoint
            else:
                reference_checkpoint = torch.load(
                    str(reference_path), map_location=self.device, weights_only=False
                )
            self.domain_reference_model.load_state_dict(reference_checkpoint["model"])
            self.domain_reference_model.eval()
            self.domain_reference_model.requires_grad_(False)
            print(f"Frozen teacher-free domain reference: {reference_path}")

        if bool(self.lidar_domain_config.get("backbone_only", False)):
            self.model.requires_grad_(False)
            self.model.image_backbone.requires_grad_(True)
            print("LiDAR adaptation trainable modules: image_backbone only")

        self.cost_evaluator = TrajectoryCostEvaluator(
            config["depth_safety"],
            depth_safety_weight=float(config["loss_weights"]["depth_safety"]),
        ).to(self.device)
        self.cost_evaluator.requires_grad_(False)
        self.target_refiner = SelfTargetRefiner(
            self.cost_evaluator, config["target_refinement"]
        )

        training = config["training"]
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters selected for domain adaptation")
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
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
        print("Teacher-free training: no teacher model loaded")
        print(
            "LiDAR domain adaptation: "
            f"enabled={self.lidar_domain.enabled}, "
            f"target_far=[{self.lidar_domain.far_min:.2f},"
            f"{self.lidar_domain.far_max:.2f}], "
            f"val_far={self.lidar_domain.validation_far_ratio:.2f}"
        )
        print(
            f"Map split train={config['data']['train_maps']}, "
            f"valid={config['data']['valid_maps']}, test={config['data']['test_maps']}"
        )
        print(
            f"Train batches: {len(self.train_loader)}; "
            f"validation batches: {len(self.val_loader)}"
        )

    def vertical_trajectory_positions(
        self,
        endstate: torch.Tensor,
        obs_body: torch.Tensor,
    ) -> torch.Tensor:
        """Sample body-frame z along the same quintic boundary-value curve."""

        samples = int(self.lidar_domain_config.get("trajectory_samples", 11))
        duration = float(
            self.lidar_domain_config.get(
                "trajectory_duration",
                self.model.state_transform.lattice_primitive.segment_time,
            )
        )
        s = torch.linspace(
            0.0,
            1.0,
            samples,
            device=endstate.device,
            dtype=endstate.dtype,
        ).view(1, samples, 1, 1)
        s2, s3 = s.square(), s.pow(3)
        s4, s5 = s.pow(4), s.pow(5)
        h00 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5
        h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5
        h20 = 0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5)
        h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
        h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5
        h21 = 0.5 * (s3 - 2.0 * s4 + s5)

        position0 = torch.zeros_like(obs_body[:, 2]).view(-1, 1, 1, 1)
        velocity0 = obs_body[:, 2].view(-1, 1, 1, 1)
        acceleration0 = obs_body[:, 5].view(-1, 1, 1, 1)
        position1 = endstate[:, 2].unsqueeze(1)
        velocity1 = endstate[:, 5].unsqueeze(1)
        acceleration1 = endstate[:, 8].unsqueeze(1)
        return (
            h00 * position0
            + h10 * duration * velocity0
            + h20 * duration**2 * acceleration0
            + h01 * position1
            + h11 * duration * velocity1
            + h21 * duration**2 * acceleration1
        )

    def domain_consistency_loss(
        self,
        dense_depth: torch.Tensor,
        prepared_obs: torch.Tensor,
        obs_body: torch.Tensor,
        sparse_features: torch.Tensor,
        sparse_raw: torch.Tensor,
        sparse_score: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not self.lidar_domain.enabled or self.domain_reference_model is None:
            zero = sparse_raw.new_zeros(())
            return zero, zero, zero, zero, zero, zero
        with torch.no_grad():
            reference_features = self.domain_reference_model.encode_depth(dense_depth)
            reference_raw = self.domain_reference_model.integrate_from_features(
                reference_features,
                prepared_obs,
                self.domain_reference_model.integration_steps,
            )
            reference_score = self.domain_reference_model.score_from_features(
                reference_features, prepared_obs, reference_raw
            )
            reference_flow_state = self.domain_reference_model.raw_to_flow_state(
                reference_raw
            )
            reference_endstate = (
                self.domain_reference_model.state_transform.pred_to_endstate(
                    reference_raw
                )
            )
            reference_curve = self.vertical_trajectory_positions(
                reference_endstate, obs_body
            )

        feature_loss = F.smooth_l1_loss(
            sparse_features.float(), reference_features.detach().float()
        )
        sparse_flow_state = self.model.raw_to_flow_state(sparse_raw)
        endpoint_loss = F.smooth_l1_loss(
            sparse_flow_state.float(), reference_flow_state.detach().float()
        )
        sparse_centered = sparse_score - sparse_score.mean(
            dim=(1, 2), keepdim=True
        )
        reference_centered = reference_score.detach() - reference_score.detach().mean(
            dim=(1, 2), keepdim=True
        )
        score_loss = F.smooth_l1_loss(
            sparse_centered.float(), reference_centered.float()
        )
        vertical_loss = F.smooth_l1_loss(
            sparse_flow_state[:, (2, 5, 8)].float(),
            reference_flow_state[:, (2, 5, 8)].detach().float(),
        )
        sparse_endstate = self.model.state_transform.pred_to_endstate(sparse_raw)
        sparse_curve = self.vertical_trajectory_positions(sparse_endstate, obs_body)
        curve_loss = F.smooth_l1_loss(
            sparse_curve.float() / float(self.model.position_scale),
            reference_curve.detach().float() / float(self.model.position_scale),
        )
        total = (
            float(self.lidar_domain_config.get("feature_weight", 1.0))
            * feature_loss
            + float(self.lidar_domain_config.get("endpoint_weight", 1.0))
            * endpoint_loss
            + float(self.lidar_domain_config.get("score_weight", 0.25))
            * score_loss
            + float(self.lidar_domain_config.get("vertical_weight", 1.0))
            * vertical_loss
            + float(self.lidar_domain_config.get("trajectory_vertical_weight", 1.0))
            * curve_loss
        )
        return (
            total,
            feature_loss,
            endpoint_loss,
            score_loss,
            vertical_loss,
            curve_loss,
        )

    def train_batch(self, batch, epoch: int) -> dict[str, float]:
        depth, position, rotation, obs_body, map_id = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        perception_depth, lidar_stats = self.lidar_domain(
            depth,
            position_world=position,
            rotation_world_body=rotation,
        )

        prepared_obs = self.model.prepare_observation(obs_body)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            depth_features = self.model.encode_depth(perception_depth)
            student_seed = self.model.integrate_from_features(
                depth_features.detach(), prepared_obs, self.model.integration_steps
            )

        # The target optimizer is deliberately float32 and detached from the
        # policy graph. It uses maps/ESDF only inside this training-time block.
        refined = self.target_refiner.refine(
            depth.float(),
            position.float(),
            rotation.float(),
            obs_body.float(),
            map_id,
            student_seed=student_seed.float(),
            epoch=epoch,
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
                perception_depth, obs_body, predicted_raw, predicted_score
            )
        )
        (
            domain_consistency,
            domain_feature,
            domain_endpoint,
            domain_score,
            domain_vertical,
            domain_vertical_curve,
        ) = self.domain_consistency_loss(
            depth,
            prepared_obs,
            obs_body,
            depth_features,
            predicted_raw,
            predicted_score,
        )
        curvature_loss = self.lattice_curvature(predicted_raw.float())
        total_loss = (
            float(self.loss_weights["flow"]) * flow_loss
            + float(self.loss_weights["endpoint"]) * endpoint_loss
            + float(self.loss_weights["trajectory"]) * trajectory_loss
            + float(self.loss_weights["score"]) * score_loss
            + float(self.loss_weights.get("local_consistency", 0.0))
            * consistency_loss
            + float(self.loss_weights.get("lattice_curvature", 0.0))
            * curvature_loss
            + float(self.loss_weights.get("domain_consistency", 0.0))
            * domain_consistency
        )

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            predicted_choice = predicted_score.reshape(
                predicted_score.shape[0], -1
            ).argmin(dim=1)
            predicted_endstate = self.model.state_transform.pred_to_endstate(
                predicted_raw
            )
            selected_vertical_position = gather_lattice(
                predicted_endstate[:, 2], predicted_choice
            ).abs().mean()

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
            "train/domain_consistency_loss": domain_consistency.item(),
            "train/domain_feature_loss": domain_feature.item(),
            "train/domain_endpoint_loss": domain_endpoint.item(),
            "train/domain_score_loss": domain_score.item(),
            "train/domain_vertical_loss": domain_vertical.item(),
            "train/domain_vertical_curve_loss": domain_vertical_curve.item(),
            "train/domain_selected_abs_endpoint_z": selected_vertical_position.item(),
            "domain/applied_fraction": lidar_stats.applied_fraction.item(),
            "domain/input_far_ratio": lidar_stats.input_far_ratio.item(),
            "domain/output_far_ratio": lidar_stats.output_far_ratio.item(),
            "domain/retained_nonfar_ratio": lidar_stats.retained_nonfar_ratio.item(),
            "train/gradient_norm": float(grad_norm),
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            "train/epoch": float(epoch),
            "target/improvement_vs_anchor": (
                refined.anchor_costs.total.mean() - refined.costs.total.mean()
            ).item(),
            "target/improvement_vs_student_seed": (
                refined.student_costs.total.mean() - refined.costs.total.mean()
            ).item(),
            "target/improvement_from_gradient": (
                refined.selected_seed_costs.total.mean() - refined.costs.total.mean()
            ).item(),
            "target/gradient_accept_rate": refined.gradient_accept_rate,
        }
        for label, fraction in refined.source_fractions.items():
            metrics[f"target/source_fraction_{label}"] = fraction
        metrics.update(self._cost_means(predicted_costs, "student"))
        metrics.update(self._cost_means(refined.costs, "target"))
        metrics.update(self._cost_means(refined.anchor_costs, "anchor"))
        metrics.update(self._cost_means(refined.student_costs, "student_seed"))
        self._write_scalars(metrics, self.global_step)
        self.global_step += 1
        return metrics

    @torch.inference_mode()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        accumulator = MeanMetrics()
        progress = tqdm(
            self.val_loader, desc=f"Validate epoch {epoch}", dynamic_ncols=True
        )
        for batch_idx, batch in enumerate(progress):
            if self._batch_limit_reached(batch_idx, "max_val_batches"):
                break
            depth, position, rotation, obs_body, map_id = self._move_batch(batch)
            prepared_obs = self.model.prepare_observation(obs_body)
            predicted_raw, predicted_score = self.model(depth, prepared_obs)
            lidar_depth, lidar_stats = self.lidar_domain(
                depth,
                position_world=position,
                rotation_world_body=rotation,
                deterministic=True,
            )
            lidar_raw, lidar_score = self.model(lidar_depth, prepared_obs)
            student_costs = self.cost_evaluator(
                predicted_raw.float(), depth, position, rotation, obs_body, map_id
            )
            anchor_raw = torch.zeros_like(predicted_raw)
            anchor_costs = self.cost_evaluator(
                anchor_raw, depth, position, rotation, obs_body, map_id
            )
            lidar_costs = self.cost_evaluator(
                lidar_raw.float(), depth, position, rotation, obs_body, map_id
            )
            reference_raw = None
            reference_score = None
            reference_costs = None
            if self.domain_reference_model is not None:
                reference_raw, reference_score = self.domain_reference_model(
                    depth, prepared_obs
                )
                reference_costs = self.cost_evaluator(
                    reference_raw.float(),
                    depth,
                    position,
                    rotation,
                    obs_body,
                    map_id,
                )
            student_choice = predicted_score.reshape(depth.shape[0], -1).argmin(dim=1)
            lidar_choice = lidar_score.reshape(depth.shape[0], -1).argmin(dim=1)
            lidar_endstate = self.model.state_transform.pred_to_endstate(lidar_raw)
            lidar_selected_z = gather_lattice(
                lidar_endstate[:, 2], lidar_choice
            ).abs()
            level_mask = obs_body[:, 8].abs() < float(
                self.lidar_domain_config.get("level_goal_z_threshold", 0.5)
            )
            level_selected_z = (
                lidar_selected_z[level_mask].mean()
                if bool(level_mask.any())
                else lidar_selected_z.new_zeros(())
            )
            lidar_selected_cost = gather_lattice(
                lidar_costs.total, lidar_choice
            ).mean()
            if reference_raw is not None and reference_score is not None:
                reference_choice = reference_score.reshape(
                    depth.shape[0], -1
                ).argmin(dim=1)
                reference_selected_cost = gather_lattice(
                    reference_costs.total, reference_choice
                ).mean()
                lidar_flow_state = self.model.raw_to_flow_state(lidar_raw)
                reference_flow_state = (
                    self.domain_reference_model.raw_to_flow_state(reference_raw)
                )
                domain_endpoint_mae = F.l1_loss(
                    lidar_flow_state, reference_flow_state
                )
                domain_vertical_mae = F.l1_loss(
                    lidar_flow_state[:, (2, 5, 8)],
                    reference_flow_state[:, (2, 5, 8)],
                )
                lidar_curve = self.vertical_trajectory_positions(
                    lidar_endstate, obs_body
                )
                reference_endstate = (
                    self.domain_reference_model.state_transform.pred_to_endstate(
                        reference_raw
                    )
                )
                reference_curve = self.vertical_trajectory_positions(
                    reference_endstate, obs_body
                )
                domain_vertical_curve_mae = F.l1_loss(
                    lidar_curve, reference_curve
                )
                lidar_centered = lidar_score - lidar_score.mean(
                    dim=(1, 2), keepdim=True
                )
                reference_centered = reference_score - reference_score.mean(
                    dim=(1, 2), keepdim=True
                )
                domain_score_mae = F.l1_loss(
                    lidar_centered, reference_centered
                )
                domain_choice_agreement = (
                    lidar_choice == reference_choice
                ).float().mean()
            else:
                zero = lidar_selected_cost.new_zeros(())
                reference_selected_cost = zero
                domain_endpoint_mae = zero
                domain_vertical_mae = zero
                domain_vertical_curve_mae = zero
                domain_score_mae = zero
                domain_choice_agreement = zero
            domain_objective = (
                lidar_selected_cost
                + float(
                    self.lidar_domain_config.get(
                        "objective_endpoint_weight", 1.0
                    )
                )
                * domain_endpoint_mae
                + float(
                    self.lidar_domain_config.get(
                        "objective_vertical_curve_weight", 1.0
                    )
                )
                * domain_vertical_curve_mae
                + float(
                    self.lidar_domain_config.get("objective_score_weight", 0.25)
                )
                * domain_score_mae
            )
            metrics = {
                "val/student_selected_cost": gather_lattice(
                    student_costs.total, student_choice
                ).mean().item(),
                "val/student_oracle_cost": student_costs.total.flatten(1)
                .amin(dim=1)
                .mean()
                .item(),
                "val/anchor_oracle_cost": anchor_costs.total.flatten(1)
                .amin(dim=1)
                .mean()
                .item(),
                "val/score_mae": F.l1_loss(
                    predicted_score, robust_cost(student_costs.total)
                ).item(),
                "val/endpoint_mse_to_anchor": F.mse_loss(
                    predicted_raw, anchor_raw
                ).item(),
                "val_lidar/student_selected_cost": lidar_selected_cost.item(),
                "val_lidar/reference_selected_cost": reference_selected_cost.item(),
                "val_lidar/domain_objective": domain_objective.item(),
                "val_lidar/domain_endpoint_mae": domain_endpoint_mae.item(),
                "val_lidar/domain_vertical_mae": domain_vertical_mae.item(),
                "val_lidar/domain_vertical_curve_mae": (
                    domain_vertical_curve_mae.item()
                ),
                "val_lidar/domain_score_mae": domain_score_mae.item(),
                "val_lidar/domain_choice_agreement": domain_choice_agreement.item(),
                "val_lidar/student_oracle_cost": lidar_costs.total.flatten(1)
                .amin(dim=1)
                .mean()
                .item(),
                "val_lidar/score_mae": F.l1_loss(
                    lidar_score, robust_cost(lidar_costs.total)
                ).item(),
                "val_lidar/selected_abs_endpoint_z": lidar_selected_z.mean().item(),
                "val_lidar/level_selected_abs_endpoint_z": level_selected_z.item(),
                "val_lidar/depth_far_ratio": lidar_stats.output_far_ratio.item(),
            }
            metrics.update(self._cost_means(student_costs, "val_student"))
            metrics.update(self._cost_means(anchor_costs, "val_anchor"))
            metrics.update(self._cost_means(lidar_costs, "val_lidar_student"))
            if reference_costs is not None:
                metrics.update(
                    self._cost_means(reference_costs, "val_domain_reference")
                )
            accumulator.update(metrics)
            progress.set_postfix(
                dense=f"{metrics['val/student_selected_cost']:.3f}",
                lidar=f"{metrics['val_lidar/student_selected_cost']:.3f}",
            )

        mean_metrics = accumulator.mean()
        self._write_scalars(mean_metrics, self.global_step)
        return mean_metrics

    def fit(self) -> Path:
        training = self.config["training"]
        epochs = int(training["epochs"])
        last_path: Path | None = None
        for epoch in range(self.start_epoch, epochs):
            start = time.time()
            train_metrics = self.train_epoch(epoch)
            print(
                f"Epoch {epoch}: train loss="
                f"{train_metrics.get('train/total_loss', math.nan):.4f}, "
                f"student cost={train_metrics.get('student/total_cost', math.nan):.4f}, "
                f"target gain={train_metrics.get('target/improvement_vs_anchor', math.nan):.4f}, "
                f"elapsed={time.time() - start:.1f}s"
            )
            if (epoch + 1) % int(training["validate_every"]) == 0:
                val_metrics = self.validate(epoch)
                selection_metric = str(
                    self.lidar_domain_config.get(
                        "selection_metric", "val_lidar/student_selected_cost"
                    )
                )
                selected_cost = val_metrics.get(selection_metric, float("inf"))
                print(
                    f"Epoch {epoch}: val metric {selection_metric}="
                    f"{selected_cost:.4f}, dense="
                    f"{val_metrics.get('val/student_selected_cost', math.nan):.4f}, "
                    f"anchor oracle="
                    f"{val_metrics.get('val/anchor_oracle_cost', math.nan):.4f}"
                )
                if selected_cost < self.best_val_cost:
                    self.best_val_cost = selected_cost
                    last_path = self.save_checkpoint(epoch + 1, "best.pt")
            if (
                (epoch + 1) % int(training["save_every"]) == 0
                or epoch + 1 == epochs
            ):
                last_path = self.save_checkpoint(epoch + 1)
            self.writer.flush()

        self.writer.close()
        if last_path is None:
            last_path = self.save_checkpoint(epochs, "last.pt")
        return last_path
