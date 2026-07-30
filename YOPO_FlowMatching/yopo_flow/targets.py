"""Teacher-guided, cost-refined targets for conditional flow matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .bootstrap import add_original_yopo_to_path
from .costs import CostBundle, TrajectoryCostEvaluator, robust_cost

add_original_yopo_to_path()

from policy.yopo_network import YopoNetwork  # noqa: E402
from policy.state_transform import StateTransform  # noqa: E402


class YOPOTeacher(torch.nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        self.state_transform = StateTransform()
        self.policy = YopoNetwork().to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        self.policy.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, depth: torch.Tensor, obs_body: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.state_transform.normalize_obs(obs_body.clone())
        prepared = self.state_transform.prepare_input(normalized)
        return self.policy(depth, prepared)


@dataclass
class RefinedTarget:
    raw_endstate: torch.Tensor
    costs: CostBundle
    teacher_costs: CostBundle


class TargetRefiner:
    """Policy-improvement target construction around a frozen YOPO teacher.

    Candidate selection performs inexpensive local exploration around the
    teacher and current student. A small differentiable cost-descent inner loop
    then refines the selected endpoint. The resulting endpoint is detached and
    used as the flow matching target x1.
    """

    def __init__(self, evaluator: TrajectoryCostEvaluator, config: dict):
        self.evaluator = evaluator
        self.enabled = bool(config["enabled"])
        self.include_student_seed = bool(config["include_student_seed"])
        self.noise_candidates = int(config["noise_candidates"])
        self.noise_std = float(config["noise_std"])
        self.gradient_steps = int(config["gradient_steps"])
        self.gradient_step_size = float(config["gradient_step_size"])

    def _evaluate(
        self,
        raw: torch.Tensor,
        depth: torch.Tensor,
        position: torch.Tensor,
        rotation: torch.Tensor,
        obs_body: torch.Tensor,
        map_id: torch.Tensor,
    ) -> CostBundle:
        return self.evaluator(raw, depth, position, rotation, obs_body, map_id)

    def _select_candidates(
        self,
        candidates: torch.Tensor,
        depth: torch.Tensor,
        position: torch.Tensor,
        rotation: torch.Tensor,
        obs_body: torch.Tensor,
        map_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, candidate_count, channels, vertical, horizontal = candidates.shape
        flat = candidates.reshape(batch_size * candidate_count, channels, vertical, horizontal)
        repeated = lambda tensor: tensor.repeat_interleave(candidate_count, dim=0)
        with torch.no_grad():
            costs = self._evaluate(
                flat,
                repeated(depth),
                repeated(position),
                repeated(rotation),
                repeated(obs_body),
                repeated(map_id),
            ).total.reshape(batch_size, candidate_count, vertical, horizontal)
            best = costs.argmin(dim=1)
            gather_index = best[:, None, None, :, :].expand(-1, 1, channels, -1, -1)
            return candidates.gather(1, gather_index).squeeze(1)

    def refine(
        self,
        teacher_raw: torch.Tensor,
        depth: torch.Tensor,
        position: torch.Tensor,
        rotation: torch.Tensor,
        obs_body: torch.Tensor,
        map_id: torch.Tensor,
        student_seed: torch.Tensor | None = None,
    ) -> RefinedTarget:
        teacher_raw = teacher_raw.detach()
        with torch.no_grad():
            teacher_costs = self._evaluate(
                teacher_raw, depth, position, rotation, obs_body, map_id
            ).detached()

        if not self.enabled:
            return RefinedTarget(teacher_raw, teacher_costs, teacher_costs)

        seeds = [teacher_raw]
        if self.include_student_seed and student_seed is not None:
            seeds.append(student_seed.detach().clamp(-1.0, 1.0))
        for _ in range(self.noise_candidates):
            seeds.append(
                (teacher_raw + self.noise_std * torch.randn_like(teacher_raw)).clamp(-1.0, 1.0)
            )
        candidates = torch.stack(seeds, dim=1)
        refined = self._select_candidates(
            candidates, depth, position, rotation, obs_body, map_id
        ).detach()

        for _ in range(self.gradient_steps):
            refined.requires_grad_(True)
            objective = robust_cost(
                self._evaluate(
                    refined, depth, position, rotation, obs_body, map_id
                ).total
            ).mean()
            gradient = torch.autograd.grad(objective, refined, only_inputs=True)[0]
            # Normalize each sample to make the configured step size stable
            # across maps and across the differently scaled YOPO cost terms.
            gradient = torch.nan_to_num(gradient, nan=0.0, posinf=1.0, neginf=-1.0)
            scale = gradient.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
            refined = (refined - self.gradient_step_size * gradient / scale).clamp(-1.0, 1.0).detach()

        with torch.no_grad():
            refined_costs = self._evaluate(
                refined, depth, position, rotation, obs_body, map_id
            ).detached()
        return RefinedTarget(refined, refined_costs, teacher_costs)
