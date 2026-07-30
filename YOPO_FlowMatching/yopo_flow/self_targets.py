"""Teacher-free, cost-refined targets for physical-anchor flow matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .costs import CostBundle, TrajectoryCostEvaluator, robust_cost


@dataclass
class SelfRefinedTarget:
    raw_endstate: torch.Tensor
    costs: CostBundle
    anchor_costs: CostBundle
    student_costs: CostBundle
    selected_seed_costs: CostBundle
    source_fractions: dict[str, float]
    gradient_accept_rate: float


class SelfTargetRefiner:
    """Build detached x1 targets using no teacher or pretrained policy.

    Candidate costs are evaluated once as one expanded batch and reused for
    anchor, student, and selected-seed diagnostics. Gradient proposals are
    accepted independently per cell only when the true cost decreases.
    """

    def __init__(self, evaluator: TrajectoryCostEvaluator, config: dict):
        self.evaluator = evaluator
        self.enabled = bool(config.get("enabled", True))
        self.include_student_seed = bool(config.get("include_student_seed", True))
        self.student_start_epoch = int(config.get("student_start_epoch", 0))
        self.anchor_noise_candidates = int(config.get("anchor_noise_candidates", 2))
        self.student_noise_candidates = int(config.get("student_noise_candidates", 1))
        self.noise_std = float(config.get("noise_std", 0.15))
        self.gradient_steps = int(config.get("gradient_steps", 2))
        self.gradient_step_size = float(config.get("gradient_step_size", 0.04))

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

    @staticmethod
    def _merge_costs(
        previous: CostBundle,
        proposal: CostBundle,
        accept: torch.Tensor,
    ) -> CostBundle:
        return CostBundle(
            **{
                name: torch.where(
                    accept, getattr(proposal, name), getattr(previous, name)
                ).detach()
                for name in previous.__dict__
            }
        )

    @staticmethod
    def _candidate_at(costs: CostBundle, index: int) -> CostBundle:
        return CostBundle(
            **{name: value[:, index].detach() for name, value in costs.__dict__.items()}
        )

    @staticmethod
    def _gather_candidate(costs: CostBundle, best: torch.Tensor) -> CostBundle:
        gather_index = best[:, None]
        return CostBundle(
            **{
                name: value.gather(1, gather_index).squeeze(1).detach()
                for name, value in costs.__dict__.items()
            }
        )

    def _select_candidates(
        self,
        candidates: torch.Tensor,
        depth: torch.Tensor,
        position: torch.Tensor,
        rotation: torch.Tensor,
        obs_body: torch.Tensor,
        map_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, CostBundle]:
        batch_size, candidate_count, channels, vertical, horizontal = candidates.shape
        flat = candidates.reshape(
            batch_size * candidate_count, channels, vertical, horizontal
        )

        def repeated(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat_interleave(candidate_count, dim=0)

        with torch.no_grad():
            flat_costs = self._evaluate(
                flat,
                repeated(depth),
                repeated(position),
                repeated(rotation),
                repeated(obs_body),
                repeated(map_id),
            ).detached()
            candidate_costs = CostBundle(
                **{
                    name: value.reshape(
                        batch_size, candidate_count, vertical, horizontal
                    )
                    for name, value in flat_costs.__dict__.items()
                }
            )
            best = candidate_costs.total.argmin(dim=1)
            gather_index = best[:, None, None, :, :].expand(
                -1, 1, channels, -1, -1
            )
            selected = candidates.gather(1, gather_index).squeeze(1)
        return selected.detach(), best.detach(), candidate_costs

    def refine(
        self,
        depth: torch.Tensor,
        position: torch.Tensor,
        rotation: torch.Tensor,
        obs_body: torch.Tensor,
        map_id: torch.Tensor,
        student_seed: torch.Tensor,
        epoch: int = 0,
    ) -> SelfRefinedTarget:
        student = student_seed.detach().float().clamp(-1.0, 1.0)
        anchor = torch.zeros_like(student)
        use_student = self.include_student_seed and epoch >= self.student_start_epoch

        seeds = [anchor]
        labels = ["anchor"]
        student_index = None
        if use_student:
            student_index = len(seeds)
            seeds.append(student)
            labels.append("student")
        for index in range(self.anchor_noise_candidates):
            seeds.append(
                (anchor + self.noise_std * torch.randn_like(anchor)).clamp(-1.0, 1.0)
            )
            labels.append(f"anchor_noise_{index}")
        if use_student:
            for index in range(self.student_noise_candidates):
                seeds.append(
                    (student + self.noise_std * torch.randn_like(student)).clamp(
                        -1.0, 1.0
                    )
                )
                labels.append(f"student_noise_{index}")

        candidates = torch.stack(seeds, dim=1)
        refined, best_source, candidate_costs = self._select_candidates(
            candidates, depth, position, rotation, obs_body, map_id
        )
        anchor_costs = self._candidate_at(candidate_costs, 0)
        if student_index is None:
            with torch.no_grad():
                student_costs = self._evaluate(
                    student, depth, position, rotation, obs_body, map_id
                ).detached()
        else:
            student_costs = self._candidate_at(candidate_costs, student_index)
        selected_seed_costs = self._gather_candidate(
            candidate_costs, best_source
        )

        if not self.enabled:
            return SelfRefinedTarget(
                raw_endstate=anchor,
                costs=anchor_costs,
                anchor_costs=anchor_costs,
                student_costs=student_costs,
                selected_seed_costs=anchor_costs,
                source_fractions={"anchor": 1.0},
                gradient_accept_rate=0.0,
            )

        current_costs = selected_seed_costs
        accepted = []
        for _ in range(self.gradient_steps):
            refined.requires_grad_(True)
            objective = robust_cost(
                self._evaluate(
                    refined, depth, position, rotation, obs_body, map_id
                ).total
            ).mean()
            gradient = torch.autograd.grad(objective, refined, only_inputs=True)[0]
            gradient = torch.nan_to_num(
                gradient, nan=0.0, posinf=1.0, neginf=-1.0
            )
            scale = gradient.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            proposal = (
                refined.detach() - self.gradient_step_size * gradient / scale
            ).clamp(-1.0, 1.0)
            with torch.no_grad():
                proposal_costs = self._evaluate(
                    proposal, depth, position, rotation, obs_body, map_id
                ).detached()
                accept = proposal_costs.total < current_costs.total
                refined = torch.where(accept[:, None], proposal, refined.detach())
                current_costs = self._merge_costs(
                    current_costs, proposal_costs, accept
                )
                accepted.append(float(accept.float().mean().item()))

        source_fractions = {
            label: float((best_source == index).float().mean().item())
            for index, label in enumerate(labels)
        }
        return SelfRefinedTarget(
            raw_endstate=refined.detach(),
            costs=current_costs,
            anchor_costs=anchor_costs,
            student_costs=student_costs,
            selected_seed_costs=selected_seed_costs,
            source_fractions=source_fractions,
            gradient_accept_rate=(sum(accepted) / len(accepted) if accepted else 0.0),
        )
