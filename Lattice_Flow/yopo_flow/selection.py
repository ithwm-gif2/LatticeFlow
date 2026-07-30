"""Continuity-aware selection over the fixed YOPO motion-primitive lattice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionResult:
    index: int
    raw_best_index: int
    endpoint_jump: float
    switched: bool
    adjusted_scores: np.ndarray


class ContinuityAwareSelector:
    """Add endpoint continuity and hysteresis after per-frame network inference.

    The neural policy remains conditioned only on the current depth image and
    ego state.  This selector operates on its 15 YOPO-compatible candidates and
    can be disabled for a clean ablation.  Hysteresis is released whenever the
    previous lattice cell is worse than the current best by more than the
    configured score margin.
    """

    def __init__(
        self,
        endpoint_weight: float = 0.02,
        hysteresis_margin: float = 0.05,
    ):
        self.endpoint_weight = float(endpoint_weight)
        self.hysteresis_margin = float(hysteresis_margin)
        self.previous_index: int | None = None
        self.previous_endpoint: np.ndarray | None = None
        self.switch_count = 0
        self.selection_count = 0
        self.endpoint_jumps: list[float] = []

    def reset(self) -> None:
        self.previous_index = None
        self.previous_endpoint = None
        self.switch_count = 0
        self.selection_count = 0
        self.endpoint_jumps.clear()

    def select(self, scores: np.ndarray, endpoints: np.ndarray) -> SelectionResult:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        endpoints = np.asarray(endpoints, dtype=np.float64)
        if endpoints.ndim != 2 or endpoints.shape[0] != scores.size:
            raise ValueError(
                f"Expected endpoints [N,D] aligned with {scores.size} scores, "
                f"got {endpoints.shape}"
            )
        raw_best = int(np.argmin(scores))
        adjusted = scores.copy()
        endpoint_jump = 0.0

        if self.previous_endpoint is not None:
            jumps = np.linalg.norm(endpoints[:, :3] - self.previous_endpoint[:3], axis=1)
            adjusted = adjusted + self.endpoint_weight * jumps
            candidate = int(np.argmin(adjusted))
            if (
                self.previous_index is not None
                and scores[self.previous_index] <= scores[raw_best] + self.hysteresis_margin
            ):
                candidate = int(self.previous_index)
            selected = candidate
            endpoint_jump = float(jumps[selected])
        else:
            selected = raw_best

        switched = self.previous_index is not None and selected != self.previous_index
        if switched:
            self.switch_count += 1
        self.selection_count += 1
        if self.previous_endpoint is not None:
            self.endpoint_jumps.append(endpoint_jump)
        self.previous_index = selected
        self.previous_endpoint = endpoints[selected].copy()
        return SelectionResult(
            index=selected,
            raw_best_index=raw_best,
            endpoint_jump=endpoint_jump,
            switched=switched,
            adjusted_scores=adjusted,
        )

    @property
    def switch_rate(self) -> float:
        return self.switch_count / max(self.selection_count - 1, 1)

    @property
    def mean_endpoint_jump(self) -> float:
        return float(np.mean(self.endpoint_jumps)) if self.endpoint_jumps else 0.0

    @property
    def max_endpoint_jump(self) -> float:
        return float(np.max(self.endpoint_jumps)) if self.endpoint_jumps else 0.0
