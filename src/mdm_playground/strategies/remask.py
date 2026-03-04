"""Remask strategies — may undo previously committed tokens."""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import BaseStrategy, SelectResult, StepState, pad_indices


# ---------------------------------------------------------------------------
# Shared helper: select k_new tokens to unmask this step
# ---------------------------------------------------------------------------

def _new_unmask(state: StepState) -> tuple[int, Tensor]:
    """Return ``(k_new, scores)`` for the unmask part of a step."""
    already = int(state.committed.sum(dim=-1).min().item())
    k_new = max(state.num_to_transfer - already, 0)
    k_new = min(k_new, int(state.mask_index.sum(dim=-1).min().item()))
    scores = state.confidence.clone().masked_fill(~state.mask_index, float("-inf"))
    return k_new, scores


# ---------------------------------------------------------------------------
# Confidence threshold (PRISM-inspired)
# ---------------------------------------------------------------------------

class ConfidenceThresholdRemaskStrategy(BaseStrategy):
    """Unmask new top-k positions; remask committed tokens with confidence < tau.

    Args:
        tau: Threshold in ``[0, 1]``.
    """

    def __init__(self, tau: float = 0.3):
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        self.tau = tau

    def select(self, state: StepState) -> SelectResult:
        B = state.x_t.shape[0]
        k_new, scores = _new_unmask(state)

        unmask_pos = (
            torch.topk(scores, k=k_new, dim=-1).indices
            if k_new > 0
            else torch.zeros(B, 0, dtype=torch.long, device=state.x_t.device)
        )

        low_conf = (state.confidence < self.tau) & state.committed
        rows = [low_conf[b].nonzero(as_tuple=True)[0] for b in range(B)]
        if all(r.shape[0] == 0 for r in rows):
            return unmask_pos, None
        return unmask_pos, pad_indices(rows)


# ---------------------------------------------------------------------------
# Top-k low confidence
# ---------------------------------------------------------------------------

class TopKLowConfidenceRemaskStrategy(BaseStrategy):
    """Each step remask the ``k_remask`` lowest-confidence committed tokens.

    Args:
        k_remask: Number of committed tokens to remask per step.
    """

    def __init__(self, k_remask: int = 5):
        if k_remask < 0:
            raise ValueError(f"k_remask must be ≥ 0, got {k_remask}")
        self.k_remask = k_remask

    def select(self, state: StepState) -> SelectResult:
        B = state.x_t.shape[0]
        k_new, scores = _new_unmask(state)

        unmask_pos = (
            torch.topk(scores, k=k_new, dim=-1).indices
            if k_new > 0
            else torch.zeros(B, 0, dtype=torch.long, device=state.x_t.device)
        )

        k_r = min(self.k_remask, int(state.committed.sum(dim=-1).min().item()))
        if k_r <= 0:
            return unmask_pos, None

        committed_conf = state.confidence.clone().masked_fill(~state.committed, float("inf"))
        remask_pos = torch.topk(committed_conf, k=k_r, largest=False, dim=-1).indices
        return unmask_pos, remask_pos


# ---------------------------------------------------------------------------
# Scheduled remask (ReMDM-inspired)
# ---------------------------------------------------------------------------

class ScheduledRemaskStrategy(BaseStrategy):
    """Remask a decaying fraction of committed tokens each step.

    Schedules:
    - ``linear``:  ``p(t) = max_prob * (1 − t/T)``
    - ``cosine``:  ``p(t) = max_prob * 0.5 * (1 + cos(π t/T))``

    Args:
        max_remask_prob: Probability at step 0.
        schedule:        ``"linear"`` or ``"cosine"``.
    """

    def __init__(self, max_remask_prob: float = 0.1, schedule: str = "cosine"):
        if schedule not in ("linear", "cosine"):
            raise ValueError(f"schedule must be 'linear' or 'cosine', got {schedule!r}")
        self.max_remask_prob = max_remask_prob
        self.schedule = schedule

    def _prob(self, step: int, total: int) -> float:
        t = step / max(total - 1, 1)
        if self.schedule == "linear":
            return self.max_remask_prob * (1.0 - t)
        return self.max_remask_prob * 0.5 * (1.0 + math.cos(math.pi * t))

    def select(self, state: StepState) -> SelectResult:
        B = state.x_t.shape[0]
        k_new, scores = _new_unmask(state)

        unmask_pos = (
            torch.topk(scores, k=k_new, dim=-1).indices
            if k_new > 0
            else torch.zeros(B, 0, dtype=torch.long, device=state.x_t.device)
        )

        p = self._prob(state.step, state.total_steps)
        if p <= 0.0 or not state.committed.any():
            return unmask_pos, None

        noise = torch.rand_like(state.confidence)
        to_remask = (noise < p) & state.committed
        rows = [to_remask[b].nonzero(as_tuple=True)[0] for b in range(B)]
        if all(r.shape[0] == 0 for r in rows):
            return unmask_pos, None
        return unmask_pos, pad_indices(rows)
