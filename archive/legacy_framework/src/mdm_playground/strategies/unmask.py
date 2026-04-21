"""Pure unmask strategies — never remask committed tokens."""
from __future__ import annotations

import torch
from torch import Tensor

from .base import BaseStrategy, SelectResult, StepState


class BaselineUnmaskStrategy(BaseStrategy):
    """Standard MDM — commit tokens one-way, never remask.

    At each step unmask the top-``k_new`` highest-confidence masked positions,
    where ``k_new = num_to_transfer - already_committed``.
    """

    def select(self, state: StepState) -> SelectResult:
        B, L = state.x_t.shape
        already = int(state.committed.sum(dim=-1).min().item())
        k_new = max(state.num_to_transfer - already, 0)

        if k_new == 0:
            return torch.zeros(B, 0, dtype=torch.long, device=state.x_t.device), None

        scores = state.confidence.clone().masked_fill(~state.mask_index, float("-inf"))
        k_new = min(k_new, int(state.mask_index.sum(dim=-1).min().item()))
        k_new = max(k_new, 1)
        return torch.topk(scores, k=k_new, dim=-1).indices, None
