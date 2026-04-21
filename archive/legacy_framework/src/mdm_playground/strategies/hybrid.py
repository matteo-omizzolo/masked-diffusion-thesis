"""Hybrid strategies — combine unmask + remask in one policy (RemeDi-style)."""
from __future__ import annotations

import torch

from .base import BaseStrategy, SelectResult, StepState, pad_indices


class RemediPolicyStrategy(BaseStrategy):
    """Replicates the published RemeDi inference loop exactly.

    At each step the top ``num_to_transfer`` positions by confidence are kept
    unmasked; *all other positions* (including previously committed ones) are
    remasked.  This is the self-reflective remasking from the ICLR 2026 paper.
    """

    def select(self, state: StepState) -> SelectResult:
        k = state.num_to_transfer
        B, L = state.x_t.shape

        unmask_pos = torch.topk(state.confidence, k=k, dim=-1).indices  # [B, k]

        keep = torch.zeros(B, L, dtype=torch.bool, device=state.x_t.device)
        keep.scatter_(1, unmask_pos, True)
        remask_pos = pad_indices(
            [keep[b].logical_not().nonzero(as_tuple=True)[0] for b in range(B)]
        )
        return unmask_pos, remask_pos
