"""Base types shared by all strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# Return type: (unmask_indices [B,k], remask_indices [B,m] | None)
SelectResult = Tuple[Tensor, Optional[Tensor]]


@dataclass
class StepState:
    """Snapshot of diffusion state at one step — passed to every strategy.

    Attributes
    ----------
    x_t:           ``[B, L]`` current tokens; masked positions = ``mask_token_id``
    x0:            ``[B, L]`` argmax-predicted tokens (TPS head)
    token_logits:  ``[B, L, V]`` raw TPS logits
    confidence:    ``[B, L]`` normalised confidence in [0, 1]
    mask_index:    ``[B, L]`` bool — True where currently MASKED
    committed:     ``[B, L]`` bool — True where token has been committed
    step:          0-based step index within the current block
    total_steps:   Total steps per block
    num_to_transfer: Cumulative tokens to unmask by end of this step
    mask_token_id: Vocabulary id used as MASK
    """

    x_t: Tensor
    x0: Tensor
    token_logits: Tensor
    confidence: Tensor
    mask_index: Tensor
    committed: Tensor
    step: int
    total_steps: int
    num_to_transfer: int
    mask_token_id: int


class BaseStrategy(ABC):
    """Abstract base for all inference strategies."""

    @abstractmethod
    def select(self, state: StepState) -> SelectResult:
        """Return ``(unmask_pos, remask_pos)`` index tensors."""
        ...


def pad_indices(tensors: list[Tensor]) -> Tensor:
    """Stack variable-length 1-D tensors into ``[B, max_len]``, padding with ``-1``."""
    if not tensors:
        return torch.zeros(0, 0, dtype=torch.long)
    max_len = max(t.shape[0] for t in tensors)
    if max_len == 0:
        return torch.zeros(len(tensors), 0, dtype=torch.long, device=tensors[0].device)
    return torch.stack(
        [F.pad(t, (0, max_len - t.shape[0]), value=-1) for t in tensors]
    )
