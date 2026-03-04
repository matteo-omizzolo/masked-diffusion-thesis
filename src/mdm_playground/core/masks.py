"""Shared masking operations used across all methods."""
from __future__ import annotations

import torch
from torch import Tensor


def make_mask(shape: tuple[int, ...], mask_id: int, device: torch.device) -> Tensor:
    """Create a fully-masked token tensor of the given shape."""
    return torch.full(shape, fill_value=mask_id, dtype=torch.long, device=device)


def apply_mask(x: Tensor, positions: Tensor, mask_id: int) -> Tensor:
    """Set ``positions`` in ``x`` to ``mask_id`` (in-place, returns x)."""
    x.scatter_(1, positions, mask_id)
    return x


def mask_fraction(x: Tensor, mask_id: int) -> Tensor:
    """Return the fraction of masked tokens per batch row; shape ``[B]``."""
    return (x == mask_id).float().mean(dim=-1)


def gather_topk_masked(
    scores: Tensor,
    mask_index: Tensor,
    k: int,
) -> Tensor:
    """Return indices of the top-k scoring *masked* positions.

    Args:
        scores:     ``[B, L]`` score tensor (higher = prefer to unmask).
        mask_index: ``[B, L]`` bool — True where position is currently masked.
        k:          Number of positions to select.

    Returns:
        ``[B, k]`` index tensor.
    """
    masked_scores = scores.masked_fill(~mask_index, float("-inf"))
    k = min(k, int(mask_index.sum(dim=-1).min().item()))
    k = max(k, 1)
    return torch.topk(masked_scores, k=k, dim=-1).indices
