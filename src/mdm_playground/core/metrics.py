"""Evaluation metrics (stub — extend as needed)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def mask_frac_curve(trajectory: List[Dict[str, Any]]) -> np.ndarray:
    """Return the fraction of masked tokens at each step.

    Args:
        trajectory: List of per-step dicts from a sampler result.

    Returns:
        ``[n_steps]`` float32 array.
    """
    fracs = []
    for step in trajectory:
        n_mask = len(step.get("mask_positions", []))
        n_total = max(len(step.get("tokens", [])), 1)
        fracs.append(n_mask / n_total)
    return np.array(fracs, dtype=np.float32)


def mean_confidence_curve(trajectory: List[Dict[str, Any]]) -> np.ndarray:
    """Mean per-token confidence at each step.

    Returns:
        ``[n_steps]`` float32 array.
    """
    return np.array(
        [np.mean(step.get("confidence", [0.0])) for step in trajectory],
        dtype=np.float32,
    )


def remask_count_curve(trajectory: List[Dict[str, Any]]) -> np.ndarray:
    """Number of positions remasked at each step.

    Returns:
        ``[n_steps]`` int32 array.
    """
    return np.array(
        [len(step.get("remask_indices", [])) for step in trajectory],
        dtype=np.int32,
    )
