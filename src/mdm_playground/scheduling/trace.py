"""Trace objects for Phase 0 corrector-scheduling audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationTrace:
    """Minimal backend trace needed for PF1-PF8 pre-flight checks.

    Arrays are intentionally typed as ``Any`` so CPU tests, NumPy-backed
    surrogates, and torch-backed real backends can share the same structure.
    """

    seed: int
    schedule: tuple[int, ...]
    tokens_by_step: list[Any]
    masks_by_step: list[Any] | None
    revisable_sets_by_step: list[Any] | None
    corrected_positions_by_step: list[Any] | None
    signal_masks_by_step: list[Any] | None
    signals_by_step: dict[str, list[float]] | None
    forward_pass_count: int | None
    rng_fingerprint_by_step: list[str] | None
    final_tokens: Any
    score: float | None
