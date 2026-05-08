"""Scheduling package for signal-adaptive corrector allocation.

This package provides the four modular abstractions used by the Phase 1
entropy-proxy experiment (`docs/experiments/entropy_proxy_experiment.md`)
and Theorem A empirical calibration (`research/candidate_theorems.md`).

Pure-Python; no torch dependency at the top level so analysis scripts can
import without a GPU. Backend modules under `backends/` bring in torch
where actually needed.

Public API:

- `compute_signals(state, logits, meta)` → dict of aggregate signals
- `estimate_single_step_gain(y_base, y_branch, F, meta)` → per-step gain dict
- `allocate_budget(signal_trace, budget, policy_name, policy_kwargs)` → allocation
- `evaluate_schedule(allocation, generator, F)` → metrics
- `GenerationTrace` → optional backend trace for Phase 0 pre-flight audits
"""

from .signals import compute_signals
from .gain import estimate_single_step_gain
from .allocation import allocate_budget, ALLOCATION_POLICIES
from .evaluate import evaluate_schedule
from .trace import GenerationTrace

__all__ = [
    "compute_signals",
    "estimate_single_step_gain",
    "allocate_budget",
    "evaluate_schedule",
    "GenerationTrace",
    "ALLOCATION_POLICIES",
]
