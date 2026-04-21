"""Schedule evaluation: compute G(S) and A(S) for an allocation.

`evaluate_schedule` runs a generator with a given corrector allocation,
computes G(S) = F(y^S) − F(y_base) and the additive surrogate
A(S) = ∑_{t ∈ S} Δ_t, and returns both alongside residual r = G − A.

The generator object must implement a minimal interface::

    class Generator(Protocol):
        def run_base(self, seed: int) -> TrajectoryResult: ...
        def run_with_schedule(
            self, allocation: dict, seed: int
        ) -> TrajectoryResult: ...

where TrajectoryResult is any dict with at least::

    {
        "tokens": np.ndarray (D,),
        "neg_nll": float,          # or another F key
        "per_step_signals": list[dict],  # length T, each has signal values
    }

This module is backend-agnostic; the surrogate and the real MDLM backend
both implement the Generator protocol.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from .gain import estimate_single_step_gain


def evaluate_schedule(
    allocation: Dict[int, int],
    delta_trace: Dict[int, float],
    generator: Any,
    F: Union[str, Callable] = "neg_nll",
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate a corrector allocation schedule.

    Parameters
    ----------
    allocation : dict {step_index: 1}
        Corrector allocation from `allocate_budget`.
    delta_trace : dict {step_index: float}
        Pre-computed per-step Δ_t from Protocol A. Used to compute
        the additive surrogate A(S) = ∑_{t ∈ S} Δ_t.
    generator : object implementing Generator protocol
        Produces trajectory samples.
    F : str or callable
        Quality functional used to compute G(S).
    seed : int
        Random seed for the joint run.

    Returns
    -------
    metrics : dict
        - ``G``          : float — G(S) = F(y^S) − F(y_base)
        - ``A``          : float — A(S) = ∑_{t ∈ S} Δ_t
        - ``residual``   : float — G − A
        - ``f_base``     : float
        - ``f_schedule`` : float
        - ``schedule_steps`` : list[int]
        - ``budget``     : int
        - ``wall_time``  : float — seconds
    """
    t0 = time.time()

    y_base = generator.run_base(seed=seed)
    y_sched = generator.run_with_schedule(allocation=allocation, seed=seed)

    gain_info = estimate_single_step_gain(y_base, y_sched, F=F)
    G = gain_info["delta"]
    f_base = gain_info["f_base"]
    f_sched = gain_info["f_branch"]

    A = sum(delta_trace.get(t, 0.0) for t in allocation)
    residual = G - A

    return {
        "G": G,
        "A": A,
        "residual": residual,
        "f_base": f_base,
        "f_schedule": f_sched,
        "schedule_steps": sorted(allocation.keys()),
        "budget": len(allocation),
        "wall_time": time.time() - t0,
    }
