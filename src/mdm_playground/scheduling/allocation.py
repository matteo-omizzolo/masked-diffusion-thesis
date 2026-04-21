"""Budget allocation policies.

`allocate_budget(signal_trace, budget, policy_name, policy_kwargs)` maps a
sequence of per-step signals to a binary corrector allocation k_t ∈ {0, 1}
with ∑_t k_t = budget.

Policy names (ALLOCATION_POLICIES):

- ``uniform``          : place correctors at evenly spaced steps
- ``top_B``            : top-B steps by raw signal value (highest signal = most budget)
- ``bottom_B``         : bottom-B steps by signal (inverted top-B; useful as a sanity check)
- ``entropy_prop``     : probabilistic allocation proportional to signal (entropy)
- ``burn_in_gated``    : top-B from non-T_low steps; T_low defined by signal threshold or
                         explicit set
- ``margin_top_B``     : alias for top_B, intended for inverse-margin signal
- ``front``            : first B steps
- ``back``             : last B steps
- ``middle``           : center B steps
- ``random``           : random B steps (seed-controlled)

All policies return a dict {t: 1} for steps that receive a corrector loop.
Steps not in the output receive k_t = 0.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _top_B_indices(signal: np.ndarray, B: int, descending: bool = True) -> List[int]:
    """Return indices of the top-B values in `signal`."""
    order = np.argsort(signal)
    if descending:
        order = order[::-1]
    return list(int(i) for i in order[:B])


def _uniform_indices(T: int, B: int) -> List[int]:
    if B <= 0:
        return []
    if B >= T:
        return list(range(T))
    step = T / B
    return sorted(set(int(i * step) for i in range(B)))


def _middle_indices(T: int, B: int) -> List[int]:
    if B <= 0:
        return []
    center = T // 2
    half = B // 2
    start = max(0, center - half)
    end = min(T, start + B)
    start = max(0, end - B)
    return list(range(start, end))


def allocate_budget(
    signal_trace: Sequence[float],
    budget: int,
    policy_name: str = "top_B",
    policy_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[int, int]:
    """Allocate corrector budget across trajectory steps.

    Parameters
    ----------
    signal_trace : sequence of float, length T
        Per-step aggregate signal (e.g., entropy or inverse margin). Index 0
        corresponds to step t=1 (first predictor step).
    budget : int
        Total corrector budget B = ∑_t k_t. Must be ≤ T.
    policy_name : str
        Allocation policy. See module docstring for options.
    policy_kwargs : dict, optional
        Policy-specific arguments:
        - ``burn_in_gated``: ``low_gain_threshold`` (float, default 0.0) —
          exclude steps where signal ≤ this value;
          ``low_gain_steps`` (list[int], optional) — explicit T_low indices.
        - ``entropy_prop``: ``temperature`` (float, default 1.0) — softmax
          temperature for proportional sampling; ``seed`` (int, default 0).
        - ``random``: ``seed`` (int, default 0).

    Returns
    -------
    allocation : dict {step_index (0-based): 1}
        Steps assigned one corrector loop. len(allocation) == budget.
    """
    signal = np.asarray(signal_trace, dtype=float)
    T = len(signal)
    budget = min(budget, T)
    policy_kwargs = policy_kwargs or {}

    if budget <= 0:
        return {}

    if policy_name in (
        "top_B",
        "margin_top_B",
        "quality_top_B",
        "top_B_per_trajectory",
        "margin_top_B_per_trajectory",
        "quality_top_B_per_trajectory",
    ):
        indices = _top_B_indices(signal, budget, descending=True)

    elif policy_name in ("bottom_B", "bottom_B_per_trajectory"):
        indices = _top_B_indices(signal, budget, descending=False)

    elif policy_name == "uniform":
        indices = _uniform_indices(T, budget)

    elif policy_name == "front":
        indices = list(range(min(budget, T)))

    elif policy_name == "back":
        indices = list(range(max(0, T - budget), T))

    elif policy_name == "middle":
        indices = _middle_indices(T, budget)

    elif policy_name == "random":
        seed = policy_kwargs.get("seed", 0)
        rng = np.random.default_rng(seed)
        indices = sorted(int(i) for i in rng.choice(T, size=budget, replace=False))

    elif policy_name == "burn_in_gated":
        threshold = policy_kwargs.get("low_gain_threshold", 0.0)
        explicit_low = policy_kwargs.get("low_gain_steps", None)
        if explicit_low is not None:
            low_set = set(explicit_low)
        else:
            low_set = set(int(i) for i in np.where(signal <= threshold)[0])
        eligible = [i for i in range(T) if i not in low_set]
        if len(eligible) == 0:
            # Fallback: use uniform if all steps are in T_low
            eligible = list(range(T))
        eligible_signal = signal[eligible]
        sub_indices = _top_B_indices(eligible_signal, min(budget, len(eligible)))
        indices = sorted(eligible[i] for i in sub_indices)

    elif policy_name == "entropy_prop":
        temperature = policy_kwargs.get("temperature", 1.0)
        seed = policy_kwargs.get("seed", 0)
        rng = np.random.default_rng(seed)
        s = signal - signal.max()
        logits = s / max(temperature, 1e-9)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        indices = sorted(
            int(i)
            for i in rng.choice(T, size=min(budget, T), replace=False, p=probs)
        )

    else:
        raise ValueError(
            f"Unknown policy '{policy_name}'. "
            f"Choose from {list(ALLOCATION_POLICIES)}."
        )

    # Deduplicate and cap at budget
    seen, result = set(), {}
    for idx in indices:
        if idx not in seen and len(seen) < budget:
            result[int(idx)] = 1
            seen.add(idx)

    return result


ALLOCATION_POLICIES = {
    "top_B": "Top-B steps by signal (primary proxy-top-B from Theorem A)",
    "margin_top_B": "Top-B by inverse confidence margin (alias for top_B)",
    "quality_top_B": "Top-B by quality mass proxy (alias for top_B)",
    "top_B_per_trajectory": (
        "Top-B by per-trajectory signal (caller supplies per-seed signal_trace; "
        "avoids mean-profile collapse to positional policies). Phase 2b primary."
    ),
    "margin_top_B_per_trajectory": "Per-trajectory inverse-margin top-B.",
    "quality_top_B_per_trajectory": "Per-trajectory quality-mass top-B.",
    "bottom_B": "Bottom-B steps by signal (sanity-check inverted policy)",
    "bottom_B_per_trajectory": (
        "Per-trajectory bottom-B (inverted signal; tracks the Phase 1 finding on "
        "ProSeCo-OWT where entropy_bot_B apparently beat uniform under the "
        "A-proxy ranking — true G(S) effect is under audit)."
    ),
    "uniform": "Uniformly spaced steps (Theorem A baseline)",
    "front": "First B steps (front-loaded)",
    "back": "Last B steps (back-loaded)",
    "middle": "Middle B steps (middle-loaded)",
    "random": "Random B steps (seed-controlled)",
    "burn_in_gated": "Top-B excluding low-gain region T_low (Proposition B)",
    "entropy_prop": "Stochastic, proportional to signal (temperature-scaled)",
}
