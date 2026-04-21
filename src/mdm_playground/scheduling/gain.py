"""Per-step one-loop marginal gain estimation (Protocol A).

`estimate_single_step_gain` takes the outputs of a base trajectory and a
branched trajectory (corrector applied at step t) and computes:

- Δ_t  = F(y_branch) − F(y_base)    one-loop quality gain
- TCR_t = Hamming(y_branch, y_base) / D    token-change rate

These are kept strictly separate: TCR_t is not Δ_t. See Q8 in
research/open_questions.md.

Quality functional F options (passed as a callable or a string key):
- 'neg_nll'   : negative per-token NLL under a reference scorer
                (requires `ref_nll` to be pre-computed externally and
                passed in the trajectory dict, to avoid loading a second
                model inside this module)
- 'token_acc' : fraction of positions matching a reference (for unit tests)
- callable    : F(tokens) -> float; must be comparable across branches

For real runs the caller computes NLL externally (on the HPC, using the
MDLM checkpoint as reference scorer) and passes it in the trajectory dict.
The surrogate mode passes a synthetic NLL.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Union

import numpy as np


def estimate_single_step_gain(
    y_base: Dict[str, Any],
    y_branch: Dict[str, Any],
    F: Union[str, Callable] = "neg_nll",
) -> Dict[str, float]:
    """Estimate per-step gain Δ_t and token-change rate TCR_t.

    Parameters
    ----------
    y_base : dict
        Output of the base trajectory (no correction). Required keys:
        - ``tokens``   : np.ndarray (D,) int — final token sequence
        - ``neg_nll``  : float — if F='neg_nll', the negative NLL of tokens
                          under a reference scorer (lower is better quality;
                          more negative means higher quality, i.e. lower NLL)
    y_branch : dict
        Output of the branched trajectory (one corrector loop at step t).
        Same keys as y_base.
    F : str or callable
        Quality functional. If str, must be a key in both trajectory dicts.
        If callable, F(trajectory_dict) -> float.

    Returns
    -------
    dict with keys:
        - ``delta``     : float — Δ_t = F(y_branch) − F(y_base)
        - ``tcr``       : float — token-change rate TCR_t ∈ [0, 1]
        - ``f_base``    : float — F(y_base)
        - ``f_branch``  : float — F(y_branch)
        - ``n_changed`` : int   — number of positions that changed
        - ``D``         : int   — sequence length
    """
    tokens_base = np.asarray(y_base["tokens"])
    tokens_branch = np.asarray(y_branch["tokens"])
    D = len(tokens_base)

    n_changed = int((tokens_base != tokens_branch).sum())
    tcr = n_changed / D if D > 0 else 0.0

    if callable(F):
        f_base = float(F(y_base))
        f_branch = float(F(y_branch))
    elif isinstance(F, str):
        f_base = float(y_base[F])
        f_branch = float(y_branch[F])
    else:
        raise ValueError(f"F must be str or callable, got {type(F)}")

    delta = f_branch - f_base

    return {
        "delta": delta,
        "tcr": tcr,
        "f_base": f_base,
        "f_branch": f_branch,
        "n_changed": n_changed,
        "D": D,
    }
