"""Aggregate trajectory signals used as candidate proxies ψ(s_t).

Each signal is a scalar function of the current state Z_t and the model
logits at step t. Signals are computed only over **revisable** positions
(i.e., unmasked positions that a corrector could resample); masked
positions are ignored.

Signals returned by `compute_signals`:

- `entropy` — mean conditional entropy H(x_i | x_{-i}, Z_t) over revisable i.
- `inverse_margin` — 1 − mean(p_1(i) − p_2(i)) over revisable i.
- `quality_mass_proxy` — 1 − mean(p_argmax(i)); crude quality proxy when
  no quality head is available.
- `unmasked_fraction` — u_t = |revisable| / D. Context scalar.
- `n_revisable` — raw count of revisable positions.

These are numpy-only so they can be computed from logits dumped to disk
without re-loading torch.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


_EPS = 1e-12


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=axis, keepdims=True)
    return e / np.clip(s, _EPS, None)


def _entropy_over_positions(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per position (nats). `probs` shape (*, V)."""
    p = np.clip(probs, _EPS, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def _top2_margin(probs: np.ndarray) -> np.ndarray:
    """p_1 − p_2 per position. `probs` shape (*, V)."""
    part = np.partition(probs, -2, axis=-1)
    p1 = part[..., -1]
    p2 = part[..., -2]
    return p1 - p2


def compute_signals(
    state: Dict[str, Any],
    logits: np.ndarray,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Compute aggregate trajectory signals at one step.

    Parameters
    ----------
    state : dict
        Must contain:
        - `tokens` (np.ndarray, shape (D,), int): current token ids; -1
          or `mask_id` for masked positions.
        - `mask_id` (int or None): id of the [MASK] token.
        - `revisable_mask` (np.ndarray, shape (D,), bool, optional): if
          provided, overrides the default "unmasked positions" definition
          of revisable.
    logits : np.ndarray, shape (D, V)
        Predictive logits at each position.
    meta : dict, optional
        Additional keys for future signals (e.g. quality scores from a
        quality head). Currently only `quality_scores` is read:
        - `quality_scores` (np.ndarray, shape (D,), float in [0, 1]):
          token-level quality estimate; if present, used for quality_mass.

    Returns
    -------
    signals : dict
        Scalar signals: entropy, inverse_margin, quality_mass_proxy,
        unmasked_fraction, n_revisable.
    """
    meta = meta or {}
    tokens = np.asarray(state["tokens"])
    mask_id = state.get("mask_id")

    D = tokens.shape[0]
    if "revisable_mask" in state:
        revisable = np.asarray(state["revisable_mask"], dtype=bool)
    else:
        if mask_id is None:
            revisable = np.ones(D, dtype=bool)
        else:
            revisable = tokens != mask_id

    n_rev = int(revisable.sum())
    if n_rev == 0:
        return {
            "entropy": 0.0,
            "inverse_margin": 0.0,
            "quality_mass_proxy": 0.0,
            "unmasked_fraction": 0.0,
            "n_revisable": 0,
        }

    probs = _softmax(logits[revisable], axis=-1)
    H = _entropy_over_positions(probs).mean()
    margin = _top2_margin(probs).mean()
    # quality mass proxy: 1 - mean probability of argmax token.
    p_argmax = probs.max(axis=-1).mean()

    quality_scores = meta.get("quality_scores")
    if quality_scores is not None:
        q = np.asarray(quality_scores)[revisable]
        quality_mass = float(1.0 - q.mean())
    else:
        quality_mass = float(1.0 - p_argmax)

    return {
        "entropy": float(H),
        "inverse_margin": float(1.0 - margin),
        "quality_mass_proxy": float(quality_mass),
        "unmasked_fraction": float(n_rev / D),
        "n_revisable": n_rev,
    }
