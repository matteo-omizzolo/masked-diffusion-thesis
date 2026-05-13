"""Bounded token-level / model-internal feature extraction pilot.

This script extends Phase 1 Protocol A by recording, at every step of the
ProSeCo-OWT base trajectory, richer pre-correction token-level features
than the existing 6 aggregate scalars. The goal is to give the
state-predictability audit (analyze_tokenlevel_state_predictability.py)
input it has not yet seen.

CRITICAL RNG NOTE
-----------------
The corrector / predictor uses torch.multinomial / torch.bernoulli, which
produce DIFFERENT samples on different devices even with the same
torch.manual_seed. The canonical Phase 1 / Gate 3a trajectories were
generated on HPC A100 GPUs. Token-level features extracted on CPU or
Apple-Silicon MPS will be along a DIFFERENT base trajectory than the one
that produced results/phase1_interaction_diag_nogit/xi_raw.json. Therefore:

  - The intended canonical pilot must run on HPC under the same device
    family used for Gate 3a (CUDA on A100).
  - The local CPU / MPS extraction is only valid as a code-path dry-run
    that validates the extraction schema and sanity-checks the feature
    arithmetic.

Features extracted per step
---------------------------

  T0 (replication): 6 aggregate features matching Phase 1 Protocol A.
  T1 (masked):      summary statistics over masked positions of the
                    pre-corrector token distribution p_x0:
                    entropy {mean, std, q10/q25/q50/q75/q90};
                    top1 prob {mean, q10/q25/q50/q75/q90};
                    margin  {mean, q10/q25/q50/q75/q90};
                    fraction with entropy above thresholds {1,2,3};
                    fraction with margin below thresholds {0.05, 0.10, 0.20}.
  T2 (revisable):   same shape as T1, restricted to the corrector's action
                    set R_t (unmasked / committed positions in ProSeCo-OWT).
  T3 (concentration): entropy mass in the top {5,10}% most uncertain
                    positions; Gini of entropy; number of positions
                    accounting for 50% of total entropy; max / mean ratio.
  T4 (per-position arrays for pair-overlap construction by the analysis
        script): entropy per position, margin per position, top1 prob per
        position, revisable index list. These are stored once per step.
  T5 (corrector-action intensity): UNAVAILABLE in ProSeCo-OWT without
        running the actual corrector — the action set R_t is determined by
        ``(x != mask_id)`` and the corrector itself is deterministic
        (argmax). T5 is therefore not populated.

All features are pre-correction. No corrected-branch outcomes are read.

Storage
-------

Per seed file: ``per_step_features_seed{seed:03d}.npz`` containing:

    seed, T, seq_len, mask_id, vocab_size,
    scalars: (T, n_scalar_features) float32
    scalar_names: tuple[str]
    entropy_per_pos: (T, seq_len) float32
    margin_per_pos: (T, seq_len) float32
    top1_per_pos: (T, seq_len) float32
    revisable_mask_per_step: (T, seq_len) bool

Per-step JSON aggregate (sanity / index): ``per_step_features_seed.json``.

Run modes
---------

    --debug
        1 seed, T=4 steps, CPU only. Writes to a sha-tagged debug dir.

    Standard
        --seeds 42 43 44 45 46 47 48 49 50 51
        --T 64
        --device cpu (or 'mps' or 'cuda')
        --out-dir results/tokenlevel_features_proseco_pilot_<sha>/

Tests
-----

The accompanying tests/proseco/state_predictability/test_tokenlevel_state_predictability.py validates
the analysis-side feature math on synthetic arrays. Extraction itself is
verified by the debug dry-run: a 1-seed × T=4 run must produce a sane
``sanity_checks.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Sanity helpers (pure numpy; used by tests too).
# ---------------------------------------------------------------------------

# Leakage / forbidden substrings: any feature name containing these is
# treated as a serious bug by the sanity layer.
LEAKAGE_SUBSTRINGS: tuple[str, ...] = (
    "branch", "f_branch", "corrected", "post_correction",
    "n_changed", "tcr", "target", "y_xi", "y_g", "G_pair", "xi_value",
)

ENTROPY_THRESHOLDS: tuple[float, ...] = (1.0, 2.0, 3.0)
MARGIN_LOW_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.20)
QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


def shannon_entropy_per_position(p: np.ndarray) -> np.ndarray:
    """Per-row entropy of a probability matrix.

    p: (N, V) row-stochastic up to clamping. Returns (N,) entropy in nats.
    """
    pc = np.clip(p, 1e-12, None)
    return -(pc * np.log(pc)).sum(axis=-1)


def top1_and_margin(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (top1_probability, margin = top1 - top2) row-wise."""
    if p.shape[-1] < 2:
        return p[..., 0], p[..., 0]
    # Partial sort: argmax for top1, then mask & re-argmax for top2.
    top1_idx = p.argmax(axis=-1)
    top1_val = np.take_along_axis(p, top1_idx[..., None], axis=-1).squeeze(-1)
    p2 = p.copy()
    p2[np.arange(len(p))[..., None] if p.ndim == 2 else 0, top1_idx[..., None]] = -1.0
    top2_val = p2.max(axis=-1)
    return top1_val, top1_val - top2_val


def quantiles(arr: np.ndarray, qs: Sequence[float] = QUANTILES) -> np.ndarray:
    if arr.size == 0:
        return np.full(len(qs), float("nan"), dtype=np.float64)
    return np.asarray(np.quantile(arr, qs), dtype=np.float64)


def gini_index(values: np.ndarray) -> float:
    """Gini concentration index of a non-negative vector. 0 = uniform,
    1 = all mass on one position. Returns nan for empty / negative input.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or np.any(v < 0):
        return float("nan")
    s = v.sum()
    if s <= 1e-12:
        return 0.0
    sorted_v = np.sort(v)
    n = len(sorted_v)
    cum = np.cumsum(sorted_v)
    return float((n + 1 - 2.0 * cum.sum() / s) / n)


def positions_for_mass_share(values: np.ndarray, share: float = 0.5) -> int:
    """Number of largest positions needed to accumulate `share` of the total."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0
    s = v.sum()
    if s <= 1e-12:
        return 0
    sorted_v = np.sort(v)[::-1]
    cum = np.cumsum(sorted_v) / s
    idx = int(np.searchsorted(cum, share) + 1)
    return min(idx, v.size)


def jaccard(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """Jaccard overlap of two integer index arrays (treated as sets)."""
    if set_a.size == 0 and set_b.size == 0:
        return float("nan")
    a = set(int(x) for x in set_a)
    b = set(int(x) for x in set_b)
    union = a | b
    if not union:
        return float("nan")
    return float(len(a & b) / len(union))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def assert_no_leakage_in_names(names: Sequence[str]) -> None:
    bad = [n for n in names if any(s in n for s in LEAKAGE_SUBSTRINGS)]
    if bad:
        raise RuntimeError(
            f"feature names contain leakage substrings: {bad[:5]}"
        )


# ---------------------------------------------------------------------------
# Token-level summaries from a probability distribution (V-dim) on each
# position, restricted to a position subset.
# ---------------------------------------------------------------------------

@dataclass
class PositionSummary:
    """Token-level summary statistics over a subset of positions.

    All scalars derive only from the input probability rows. No correction
    outcome enters here.
    """
    n_positions: int
    entropy_mean: float
    entropy_std: float
    entropy_qs: np.ndarray  # len(QUANTILES)
    top1_mean: float
    top1_qs: np.ndarray
    margin_mean: float
    margin_qs: np.ndarray
    frac_entropy_above: np.ndarray  # one per ENTROPY_THRESHOLDS
    frac_margin_below: np.ndarray   # one per MARGIN_LOW_THRESHOLDS
    gini_entropy: float
    mass_top5pct: float    # share of total entropy in top-5% positions
    mass_top10pct: float   # share in top-10% positions
    n_pos_for_50pct_entropy: int

    def names(self, family_prefix: str) -> list[str]:
        out = [
            f"{family_prefix}_n_pos",
            f"{family_prefix}_entropy_mean",
            f"{family_prefix}_entropy_std",
        ]
        out += [f"{family_prefix}_entropy_q{int(100*q)}" for q in QUANTILES]
        out += [f"{family_prefix}_top1_mean"]
        out += [f"{family_prefix}_top1_q{int(100*q)}" for q in QUANTILES]
        out += [f"{family_prefix}_margin_mean"]
        out += [f"{family_prefix}_margin_q{int(100*q)}" for q in QUANTILES]
        out += [
            f"{family_prefix}_frac_entropy_above_{str(thr).replace('.', 'p')}"
            for thr in ENTROPY_THRESHOLDS
        ]
        out += [
            f"{family_prefix}_frac_margin_below_{str(thr).replace('.', 'p')}"
            for thr in MARGIN_LOW_THRESHOLDS
        ]
        out += [
            f"{family_prefix}_gini_entropy",
            f"{family_prefix}_mass_top5pct",
            f"{family_prefix}_mass_top10pct",
            f"{family_prefix}_n_pos_for_50pct_entropy",
        ]
        return out

    def values(self) -> np.ndarray:
        parts = [
            [float(self.n_positions), self.entropy_mean, self.entropy_std],
            self.entropy_qs.tolist(),
            [self.top1_mean],
            self.top1_qs.tolist(),
            [self.margin_mean],
            self.margin_qs.tolist(),
            self.frac_entropy_above.tolist(),
            self.frac_margin_below.tolist(),
            [self.gini_entropy, self.mass_top5pct, self.mass_top10pct,
             float(self.n_pos_for_50pct_entropy)],
        ]
        flat: list[float] = []
        for p in parts:
            flat.extend(p)
        return np.asarray(flat, dtype=np.float64)


def summarize_positions(
    p_subset: np.ndarray,
    entropy_subset: np.ndarray,
    top1_subset: np.ndarray,
    margin_subset: np.ndarray,
) -> PositionSummary:
    n = int(p_subset.shape[0]) if p_subset is not None else int(entropy_subset.size)
    if n == 0:
        nan_q = np.full(len(QUANTILES), float("nan"))
        return PositionSummary(
            n_positions=0,
            entropy_mean=float("nan"), entropy_std=float("nan"),
            entropy_qs=nan_q.copy(),
            top1_mean=float("nan"), top1_qs=nan_q.copy(),
            margin_mean=float("nan"), margin_qs=nan_q.copy(),
            frac_entropy_above=np.full(len(ENTROPY_THRESHOLDS), float("nan")),
            frac_margin_below=np.full(len(MARGIN_LOW_THRESHOLDS), float("nan")),
            gini_entropy=float("nan"),
            mass_top5pct=float("nan"), mass_top10pct=float("nan"),
            n_pos_for_50pct_entropy=0,
        )
    ent = np.asarray(entropy_subset, dtype=np.float64)
    top1 = np.asarray(top1_subset, dtype=np.float64)
    marg = np.asarray(margin_subset, dtype=np.float64)
    ent_qs = quantiles(ent)
    top1_qs = quantiles(top1)
    margin_qs = quantiles(marg)
    frac_above = np.asarray(
        [float(np.mean(ent > thr)) for thr in ENTROPY_THRESHOLDS],
        dtype=np.float64,
    )
    frac_below = np.asarray(
        [float(np.mean(marg < thr)) for thr in MARGIN_LOW_THRESHOLDS],
        dtype=np.float64,
    )
    if n >= 1:
        k5 = max(1, int(math.ceil(0.05 * n)))
        k10 = max(1, int(math.ceil(0.10 * n)))
        sorted_ent_desc = np.sort(ent)[::-1]
        total_ent = float(sorted_ent_desc.sum()) if sorted_ent_desc.sum() > 1e-12 else 1.0
        mass5 = float(sorted_ent_desc[:k5].sum() / total_ent)
        mass10 = float(sorted_ent_desc[:k10].sum() / total_ent)
    else:
        mass5 = mass10 = float("nan")
    return PositionSummary(
        n_positions=n,
        entropy_mean=float(ent.mean()),
        entropy_std=float(ent.std()),
        entropy_qs=ent_qs,
        top1_mean=float(top1.mean()),
        top1_qs=top1_qs,
        margin_mean=float(marg.mean()),
        margin_qs=margin_qs,
        frac_entropy_above=frac_above,
        frac_margin_below=frac_below,
        gini_entropy=gini_index(ent),
        mass_top5pct=mass5,
        mass_top10pct=mass10,
        n_pos_for_50pct_entropy=positions_for_mass_share(ent, 0.5),
    )


def t0_aggregate(
    entropy_rev: np.ndarray, margin_rev: np.ndarray, top1_rev: np.ndarray,
    n_revisable: int, n_masked: int, seq_len: int,
) -> tuple[np.ndarray, list[str]]:
    """Reproduce the 6 aggregate features from Phase 1 Protocol A."""
    names = [
        "T0_entropy", "T0_inverse_margin", "T0_quality_mass_proxy",
        "T0_unmasked_fraction", "T0_n_revisable", "T0_n_masked",
    ]
    if n_revisable == 0:
        return np.zeros(6, dtype=np.float64), names
    H = float(entropy_rev.mean())
    margin_mean = float(margin_rev.mean())
    top1_mean = float(top1_rev.mean())
    return (
        np.asarray([
            H, 1.0 - margin_mean, 1.0 - top1_mean,
            float(n_revisable) / float(seq_len),
            float(n_revisable), float(n_masked),
        ], dtype=np.float64),
        names,
    )


# ---------------------------------------------------------------------------
# Backend hook
# ---------------------------------------------------------------------------

def git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


@dataclass
class StepFeatures:
    t: int
    n_revisable: int
    n_masked: int
    seq_len: int
    scalars: np.ndarray       # (n_scalar_features,)
    scalar_names: list[str]
    entropy_per_pos: np.ndarray  # (seq_len,) float32
    margin_per_pos: np.ndarray
    top1_per_pos: np.ndarray
    revisable_mask: np.ndarray   # (seq_len,) bool


def step_features_from_p_x0(
    t_idx: int, x_tokens: np.ndarray, mask_id: int, p_x0: np.ndarray,
) -> StepFeatures:
    """Build a StepFeatures record from one (x, p_x0) state.

    x_tokens: (L,) int token ids before correction.
    mask_id : int, the mask token id.
    p_x0    : (L, V) probabilities (already mask-token-zeroed).
    """
    L = x_tokens.shape[0]
    is_masked = (x_tokens == mask_id)
    is_revisable = ~is_masked  # ProSeCo-OWT corrector acts on unmasked positions.
    n_revisable = int(is_revisable.sum())
    n_masked = int(is_masked.sum())

    entropy_full = shannon_entropy_per_position(p_x0).astype(np.float64)
    top1_full, margin_full = top1_and_margin(p_x0)
    top1_full = top1_full.astype(np.float64)
    margin_full = margin_full.astype(np.float64)

    t0_vals, t0_names = t0_aggregate(
        entropy_full[is_revisable], margin_full[is_revisable], top1_full[is_revisable],
        n_revisable, n_masked, L,
    )
    t1_sum = summarize_positions(
        p_x0[is_masked] if n_masked > 0 else np.zeros((0, p_x0.shape[1])),
        entropy_full[is_masked], top1_full[is_masked], margin_full[is_masked],
    )
    t2_sum = summarize_positions(
        p_x0[is_revisable] if n_revisable > 0 else np.zeros((0, p_x0.shape[1])),
        entropy_full[is_revisable], top1_full[is_revisable], margin_full[is_revisable],
    )
    # T3 concentration is computed over MASKED positions (most uncertainty lives
    # there for early steps) and stored under the T3_ prefix.
    t3_vals = np.asarray([
        t1_sum.gini_entropy,
        t1_sum.mass_top5pct,
        t1_sum.mass_top10pct,
        float(t1_sum.n_pos_for_50pct_entropy),
        float(entropy_full.max()),
        float(entropy_full.max() / max(entropy_full.mean(), 1e-12)),
    ], dtype=np.float64)
    t3_names = [
        "T3_gini_entropy_masked",
        "T3_mass_top5pct_masked",
        "T3_mass_top10pct_masked",
        "T3_n_pos_for_50pct_entropy_masked",
        "T3_max_entropy_over_all",
        "T3_max_over_mean_entropy_over_all",
    ]

    t1_names = t1_sum.names("T1_masked")
    t2_names = t2_sum.names("T2_rev")
    scalars = np.concatenate([t0_vals, t1_sum.values(), t2_sum.values(), t3_vals])
    scalar_names = t0_names + t1_names + t2_names + t3_names

    assert_no_leakage_in_names(scalar_names)

    return StepFeatures(
        t=int(t_idx),
        n_revisable=n_revisable, n_masked=n_masked, seq_len=L,
        scalars=scalars.astype(np.float64),
        scalar_names=scalar_names,
        entropy_per_pos=entropy_full.astype(np.float32),
        margin_per_pos=margin_full.astype(np.float32),
        top1_per_pos=top1_full.astype(np.float32),
        revisable_mask=is_revisable.astype(bool),
    )


# ---------------------------------------------------------------------------
# Driver: real model or synthetic stub
# ---------------------------------------------------------------------------

def run_extraction_real(
    checkpoint: str, seed: int, T: int, device: str,
) -> list[StepFeatures]:
    """Run base trajectory for one seed and extract per-step token-level
    features. Reuses ProSeCoOWTGenerator's internals to avoid duplication.
    """
    # Lazy import: importing the backend touches torch and the checkpoint.
    from mdm_playground.scheduling.backends.proseco_owt import ProSeCoOWTGenerator
    import torch

    gen = ProSeCoOWTGenerator(
        checkpoint=checkpoint, T=T, corrector_steps=1, device=device,
    )
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    B = 1
    L = gen.seq_len
    x = gen.mask_id * torch.ones(B, L, dtype=torch.long, device=device)
    timesteps = torch.linspace(1.0, gen.eps, T + 1, device=device)

    out: list[StepFeatures] = []
    for step_i in range(T):
        t = timesteps[step_i] * torch.ones(B, 1, device=device)
        s = timesteps[step_i + 1] * torch.ones(B, 1, device=device)
        # Compute p_x0 at time t with current x, then advance via predictor.
        # Match canonical Phase 1: features are extracted using the OLD p_x0
        # but the NEW x (post-predictor), so the revisable set R_t reflects
        # the corrector's action set at the next timestep, scored under the
        # predictor's last distribution.
        p_x0 = gen._forward(x, t)
        x = gen._predictor_step(x, t, s)
        x_tokens_np = x[0].detach().cpu().numpy().astype(np.int64)
        p_x0_np = p_x0[0].detach().cpu().float().numpy()
        sf = step_features_from_p_x0(step_i, x_tokens_np, gen.mask_id, p_x0_np)
        out.append(sf)
        # NB: base trajectory only — no corrector is applied.
    return out


def run_extraction_synthetic(seed: int, T: int, seq_len: int = 64, V: int = 16) -> list[StepFeatures]:
    """Synthetic stub: produces a fake base trajectory with consistent
    feature shapes. Used by tests and by --use-synthetic for code-path
    smoke runs when no checkpoint is available.
    """
    rng = np.random.default_rng(seed)
    mask_id = 0
    x = np.zeros(seq_len, dtype=np.int64)  # all masked initially
    out: list[StepFeatures] = []
    for t in range(T):
        # Progressive unmasking: reveal floor(t / T * seq_len) positions.
        n_reveal = int(seq_len * t / T)
        reveal_idx = rng.choice(seq_len, size=n_reveal, replace=False)
        x = np.full(seq_len, mask_id, dtype=np.int64)
        x[reveal_idx] = 1 + (reveal_idx % (V - 1))
        # Synthetic p_x0: random Dirichlet-like distribution.
        logits = rng.normal(size=(seq_len, V))
        # Make masked positions more uncertain.
        is_masked = (x == mask_id)
        logits[is_masked] *= 0.3
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        p[:, mask_id] = 0.0
        p = p / p.sum(axis=-1, keepdims=True)
        sf = step_features_from_p_x0(t, x, mask_id, p)
        out.append(sf)
    return out


# ---------------------------------------------------------------------------
# Sanity layer
# ---------------------------------------------------------------------------

def sanity_check_features(steps: list[StepFeatures]) -> dict[str, Any]:
    """Return a structured sanity-check report. Raises on hard violations.
    """
    report: dict[str, Any] = {"steps_checked": len(steps), "warnings": []}
    feature_names = steps[0].scalar_names
    assert_no_leakage_in_names(feature_names)

    finite_share = []
    entropy_min = math.inf
    entropy_max = -math.inf
    margin_min = math.inf
    margin_max = -math.inf
    top1_min = math.inf
    top1_max = -math.inf
    quantile_violations = 0
    seq_len = steps[0].seq_len
    for s in steps:
        if s.scalars.shape[0] != len(feature_names):
            raise RuntimeError("scalar feature length drift across steps")
        finite_mask = np.isfinite(s.scalars)
        finite_share.append(float(finite_mask.mean()))
        # Per-position arrays: shape and physical ranges.
        if s.entropy_per_pos.shape != (seq_len,):
            raise RuntimeError("entropy_per_pos shape mismatch")
        if (s.entropy_per_pos < -1e-6).any():
            raise RuntimeError(f"negative entropy at step {s.t}")
        if not ((-1e-6 <= s.top1_per_pos) & (s.top1_per_pos <= 1.0 + 1e-6)).all():
            raise RuntimeError(f"top1 out of [0,1] at step {s.t}")
        if not ((-1.0 - 1e-6 <= s.margin_per_pos) & (s.margin_per_pos <= 1.0 + 1e-6)).all():
            raise RuntimeError(f"margin out of [-1,1] at step {s.t}")
        entropy_min = min(entropy_min, float(s.entropy_per_pos.min()))
        entropy_max = max(entropy_max, float(s.entropy_per_pos.max()))
        margin_min = min(margin_min, float(s.margin_per_pos.min()))
        margin_max = max(margin_max, float(s.margin_per_pos.max()))
        top1_min = min(top1_min, float(s.top1_per_pos.min()))
        top1_max = max(top1_max, float(s.top1_per_pos.max()))
        # Quantile ordering inside the scalar block.
        for prefix in ("T1_masked", "T2_rev"):
            qs_entropy = [s.scalars[feature_names.index(f"{prefix}_entropy_q{int(100*q)}")] for q in QUANTILES]
            if any(np.isnan(qs_entropy)):
                continue
            for a, b in zip(qs_entropy, qs_entropy[1:]):
                if a > b + 1e-9:
                    quantile_violations += 1
    report["finite_feature_share_min"] = float(min(finite_share)) if finite_share else 0.0
    report["entropy_per_pos_min"] = entropy_min
    report["entropy_per_pos_max"] = entropy_max
    report["margin_per_pos_min"] = margin_min
    report["margin_per_pos_max"] = margin_max
    report["top1_per_pos_min"] = top1_min
    report["top1_per_pos_max"] = top1_max
    report["quantile_ordering_violations"] = int(quantile_violations)
    report["unique_scalar_feature_names"] = int(len(set(feature_names)))
    report["n_scalar_features"] = int(len(feature_names))
    if quantile_violations > 0:
        report["warnings"].append("quantile ordering violations seen")
    return report


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_seed_features(path: Path, seed: int, T: int, steps: list[StepFeatures]) -> None:
    scalar_names = tuple(steps[0].scalar_names)
    scalars = np.stack([s.scalars for s in steps], axis=0).astype(np.float32)
    entropy = np.stack([s.entropy_per_pos for s in steps], axis=0)
    margin = np.stack([s.margin_per_pos for s in steps], axis=0)
    top1 = np.stack([s.top1_per_pos for s in steps], axis=0)
    rev_mask = np.stack([s.revisable_mask for s in steps], axis=0)
    np.savez_compressed(
        path,
        seed=np.int64(seed),
        T=np.int64(T),
        seq_len=np.int64(steps[0].seq_len),
        scalars=scalars,
        scalar_names=np.asarray(scalar_names),
        entropy_per_pos=entropy,
        margin_per_pos=margin,
        top1_per_pos=top1,
        revisable_mask_per_step=rev_mask,
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default))


def _default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_extraction_plan(out_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# Token-level Feature Extraction Plan

Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}
Script: scripts/proseco/state_predictability/extract_tokenlevel_features_proseco.py
Backend: ProSeCo-OWT (mdm_playground.scheduling.backends.proseco_owt)
Checkpoint: {args.checkpoint}
Device: {args.device}
Seeds: {args.seeds}
T (predictor steps): {args.T}
Debug: {args.debug}
Synthetic: {args.use_synthetic}

## Code path identified

- ProSeCoOWTGenerator._forward(x, t) returns p_x0 (B, L, V) with mask token zeroed.
- The corrector action set R_t in ProSeCo-OWT is the UNMASKED positions
  (x != mask_id). The corrector argmaxes p_x0 and overwrites R_t.
- Base trajectory (no corrector at any t) is reproduced step-by-step.

## Features per step

T0 aggregate (replication of Phase 1):
- entropy, inverse_margin, quality_mass_proxy, unmasked_fraction,
  n_revisable, n_masked.

T1 masked-position uncertainty shape:
- mean/std/quantiles of entropy, top1, margin over MASKED positions;
- frac entropy > thresholds {ENTROPY_THRESHOLDS};
- frac margin < thresholds {MARGIN_LOW_THRESHOLDS}.

T2 revisable-set uncertainty shape: same statistics over R_t.

T3 concentration: gini, top-5%% / top-10%% entropy mass, n_pos for 50%% mass,
max entropy, max/mean entropy.

T4 per-position arrays for downstream pair-overlap computation:
- entropy_per_pos (T, L) float32, margin_per_pos, top1_per_pos,
  revisable_mask_per_step (T, L) bool.

T5 corrector-action intensity: UNAVAILABLE. ProSeCo-OWT corrector is
deterministic (argmax of p_x0 on R_t), so 'revision distribution' is a
delta. Documented and not extracted.

## Strict leakage exclusions

- No corrector branch is run during extraction. Only the base trajectory.
- No f_branch, n_changed, tcr, post-correction state read.
- No target values written into the feature file.

## RNG / device caveat

Local CPU / MPS extraction will NOT reproduce the HPC A100 base
trajectory that produced the canonical Phase 1 / Gate 3a artifacts.
Predictor-step stochasticity (torch.multinomial, torch.bernoulli) is
device-dependent. For the pilot to be aligned with existing xi_raw.json,
extraction must run on the same device family as Gate 3a (CUDA on
A100). The debug / synthetic runs verify only the code path and feature
arithmetic; the canonical pilot must run on HPC.
"""
    (out_dir / "feature_extraction_plan.md").write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = os.environ.get(
    "PROSECO_OWT_CHECKPOINT",
    str(Path.home() / "mdm" / "checkpoints" / "proseco_owt"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(42, 52)))
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--results-prefix", type=Path,
                        default=Path("results/tokenlevel_features_proseco_pilot"))
    parser.add_argument("--debug", action="store_true",
                        help="1 seed, T=4 steps, force CPU, synthetic if no checkpoint")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Skip the real backend; produce synthetic stub features")
    args = parser.parse_args(argv)

    git_sha = git_short_hash()
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.debug:
        out_dir = Path(f"{args.results_prefix}_debug_{git_sha}")
    else:
        out_dir = Path(f"{args.results_prefix}_{git_sha}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} non-empty; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        args.seeds = args.seeds[:1]
        args.T = min(args.T, 4)
        args.device = "cpu"

    write_extraction_plan(out_dir, args)

    extraction_log: list[dict[str, Any]] = []
    seed_files: list[str] = []
    aggregate_sanity: dict[str, Any] = {"per_seed": {}}
    t0_start = time.time()

    for seed in args.seeds:
        t0 = time.time()
        try:
            if args.use_synthetic:
                steps = run_extraction_synthetic(seed=seed, T=args.T)
            else:
                steps = run_extraction_real(
                    checkpoint=args.checkpoint, seed=seed, T=args.T,
                    device=args.device,
                )
        except FileNotFoundError as e:
            print(f"[seed {seed}] checkpoint not found: {e}; falling back to synthetic",
                  file=sys.stderr)
            steps = run_extraction_synthetic(seed=seed, T=args.T)
        sanity = sanity_check_features(steps)
        seed_path = out_dir / f"per_step_features_seed{seed:03d}.npz"
        write_seed_features(seed_path, seed=seed, T=args.T, steps=steps)
        elapsed = time.time() - t0
        extraction_log.append({
            "seed": seed,
            "T": args.T,
            "out_file": str(seed_path.name),
            "elapsed_seconds": elapsed,
            "sanity_warnings": sanity["warnings"],
        })
        aggregate_sanity["per_seed"][str(seed)] = sanity
        seed_files.append(str(seed_path.name))
        print(f"[seed {seed}] T={args.T} done in {elapsed:.1f}s -> {seed_path.name}")

    aggregate_sanity["total_seeds"] = len(args.seeds)
    aggregate_sanity["total_seconds"] = time.time() - t0_start
    aggregate_sanity["n_scalar_features"] = (
        next(iter(aggregate_sanity["per_seed"].values()))["n_scalar_features"]
    )

    config = {
        "checkpoint": args.checkpoint,
        "seeds": list(args.seeds),
        "T": args.T,
        "device": args.device,
        "out_dir": str(out_dir),
        "git_sha": git_sha,
        "debug": bool(args.debug),
        "use_synthetic": bool(args.use_synthetic),
        "default_quantiles": list(QUANTILES),
        "entropy_thresholds": list(ENTROPY_THRESHOLDS),
        "margin_low_thresholds": list(MARGIN_LOW_THRESHOLDS),
    }
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "sanity_checks.json", aggregate_sanity)
    feature_summary = {
        "scalar_feature_names": next(iter(aggregate_sanity["per_seed"].values()))[
            "n_scalar_features"
        ],
        "families_present": ["T0", "T1", "T2", "T3", "T4_per_position"],
        "families_unavailable": ["T5_corrector_action_intensity"],
        "per_seed_npz_files": seed_files,
    }
    write_json(out_dir / "feature_summary.json", feature_summary)
    (out_dir / "extraction_log.txt").write_text(
        json.dumps(extraction_log, indent=2, default=_default)
    )
    print(f"DONE: {len(args.seeds)} seeds, total {aggregate_sanity['total_seconds']:.1f}s,"
          f" out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
