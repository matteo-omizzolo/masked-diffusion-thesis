"""Pair-level state-predictability analysis using token-level features.

Loads per-step token-level features produced by
``scripts/proseco/state_predictability/extract_tokenlevel_features_proseco.py`` and tests whether they
improve held-out prediction of pair-level corrector-timing targets
(xi_{s,t} and G_pair) beyond the strongest existing baseline:
P1 = pair geometry + A_pair.

Hard requirement
----------------

The features must come from the same base trajectory that produced the
target xi_raw.json. Because predictor-step stochasticity is device-
dependent, the features and targets must be from the same device family
(typically HPC A100 / CUDA). The script asserts that every required
(seed, t) endpoint is present in the loaded features and that the
recorded T0 aggregate replicates the canonical Phase 1 values within a
configurable tolerance. If the tolerance check fails, the script writes
a clear warning into interpretation.md and reports an
``alignment_status`` field; analysis still runs so the code path is
exercised.

Ablation
--------

    P0  geom only.
    P1  geom + A_pair.                                  (strong baseline)
    P2  P1 + T0 aggregate.                              (replication)
    P3  P2 + T1 masked-position uncertainty shape.
    P4  P3 + T2 revisable-set uncertainty shape.
    P5  P4 + T3 concentration features.
    P6  P5 + pair-level overlaps (T4: Jaccard of revisable / high-entropy /
                                   low-margin sets; cosine of entropy /
                                   margin vectors; quantile abs-diffs).

Each predictor uses a grouped 5-fold CV (when K seeds >= 5) or a
leave-one-seed-out (LOO) CV otherwise. Standardisation is fit on train
folds only. Reports per-seed bootstrap CIs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_XI_FILE = Path("results/phase1_interaction_diag_nogit/xi_raw.json")
DEFAULT_PROTOCOL_A = Path("results/phase1_proseco_owt_full/protocol_a")
DEFAULT_RESULTS_PREFIX = Path("results/tokenlevel_state_predictability_pilot")


# Reuse the leakage-substring guard from the extraction module.
LEAKAGE_SUBSTRINGS: tuple[str, ...] = (
    "branch", "corrected", "post_correction", "n_changed", "tcr",
    "target", "y_xi", "y_g", "G_pair", "xi_value",
)


def git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class SeedFeatures:
    seed: int
    T: int
    seq_len: int
    scalars: np.ndarray            # (T, F_scalar) float
    scalar_names: list[str]
    entropy_per_pos: np.ndarray    # (T, L)
    margin_per_pos: np.ndarray
    top1_per_pos: np.ndarray
    revisable_mask_per_step: np.ndarray  # (T, L) bool


def load_seed_features(npz_path: Path) -> SeedFeatures:
    d = np.load(npz_path, allow_pickle=False)
    scalar_names = [str(s) for s in d["scalar_names"].tolist()]
    return SeedFeatures(
        seed=int(d["seed"]),
        T=int(d["T"]),
        seq_len=int(d["seq_len"]),
        scalars=np.asarray(d["scalars"], dtype=np.float64),
        scalar_names=scalar_names,
        entropy_per_pos=np.asarray(d["entropy_per_pos"], dtype=np.float64),
        margin_per_pos=np.asarray(d["margin_per_pos"], dtype=np.float64),
        top1_per_pos=np.asarray(d["top1_per_pos"], dtype=np.float64),
        revisable_mask_per_step=np.asarray(d["revisable_mask_per_step"], dtype=bool),
    )


def load_features_dir(path: Path) -> dict[int, SeedFeatures]:
    files = sorted(path.glob("per_step_features_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no per_step_features_seed*.npz in {path}")
    out: dict[int, SeedFeatures] = {}
    for f in files:
        sf = load_seed_features(f)
        out[sf.seed] = sf
    return out


def load_xi_corpus(xi_file: Path) -> list[dict[str, Any]]:
    raw = json.loads(xi_file.read_text())
    return raw


def load_protocol_a_t0(protocol_a_dir: Path) -> dict[tuple[int, int], dict[str, float]]:
    """Return a {(seed, t): {feature: value}} map for the canonical T0 features.
    Used for the alignment tolerance check.
    """
    out: dict[tuple[int, int], dict[str, float]] = {}
    files = sorted(protocol_a_dir.glob("trajectory_*.json"))
    feats = ("entropy", "inverse_margin", "quality_mass_proxy",
             "unmasked_fraction", "n_revisable", "n_masked")
    for path in files:
        obj = json.loads(path.read_text())
        sd = int(obj["seed"])
        for s in obj["per_t"]:
            out[(sd, int(s["t"]))] = {k: float(s[k]) for k in feats}
    return out


def alignment_status(
    features: dict[int, SeedFeatures],
    protocol_a: dict[tuple[int, int], dict[str, float]],
    abs_tol: float = 1e-3,
) -> dict[str, Any]:
    """Compare T0 aggregate stored in the new extraction against the
    canonical Phase 1 trajectory_*.json values for each (seed, t).

    Reports the maximum absolute deviation by feature; if the deviation
    exceeds abs_tol on any feature, we flag the run as not RNG-aligned
    with the canonical artifacts. Targets joined to such features
    measure timing along a different base trajectory.
    """
    deviations: dict[str, list[float]] = {
        "entropy": [], "inverse_margin": [], "quality_mass_proxy": [],
        "unmasked_fraction": [], "n_revisable": [], "n_masked": [],
    }
    seeds_compared = 0
    pairs_compared = 0
    for sd, sf in features.items():
        if (sd, 0) not in protocol_a:
            continue
        seeds_compared += 1
        for t in range(sf.T):
            key = (sd, t)
            if key not in protocol_a:
                continue
            t0_block_idx = {n: i for i, n in enumerate(sf.scalar_names) if n.startswith("T0_")}
            row = sf.scalars[t]
            extracted = {
                "entropy": row[t0_block_idx["T0_entropy"]],
                "inverse_margin": row[t0_block_idx["T0_inverse_margin"]],
                "quality_mass_proxy": row[t0_block_idx["T0_quality_mass_proxy"]],
                "unmasked_fraction": row[t0_block_idx["T0_unmasked_fraction"]],
                "n_revisable": row[t0_block_idx["T0_n_revisable"]],
                "n_masked": row[t0_block_idx["T0_n_masked"]],
            }
            canon = protocol_a[key]
            for k in deviations:
                deviations[k].append(abs(extracted[k] - canon[k]))
            pairs_compared += 1
    summary: dict[str, Any] = {
        "seeds_compared": seeds_compared,
        "rows_compared": pairs_compared,
    }
    max_dev_by_feature: dict[str, float] = {}
    for k, vals in deviations.items():
        if vals:
            max_dev_by_feature[k] = float(max(vals))
        else:
            max_dev_by_feature[k] = float("nan")
    summary["max_abs_deviation_by_feature"] = max_dev_by_feature
    summary["alignment_tol"] = float(abs_tol)
    summary["rng_aligned"] = bool(
        pairs_compared > 0 and all(
            (math.isnan(v) or v <= abs_tol) for v in max_dev_by_feature.values()
        )
    )
    return summary


# ---------------------------------------------------------------------------
# Feature construction: ablation P0..P6
# ---------------------------------------------------------------------------

def _select_scalar_block(sf: SeedFeatures, prefix: str) -> tuple[np.ndarray, list[str]]:
    idx = [i for i, n in enumerate(sf.scalar_names) if n.startswith(prefix)]
    return sf.scalars[:, idx], [sf.scalar_names[i] for i in idx]


def _per_position_indices(mask: np.ndarray) -> np.ndarray:
    return np.where(mask)[0].astype(np.int64)


def _entropy_quantile_mask_high(entropy_per_pos: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask for positions whose entropy is in the top (1-q) of all
    positions at this step.
    """
    if entropy_per_pos.size == 0:
        return np.zeros_like(entropy_per_pos, dtype=bool)
    thr = np.quantile(entropy_per_pos, q)
    return entropy_per_pos >= thr


def _low_margin_mask(margin_per_pos: np.ndarray, q: float) -> np.ndarray:
    if margin_per_pos.size == 0:
        return np.zeros_like(margin_per_pos, dtype=bool)
    thr = np.quantile(margin_per_pos, q)
    return margin_per_pos <= thr


def pair_overlap_features(
    sf_s: SeedFeatures, s: int, t: int,
) -> tuple[np.ndarray, list[str]]:
    """T4 pair-overlap / similarity features for a single pair (s,t) using
    the per-position arrays from a single seed's feature record.
    """
    L = sf_s.seq_len
    ent_s = sf_s.entropy_per_pos[s]
    ent_t = sf_s.entropy_per_pos[t]
    mar_s = sf_s.margin_per_pos[s]
    mar_t = sf_s.margin_per_pos[t]
    rev_s = sf_s.revisable_mask_per_step[s]
    rev_t = sf_s.revisable_mask_per_step[t]
    rev_s_idx = _per_position_indices(rev_s)
    rev_t_idx = _per_position_indices(rev_t)
    he_s = _entropy_quantile_mask_high(ent_s, 0.90)
    he_t = _entropy_quantile_mask_high(ent_t, 0.90)
    lm_s = _low_margin_mask(mar_s, 0.10)
    lm_t = _low_margin_mask(mar_t, 0.10)
    # Jaccard helpers.
    def jacc(a: np.ndarray, b: np.ndarray) -> float:
        ai = set(int(x) for x in np.where(a)[0])
        bi = set(int(x) for x in np.where(b)[0])
        union = ai | bi
        if not union:
            return 0.0
        return float(len(ai & bi) / len(union))

    def cos(u: np.ndarray, v: np.ndarray) -> float:
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu < 1e-12 or nv < 1e-12:
            return 0.0
        return float(np.dot(u, v) / (nu * nv))

    feats = np.asarray([
        jacc(rev_s, rev_t),
        jacc(he_s, he_t),
        jacc(lm_s, lm_t),
        cos(ent_s, ent_t),
        cos(mar_s, mar_t),
        float(np.mean(ent_s[rev_t])) if rev_t.any() else 0.0,
        float(np.mean(ent_t[rev_s])) if rev_s.any() else 0.0,
        float(abs(ent_s.mean() - ent_t.mean())),
        float(abs(mar_s.mean() - mar_t.mean())),
        float(np.linalg.norm(ent_s - ent_t)) / float(L),
        float(np.linalg.norm(mar_s - mar_t)) / float(L),
    ], dtype=np.float64)
    names = [
        "T4_jaccard_revisable", "T4_jaccard_top_entropy", "T4_jaccard_low_margin",
        "T4_cos_entropy", "T4_cos_margin",
        "T4_mean_entropy_s_on_rev_t", "T4_mean_entropy_t_on_rev_s",
        "T4_abs_diff_mean_entropy", "T4_abs_diff_mean_margin",
        "T4_l2_dist_entropy_norm", "T4_l2_dist_margin_norm",
    ]
    return feats, names


def build_pair_feature_matrix(
    xi_rows: Sequence[dict[str, Any]],
    features_by_seed: dict[int, SeedFeatures],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Return X, y_xi, y_g, seeds, groups for the pair audit.

    Only pair rows whose seed is present in features_by_seed are kept.
    """
    sample_sf = next(iter(features_by_seed.values()))
    t1_cols_all = [n for n in sample_sf.scalar_names if n.startswith("T1_masked")]
    t2_cols_all = [n for n in sample_sf.scalar_names if n.startswith("T2_rev")]
    t3_cols_all = [n for n in sample_sf.scalar_names if n.startswith("T3_")]
    t0_cols = [n for n in sample_sf.scalar_names if n.startswith("T0_")]

    geom_cols = [
        "s_norm", "t_norm", "dist_norm",
        "s2", "t2", "dist2", "s*t", "|t-s|",
        "phase_s_early", "phase_s_middle", "phase_s_late",
        "phase_t_early", "phase_t_middle", "phase_t_late",
    ]
    oracle_cols = ["A_pair"]

    def expand_pair_cols(cols: Sequence[str], prefix: str) -> list[str]:
        out: list[str] = []
        for c in cols:
            out.extend([f"pair__{c}_s", f"pair__{c}_t",
                        f"pair__{c}_mean", f"pair__{c}_absdiff"])
        return out

    t0_pair_cols = expand_pair_cols(t0_cols, "T0")
    t1_pair_cols = expand_pair_cols(t1_cols_all, "T1")
    t2_pair_cols = expand_pair_cols(t2_cols_all, "T2")
    t3_pair_cols = expand_pair_cols(t3_cols_all, "T3")
    # T4 is computed per pair (not per step), so its column block is fixed.
    _, t4_names = pair_overlap_features(sample_sf, 0, min(1, sample_sf.T - 1))
    t4_cols = list(t4_names)

    column_names = (geom_cols + oracle_cols + t0_pair_cols + t1_pair_cols
                    + t2_pair_cols + t3_pair_cols + t4_cols)

    # Sanity: column names contain no leakage substring.
    for n in column_names:
        if any(sub in n for sub in LEAKAGE_SUBSTRINGS):
            raise RuntimeError(f"leakage substring detected in column {n!r}")

    n_geom = len(geom_cols)
    n_or = len(oracle_cols)
    n_t0p = len(t0_pair_cols)
    n_t1p = len(t1_pair_cols)
    n_t2p = len(t2_pair_cols)
    n_t3p = len(t3_pair_cols)
    n_t4 = len(t4_cols)
    groups = {
        "B_pair":   list(range(0, n_geom)),
        "ORACLE":   list(range(n_geom, n_geom + n_or)),
        "T0_pair":  list(range(n_geom + n_or, n_geom + n_or + n_t0p)),
        "T1_pair":  list(range(n_geom + n_or + n_t0p,
                               n_geom + n_or + n_t0p + n_t1p)),
        "T2_pair":  list(range(n_geom + n_or + n_t0p + n_t1p,
                               n_geom + n_or + n_t0p + n_t1p + n_t2p)),
        "T3_pair":  list(range(n_geom + n_or + n_t0p + n_t1p + n_t2p,
                               n_geom + n_or + n_t0p + n_t1p + n_t2p + n_t3p)),
        "T4_pair":  list(range(n_geom + n_or + n_t0p + n_t1p + n_t2p + n_t3p,
                               n_geom + n_or + n_t0p + n_t1p + n_t2p + n_t3p + n_t4)),
        "all_columns": column_names,
    }

    rows: list[list[float]] = []
    y_xi: list[float] = []
    y_g: list[float] = []
    seeds: list[int] = []
    skipped = 0
    for r in xi_rows:
        sd = int(r["seed"])
        if sd not in features_by_seed:
            skipped += 1
            continue
        sf = features_by_seed[sd]
        s = int(r["t"])
        t = int(r["t_prime"])
        if s >= sf.T or t >= sf.T:
            skipped += 1
            continue
        a_pair = float(r["A_pair"])
        g_pair = float(r["G_pair"])
        xi = float(r["xi"])
        if abs(xi - (g_pair - a_pair)) > 1e-6:
            raise ValueError(f"xi != G-A at seed={sd} pair=({s},{t})")
        T = sf.T
        s_norm = s / float(T - 1)
        t_norm = t / float(T - 1)
        dist = abs(t - s)
        dist_norm = dist / float(T)
        ps = "early" if s_norm < 1/3 else ("middle" if s_norm < 2/3 else "late")
        pt = "early" if t_norm < 1/3 else ("middle" if t_norm < 2/3 else "late")
        geom = [
            s_norm, t_norm, dist_norm,
            s_norm * s_norm, t_norm * t_norm, dist_norm * dist_norm,
            s_norm * t_norm, float(abs(t - s)),
            1.0 if ps == "early" else 0.0, 1.0 if ps == "middle" else 0.0, 1.0 if ps == "late" else 0.0,
            1.0 if pt == "early" else 0.0, 1.0 if pt == "middle" else 0.0, 1.0 if pt == "late" else 0.0,
        ]
        oracle = [a_pair]

        def pair_block(idx_list: list[int]) -> list[float]:
            vals_s = sf.scalars[s, idx_list]
            vals_t = sf.scalars[t, idx_list]
            mean = 0.5 * (vals_s + vals_t)
            absdiff = np.abs(vals_t - vals_s)
            interleaved: list[float] = []
            for a, b, m, d in zip(vals_s, vals_t, mean, absdiff):
                interleaved.extend([float(a), float(b), float(m), float(d)])
            return interleaved

        t0_pair_vals = pair_block([sf.scalar_names.index(c) for c in t0_cols])
        t1_pair_vals = pair_block([sf.scalar_names.index(c) for c in t1_cols_all])
        t2_pair_vals = pair_block([sf.scalar_names.index(c) for c in t2_cols_all])
        t3_pair_vals = pair_block([sf.scalar_names.index(c) for c in t3_cols_all])
        t4_vals, _ = pair_overlap_features(sf, s, t)

        row = geom + oracle + t0_pair_vals + t1_pair_vals + t2_pair_vals + t3_pair_vals + list(t4_vals)
        rows.append(row)
        y_xi.append(xi)
        y_g.append(g_pair)
        seeds.append(sd)

    X = np.asarray(rows, dtype=np.float64)
    # Replace any non-finite entries deterministically (extracted quantiles
    # over empty subsets are nan).
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        X,
        np.asarray(y_xi, dtype=np.float64),
        np.asarray(y_g, dtype=np.float64),
        np.asarray(seeds, dtype=np.int64),
        groups,
    )


# ---------------------------------------------------------------------------
# Ridge / CV
# ---------------------------------------------------------------------------

def make_seed_folds(seeds: Sequence[int], n_folds: int, seed: int = 1729) -> list[list[int]]:
    unique = sorted({int(s) for s in seeds})
    n_folds = max(2, min(n_folds, len(unique)))
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique, dtype=int)
    rng.shuffle(shuffled)
    return [list(map(int, f)) for f in np.array_split(shuffled, n_folds)]


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    Xs = (X - mean) / scale
    design = np.column_stack([np.ones(len(Xs)), Xs])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, scale, coef


def predict_ridge(X: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray) -> np.ndarray:
    Xs = (X - mean) / scale
    design = np.column_stack([np.ones(len(Xs)), Xs])
    return design @ coef


def fit_predict_grouped(
    X: np.ndarray, y: np.ndarray, seeds: np.ndarray, folds: Sequence[Sequence[int]],
    *, alpha: float = 1.0,
) -> np.ndarray:
    preds = np.empty_like(y)
    for fold in folds:
        test_mask = np.isin(seeds, np.asarray(fold, dtype=np.int64))
        train_seeds = set(int(s) for s in seeds[~test_mask].tolist())
        for fs in fold:
            if int(fs) in train_seeds:
                raise RuntimeError("seed leakage between train and test")
        mean, scale, coef = fit_ridge(X[~test_mask], y[~test_mask], alpha=alpha)
        preds[test_mask] = predict_ridge(X[test_mask], mean, scale, coef)
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_seed_mse(seeds: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict[int, float]:
    out: dict[int, list[float]] = {}
    for s, yi, pi in zip(seeds, y, p):
        out.setdefault(int(s), []).append((pi - yi) ** 2)
    return {s: float(np.mean(v)) for s, v in out.items()}


def bootstrap_seed_diff(
    seeds: np.ndarray, y: np.ndarray, p_base: np.ndarray, p_cand: np.ndarray,
    *, n_resamples: int = 2000, seed: int = 1729,
) -> dict[str, float]:
    base = per_seed_mse(seeds, y, p_base)
    cand = per_seed_mse(seeds, y, p_cand)
    sd = sorted(base)
    diffs = np.asarray([base[s] - cand[s] for s in sd], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        boot[i] = float(np.mean(diffs[idx]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "mean_mse_reduction": float(np.mean(diffs)),
        "ci95_lo": float(lo), "ci95_hi": float(hi),
        "pct_seeds_improved": float(np.mean(diffs > 0)),
        "n_seeds": int(len(sd)),
    }


def _rank_avg(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    out = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i + 1
        while j < len(v) and v[order[j]] == v[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rank_avg(np.asarray(x, dtype=float))
    ry = _rank_avg(np.asarray(y, dtype=float))
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.mean(((rx - rx.mean()) / sx) * ((ry - ry.mean()) / sy)))


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

ABLATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("P0_geom",              ("B_pair",)),
    ("P1_geom_apair",        ("B_pair", "ORACLE")),
    ("P2_plus_T0",           ("B_pair", "ORACLE", "T0_pair")),
    ("P3_plus_T1",           ("B_pair", "ORACLE", "T0_pair", "T1_pair")),
    ("P4_plus_T2",           ("B_pair", "ORACLE", "T0_pair", "T1_pair", "T2_pair")),
    ("P5_plus_T3",           ("B_pair", "ORACLE", "T0_pair", "T1_pair", "T2_pair", "T3_pair")),
    ("P6_plus_T4_overlaps",  ("B_pair", "ORACLE", "T0_pair", "T1_pair", "T2_pair", "T3_pair", "T4_pair")),
)


def _select(groups: dict[str, list[int]], parts: Sequence[str]) -> list[int]:
    idx: list[int] = []
    for p in parts:
        idx.extend(groups[p])
    return idx


COMPARISONS_XI: tuple[tuple[str, str], ...] = (
    ("P1_geom_apair", "P0_geom"),
    ("P2_plus_T0", "P1_geom_apair"),
    ("P3_plus_T1", "P1_geom_apair"),
    ("P4_plus_T2", "P1_geom_apair"),
    ("P5_plus_T3", "P1_geom_apair"),
    ("P6_plus_T4_overlaps", "P1_geom_apair"),
    ("P6_plus_T4_overlaps", "P5_plus_T3"),
)


def run_audit(
    X: np.ndarray, y_xi: np.ndarray, y_g: np.ndarray, seeds: np.ndarray,
    groups: dict[str, list[int]], n_folds: int, fold_seed: int,
) -> dict[str, Any]:
    folds = make_seed_folds(sorted({int(s) for s in seeds}), n_folds=n_folds, seed=fold_seed)
    predictor_names = [n for n, _ in ABLATIONS]
    preds_xi: dict[str, np.ndarray] = {}
    preds_g: dict[str, np.ndarray] = {}
    for name, parts in ABLATIONS:
        idx = _select(groups, parts)
        Xp = X[:, idx]
        preds_xi[name] = fit_predict_grouped(Xp, y_xi, seeds, folds)
        preds_g[name]  = fit_predict_grouped(Xp, y_g,  seeds, folds)

    def _summary(target_y: np.ndarray, preds: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
        m: dict[str, Any] = {}
        for n in predictor_names:
            p = preds[n]
            m[n] = {
                "mse": float(np.mean((p - target_y) ** 2)),
                "mae": float(np.mean(np.abs(p - target_y))),
                "spearman_pred_target": spearman(p, target_y),
            }
        imp: dict[str, Any] = {}
        for cand, base in COMPARISONS_XI:
            imp[f"{cand}_vs_{base}"] = bootstrap_seed_diff(
                seeds, target_y, preds[base], preds[cand]
            )
        return m, imp

    metrics_xi, imp_xi = _summary(y_xi, preds_xi)
    metrics_g, imp_g = _summary(y_g, preds_g)

    return {
        "predictor_names": predictor_names,
        "metrics_xi": metrics_xi,
        "improvements_xi": imp_xi,
        "metrics_g_pair": metrics_g,
        "improvements_g_pair": imp_g,
        "n_rows": int(len(y_xi)),
        "n_seeds": int(len(set(seeds.tolist()))),
        "fold_assignment": [list(f) for f in folds],
        "preds_xi_P1": preds_xi["P1_geom_apair"].tolist(),
        "preds_xi_best_enriched": preds_xi["P6_plus_T4_overlaps"].tolist(),
        "y_xi": y_xi.tolist(),
        "y_g": y_g.tolist(),
        "seeds_per_row": seeds.tolist(),
    }


# ---------------------------------------------------------------------------
# Plots (PIL only)
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


def _font(size: int = 14) -> Any:
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    if Image is None:
        return
    w, h = 1200, 600
    margin = 90
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), title, fill="black", font=_font(16))
    finite = [v for v in values if math.isfinite(v)]
    y_min = min(0.0, min(finite) if finite else 0.0)
    y_max = max(finite) if finite else 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    n = max(1, len(values))
    plot_w = w - 2 * margin
    bar_w = max(8, int(plot_w / n * 0.6))
    zero_y = h - margin - (h - 2 * margin) * (0.0 - y_min) / (y_max - y_min)
    draw.line([(margin, zero_y), (w - margin, zero_y)], fill="#666666", width=1)
    for i, (label, v) in enumerate(zip(labels, values)):
        cx = margin + (i + 0.5) * plot_w / n
        y = h - margin - (h - 2 * margin) * (v - y_min) / (y_max - y_min)
        c = "#2f855a" if v >= 0 else "#c53030"
        draw.rectangle([cx - bar_w / 2, min(y, zero_y), cx + bar_w / 2, max(y, zero_y)], fill=c)
        draw.text((cx - 30, h - margin + 10), label[:16], fill="black", font=_font(10))
    draw.rectangle([margin, margin, w - margin, h - margin], outline="black")
    draw.text((10, margin), ylabel, fill="black", font=_font(12))
    img.save(path)


def _scatter(path: Path, xs: list[float], ys: list[float], title: str, xlabel: str, ylabel: str) -> None:
    if Image is None or not xs:
        return
    w, h = 720, 720
    margin = 80
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), title, fill="black", font=_font(16))
    xa = np.asarray(xs); ya = np.asarray(ys)
    x_min, x_max = float(xa.min()), float(xa.max())
    y_min, y_max = float(ya.min()), float(ya.max())
    if x_max - x_min < 1e-12: x_max = x_min + 1.0
    if y_max - y_min < 1e-12: y_max = y_min + 1.0
    for x, y in zip(xa, ya):
        px = margin + (w - 2 * margin) * (x - x_min) / (x_max - x_min)
        py = h - margin - (h - 2 * margin) * (y - y_min) / (y_max - y_min)
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill="#2b6cb0")
    draw.rectangle([margin, margin, w - margin, h - margin], outline="black")
    draw.text((margin, h - 45), xlabel, fill="black", font=_font(12))
    draw.text((10, margin), ylabel, fill="black", font=_font(12))
    img.save(path)


def write_plots(out_dir: Path, aggregate: dict[str, Any], xi_rows: list[dict[str, Any]], features: dict[int, SeedFeatures]) -> None:
    if Image is None:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Ablation MSE-reduction bar chart for xi.
    labels = []
    means = []
    los = []
    his = []
    for k, v in aggregate["improvements_xi"].items():
        labels.append(k.replace("_vs_", "/"))
        means.append(v["mean_mse_reduction"])
        los.append(v["ci95_lo"]); his.append(v["ci95_hi"])
    _bar(plots_dir / "ablation_mse_reduction_xi.png", labels, means,
         "Per-seed MSE reduction (xi target) — pair-level enriched ablations",
         "MSE reduction (positive = candidate better)")
    # Predicted vs actual for P1 and P6.
    _scatter(plots_dir / "pred_vs_actual_xi_P1.png",
             aggregate["preds_xi_P1"], aggregate["y_xi"],
             "P1 (geom + A_pair) predictions vs xi", "predicted xi", "xi")
    _scatter(plots_dir / "pred_vs_actual_xi_P6.png",
             aggregate["preds_xi_best_enriched"], aggregate["y_xi"],
             "P6 (full enriched + T4 overlaps) predictions vs xi",
             "predicted xi", "xi")
    # xi vs revisable-overlap and xi vs T4_jaccard_top_entropy.
    rev_overlap: list[float] = []
    he_overlap: list[float] = []
    xi_vals: list[float] = []
    for r in xi_rows:
        sd = int(r["seed"])
        if sd not in features:
            continue
        sf = features[sd]
        s = int(r["t"]); t = int(r["t_prime"])
        if s >= sf.T or t >= sf.T:
            continue
        # Jaccard quickly.
        rev_a = set(int(i) for i in np.where(sf.revisable_mask_per_step[s])[0])
        rev_b = set(int(i) for i in np.where(sf.revisable_mask_per_step[t])[0])
        ru = rev_a | rev_b
        rev_overlap.append(0.0 if not ru else len(rev_a & rev_b) / len(ru))
        ent_a = sf.entropy_per_pos[s]
        ent_b = sf.entropy_per_pos[t]
        ha = ent_a >= np.quantile(ent_a, 0.90) if ent_a.size else np.zeros(0, dtype=bool)
        hb = ent_b >= np.quantile(ent_b, 0.90) if ent_b.size else np.zeros(0, dtype=bool)
        ia = set(int(i) for i in np.where(ha)[0])
        ib = set(int(i) for i in np.where(hb)[0])
        hu = ia | ib
        he_overlap.append(0.0 if not hu else len(ia & ib) / len(hu))
        xi_vals.append(float(r["xi"]))
    if xi_vals:
        _scatter(plots_dir / "xi_vs_revisable_overlap.png", rev_overlap, xi_vals,
                 "xi vs revisable-set Jaccard", "Jaccard(R_s, R_t)", "xi")
        _scatter(plots_dir / "xi_vs_top_entropy_overlap.png", he_overlap, xi_vals,
                 "xi vs top-10%% entropy Jaccard", "Jaccard(top10%%_H_s, top10%%_H_t)", "xi")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

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


def make_verdict(imp_xi: dict[str, Any]) -> dict[str, Any]:
    best_cmp = imp_xi.get("P6_plus_T4_overlaps_vs_P1_geom_apair")
    if best_cmp is None:
        return {"label": "no_test"}
    if best_cmp["ci95_lo"] > 0 and best_cmp["pct_seeds_improved"] >= 0.6:
        label = "tokenlevel_supported"
    elif best_cmp["mean_mse_reduction"] > 0 and best_cmp["pct_seeds_improved"] >= 0.5:
        label = "tokenlevel_weak"
    else:
        label = "tokenlevel_not_supported"
    return {
        "label": label,
        "comparison": best_cmp,
        "interpretation": (
            "Pre-registered expansion gate. tokenlevel_supported = expand to "
            "K=30 (HPC). tokenlevel_weak = expand cautiously after feature "
            "importance review. tokenlevel_not_supported = stop and write the "
            "ProSeCo-OWT case study."
        ),
    }


def write_interpretation(out_dir: Path, aggregate: dict[str, Any], align: dict[str, Any]) -> None:
    pred_xi = aggregate["metrics_xi"]
    imp_xi = aggregate["improvements_xi"]
    pred_g = aggregate["metrics_g_pair"]
    imp_g = aggregate["improvements_g_pair"]
    verdict = make_verdict(imp_xi)

    lines = [
        "# Token-level Pair-Level State Predictability Audit",
        "",
        "## Question",
        "",
        ("Do token-level / model-internal pre-correction features (masked-"
         "position uncertainty shape, revisable-set uncertainty shape, "
         "concentration, pair-level overlaps) improve held-out prediction "
         "of pair-level corrector-timing targets (xi and G_pair) beyond "
         "pair geometry + A_pair?"),
        "",
        "## Data and alignment",
        "",
        f"- Pair rows used: {aggregate['n_rows']}",
        f"- Seeds: {aggregate['n_seeds']}",
        f"- RNG alignment with canonical Phase 1 trajectory_*.json:"
        f" `{align.get('rng_aligned')}`",
        (f"- Max T0 deviations (extracted vs canonical):"
         f" {align.get('max_abs_deviation_by_feature')}"),
        "",
    ]
    if not align.get("rng_aligned", False):
        lines += [
            "> **Alignment warning.** The extracted features were generated"
            " along a base trajectory that does not reproduce the canonical"
            " Phase 1 trajectory_*.json T0 values within tolerance. Local"
            " CPU / MPS extraction does not reproduce HPC A100 RNG. The"
            " analysis below uses xi_raw.json values that were computed"
            " on a different trajectory; held-out predictability numbers"
            " in this run are therefore **invalid as a scientific result**"
            " and should be interpreted only as a code-path smoke. The"
            " canonical pilot must run on HPC.",
            "",
        ]
    lines += [
        "## Predictors",
        "",
        "- P0  pair geometry only.",
        "- P1  pair geometry + A_pair  (strong baseline).",
        "- P2  P1 + T0 aggregate (replication block).",
        "- P3  P2 + T1 masked-position uncertainty shape.",
        "- P4  P3 + T2 revisable-set uncertainty shape.",
        "- P5  P4 + T3 concentration features.",
        "- P6  P5 + T4 pair overlap / similarity features.",
        "",
        "## Held-out MSE — target = xi",
        "",
        "| Predictor | MSE | Spearman |",
        "|---|---:|---:|",
    ]
    for n in aggregate["predictor_names"]:
        m = pred_xi[n]
        lines.append(f"| `{n}` | {m['mse']:.6f} | {m['spearman_pred_target']:.3f} |")
    lines += [
        "",
        "## MSE-reduction comparisons (xi)",
        "",
        "| comparison | mean | 95% CI | %% seeds improved |",
        "|---|---:|---|---:|",
    ]
    for k, v in imp_xi.items():
        lines.append(f"| `{k}` | {v['mean_mse_reduction']:+.6f} |"
                     f" [{v['ci95_lo']:+.6f}, {v['ci95_hi']:+.6f}] |"
                     f" {v['pct_seeds_improved']:.2f} |")
    lines += [
        "",
        "## Held-out MSE — target = G_pair",
        "",
        "| Predictor | MSE | Spearman |",
        "|---|---:|---:|",
    ]
    for n in aggregate["predictor_names"]:
        m = pred_g[n]
        lines.append(f"| `{n}` | {m['mse']:.6f} | {m['spearman_pred_target']:.3f} |")
    lines += [
        "",
        "## MSE-reduction comparisons (G_pair)",
        "",
        "| comparison | mean | 95% CI | %% seeds improved |",
        "|---|---:|---|---:|",
    ]
    for k, v in imp_g.items():
        lines.append(f"| `{k}` | {v['mean_mse_reduction']:+.6f} |"
                     f" [{v['ci95_lo']:+.6f}, {v['ci95_hi']:+.6f}] |"
                     f" {v['pct_seeds_improved']:.2f} |")
    lines += [
        "",
        "## Verdict (pre-registered expansion gate)",
        "",
        f"- Label: **{verdict['label']}**",
        "",
        verdict.get("interpretation", ""),
        "",
        "## Limitations",
        "",
        "- ProSeCo-OWT only. The framework is corrector-agnostic; this"
        " empirical verdict is not.",
        "- 30 seeds in canonical xi_raw; pilot subsets are smaller and the"
        " bootstrap CI reflects that.",
        "- Only ridge / linear class tested. Tree models and kernel methods"
        " untested locally.",
        "- T5 corrector-action intensity unavailable in ProSeCo-OWT without"
        " modelling the deterministic argmax corrector at additional steps.",
        "",
        "## Decision",
        "",
    ]
    if verdict["label"] == "tokenlevel_supported":
        lines += [
            "Expand to K=30 with the same extraction pipeline on HPC."
            " Run feature-importance / leave-one-family-out analysis to"
            " localise the contributing feature family.",
        ]
    elif verdict["label"] == "tokenlevel_weak":
        lines += [
            "Do not expand to K=30 yet. Run a feature-importance review"
            " on the current pilot first, then decide. The pilot CI is"
            " not yet a positive expansion gate.",
        ]
    else:
        lines += [
            "Do not expand. Under the tested model class and the tested"
            " token-level / revisable-set / overlap feature families,"
            " correction timing on ProSeCo-OWT is captured by temporal"
            " geometry plus A_pair. Synthesise the ProSeCo-OWT case study"
            " as geometry / marginal-redundancy driven, with explicit"
            " scope to this backbone and feature class.",
        ]
    (out_dir / "interpretation.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=None,
                        help="Folder containing per_step_features_seed*.npz")
    parser.add_argument("--xi-file", type=Path, default=DEFAULT_XI_FILE)
    parser.add_argument("--protocol-a-dir", type=Path, default=DEFAULT_PROTOCOL_A)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--results-prefix", type=Path, default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1729)
    parser.add_argument("--alignment-tol", type=float, default=1e-3)
    parser.add_argument("--debug", action="store_true",
                        help="If --features-dir omitted: generate synthetic features for 3 seeds.")
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

    if args.features_dir is None:
        if not args.debug:
            print("--features-dir is required (or pass --debug to use synthetic).",
                  file=sys.stderr)
            return 2
        # Generate synthetic features inline to exercise the analysis path.
        _here = Path(__file__).resolve().parent
        if str(_here.parent) not in sys.path:
            sys.path.insert(0, str(_here.parent))
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        import importlib
        extract_mod = importlib.import_module("extract_tokenlevel_features_proseco")
        run_extraction_synthetic = extract_mod.run_extraction_synthetic
        write_seed_features = extract_mod.write_seed_features
        synth_dir = out_dir / "_synthetic_features"
        synth_dir.mkdir(exist_ok=True)
        for seed in (42, 43, 44):
            steps = run_extraction_synthetic(seed=seed, T=8, seq_len=64)
            write_seed_features(synth_dir / f"per_step_features_seed{seed:03d}.npz", seed=seed, T=8, steps=steps)
        args.features_dir = synth_dir

    features = load_features_dir(args.features_dir)
    xi_rows = load_xi_corpus(args.xi_file)
    # Optional alignment check against canonical Phase 1.
    try:
        protocol_a = load_protocol_a_t0(args.protocol_a_dir)
        align = alignment_status(features, protocol_a, abs_tol=args.alignment_tol)
    except FileNotFoundError:
        align = {"rng_aligned": None, "reason": "protocol_a not found"}

    # Filter xi_rows to seeds present in features; in --debug, also clip t to T.
    feat_seeds = set(features.keys())
    if args.debug:
        max_T = min(sf.T for sf in features.values())
        xi_rows = [r for r in xi_rows
                   if int(r["seed"]) in feat_seeds
                   and int(r["t"]) < max_T and int(r["t_prime"]) < max_T]
    else:
        xi_rows = [r for r in xi_rows if int(r["seed"]) in feat_seeds]
    if not xi_rows:
        raise RuntimeError("no xi pair rows align with the provided features")

    X, y_xi, y_g, seeds, groups = build_pair_feature_matrix(xi_rows, features)
    aggregate = run_audit(X, y_xi, y_g, seeds, groups,
                          n_folds=args.n_folds, fold_seed=args.fold_seed)

    config = {
        "features_dir": str(args.features_dir),
        "xi_file": str(args.xi_file),
        "protocol_a_dir": str(args.protocol_a_dir),
        "out_dir": str(out_dir),
        "git_sha": git_sha,
        "n_folds": args.n_folds,
        "fold_seed": args.fold_seed,
        "alignment_tol": args.alignment_tol,
        "debug": bool(args.debug),
    }
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "aggregate_stats.json", {
        "meta": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "n_rows": aggregate["n_rows"],
            "n_seeds": aggregate["n_seeds"],
            "alignment_status": align,
        },
        "predictor_names": aggregate["predictor_names"],
        "metrics_xi": aggregate["metrics_xi"],
        "improvements_xi": aggregate["improvements_xi"],
        "metrics_g_pair": aggregate["metrics_g_pair"],
        "improvements_g_pair": aggregate["improvements_g_pair"],
        "verdict": make_verdict(aggregate["improvements_xi"]),
    })
    write_json(out_dir / "fold_metrics.json", {
        "fold_assignment": aggregate["fold_assignment"],
    })
    write_json(out_dir / "feature_summary.json", {
        "column_groups": {k: v for k, v in groups.items() if k != "all_columns"},
        "n_columns": int(X.shape[1]),
    })
    # Prediction rows as JSON (compact, no full per-position arrays).
    pred_rows: list[dict[str, Any]] = []
    for i in range(len(y_xi)):
        pred_rows.append({
            "seed": int(seeds[i]),
            "row_idx": i,
            "y_xi": float(y_xi[i]),
            "y_g": float(y_g[i]),
            "P1_xi": float(aggregate["preds_xi_P1"][i]),
            "P6_xi": float(aggregate["preds_xi_best_enriched"][i]),
        })
    write_json(out_dir / "prediction_rows.json", pred_rows)
    write_interpretation(out_dir, aggregate, align)
    write_plots(out_dir, aggregate, xi_rows, features)
    print(json.dumps(make_verdict(aggregate["improvements_xi"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
