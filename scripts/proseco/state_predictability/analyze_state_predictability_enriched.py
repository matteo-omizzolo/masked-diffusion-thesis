"""Enriched-feature state-predictability audit for corrector timing.

This script extends the marginal audit (analyze_state_predictability.py) and
the pair audit (analyze_state_predictability_pair.py) by testing whether a
*richer* family of locally derived pre-correction features carries
incremental predictive signal beyond the cubic-time / pair-geometry
baselines.

Three target levels are evaluated under grouped seed K-fold:

    Marginal:   Y = delta_{i,t}                  (Phase 1 Protocol A)
    Pair:       Y = xi_{i,s,t}  and  Y = G_{i,{s,t}}  (Gate 3a + Phase 1)
    Schedule:   Y = G_i(S)                       (Phase 2b MC pool + Phase 1)

No new inference is run. All features are computed from existing artifacts.

Feature families (interpretable, statistically motivated):

  B  Baseline geometry (time / pair / schedule).
  S0 Original 6 aggregate pre-correction state features.
  S1 Revisable-set intensity ratios.
  S2 Causal temporal derivatives (h_t - h_{t-k}, rolling means).
  S3 Forward-looking temporal features (offline pre-decision: h_{t+k} - h_t,
     local curvature). These require running the base trajectory once,
     which is compatible with the open-loop scheduling design of the thesis.
  S4 Pair-specific contrasts (|h_s - h_t|, mean, similarity proxies)
     — used only for pair / schedule targets.

Models: ridge regression (alpha=1.0) with train-only standardization, and
a curated polynomial-degree-2 ridge over a small subset, used as an
interpretable non-linear probe.

Outputs land under results/state_predictability_enriched_<gitsha>/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_PROTOCOL_A = Path("results/phase1_proseco_owt_full/protocol_a")
DEFAULT_XI_FILE = Path("results/phase1_interaction_diag_nogit/xi_raw.json")
DEFAULT_MC_FILE = Path("results/phase2b_proseco_owt/mc_raw.json")
DEFAULT_RESULTS_PREFIX = Path("results/state_predictability_enriched")

STATE_FEATURES: tuple[str, ...] = (
    "entropy",
    "inverse_margin",
    "quality_mass_proxy",
    "unmasked_fraction",
    "n_revisable",
    "n_masked",
)

# Excluded as leakage: derived from the corrected branch or terminal outcome.
LEAKAGE_FIELDS: tuple[str, ...] = (
    "n_changed",       # tokens changed by corrector
    "tcr",             # token change rate (post-correction)
    "f_branch",        # corrected branch terminal F
)

# f_base is constant per seed (verified empirically). It is therefore not a
# per-step feature; including it would be a pure seed indicator and is
# excluded to avoid trivial-seed-effect leakage in the grouped CV.
SEED_CONSTANT_FIELDS: tuple[str, ...] = ("f_base",)


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
# Schema and feature audit
# ---------------------------------------------------------------------------

FEATURE_AUDIT: dict[str, dict[str, Any]] = {
    "entropy":            {"class": "deployable_offline_pre_correction", "family": "S0"},
    "inverse_margin":     {"class": "deployable_offline_pre_correction", "family": "S0"},
    "quality_mass_proxy": {"class": "deployable_offline_pre_correction", "family": "S0"},
    "unmasked_fraction":  {"class": "deployable_offline_pre_correction", "family": "S0"},
    "n_revisable":        {"class": "deployable_offline_pre_correction", "family": "S0"},
    "n_masked":           {"class": "deployable_offline_pre_correction", "family": "S0"},
    "delta":              {"class": "single_step_correction_value",      "family": "target/oracle_diagnostic"},
    "tcr":                {"class": "LEAKAGE_post_correction",            "family": "excluded"},
    "f_branch":           {"class": "LEAKAGE_post_correction",            "family": "excluded"},
    "n_changed":          {"class": "LEAKAGE_post_correction",            "family": "excluded"},
    "f_base":             {"class": "seed_constant_pre_correction",       "family": "excluded (per-seed only)"},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepRow:
    seed: int
    t: int
    T: int
    delta: float
    state: dict[str, float]

    @property
    def t_norm(self) -> float:
        return self.t / float(self.T - 1) if self.T > 1 else 0.0


@dataclass
class TrajectoryEnriched:
    """Cached enriched per-step features for one seed."""
    seed: int
    T: int
    raw: dict[str, np.ndarray]                # field -> length-T array
    derived: dict[str, np.ndarray] = field(default_factory=dict)

    def deltas(self) -> np.ndarray:
        return self.raw["delta"]

    def enriched(self, t: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, arr in self.raw.items():
            if name in LEAKAGE_FIELDS or name in SEED_CONSTANT_FIELDS or name == "delta":
                continue
            out[name] = float(arr[t])
        for name, arr in self.derived.items():
            out[name] = float(arr[t])
        return out


def load_trajectories(input_dir: Path) -> list[TrajectoryEnriched]:
    files = sorted(input_dir.glob("trajectory_*.json"))
    if not files:
        raise FileNotFoundError(f"no trajectory_*.json files in {input_dir}")
    out: list[TrajectoryEnriched] = []
    for path in files:
        obj = json.loads(path.read_text())
        seed = int(obj["seed"])
        T = int(obj["T"])
        raw: dict[str, list[float]] = {}
        for step in obj["per_t"]:
            for name in STATE_FEATURES + ("delta",):
                raw.setdefault(name, []).append(float(step[name]))
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
        out.append(TrajectoryEnriched(seed=seed, T=T, raw=arrays))
    return out


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

DERIVATIVE_BASE: tuple[str, ...] = (
    "entropy", "inverse_margin", "quality_mass_proxy",
    "unmasked_fraction", "n_revisable",
)


def _shift(arr: np.ndarray, k: int) -> np.ndarray:
    """Return arr shifted by k positions (k>0 => past, k<0 => future).

    Boundary values fall back to arr[0] for past shifts and arr[-1] for
    future shifts, so no NaNs are produced.
    """
    out = np.empty_like(arr)
    if k > 0:
        out[:k] = arr[0]
        out[k:] = arr[:-k]
    elif k < 0:
        kk = -k
        out[-kk:] = arr[-1]
        out[:-kk] = arr[kk:]
    else:
        out = arr.copy()
    return out


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr.copy()
    pad = np.concatenate([np.full(w - 1, arr[0]), arr])
    csum = np.cumsum(pad, dtype=np.float64)
    out = (csum[w - 1:] - np.concatenate([[0.0], csum[:-w]])) / float(w)
    return out


def compute_derived(traj: TrajectoryEnriched) -> None:
    """Populate `traj.derived` with families S1 (ratios), S2 (causal
    derivatives + rolling means), and S3 (forward derivatives + curvature).
    """
    L_seq = 1024.0  # ProSeCo-OWT base sequence length (consistent across runs).
    n_rev = traj.raw["n_revisable"]
    n_mask = traj.raw["n_masked"]
    uf = traj.raw["unmasked_fraction"]
    # Family S1 — revisable-set intensity ratios.
    traj.derived["S1__rev_over_seq"] = n_rev / L_seq
    traj.derived["S1__rev_over_masked"] = n_rev / np.maximum(n_mask, 1.0)
    traj.derived["S1__rev_over_unmasked"] = n_rev / np.maximum(L_seq - n_mask, 1.0)
    traj.derived["S1__mask_fraction"] = n_mask / L_seq
    traj.derived["S1__entropy_per_rev"] = traj.raw["entropy"] * (n_rev / L_seq)

    # Families S2 (causal) and S3 (forward) on the listed base features.
    for name in DERIVATIVE_BASE:
        arr = traj.raw[name]
        # Causal first/second differences.
        traj.derived[f"S2__{name}_diff1_back"] = arr - _shift(arr, 1)
        traj.derived[f"S2__{name}_diff2_back"] = arr - _shift(arr, 2)
        # Causal rolling mean over previous 4 steps.
        traj.derived[f"S2__{name}_roll4_back"] = _rolling_mean(arr, 4)
        # Forward differences (offline pre-decision: requires base trajectory).
        traj.derived[f"S3__{name}_diff1_fwd"] = _shift(arr, -1) - arr
        traj.derived[f"S3__{name}_curv"] = _shift(arr, 1) - 2.0 * arr + _shift(arr, -1)


def enriched_feature_names() -> dict[str, list[str]]:
    """Return feature names grouped by family."""
    s0 = list(STATE_FEATURES)
    s1 = [
        "S1__rev_over_seq", "S1__rev_over_masked", "S1__rev_over_unmasked",
        "S1__mask_fraction", "S1__entropy_per_rev",
    ]
    s2 = [
        f"S2__{n}_{kind}" for n in DERIVATIVE_BASE
        for kind in ("diff1_back", "diff2_back", "roll4_back")
    ]
    s3 = [
        f"S3__{n}_{kind}" for n in DERIVATIVE_BASE
        for kind in ("diff1_fwd", "curv")
    ]
    return {"S0": s0, "S1": s1, "S2": s2, "S3": s3}


# ---------------------------------------------------------------------------
# Ridge with train-only standardization
# ---------------------------------------------------------------------------

@dataclass
class RidgeFit:
    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != len(self.columns):
            raise ValueError("predict X column mismatch")
        Xs = (X - self.mean) / self.scale
        design = np.column_stack([np.ones(len(Xs)), Xs])
        return design @ self.coef


def fit_ridge(X: np.ndarray, y: np.ndarray, columns: Sequence[str], *, alpha: float = 1.0) -> RidgeFit:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    Xs = (X - mean) / scale
    design = np.column_stack([np.ones(len(Xs)), Xs])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeFit(tuple(columns), mean, scale, coef, alpha)


# ---------------------------------------------------------------------------
# Splits and metrics
# ---------------------------------------------------------------------------

def make_seed_folds(seeds: Sequence[int], n_folds: int, seed: int = 1729) -> list[list[int]]:
    unique = sorted({int(s) for s in seeds})
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_folds > len(unique):
        raise ValueError("n_folds cannot exceed number of seeds")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique, dtype=int)
    rng.shuffle(shuffled)
    folds = [list(map(int, fold)) for fold in np.array_split(shuffled, n_folds)]
    if any(not f for f in folds):
        raise ValueError("empty fold produced")
    return folds


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if len(xa) < 2:
        return float("nan")
    rx = _rank_average(xa)
    ry = _rank_average(ya)
    sx = rx.std()
    sy = ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.mean(((rx - rx.mean()) / sx) * ((ry - ry.mean()) / sy)))


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
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        boot[i] = float(np.mean(diffs[idx]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "mean_mse_reduction": float(np.mean(diffs)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "pct_seeds_improved": float(np.mean(diffs > 0)),
        "n_seeds": int(len(sd)),
    }


# ---------------------------------------------------------------------------
# Marginal audit
# ---------------------------------------------------------------------------

def build_marginal_feature_blocks(
    trajs: Sequence[TrajectoryEnriched],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Return (X_full, y, seeds, t_array, group_indices)."""
    rows: list[list[float]] = []
    seeds: list[int] = []
    targets: list[float] = []
    ts: list[int] = []
    s0_cols = list(STATE_FEATURES)
    s1_cols = [c for c in enriched_feature_names()["S1"]]
    s2_cols = [c for c in enriched_feature_names()["S2"]]
    s3_cols = [c for c in enriched_feature_names()["S3"]]
    geom_cols = ["t_norm", "t_norm2", "t_norm3", "phase_early", "phase_middle", "phase_late"]

    for traj in trajs:
        compute_derived(traj)
        for t in range(traj.T):
            feats = traj.enriched(t)
            t_norm = t / (traj.T - 1) if traj.T > 1 else 0.0
            phase = "early" if t_norm < 1/3 else ("middle" if t_norm < 2/3 else "late")
            geom = [
                t_norm, t_norm * t_norm, t_norm * t_norm * t_norm,
                1.0 if phase == "early" else 0.0,
                1.0 if phase == "middle" else 0.0,
                1.0 if phase == "late" else 0.0,
            ]
            row = (
                geom
                + [feats[c] for c in s0_cols]
                + [feats[c] for c in s1_cols]
                + [feats[c] for c in s2_cols]
                + [feats[c] for c in s3_cols]
            )
            rows.append(row)
            seeds.append(traj.seed)
            targets.append(float(traj.deltas()[t]))
            ts.append(t)
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    s = np.asarray(seeds, dtype=np.int64)
    t_arr = np.asarray(ts, dtype=np.int64)
    n_geom = len(geom_cols)
    n_s0 = len(s0_cols)
    n_s1 = len(s1_cols)
    n_s2 = len(s2_cols)
    n_s3 = len(s3_cols)
    groups = {
        "B":     list(range(0, n_geom)),
        "S0":    list(range(n_geom, n_geom + n_s0)),
        "S1":    list(range(n_geom + n_s0, n_geom + n_s0 + n_s1)),
        "S2":    list(range(n_geom + n_s0 + n_s1, n_geom + n_s0 + n_s1 + n_s2)),
        "S3":    list(range(n_geom + n_s0 + n_s1 + n_s2, n_geom + n_s0 + n_s1 + n_s2 + n_s3)),
        "all_columns": geom_cols + s0_cols + s1_cols + s2_cols + s3_cols,
    }
    return X, y, s, t_arr, groups


MARGINAL_ABLATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("M0_geom",              ("B",)),
    ("M1_geom_S0",           ("B", "S0")),
    ("M2_geom_S0_S1",        ("B", "S0", "S1")),
    ("M3_geom_S0_S1_S2",     ("B", "S0", "S1", "S2")),
    ("M4_full_linear",       ("B", "S0", "S1", "S2", "S3")),
)

PAIR_ABLATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("P0_geom",                       ("B_pair",)),
    ("P1_geom_apair",                 ("B_pair", "ORACLE_apair")),
    ("P2_geom_apair_S0",              ("B_pair", "ORACLE_apair", "S0_pair")),
    ("P3_geom_apair_S0_S1",           ("B_pair", "ORACLE_apair", "S0_pair", "S1_pair")),
    ("P4_full_no_S3",                 ("B_pair", "ORACLE_apair", "S0_pair", "S1_pair", "S2_pair")),
    ("P5_full_linear",                ("B_pair", "ORACLE_apair", "S0_pair", "S1_pair", "S2_pair", "S3_pair")),
)

SCHEDULE_ABLATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("S0_sched_geom",                 ("B_sched",)),
    ("S1_sched_geom_AofS",            ("B_sched", "ORACLE_AofS")),
    ("S2_sched_geom_AofS_state",      ("B_sched", "ORACLE_AofS", "STATE_summaries")),
    ("S3_sched_geom_AofS_state_pair", ("B_sched", "ORACLE_AofS", "STATE_summaries", "PAIR_summaries")),
)


def _select_columns(groups: dict[str, list[int]], parts: Sequence[str]) -> list[int]:
    idx: list[int] = []
    for p in parts:
        if p in groups:
            idx.extend(groups[p])
    if not idx:
        raise ValueError(f"no columns selected for parts={parts}")
    return idx


def fit_predict_grouped(
    X: np.ndarray, y: np.ndarray, seeds: np.ndarray, folds: Sequence[Sequence[int]],
    *, alpha: float = 1.0, columns: Sequence[str] | None = None,
) -> np.ndarray:
    preds = np.empty_like(y)
    for fold in folds:
        test_mask = np.isin(seeds, np.asarray(fold, dtype=np.int64))
        if not np.any(test_mask):
            raise RuntimeError("empty fold mask")
        if np.any(np.isin(seeds[~test_mask], np.asarray(fold, dtype=np.int64))):
            raise RuntimeError("seed leakage between train and test")
        X_train = X[~test_mask]
        y_train = y[~test_mask]
        X_test = X[test_mask]
        cols = list(columns) if columns is not None else [f"x{i}" for i in range(X.shape[1])]
        model = fit_ridge(X_train, y_train, cols, alpha=alpha)
        preds[test_mask] = model.predict(X_test)
    return preds


def run_marginal_audit(trajs: Sequence[TrajectoryEnriched], n_folds: int, fold_seed: int) -> dict[str, Any]:
    X, y, seeds, _, groups = build_marginal_feature_blocks(trajs)
    fold_assignment = make_seed_folds(sorted({int(s) for s in seeds}), n_folds, seed=fold_seed)
    column_names = groups["all_columns"]

    predictions: dict[str, np.ndarray] = {}
    for name, parts in MARGINAL_ABLATIONS:
        cols_idx = _select_columns(groups, parts)
        X_sub = X[:, cols_idx]
        col_sub = [column_names[i] for i in cols_idx]
        predictions[name] = fit_predict_grouped(
            X_sub, y, seeds, fold_assignment, columns=col_sub
        )

    # Non-linear probe (poly-2 over a small curated subset of S2/S3 + S0 means).
    nonlin_subset = ["entropy", "inverse_margin", "n_revisable",
                     "S1__rev_over_masked", "S2__entropy_diff1_back",
                     "S3__entropy_curv"]
    nonlin_idx = [column_names.index(c) for c in nonlin_subset]
    X_nl = X[:, nonlin_idx]
    poly_cols: list[str] = list(nonlin_subset)
    poly_data: list[np.ndarray] = [X_nl[:, i] for i in range(X_nl.shape[1])]
    for i in range(X_nl.shape[1]):
        for j in range(i, X_nl.shape[1]):
            poly_cols.append(f"poly2:{nonlin_subset[i]}*{nonlin_subset[j]}")
            poly_data.append(X_nl[:, i] * X_nl[:, j])
    X_nl_full = np.column_stack(poly_data)
    # Combine with linear geometry to keep a fair baseline.
    B_idx = groups["B"]
    X_probe = np.column_stack([X[:, B_idx], X_nl_full])
    probe_cols = [column_names[i] for i in B_idx] + poly_cols
    predictions["M5_nonlinear_probe"] = fit_predict_grouped(
        X_probe, y, seeds, fold_assignment, columns=probe_cols
    )

    predictor_names = [n for n, _ in MARGINAL_ABLATIONS] + ["M5_nonlinear_probe"]
    metrics, improvements = _summarize(seeds, y, predictions, predictor_names, [
        ("M1_geom_S0", "M0_geom"),
        ("M2_geom_S0_S1", "M1_geom_S0"),
        ("M3_geom_S0_S1_S2", "M1_geom_S0"),
        ("M4_full_linear", "M1_geom_S0"),
        ("M4_full_linear", "M0_geom"),
        ("M5_nonlinear_probe", "M4_full_linear"),
    ])
    return {
        "predictor_names": predictor_names,
        "metrics": metrics,
        "improvements": improvements,
        "n_rows": int(len(y)),
        "n_seeds": int(len(set(seeds.tolist()))),
        "fold_assignment": [list(f) for f in fold_assignment],
    }


def _summarize(
    seeds: np.ndarray, y: np.ndarray, preds: Mapping[str, np.ndarray],
    predictor_names: Sequence[str], comparisons: Sequence[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    for name in predictor_names:
        p = preds[name]
        metrics[name] = {
            "mse": float(np.mean((p - y) ** 2)),
            "mae": float(np.mean(np.abs(p - y))),
            "spearman_pred_target": spearman(p, y),
        }
    improvements: dict[str, Any] = {}
    for cand, base in comparisons:
        improvements[f"{cand}_vs_{base}"] = bootstrap_seed_diff(
            seeds, y, preds[base], preds[cand]
        )
    return metrics, improvements


# ---------------------------------------------------------------------------
# Pair audit
# ---------------------------------------------------------------------------

def load_pair_with_enriched(
    xi_file: Path, trajs_by_seed: dict[int, TrajectoryEnriched],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Build pair-level (X, y_xi, y_gpair, seeds, groups)."""
    raw = json.loads(xi_file.read_text())

    enriched_names = enriched_feature_names()
    s0 = enriched_names["S0"]
    s1 = enriched_names["S1"]
    s2 = enriched_names["S2"]
    s3 = enriched_names["S3"]

    # Per-pair: geom, A_pair, then for S0/S1/S2/S3 the (h_s, h_t, mean, absdiff).
    rows: list[list[float]] = []
    y_xi: list[float] = []
    y_g: list[float] = []
    seeds: list[int] = []

    geom_cols = [
        "s_norm", "t_norm", "dist_norm",
        "s2", "t2", "dist2",
        "s*t", "|t-s|",
        "phase_s_early", "phase_s_middle", "phase_s_late",
        "phase_t_early", "phase_t_middle", "phase_t_late",
    ]
    oracle_cols = ["A_pair"]

    def expand(name: str) -> list[str]:
        return [f"{name}_s", f"{name}_t", f"{name}_mean", f"{name}_absdiff"]

    s0_pair_cols = [c for name in s0 for c in expand(name)]
    s1_pair_cols = [c for name in s1 for c in expand(name)]
    s2_pair_cols = [c for name in s2 for c in expand(name)]
    s3_pair_cols = [c for name in s3 for c in expand(name)]

    for r in raw:
        sd = int(r["seed"])
        traj = trajs_by_seed.get(sd)
        if traj is None:
            continue
        # Verify identities up-front.
        a_pair = float(r["A_pair"])
        g_pair = float(r["G_pair"])
        xi = float(r["xi"])
        if abs(xi - (g_pair - a_pair)) > 1e-6:
            raise ValueError(f"xi != G-A at seed={sd}, pair=({r['t']},{r['t_prime']})")
        s = int(r["t"])
        t = int(r["t_prime"])
        T = traj.T
        feats_s = traj.enriched(s)
        feats_t = traj.enriched(t)
        s_norm = s / float(T - 1)
        t_norm = t / float(T - 1)
        dist = abs(t - s)
        dist_norm = dist / float(T)
        ps = "early" if s_norm < 1/3 else ("middle" if s_norm < 2/3 else "late")
        pt = "early" if t_norm < 1/3 else ("middle" if t_norm < 2/3 else "late")
        geom = [
            s_norm, t_norm, dist_norm,
            s_norm * s_norm, t_norm * t_norm, dist_norm * dist_norm,
            s_norm * t_norm, abs(t - s),
            1.0 if ps == "early" else 0.0, 1.0 if ps == "middle" else 0.0, 1.0 if ps == "late" else 0.0,
            1.0 if pt == "early" else 0.0, 1.0 if pt == "middle" else 0.0, 1.0 if pt == "late" else 0.0,
        ]
        oracle = [a_pair]
        s0_pair: list[float] = []
        for name in s0:
            a = feats_s[name]; b = feats_t[name]
            s0_pair.extend([a, b, 0.5 * (a + b), abs(b - a)])
        s1_pair: list[float] = []
        for name in s1:
            a = feats_s[name]; b = feats_t[name]
            s1_pair.extend([a, b, 0.5 * (a + b), abs(b - a)])
        s2_pair: list[float] = []
        for name in s2:
            a = feats_s[name]; b = feats_t[name]
            s2_pair.extend([a, b, 0.5 * (a + b), abs(b - a)])
        s3_pair: list[float] = []
        for name in s3:
            a = feats_s[name]; b = feats_t[name]
            s3_pair.extend([a, b, 0.5 * (a + b), abs(b - a)])
        rows.append(geom + oracle + s0_pair + s1_pair + s2_pair + s3_pair)
        y_xi.append(xi)
        y_g.append(g_pair)
        seeds.append(sd)

    X = np.asarray(rows, dtype=np.float64)
    y_xi_arr = np.asarray(y_xi, dtype=np.float64)
    y_g_arr = np.asarray(y_g, dtype=np.float64)
    sd_arr = np.asarray(seeds, dtype=np.int64)
    column_names = geom_cols + oracle_cols + s0_pair_cols + s1_pair_cols + s2_pair_cols + s3_pair_cols

    ng = len(geom_cols)
    no = len(oracle_cols)
    n_s0 = len(s0_pair_cols)
    n_s1 = len(s1_pair_cols)
    n_s2 = len(s2_pair_cols)
    n_s3 = len(s3_pair_cols)
    groups = {
        "B_pair":        list(range(0, ng)),
        "ORACLE_apair":  list(range(ng, ng + no)),
        "S0_pair":       list(range(ng + no, ng + no + n_s0)),
        "S1_pair":       list(range(ng + no + n_s0, ng + no + n_s0 + n_s1)),
        "S2_pair":       list(range(ng + no + n_s0 + n_s1, ng + no + n_s0 + n_s1 + n_s2)),
        "S3_pair":       list(range(ng + no + n_s0 + n_s1 + n_s2, ng + no + n_s0 + n_s1 + n_s2 + n_s3)),
        "all_columns":   column_names,
    }
    return X, y_xi_arr, y_g_arr, sd_arr, groups


def run_pair_audit(
    xi_file: Path, trajs: Sequence[TrajectoryEnriched], n_folds: int, fold_seed: int,
) -> dict[str, Any]:
    trajs_by_seed = {t.seed: t for t in trajs}
    for t in trajs:
        if not t.derived:
            compute_derived(t)
    X, y_xi, y_g, seeds, groups = load_pair_with_enriched(xi_file, trajs_by_seed)
    col_names = groups["all_columns"]
    fold_assignment = make_seed_folds(sorted({int(s) for s in seeds}), n_folds, seed=fold_seed)

    preds_xi: dict[str, np.ndarray] = {}
    preds_g: dict[str, np.ndarray] = {}
    for name, parts in PAIR_ABLATIONS:
        idx = _select_columns(groups, parts)
        cols = [col_names[i] for i in idx]
        Xs = X[:, idx]
        preds_xi[name] = fit_predict_grouped(Xs, y_xi, seeds, fold_assignment, columns=cols)
        preds_g[name]  = fit_predict_grouped(Xs, y_g,  seeds, fold_assignment, columns=cols)

    # Curated non-linear probe at the pair level.
    nonlin_subset = [
        "A_pair",
        "entropy_s", "entropy_t", "entropy_absdiff",
        "n_revisable_s", "n_revisable_t",
        "S1__rev_over_masked_s", "S1__rev_over_masked_t",
        "S2__entropy_diff1_back_s", "S2__entropy_diff1_back_t",
    ]
    nl_idx = [col_names.index(c) for c in nonlin_subset]
    X_nl = X[:, nl_idx]
    poly_cols = list(nonlin_subset)
    poly_data = [X_nl[:, i] for i in range(X_nl.shape[1])]
    for i in range(X_nl.shape[1]):
        for j in range(i, X_nl.shape[1]):
            poly_cols.append(f"poly2:{nonlin_subset[i]}*{nonlin_subset[j]}")
            poly_data.append(X_nl[:, i] * X_nl[:, j])
    X_probe = np.column_stack([X[:, groups["B_pair"]]] + poly_data)
    probe_cols = [col_names[i] for i in groups["B_pair"]] + poly_cols
    preds_xi["P6_nonlinear_probe"] = fit_predict_grouped(
        X_probe, y_xi, seeds, fold_assignment, columns=probe_cols
    )
    preds_g["P6_nonlinear_probe"] = fit_predict_grouped(
        X_probe, y_g, seeds, fold_assignment, columns=probe_cols
    )

    predictor_names = [n for n, _ in PAIR_ABLATIONS] + ["P6_nonlinear_probe"]
    comparisons = [
        ("P1_geom_apair", "P0_geom"),
        ("P2_geom_apair_S0", "P1_geom_apair"),
        ("P3_geom_apair_S0_S1", "P1_geom_apair"),
        ("P4_full_no_S3", "P1_geom_apair"),
        ("P5_full_linear", "P1_geom_apair"),
        ("P6_nonlinear_probe", "P1_geom_apair"),
    ]
    m_xi, i_xi = _summarize(seeds, y_xi, preds_xi, predictor_names, comparisons)
    m_g,  i_g  = _summarize(seeds, y_g,  preds_g,  predictor_names, comparisons)

    return {
        "predictor_names": predictor_names,
        "metrics_xi": m_xi,
        "improvements_xi": i_xi,
        "metrics_g_pair": m_g,
        "improvements_g_pair": i_g,
        "n_rows": int(len(y_xi)),
        "n_seeds": int(len(set(seeds.tolist()))),
        "fold_assignment": [list(f) for f in fold_assignment],
    }


# ---------------------------------------------------------------------------
# Schedule-level audit (Phase 2b MC pool)
# ---------------------------------------------------------------------------

def load_mc_pool(mc_file: Path) -> list[dict[str, Any]]:
    raw = json.loads(mc_file.read_text())
    out: list[dict[str, Any]] = []
    for r in raw:
        out.append({
            "seed": int(r["seed"]),
            "B": int(r["B"]),
            "schedule": [int(x) for x in r["schedule_steps"]],
            "A": float(r["A"]),
            "G": float(r["G"]),
        })
    return out


def build_schedule_features(
    pool: Sequence[dict[str, Any]], trajs_by_seed: dict[int, TrajectoryEnriched],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Schedule-level (X, y, seeds, B_vals, groups). One row per (seed, B, mc_idx)."""
    rows: list[list[float]] = []
    y: list[float] = []
    seeds: list[int] = []
    Bs: list[int] = []

    geom_cols = [
        "B_size", "mean_t_norm", "std_t_norm", "min_t_norm", "max_t_norm",
        "spread", "n_early", "n_middle", "n_late", "phase_diversity",
    ]
    oracle_cols = ["A_of_S"]

    state_summary_cols: list[str] = []
    summary_kinds = ("mean", "max", "min", "std")
    for fam, names in enriched_feature_names().items():
        for name in names:
            for sk in summary_kinds:
                state_summary_cols.append(f"sched_{sk}__{name}")
    pair_summary_cols = [
        "mean_pair_dist", "max_pair_dist", "min_pair_dist",
        "mean_entropy_absdiff", "mean_invmargin_absdiff",
        "n_same_phase_pairs",
    ]

    for r in pool:
        sd = r["seed"]
        traj = trajs_by_seed.get(sd)
        if traj is None:
            continue
        if not traj.derived:
            compute_derived(traj)
        S = r["schedule"]
        B = len(S)
        if B == 0:
            continue
        T = traj.T
        norms = np.asarray([t / (T - 1) for t in S], dtype=np.float64)
        n_early = float(sum(1 for x in norms if x < 1/3))
        n_middle = float(sum(1 for x in norms if 1/3 <= x < 2/3))
        n_late = float(sum(1 for x in norms if x >= 2/3))
        phase_div = float(int(n_early > 0) + int(n_middle > 0) + int(n_late > 0))
        geom = [
            float(B), float(norms.mean()),
            float(norms.std()) if B > 1 else 0.0,
            float(norms.min()), float(norms.max()),
            float(norms.max() - norms.min()),
            n_early, n_middle, n_late, phase_div,
        ]
        oracle = [float(r["A"])]
        # Per-step enriched features for each scheduled time.
        per_step: dict[str, list[float]] = {}
        for t in S:
            feats = traj.enriched(t)
            for name, val in feats.items():
                per_step.setdefault(name, []).append(val)
        state_summary: list[float] = []
        for fam, names in enriched_feature_names().items():
            for name in names:
                arr = np.asarray(per_step.get(name, [0.0]), dtype=np.float64)
                state_summary.extend([
                    float(arr.mean()),
                    float(arr.max()),
                    float(arr.min()),
                    float(arr.std()) if len(arr) > 1 else 0.0,
                ])
        # Pair summaries within S.
        if B >= 2:
            dists: list[int] = []
            ent_diffs: list[float] = []
            invm_diffs: list[float] = []
            same_phase_pairs = 0
            phases = ["early" if x < 1/3 else ("middle" if x < 2/3 else "late") for x in norms]
            ents = [traj.enriched(t)["entropy"] for t in S]
            invs = [traj.enriched(t)["inverse_margin"] for t in S]
            for i in range(B):
                for j in range(i + 1, B):
                    dists.append(abs(S[j] - S[i]))
                    ent_diffs.append(abs(ents[j] - ents[i]))
                    invm_diffs.append(abs(invs[j] - invs[i]))
                    if phases[i] == phases[j]:
                        same_phase_pairs += 1
            pair_summary = [
                float(np.mean(dists)), float(np.max(dists)), float(np.min(dists)),
                float(np.mean(ent_diffs)), float(np.mean(invm_diffs)),
                float(same_phase_pairs),
            ]
        else:
            pair_summary = [0.0] * len(pair_summary_cols)
        rows.append(geom + oracle + state_summary + pair_summary)
        y.append(float(r["G"]))
        seeds.append(sd)
        Bs.append(r["B"])

    X = np.asarray(rows, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    sd_arr = np.asarray(seeds, dtype=np.int64)
    B_arr = np.asarray(Bs, dtype=np.int64)
    column_names = geom_cols + oracle_cols + state_summary_cols + pair_summary_cols
    ng = len(geom_cols); no = len(oracle_cols)
    nss = len(state_summary_cols); nps = len(pair_summary_cols)
    groups = {
        "B_sched":          list(range(0, ng)),
        "ORACLE_AofS":      list(range(ng, ng + no)),
        "STATE_summaries":  list(range(ng + no, ng + no + nss)),
        "PAIR_summaries":   list(range(ng + no + nss, ng + no + nss + nps)),
        "all_columns":      column_names,
    }
    return X, y_arr, sd_arr, B_arr, groups


def run_schedule_audit(
    mc_file: Path, trajs: Sequence[TrajectoryEnriched], n_folds: int, fold_seed: int,
) -> dict[str, Any]:
    trajs_by_seed = {t.seed: t for t in trajs}
    pool = load_mc_pool(mc_file)
    X, y, seeds, Bs, groups = build_schedule_features(pool, trajs_by_seed)
    col_names = groups["all_columns"]
    fold_assignment = make_seed_folds(sorted({int(s) for s in seeds}), n_folds, seed=fold_seed)

    out_by_B: dict[str, dict[str, Any]] = {}
    for B_value in sorted(set(Bs.tolist())):
        mask = Bs == B_value
        if mask.sum() < 30:
            continue
        Xb = X[mask]; yb = y[mask]; sb = seeds[mask]
        # Recompute fold assignment restricted to seeds present at this B
        # (Phase 2b is balanced so this matches).
        sub_folds = make_seed_folds(sorted({int(s) for s in sb}), n_folds, seed=fold_seed)
        preds: dict[str, np.ndarray] = {}
        for name, parts in SCHEDULE_ABLATIONS:
            idx = _select_columns(groups, parts)
            cols = [col_names[i] for i in idx]
            preds[name] = fit_predict_grouped(
                Xb[:, idx], yb, sb, sub_folds, columns=cols
            )
        predictor_names = [n for n, _ in SCHEDULE_ABLATIONS]
        metrics, improvements = _summarize(sb, yb, preds, predictor_names, [
            ("S1_sched_geom_AofS", "S0_sched_geom"),
            ("S2_sched_geom_AofS_state", "S1_sched_geom_AofS"),
            ("S3_sched_geom_AofS_state_pair", "S1_sched_geom_AofS"),
        ])
        # Top-K schedule overlap within seed: rank by predicted G, top-10 vs oracle.
        ranking: dict[str, dict[str, float]] = {}
        for name in predictor_names:
            ranking[name] = _within_seed_topk(sb, yb, preds[name], K=10)
        out_by_B[str(int(B_value))] = {
            "n_rows": int(mask.sum()),
            "metrics": metrics,
            "improvements": improvements,
            "top10_ranking": ranking,
        }
    return {
        "predictor_names": [n for n, _ in SCHEDULE_ABLATIONS],
        "by_B": out_by_B,
        "n_total_rows": int(len(y)),
        "n_seeds": int(len(set(seeds.tolist()))),
    }


def _within_seed_topk(seeds: np.ndarray, y: np.ndarray, p: np.ndarray, K: int) -> dict[str, float]:
    by_seed: dict[int, list[tuple[float, float]]] = {}
    for s, yi, pi in zip(seeds, y, p):
        by_seed.setdefault(int(s), []).append((float(pi), float(yi)))
    overlaps: list[float] = []
    close_ratios: list[float] = []
    for _, rows in by_seed.items():
        n = len(rows)
        if n == 0:
            continue
        k = min(K, n)
        oracle_idx = np.argsort([-r[1] for r in rows])[:k]
        pred_idx = np.argsort([-r[0] for r in rows])[:k]
        oset = set(int(i) for i in oracle_idx)
        pset = set(int(i) for i in pred_idx)
        overlaps.append(len(oset & pset) / float(k))
        ys = [rows[i][1] for i in range(n)]
        baseline = float(np.median(ys))
        oracle_gain = float(np.mean([rows[i][1] for i in oracle_idx]))
        pred_gain = float(np.mean([rows[i][1] for i in pred_idx]))
        denom = oracle_gain - baseline
        if abs(denom) > 1e-12:
            close_ratios.append((pred_gain - baseline) / denom)
    return {
        "K": int(K),
        "mean_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "mean_close_ratio": float(np.mean(close_ratios)) if close_ratios else float("nan"),
        "n_seeds": int(len(by_seed)),
    }


# ---------------------------------------------------------------------------
# IO
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
    raise TypeError(repr(type(o)))


def write_feature_audit(out_dir: Path) -> None:
    lines = ["# Feature/Data Audit", "", "Classification of every per-step field in the Phase 1 Protocol A artifacts.",
             "", "| Field | Class | Family |", "|---|---|---|"]
    for k, v in sorted(FEATURE_AUDIT.items()):
        lines.append(f"| `{k}` | {v['class']} | {v['family']} |")
    lines += [
        "",
        "## Derived feature families (constructed in this script)",
        "",
        "- **S1** Revisable-set intensity ratios: `rev_over_seq`, "
        "`rev_over_masked`, `rev_over_unmasked`, `mask_fraction`, "
        "`entropy_per_rev`. Deployable offline.",
        "- **S2** Causal temporal derivatives over "
        f"{', '.join(DERIVATIVE_BASE)}: `diff1_back`, `diff2_back`, "
        "`roll4_back`. Deployable offline.",
        "- **S3** Forward-looking temporal features (offline pre-decision: "
        "require base trajectory): `diff1_fwd`, `curv` (local curvature).",
        "- **S4** (pair-only) Contrast features: for each base/derived "
        "feature, take value at s and t, mean, and absolute difference.",
        "",
        "## Leakage explicitly excluded",
        "",
        f"- `{'`, `'.join(LEAKAGE_FIELDS)}` (post-correction).",
        f"- `{'`, `'.join(SEED_CONSTANT_FIELDS)}` (constant per seed; would "
        "act as a seed indicator under grouped CV).",
        "",
        "## Oracle / non-deployable",
        "",
        "- `A_pair` (= delta_s + delta_t) and `A(S)` (= sum delta_t over S):"
        " oracle-diagnostic features that assume access to marginal "
        "correction values. They are included only as **strong baselines** "
        "for the pair and schedule audits, not as deployable predictors.",
    ]
    (out_dir / "feature_audit.md").write_text("\n".join(lines) + "\n")


def write_interpretation(out_dir: Path, marg: dict[str, Any], pair: dict[str, Any], sched: dict[str, Any]) -> None:
    def _verdict(imp: dict[str, Any] | None, target_label: str) -> str:
        if imp is None:
            return "no_test"
        if imp["ci95_lo"] > 0 and imp["pct_seeds_improved"] >= 0.6:
            return f"{target_label}_supported"
        if imp["mean_mse_reduction"] > 0 and imp["pct_seeds_improved"] >= 0.5:
            return f"{target_label}_weak"
        return f"{target_label}_not_supported"

    marg_imp = marg["improvements"].get("M4_full_linear_vs_M0_geom")
    marg_S2 = marg["improvements"].get("M3_geom_S0_S1_S2_vs_M1_geom_S0")
    xi_imp = pair["improvements_xi"].get("P5_full_linear_vs_P1_geom_apair")
    g_imp = pair["improvements_g_pair"].get("P5_full_linear_vs_P1_geom_apair")
    poly_xi = pair["improvements_xi"].get("P6_nonlinear_probe_vs_P1_geom_apair")

    schedule_blocks = []
    if "by_B" in sched:
        for B, blk in sched["by_B"].items():
            schedule_blocks.append((B, blk["improvements"].get("S3_sched_geom_AofS_state_pair_vs_S1_sched_geom_AofS"),
                                    blk["improvements"].get("S2_sched_geom_AofS_state_vs_S1_sched_geom_AofS")))

    lines = [
        "# Enriched-Feature State Predictability Audit",
        "",
        "## Question",
        "",
        ("Does a richer family of locally derived pre-correction features"
         " (intensity ratios, temporal derivatives, forward curvature,"
         " pair contrasts, schedule-level summaries) improve held-out"
         " prediction of correction timing value beyond time geometry,"
         " original aggregate state, and marginal/additive baselines?"),
        "",
        "## Data",
        "",
        f"- Marginal: {marg['n_rows']} rows, {marg['n_seeds']} seeds"
        f" (Phase 1 Protocol A).",
        f"- Pair: {pair['n_rows']} rows, {pair['n_seeds']} seeds"
        f" (Gate 3a xi_raw.json joined to Phase 1).",
        f"- Schedule: {sched['n_total_rows']} rows over"
        f" {sched['n_seeds']} seeds (Phase 2b MC pool joined to Phase 1).",
        "",
        "## Validation Protocol",
        "",
        "- Grouped 5-fold by seed.",
        "- Standardization fit on train rows only.",
        "- Ridge alpha = 1.0.",
        "- Per-seed MSE differences with seed-bootstrap 95 % CI"
        " (2000 resamples).",
        "",
        "## Results — Marginal target Δ_t",
        "",
        "| Predictor | MSE | Spearman(pred, Δ) |",
        "|---|---:|---:|",
    ]
    for name in marg["predictor_names"]:
        m = marg["metrics"][name]
        lines.append(f"| `{name}` | {m['mse']:.6f} | {m['spearman_pred_target']:.3f} |")
    lines += ["", "**MSE-reduction comparisons (marginal):**", "",
              "| comparison | mean | 95 % CI | %% seeds improved |",
              "|---|---:|---|---:|"]
    for k, v in marg["improvements"].items():
        lines.append(f"| `{k}` | {v['mean_mse_reduction']:+.6f} |"
                     f" [{v['ci95_lo']:+.6f}, {v['ci95_hi']:+.6f}] |"
                     f" {v['pct_seeds_improved']:.2f} |")
    lines += ["", "## Results — Pair targets",
              "", "**Predictor MSE on ξ:**", "",
              "| Predictor | MSE | Spearman(pred, ξ) |",
              "|---|---:|---:|"]
    for name in pair["predictor_names"]:
        m = pair["metrics_xi"][name]
        lines.append(f"| `{name}` | {m['mse']:.6f} | {m['spearman_pred_target']:.3f} |")
    lines += ["", "**Predictor MSE on G_pair:**", "",
              "| Predictor | MSE | Spearman(pred, G_pair) |",
              "|---|---:|---:|"]
    for name in pair["predictor_names"]:
        m = pair["metrics_g_pair"][name]
        lines.append(f"| `{name}` | {m['mse']:.6f} | {m['spearman_pred_target']:.3f} |")
    lines += ["", "**MSE-reduction comparisons (pair):**", "",
              "| target | comparison | mean | 95 % CI | %% improved |",
              "|---|---|---:|---|---:|"]
    for k, v in pair["improvements_xi"].items():
        lines.append(f"| ξ | `{k}` | {v['mean_mse_reduction']:+.6f} |"
                     f" [{v['ci95_lo']:+.6f}, {v['ci95_hi']:+.6f}] |"
                     f" {v['pct_seeds_improved']:.2f} |")
    for k, v in pair["improvements_g_pair"].items():
        lines.append(f"| G_pair | `{k}` | {v['mean_mse_reduction']:+.6f} |"
                     f" [{v['ci95_lo']:+.6f}, {v['ci95_hi']:+.6f}] |"
                     f" {v['pct_seeds_improved']:.2f} |")
    lines += ["", "## Results — Schedule target G(S)", ""]
    if "by_B" in sched:
        for B, blk in sorted(sched["by_B"].items(), key=lambda kv: int(kv[0])):
            lines += [f"### B = {B} ({blk['n_rows']} schedules)", "",
                      "| Predictor | MSE | Spearman(pred, G) | top-10 overlap | close-ratio |",
                      "|---|---:|---:|---:|---:|"]
            for name in sched["predictor_names"]:
                m = blk["metrics"][name]
                r = blk["top10_ranking"][name]
                lines.append(f"| `{name}` | {m['mse']:.6f} |"
                             f" {m['spearman_pred_target']:.3f} |"
                             f" {r['mean_overlap']:.3f} |"
                             f" {r['mean_close_ratio']:.3f} |")
            lines += ["", "**MSE reductions:**", "",
                      "| comparison | mean | 95 % CI | %% improved |",
                      "|---|---:|---|---:|"]
            for k, v in blk["improvements"].items():
                lines.append(f"| `{k}` | {v['mean_mse_reduction']:+.6f} |"
                             f" [{v['ci95_lo']:+.6f}, {v['ci95_hi']:+.6f}] |"
                             f" {v['pct_seeds_improved']:.2f} |")
            lines.append("")
    lines += [
        "## Verdict labels",
        "",
        f"- Marginal full-linear vs geom-only: **{_verdict(marg_imp, 'marginal')}**",
        f"- Marginal S2 derivatives vs S0: **{_verdict(marg_S2, 'derivatives')}**",
        f"- Pair ξ enriched vs geom+A: **{_verdict(xi_imp, 'pair_xi')}**",
        f"- Pair G_pair enriched vs geom+A: **{_verdict(g_imp, 'pair_g')}**",
        f"- Pair non-linear probe vs geom+A: **{_verdict(poly_xi, 'pair_xi_nonlin')}**",
    ]
    for B, sched_imp, state_imp in schedule_blocks:
        lines.append(f"- Schedule B={B}, full vs geom+A(S): **{_verdict(sched_imp, f'sched_B{B}')}**")
        lines.append(f"- Schedule B={B}, state vs geom+A(S): **{_verdict(state_imp, f'sched_B{B}_stateonly')}**")
    lines += [
        "",
        "## Statistical interpretation",
        "",
        "The audit uses incremental MSE-reduction with bootstrap CIs. A"
        " 'supported' label requires CI excluding zero on the positive side"
        " AND >= 60% of seeds improved; 'weak' requires positive mean and"
        " >=50% of seeds improved; otherwise 'not_supported'. These thresholds"
        " are pre-registered; thresholds were not adjusted post hoc.",
        "",
        "## Limitations",
        "",
        "- ProSeCo-OWT only. The framework is corrector-agnostic; empirical"
        " verdicts are not.",
        "- Pre-correction features are aggregate scalars at each step."
        " Token-level features (logit quantiles, revisable-set entropy"
        " distributions) were not extracted in this round.",
        "- Polynomial-degree-2 ridge is the conservative non-linearity test."
        " Tree/kernel methods may behave differently and are not ruled out.",
        "- A(S) and A_pair are oracle-diagnostic baselines (they require"
        " marginal Δ_t values for every t in S). The S2/S3/state ablations"
        " test incremental signal *on top of* this oracle baseline; positive"
        " results would still be informative for offline scheduling.",
        "",
        "## Decision",
        "",
    ]
    if any("supported" == _verdict(imp, "x").split("_")[-1] for imp in [marg_imp, xi_imp, g_imp]):
        lines += [
            "- At least one enriched feature family yields a strong positive"
            " signal. Recommended next step: localize which family contributes,"
            " run a feature-importance analysis, and consider a small targeted"
            " inference pass for token-level features only after the cheaper"
            " enrichment is fully characterised.",
        ]
    else:
        lines += [
            "- The enriched feature families tested in this audit do not"
            " provide incremental held-out predictability beyond geometry"
            " (and A_pair / A(S) where applicable) at any of the three"
            " target levels under the tested model class. Together with"
            " the marginal and pair-only state audits, this constitutes"
            " three independent negatives over derived aggregate features.",
            "",
            "- This justifies either (a) a targeted token-level / model-internal"
            " feature extraction with a tightly bounded HPC budget, or"
            " (b) writing the ProSeCo-OWT case study as a clean negative"
            " on observable aggregate state predictability. The thesis"
            " statistical claim is consistent either way: timing structure"
            " on this backbone is dominated by temporal geometry plus the"
            " marginal-additive structure, with no incremental signal from"
            " observable aggregate trajectory state.",
        ]
    (out_dir / "interpretation.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_out_dir(prefix: Path, git_sha: str) -> Path:
    return Path(f"{prefix}_{git_sha}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-a-dir", type=Path, default=DEFAULT_PROTOCOL_A)
    parser.add_argument("--xi-file", type=Path, default=DEFAULT_XI_FILE)
    parser.add_argument("--mc-file", type=Path, default=DEFAULT_MC_FILE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--results-prefix", type=Path, default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1729)
    parser.add_argument("--debug", action="store_true",
                        help="Use first 6 seeds + 2 folds, do not run schedule audit.")
    args = parser.parse_args(argv)

    git_sha = git_short_hash()
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.debug:
        out_dir = Path(f"{args.results_prefix}_debug_{git_sha}")
    else:
        out_dir = default_out_dir(args.results_prefix, git_sha)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} non-empty; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = load_trajectories(args.protocol_a_dir)
    for t in trajs:
        compute_derived(t)

    if args.debug:
        keep = sorted({t.seed for t in trajs})[:6]
        trajs = [t for t in trajs if t.seed in keep]
        n_folds = 2
    else:
        n_folds = args.n_folds

    marg = run_marginal_audit(trajs, n_folds=n_folds, fold_seed=args.fold_seed)
    pair = run_pair_audit(args.xi_file, trajs, n_folds=n_folds, fold_seed=args.fold_seed)
    if args.debug:
        sched = {"by_B": {}, "n_total_rows": 0, "n_seeds": 0,
                 "predictor_names": [n for n, _ in SCHEDULE_ABLATIONS]}
    else:
        sched = run_schedule_audit(args.mc_file, trajs, n_folds=n_folds, fold_seed=args.fold_seed)

    aggregate = {
        "meta": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generated_by": f"scripts/proseco/state_predictability/analyze_state_predictability_enriched.py@{git_sha}",
            "n_folds": n_folds,
            "fold_seed": args.fold_seed,
        },
        "marginal": marg,
        "pair": pair,
        "schedule": sched,
        "feature_audit_summary": FEATURE_AUDIT,
        "derived_feature_groups": enriched_feature_names(),
        "leakage_excluded_fields": list(LEAKAGE_FIELDS),
        "seed_constant_fields": list(SEED_CONSTANT_FIELDS),
    }
    config = {
        "protocol_a_dir": str(args.protocol_a_dir),
        "xi_file": str(args.xi_file),
        "mc_file": str(args.mc_file),
        "out_dir": str(out_dir),
        "git_sha": git_sha,
        "n_folds": n_folds,
        "fold_seed": args.fold_seed,
        "debug": bool(args.debug),
    }
    fold_metrics = {
        "marginal_folds": marg["fold_assignment"],
        "pair_folds": pair["fold_assignment"],
    }
    feature_summary = {
        "marginal_n_columns": int(sum(len(v) for k, v in enriched_feature_names().items())),
        "pair_n_pair_features_per_base": 4,
        "derived_feature_groups": enriched_feature_names(),
    }

    write_json(out_dir / "config.json", config)
    write_json(out_dir / "aggregate_stats.json", aggregate)
    write_json(out_dir / "fold_metrics.json", fold_metrics)
    write_json(out_dir / "feature_summary.json", feature_summary)
    write_feature_audit(out_dir)
    write_interpretation(out_dir, marg, pair, sched)
    summary = {
        "marginal_verdict_key": "M4_full_linear_vs_M0_geom",
        "marginal_verdict": marg["improvements"].get("M4_full_linear_vs_M0_geom"),
        "pair_xi_verdict": pair["improvements_xi"].get("P5_full_linear_vs_P1_geom_apair"),
        "pair_g_verdict": pair["improvements_g_pair"].get("P5_full_linear_vs_P1_geom_apair"),
        "pair_nonlin_xi_verdict": pair["improvements_xi"].get("P6_nonlinear_probe_vs_P1_geom_apair"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
