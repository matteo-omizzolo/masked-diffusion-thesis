"""Held-out pair-level state-predictability audit for corrector timing.

Counterpart to scripts/proseco/state_predictability/analyze_state_predictability.py. The marginal audit
asked whether observable pre-correction state at a single step t predicts the
single-step correction value delta_t. The result there was negative on
ProSeCo-OWT. The broader Gate 3a evidence (mean xi negative; corr(A_pair, xi)
about -0.70) suggests the timing structure lives at the pair / interaction
level rather than the marginal level.

This script uses only existing artifacts:

    results/phase1_interaction_diag_nogit/xi_raw.json
    results/phase1_proseco_owt_full/protocol_a/trajectory_*.json

It does NOT run any model, does NOT call ProSeCo, and does NOT need HPC.

Primary target:
    xi_{i,s,t} = G_i({s,t}) - G_i({s}) - G_i({t}).

Secondary target:
    G_i({s,t}).

For each target, several predictors are compared under grouped 5-fold seed
holdout (30 seeds, 6 per fold):

    P0_xi   intercept (mean xi on train).
    P0_G    A_pair on train fold (the canonical additive baseline for G).
    P1      time geometry only: (s_norm, t_norm, distance_norm, polys, phase).
    P2_xi   P1 + A_pair as a scalar feature (marginal-info baseline).
    P2_G    P1 + A_pair (same feature set, applied to G).
    P3      P2 + pre-correction state features at s and t (with sum/diff
            transforms).
    P4      Degree-2 polynomial expansion of the P3 feature vector + ridge
            (a conservative non-linear arm, no sklearn dependency).

Metrics per target / predictor:
    held-out MSE,
    mean per-seed MSE reduction vs the relevant baseline (with seed
        bootstrap 95% CI),
    Spearman(prediction, target) on the held-out pool,
    fraction of seeds where the candidate beats the baseline,
    auxiliary B=2 ranking metric: among the 73 fixed candidate pairs per
        seed, what fraction of the top-K oracle G_pair set is recovered by
        predicting top-K by G_pair_hat.

Outputs land under results/state_predictability_pair_<gitsha>/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional plotting
    Image = None
    ImageDraw = None
    ImageFont = None


DEFAULT_XI_FILE = Path("results/phase1_interaction_diag_nogit/xi_raw.json")
DEFAULT_PROTOCOL_A_DIR = Path("results/phase1_proseco_owt_full/protocol_a")
DEFAULT_RESULTS_PREFIX = Path("results/state_predictability_pair")

STATE_FEATURES: tuple[str, ...] = (
    "entropy",
    "inverse_margin",
    "quality_mass_proxy",
    "unmasked_fraction",
    "n_revisable",
    "n_masked",
)

# Explicit list of disallowed fields. These either depend on the corrected
# branch (post-correction) or are derived from final utility, both of which
# would be leakage for a pre-decision audit.
LEAKAGE_FIELDS: tuple[str, ...] = (
    "n_changed",
    "tcr",
    "f_branch",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairRow:
    seed: int
    s: int
    t: int
    T: int
    distance: int
    phase_s: str
    phase_t: str
    delta_s: float
    delta_t: float
    a_pair: float
    g_pair: float
    xi: float
    state_s: dict[str, float]
    state_t: dict[str, float]

    @property
    def s_norm(self) -> float:
        return self.s / float(self.T - 1) if self.T > 1 else 0.0

    @property
    def t_norm(self) -> float:
        return self.t / float(self.T - 1) if self.T > 1 else 0.0

    @property
    def distance_norm(self) -> float:
        return self.distance / float(self.T) if self.T > 0 else 0.0


def git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def load_protocol_a_features(input_dir: Path) -> tuple[dict[tuple[int, int], dict[str, float]], dict[int, int]]:
    """Return ((seed,t) -> {feature: value}, seed -> T)."""
    files = sorted(input_dir.glob("trajectory_*.json"))
    if not files:
        raise FileNotFoundError(f"no trajectory_*.json files in {input_dir}")
    feats: dict[tuple[int, int], dict[str, float]] = {}
    seed_T: dict[int, int] = {}
    for path in files:
        obj = json.loads(path.read_text())
        seed = int(obj["seed"])
        T = int(obj["T"])
        seed_T[seed] = T
        for step in obj["per_t"]:
            for bad in LEAKAGE_FIELDS:
                if bad not in step:
                    # Not all phase-1 schemas store all of these; fine.
                    pass
            entry: dict[str, float] = {}
            for name in STATE_FEATURES:
                if name not in step:
                    raise KeyError(f"{path}:{step.get('t')} missing feature {name!r}")
                entry[name] = float(step[name])
            feats[(seed, int(step["t"]))] = entry
    return feats, seed_T


def load_pair_rows(
    xi_path: Path,
    feats: dict[tuple[int, int], dict[str, float]],
    seed_T: dict[int, int],
    *,
    strict_delta_check_tol: float = 1e-6,
) -> list[PairRow]:
    raw = json.loads(xi_path.read_text())
    rows: list[PairRow] = []
    for r in raw:
        seed = int(r["seed"])
        T = seed_T.get(seed)
        if T is None:
            raise KeyError(f"seed {seed} has no Protocol A trajectory")
        s = int(r["t"])
        t = int(r["t_prime"])
        if s == t:
            raise ValueError(f"degenerate pair s==t for seed {seed}")
        if (seed, s) not in feats or (seed, t) not in feats:
            raise KeyError(
                f"missing Protocol A features for seed {seed} at s={s} or t={t}"
            )
        delta_s = float(r["delta_t"])
        delta_t = float(r["delta_tp"])
        a_pair = float(r["A_pair"])
        if abs(a_pair - (delta_s + delta_t)) > strict_delta_check_tol:
            raise ValueError(
                f"A_pair != delta_s + delta_t at seed={seed}, (s,t)=({s},{t})"
            )
        g_pair = float(r["G_pair"])
        xi = float(r["xi"])
        if abs(xi - (g_pair - a_pair)) > strict_delta_check_tol:
            raise ValueError(
                f"xi != G_pair - A_pair at seed={seed}, (s,t)=({s},{t})"
            )
        rows.append(
            PairRow(
                seed=seed,
                s=s,
                t=t,
                T=T,
                distance=int(r["distance"]),
                phase_s=str(r["phase_t"]),
                phase_t=str(r["phase_tp"]),
                delta_s=delta_s,
                delta_t=delta_t,
                a_pair=a_pair,
                g_pair=g_pair,
                xi=xi,
                state_s=dict(feats[(seed, s)]),
                state_t=dict(feats[(seed, t)]),
            )
        )
    return sorted(rows, key=lambda r: (r.seed, r.s, r.t))


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

PHASE_LIST = ("early", "middle", "late")


def _phase_indicators(phase_s: str, phase_t: str) -> list[float]:
    out: list[float] = []
    for ph in PHASE_LIST:
        out.append(1.0 if phase_s == ph else 0.0)
    for ph in PHASE_LIST:
        out.append(1.0 if phase_t == ph else 0.0)
    return out


def geometry_features(row: PairRow) -> list[float]:
    s = row.s_norm
    t = row.t_norm
    d = row.distance_norm
    feats = [
        s, t, d,
        s * s, t * t, d * d,
        s * t, s * d, t * d,
        s * s * s, t * t * t, d * d * d,
        abs(t - s),
    ]
    feats.extend(_phase_indicators(row.phase_s, row.phase_t))
    return feats


def geometry_feature_names() -> list[str]:
    names = [
        "s", "t", "dist",
        "s2", "t2", "dist2",
        "s*t", "s*dist", "t*dist",
        "s3", "t3", "dist3",
        "|t-s|",
    ]
    for which in ("s", "t"):
        for ph in PHASE_LIST:
            names.append(f"phase_{which}_{ph}")
    return names


def state_pair_features(row: PairRow) -> list[float]:
    out: list[float] = []
    for name in STATE_FEATURES:
        fs = row.state_s[name]
        ft = row.state_t[name]
        out.extend([fs, ft, 0.5 * (fs + ft), abs(ft - fs)])
    return out


def state_pair_feature_names() -> list[str]:
    names: list[str] = []
    for name in STATE_FEATURES:
        names.extend([f"{name}_s", f"{name}_t", f"{name}_mean", f"{name}_absdiff"])
    return names


def build_feature_vector(row: PairRow, kind: str) -> list[float]:
    if kind == "geom":
        return geometry_features(row)
    if kind == "geom_apair":
        return geometry_features(row) + [row.a_pair]
    if kind == "apair":
        return [row.a_pair]
    if kind == "geom_apair_state":
        return geometry_features(row) + [row.a_pair] + state_pair_features(row)
    if kind == "geom_apair_state_poly2":
        base = geometry_features(row) + [row.a_pair] + state_pair_features(row)
        # Degree-2 polynomial expansion on a curated subset to keep p < n.
        # We expand only over the state pair-summary features plus A_pair,
        # not over the (already-rich) geometry features.
        poly_input_idx = list(
            range(len(geometry_features(row)) + 1, len(base))
        )
        poly_input_idx.append(len(geometry_features(row)))  # A_pair
        poly_inputs = [base[i] for i in poly_input_idx]
        extras: list[float] = []
        n_pi = len(poly_inputs)
        for i in range(n_pi):
            for j in range(i, n_pi):
                extras.append(poly_inputs[i] * poly_inputs[j])
        return base + extras
    raise ValueError(f"unknown feature kind {kind!r}")


def feature_names(kind: str) -> list[str]:
    if kind == "geom":
        return geometry_feature_names()
    if kind == "geom_apair":
        return geometry_feature_names() + ["A_pair"]
    if kind == "apair":
        return ["A_pair"]
    if kind == "geom_apair_state":
        return geometry_feature_names() + ["A_pair"] + state_pair_feature_names()
    if kind == "geom_apair_state_poly2":
        base = geometry_feature_names() + ["A_pair"] + state_pair_feature_names()
        poly_in = state_pair_feature_names() + ["A_pair"]
        extras = []
        for i in range(len(poly_in)):
            for j in range(i, len(poly_in)):
                extras.append(f"poly2:{poly_in[i]}*{poly_in[j]}")
        return base + extras
    raise ValueError(f"unknown feature kind {kind!r}")


def _matrix(rows: Sequence[PairRow], kind: str) -> np.ndarray:
    return np.asarray([build_feature_vector(r, kind) for r in rows], dtype=np.float64)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class RidgeModel:
    kind: str
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def predict(self, rows: Sequence[PairRow]) -> np.ndarray:
        X = _matrix(rows, self.kind)
        Xs = (X - self.mean) / self.scale
        design = np.column_stack([np.ones(len(rows)), Xs])
        return design @ self.coef


def fit_ridge(
    rows: Sequence[PairRow], kind: str, target: str, *, alpha: float = 1.0,
) -> RidgeModel:
    X = _matrix(rows, kind)
    y = _extract_target(rows, target)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    Xs = (X - mean) / scale
    design = np.column_stack([np.ones(len(rows)), Xs])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeModel(kind=kind, alpha=alpha, mean=mean, scale=scale, coef=coef)


def _extract_target(rows: Sequence[PairRow], target: str) -> np.ndarray:
    if target == "xi":
        return np.asarray([r.xi for r in rows], dtype=np.float64)
    if target == "g_pair":
        return np.asarray([r.g_pair for r in rows], dtype=np.float64)
    raise ValueError(f"unknown target {target!r}")


def predict_intercept(train: Sequence[PairRow], test: Sequence[PairRow], target: str) -> np.ndarray:
    val = float(np.mean(_extract_target(train, target)))
    return np.full(len(test), val, dtype=np.float64)


def predict_apair_only(train: Sequence[PairRow], test: Sequence[PairRow], target: str) -> np.ndarray:
    # Use A_pair as a single-feature linear baseline (intercept + slope).
    X = np.asarray([[r.a_pair] for r in train], dtype=np.float64)
    y = _extract_target(train, target)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    Xs = (X - mean) / scale
    design = np.column_stack([np.ones(len(train)), Xs])
    penalty = np.eye(design.shape[1]) * 1.0
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    Xt = np.asarray([[r.a_pair] for r in test], dtype=np.float64)
    Xts = (Xt - mean) / scale
    design_t = np.column_stack([np.ones(len(test)), Xts])
    return design_t @ coef


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def per_seed_mse(rows: Sequence[dict[str, Any]], pred_key: str, target: str) -> dict[int, float]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), []).append(r)
    out: dict[int, float] = {}
    for sd, rs in by_seed.items():
        y = np.asarray([float(r[f"y_{target}"]) for r in rs])
        p = np.asarray([float(r[f"{pred_key}__{target}"]) for r in rs])
        out[sd] = float(np.mean((p - y) ** 2))
    return out


def bootstrap_seed_diff(
    rows: Sequence[dict[str, Any]],
    baseline: str,
    candidate: str,
    target: str,
    *,
    n_resamples: int = 2000,
    seed: int = 1729,
) -> dict[str, float]:
    base_mse = per_seed_mse(rows, baseline, target)
    cand_mse = per_seed_mse(rows, candidate, target)
    seeds = sorted(base_mse)
    diffs = np.asarray(
        [base_mse[s] - cand_mse[s] for s in seeds], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        boot.append(float(np.mean(diffs[idx])))
    lo, hi = np.quantile(np.asarray(boot), [0.025, 0.975])
    return {
        "target": target,
        "baseline": baseline,
        "candidate": candidate,
        "mean_mse_reduction": float(np.mean(diffs)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "pct_seeds_improved": float(np.mean(diffs > 0.0)),
        "n_seeds": int(len(seeds)),
    }


def b2_ranking_metric(
    prediction_rows: Sequence[dict[str, Any]],
    pred_key: str,
    target_label: str = "g_pair",
    K: int = 5,
) -> dict[str, float]:
    """For each seed, compute the overlap between top-K predicted pairs and
    top-K oracle pairs by G_pair. Reports mean overlap and mean close-ratio
    of realized gain over uniform-pair (median) gain divided by oracle gap.
    """
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for r in prediction_rows:
        by_seed.setdefault(int(r["seed"]), []).append(r)
    overlaps: list[float] = []
    close_ratios: list[float] = []
    for sd, rs in by_seed.items():
        y = np.asarray([float(r["y_g_pair"]) for r in rs])
        p = np.asarray([float(r[f"{pred_key}__{target_label}"]) for r in rs])
        k = min(K, len(rs))
        oracle_idx = set(np.argsort(-y)[:k].tolist())
        pred_idx = set(np.argsort(-p)[:k].tolist())
        overlaps.append(len(oracle_idx & pred_idx) / float(k))
        oracle_gain = float(np.mean(y[list(oracle_idx)]))
        baseline_gain = float(np.median(y))
        pred_gain = float(np.mean(y[list(pred_idx)]))
        denom = oracle_gain - baseline_gain
        if abs(denom) > 1e-12:
            close_ratios.append((pred_gain - baseline_gain) / denom)
    return {
        "K": int(K),
        "mean_top_k_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "mean_close_ratio": float(np.mean(close_ratios)) if close_ratios else float("nan"),
        "n_seeds": int(len(by_seed)),
    }


# ---------------------------------------------------------------------------
# Audit driver
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


PREDICTORS: dict[str, dict[str, str]] = {
    # Each entry: kind -> feature kind label. None means hand-coded baseline.
    "intercept":         {"kind": "__intercept__"},
    "apair_only":        {"kind": "__apair_only__"},
    "ridge_geom":        {"kind": "geom"},
    "ridge_geom_apair":  {"kind": "geom_apair"},
    "ridge_geom_apair_state": {"kind": "geom_apair_state"},
    "ridge_poly2":       {"kind": "geom_apair_state_poly2"},
}


def _predict_one(
    name: str, train: Sequence[PairRow], test: Sequence[PairRow], target: str,
    *, ridge_alpha: float,
) -> np.ndarray:
    info = PREDICTORS[name]
    kind = info["kind"]
    if kind == "__intercept__":
        return predict_intercept(train, test, target)
    if kind == "__apair_only__":
        return predict_apair_only(train, test, target)
    model = fit_ridge(train, kind, target, alpha=ridge_alpha)
    return model.predict(test)


def run_audit(
    rows: Sequence[PairRow],
    *,
    n_folds: int = 5,
    fold_seed: int = 1729,
    ridge_alpha: float = 1.0,
    targets: Sequence[str] = ("xi", "g_pair"),
    predictor_names: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictor_names = list(predictor_names or PREDICTORS.keys())
    seeds = sorted({r.seed for r in rows})
    folds = make_seed_folds(seeds, n_folds=n_folds, seed=fold_seed)
    fold_assignment: dict[int, int] = {}
    for fold_id, fold_seeds in enumerate(folds):
        for s in fold_seeds:
            fold_assignment[s] = fold_id

    # Pre-check: no leakage at fold partition level.
    flat = [s for fold in folds for s in fold]
    if len(set(flat)) != len(flat):
        raise RuntimeError("fold partition has duplicated seed (leakage)")

    prediction_rows: list[dict[str, Any]] = []
    for fold_id, test_seeds in enumerate(folds):
        test_set = set(test_seeds)
        train = [r for r in rows if r.seed not in test_set]
        test = [r for r in rows if r.seed in test_set]
        preds: dict[str, dict[str, np.ndarray]] = {n: {} for n in predictor_names}
        for target in targets:
            for name in predictor_names:
                preds[name][target] = _predict_one(
                    name, train, test, target, ridge_alpha=ridge_alpha
                )
        for idx, row in enumerate(test):
            out: dict[str, Any] = {
                "seed": row.seed,
                "s": row.s,
                "t": row.t,
                "T": row.T,
                "distance": row.distance,
                "phase_s": row.phase_s,
                "phase_t": row.phase_t,
                "fold_id": fold_id,
                "a_pair": row.a_pair,
                "y_xi": row.xi,
                "y_g_pair": row.g_pair,
            }
            for name in predictor_names:
                for target in targets:
                    out[f"{name}__{target}"] = float(preds[name][target][idx])
            prediction_rows.append(out)

    aggregate = _aggregate(
        prediction_rows,
        predictor_names=predictor_names,
        targets=targets,
        folds=folds,
        fold_assignment=fold_assignment,
    )
    return prediction_rows, aggregate


def _aggregate(
    prediction_rows: Sequence[dict[str, Any]],
    *,
    predictor_names: Sequence[str],
    targets: Sequence[str],
    folds: Sequence[Sequence[int]],
    fold_assignment: dict[int, int],
) -> dict[str, Any]:
    predictive: dict[str, dict[str, Any]] = {}
    for target in targets:
        block: dict[str, Any] = {}
        y = np.asarray([float(r[f"y_{target}"]) for r in prediction_rows])
        for name in predictor_names:
            p = np.asarray([float(r[f"{name}__{target}"]) for r in prediction_rows])
            err = p - y
            block[name] = {
                "mse": float(np.mean(err * err)),
                "mae": float(np.mean(np.abs(err))),
                "spearman_pred_target": spearman(p, y),
            }
        predictive[target] = block

    # MSE-reduction comparisons.
    improvements: dict[str, dict[str, Any]] = {}
    cmps_xi = [
        ("intercept", "ridge_geom"),
        ("ridge_geom", "ridge_geom_apair"),
        ("ridge_geom_apair", "ridge_geom_apair_state"),
        ("ridge_geom_apair_state", "ridge_poly2"),
        ("apair_only", "ridge_geom_apair_state"),
    ]
    cmps_g = [
        ("apair_only", "ridge_geom_apair"),
        ("ridge_geom_apair", "ridge_geom_apair_state"),
        ("ridge_geom_apair_state", "ridge_poly2"),
        ("apair_only", "ridge_geom_apair_state"),
    ]
    cmps_by_target = {"xi": cmps_xi, "g_pair": cmps_g}
    for target in targets:
        bucket: dict[str, Any] = {}
        for baseline, candidate in cmps_by_target.get(target, []):
            key = f"{candidate}_vs_{baseline}"
            bucket[key] = bootstrap_seed_diff(
                prediction_rows, baseline, candidate, target
            )
        improvements[target] = bucket

    # B=2 ranking metric: rank candidate pairs by predicted G_pair and
    # compare to oracle (true G_pair) top-K within each seed.
    ranking_metric: dict[str, dict[str, Any]] = {}
    for name in predictor_names:
        ranking_metric[name] = b2_ranking_metric(
            prediction_rows, name, target_label="g_pair", K=5
        )

    fold_sizes = [
        {"fold_id": i, "n_test_seeds": len(fold)} for i, fold in enumerate(folds)
    ]

    verdict = _make_verdict(predictive, improvements)
    return {
        "meta": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generated_by": f"scripts/proseco/state_predictability/analyze_state_predictability_pair.py@{git_short_hash()}",
            "n_rows": len(prediction_rows),
            "n_seeds": len({int(r["seed"]) for r in prediction_rows}),
            "n_pairs_per_seed": len({(int(r["s"]), int(r["t"])) for r in prediction_rows if int(r["seed"]) == int(prediction_rows[0]["seed"])}),
            "targets": list(targets),
            "predictor_names": list(predictor_names),
            "state_features": list(STATE_FEATURES),
            "leakage_excluded_fields": list(LEAKAGE_FIELDS),
            "folds": fold_sizes,
            "fold_assignment": {str(k): int(v) for k, v in fold_assignment.items()},
        },
        "predictive_metrics": predictive,
        "mse_improvements": improvements,
        "b2_top5_ranking": ranking_metric,
        "verdict": verdict,
    }


def _make_verdict(predictive: dict[str, Any], improvements: dict[str, Any]) -> dict[str, Any]:
    # For xi: the relevant question is whether geom+apair+state beats geom+apair.
    xi_imp = improvements.get("xi", {}).get(
        "ridge_geom_apair_state_vs_ridge_geom_apair"
    )
    g_imp = improvements.get("g_pair", {}).get(
        "ridge_geom_apair_state_vs_ridge_geom_apair"
    )
    poly_xi_imp = improvements.get("xi", {}).get(
        "ridge_poly2_vs_ridge_geom_apair_state"
    )

    def label_for(imp: dict[str, Any] | None) -> str:
        if imp is None:
            return "no_test"
        if imp["ci95_lo"] > 0 and imp["pct_seeds_improved"] >= 0.6:
            return "state_predictability_supported"
        if imp["mean_mse_reduction"] > 0 and imp["pct_seeds_improved"] >= 0.5:
            return "state_predictability_weak"
        return "state_predictability_not_supported"

    return {
        "label_xi": label_for(xi_imp),
        "label_g_pair": label_for(g_imp),
        "label_poly2_xi": label_for(poly_xi_imp),
        "xi_state_vs_geom_apair": xi_imp,
        "g_state_vs_geom_apair": g_imp,
        "poly2_vs_state_xi": poly_xi_imp,
        "interpretation": (
            "Each label is the verdict for the relevant baseline-vs-candidate "
            "comparison. 'supported' means CI excludes zero on the positive "
            "side AND >=60%% of seeds improved. 'weak' means mean MSE "
            "reduction > 0 AND >=50%% of seeds improved (CI may touch zero). "
            "Otherwise 'not_supported'. Use these labels to decide whether "
            "to pursue feature enrichment, schedule-level prediction, or "
            "writing the diagnostic case study."
        ),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"unserializable: {type(o)!r}")


def write_interpretation(out_dir: Path, aggregate: dict[str, Any]) -> None:
    verdict = aggregate["verdict"]
    pred = aggregate["predictive_metrics"]
    imp = aggregate["mse_improvements"]
    rank = aggregate["b2_top5_ranking"]
    meta = aggregate["meta"]

    lines: list[str] = []
    lines += [
        "# Pair-Level State Predictability Audit",
        "",
        "## Question",
        "",
        ("Does pre-correction state at the two correction times (s,t) predict"
         " the pair interaction xi_{s,t} = G({s,t}) - delta_s - delta_t beyond"
         " time-only and additive baselines? Secondary: same question for"
         " G_pair = G({s,t})."),
        "",
        "## Data",
        "",
        f"- Rows: {meta['n_rows']} (seed x pair).",
        f"- Seeds: {meta['n_seeds']}; pairs per seed: {meta['n_pairs_per_seed']}.",
        "- Source: results/phase1_interaction_diag_nogit/xi_raw.json (Gate 3a)",
        "  joined to results/phase1_proseco_owt_full/protocol_a/trajectory_*.json.",
        "- Pre-correction state features only:"
        f" {', '.join(meta['state_features'])}.",
        "- Excluded as leakage:"
        f" {', '.join(meta['leakage_excluded_fields'])} (post-correction).",
        "",
        "## Targets",
        "",
        "- Primary: xi_{i,s,t} = G_i({s,t}) - delta_s - delta_t.",
        "- Secondary: G_i({s,t}).",
        "",
        "## Predictors",
        "",
        "- `intercept`            mean target on the train fold.",
        "- `apair_only`           A_pair = delta_s + delta_t (linear).",
        "- `ridge_geom`           cubic geometry of (s, t, |t-s|) + phase indicators.",
        "- `ridge_geom_apair`     geom + A_pair.",
        "- `ridge_geom_apair_state` geom + A_pair + pre-correction state"
        " features at s and t with sum/diff transforms.",
        "- `ridge_poly2`          degree-2 polynomial expansion of"
        " (A_pair, state features) on top of geom.",
        "",
        "## Validation Protocol",
        "",
        f"- Grouped {len(meta['folds'])}-fold by seed (no seed in both train and test).",
        f"- Folds: {meta['folds']}.",
        "- Standardization fit on train only; ridge alpha = 1.0.",
        "- Per-seed MSE reduction with seed-bootstrap 95% CI"
        " (2000 resamples).",
        "",
        "## Results",
        "",
        "### Held-out MSE per predictor",
        "",
    ]
    for target in meta["targets"]:
        lines.append(f"**Target: {target}**")
        lines.append("")
        lines.append("| Predictor | MSE | MAE | Spearman(pred,target) |")
        lines.append("|---|---:|---:|---:|")
        for name in meta["predictor_names"]:
            m = pred[target][name]
            lines.append(
                f"| `{name}` | {m['mse']:.6f} | {m['mae']:.6f} |"
                f" {m['spearman_pred_target']:.3f} |"
            )
        lines.append("")

    lines += [
        "### MSE-reduction comparisons (per-seed bootstrap)",
        "",
    ]
    for target in meta["targets"]:
        if not imp.get(target):
            continue
        lines.append(f"**Target: {target}**")
        lines.append("")
        lines.append("| Comparison | mean reduction | 95% CI | %% seeds improved |")
        lines.append("|---|---:|---|---:|")
        for key, block in imp[target].items():
            lines.append(
                f"| `{key}` | {block['mean_mse_reduction']:+.6f} |"
                f" [{block['ci95_lo']:+.6f}, {block['ci95_hi']:+.6f}] |"
                f" {block['pct_seeds_improved']:.2f} |"
            )
        lines.append("")

    lines += [
        "### Within-seed top-5 ranking on G_pair",
        "",
        "Each seed has 73 candidate pairs. We rank by predicted G_pair and"
        " compute top-5 overlap with the oracle top-5 (by true G_pair).",
        "",
        "| Predictor | mean top-5 overlap | mean close-ratio (vs median pair) |",
        "|---|---:|---:|",
    ]
    for name in meta["predictor_names"]:
        block = rank[name]
        lines.append(
            f"| `{name}` | {block['mean_top_k_overlap']:.3f} |"
            f" {block['mean_close_ratio']:.3f} |"
        )
    lines.append("")

    lines += [
        "## Interpretation",
        "",
        f"- Verdict (xi, state vs geom+A): **{verdict['label_xi']}**.",
        f"- Verdict (G_pair, state vs geom+A): **{verdict['label_g_pair']}**.",
        f"- Verdict (poly2 vs state, xi): **{verdict['label_poly2_xi']}**.",
        "",
        verdict["interpretation"],
        "",
        "Reading guide. The pair-level audit is the natural extension of the"
        " marginal audit. The marginal audit asked whether observable state"
        " predicts delta_t beyond time. This audit asks whether observable"
        " pre-correction state at (s,t) predicts the *interaction* xi_{s,t}"
        " (and the joint pair gain G_{s,t}) beyond the marginal-additive"
        " baseline. Because corr(A_pair, xi) is large and negative on Gate 3a"
        " (~ -0.70), `apair_only` is already a strong linear predictor of xi"
        " in the training distribution. The relevant question is therefore"
        " incremental: does state add anything on top of geom + A_pair?",
        "",
        "## Limitations",
        "",
        "- ProSeCo-OWT only. The framework is corrector-agnostic; the"
        " empirical verdict is not.",
        "- 73 sampled pairs (stratified, not exhaustive).",
        "- 30 seeds = 30 bootstrap units. Variance of per-seed MSE reductions"
        " is the dominant uncertainty.",
        "- Pre-correction state features are 6 aggregate scalars at each"
        " endpoint. A negative result does not rule out richer features"
        " (e.g. corrector-internal log-prob gaps over R_t, derivatives,"
        " token-resolved summaries).",
        "- Polynomial-degree-2 ridge is a conservative non-linearity test."
        " A strictly negative non-linear arm does not rule out tree models"
        " or kernel methods.",
        "- B=2 top-5 overlap is a ranking proxy *within* a fixed, sparse pool;"
        " it is not a schedule-level result.",
        "",
        "## Decision",
        "",
    ]
    decision = _make_decision_paragraph(verdict)
    lines += decision
    (out_dir / "interpretation.md").write_text("\n".join(lines) + "\n")


def _make_decision_paragraph(verdict: dict[str, Any]) -> list[str]:
    label_xi = verdict["label_xi"]
    label_g = verdict["label_g_pair"]
    label_poly = verdict["label_poly2_xi"]
    if "supported" in {label_xi, label_g} and label_xi != "state_predictability_not_supported":
        return [
            "The pair-level audit produces a non-trivial positive signal:"
            " observable pre-correction state at (s,t) adds incremental"
            " predictive value for the pair-level target on held-out seeds."
            " This motivates: (i) a feature-enrichment ablation to localize"
            " which features carry the signal, (ii) a set-level / trajectory"
            " extension to test whether the same state signal predicts G(S)"
            " for B >= 3, and (iii) supervisor consultation on whether a"
            " heavier-backbone external-validity gate is now warranted before"
            " write-up.",
        ]
    if label_xi == "state_predictability_weak" or label_g == "state_predictability_weak":
        return [
            "The pair-level audit produces a weak positive signal. The"
            " mean per-seed MSE reduction is positive but its bootstrap CI"
            " touches zero. Recommended next step: a feature-enrichment"
            " ablation focused on features that are theoretically pair-aware"
            " (e.g. log-probability gap over R_t, distributional contrast"
            " between s and t) before any new HPC run. Do not start writing"
            " the final ch7 conclusions until the ablation is in.",
        ]
    if label_poly == "state_predictability_supported":
        return [
            "Linear pair-level state predictability is not supported, but the"
            " polynomial-degree-2 expansion of state features shows"
            " incremental predictive value on xi. This is consistent with"
            " non-linear interaction structure. Recommended next step: try"
            " a tree-based estimator (RandomForest/GBM if installed) under"
            " the same grouped split before drawing the final negative.",
        ]
    return [
        "The pair-level audit does not improve on the geom + A_pair baseline."
        " Together with the marginal audit, this indicates that the current"
        " set of pre-correction scalar state features carries no incremental"
        " held-out signal for either delta_t or xi_{s,t} on ProSeCo-OWT."
        " The thesis can now safely report the corrector-timing structure as"
        " (a) real at the schedule level (MC-oracle headroom),"
        " (b) interaction-driven and mostly redundant (Gate 3a),"
        " (c) reachable by true-G search (Phase 3a), and"
        " (d) not deployable through online state-conditioning under the"
        " current state representation, at either marginal or pair level."
        " The natural follow-ups remain (i) one feature-enrichment ablation"
        " and (ii) a supervisor-elective external-validity gate on a"
        " heavier backbone, in that order.",
    ]


# ---------------------------------------------------------------------------
# Plotting (optional, Pillow only)
# ---------------------------------------------------------------------------

def _font(size: int = 14) -> Any:
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def write_plots(out_dir: Path, prediction_rows: Sequence[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    if Image is None:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_mse_bars(plot_dir / "mse_by_predictor_xi.png", aggregate, "xi")
    _plot_mse_bars(plot_dir / "mse_by_predictor_g_pair.png", aggregate, "g_pair")
    _plot_xi_distribution(plot_dir / "xi_distribution.png", prediction_rows)
    _plot_pred_vs_actual(
        plot_dir / "pred_vs_actual_xi_ridge_state.png",
        prediction_rows, "ridge_geom_apair_state", "xi",
    )
    _plot_residuals_by_distance(
        plot_dir / "residuals_by_distance_xi.png", prediction_rows,
        "ridge_geom_apair_state", "xi",
    )


def _plot_mse_bars(path: Path, aggregate: dict[str, Any], target: str) -> None:
    pred = aggregate["predictive_metrics"].get(target)
    if not pred:
        return
    names = list(pred.keys())
    vals = [pred[n]["mse"] for n in names]
    _bar_chart(path, names, vals, f"Held-out MSE — target = {target}", "MSE")


def _plot_xi_distribution(path: Path, prediction_rows: Sequence[dict[str, Any]]) -> None:
    xs = [float(r["y_xi"]) for r in prediction_rows]
    _histogram(path, xs, "xi distribution (all seeds, all pairs)", "xi")


def _plot_pred_vs_actual(path: Path, rows: Sequence[dict[str, Any]], pred_key: str, target: str) -> None:
    y = [float(r[f"y_{target}"]) for r in rows]
    p = [float(r[f"{pred_key}__{target}"]) for r in rows]
    _scatter(path, p, y, f"{pred_key} predictions vs actual {target}", "predicted", target)


def _plot_residuals_by_distance(path: Path, rows: Sequence[dict[str, Any]], pred_key: str, target: str) -> None:
    by_d: dict[int, list[float]] = {}
    for r in rows:
        y = float(r[f"y_{target}"])
        p = float(r[f"{pred_key}__{target}"])
        by_d.setdefault(int(r["distance"]), []).append(p - y)
    xs = sorted(by_d)
    means = [float(np.mean(by_d[x])) for x in xs]
    _line_plot(path, xs, means, f"Mean residual by pair distance — {pred_key} on {target}", "|t-s|", "mean residual")


def _bar_chart(path: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    if Image is None:
        return
    w, h = 1100, 580
    margin = 80
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), title, fill="black", font=_font(18))
    finite = [v for v in values if math.isfinite(v)]
    y_min = min(0.0, min(finite) if finite else 0.0)
    y_max = max(finite) if finite else 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    n = len(values)
    plot_w = w - 2 * margin
    bar_w = max(8, int(plot_w / max(n, 1) * 0.6))
    zero_y = h - margin - (h - 2 * margin) * (0.0 - y_min) / (y_max - y_min)
    draw.line([(margin, zero_y), (w - margin, zero_y)], fill="#666666", width=1)
    for i, (label, value) in enumerate(zip(labels, values)):
        cx = margin + (i + 0.5) * plot_w / n
        y = h - margin - (h - 2 * margin) * (value - y_min) / (y_max - y_min)
        color = "#2f855a" if value >= 0 else "#c53030"
        draw.rectangle([cx - bar_w / 2, min(y, zero_y), cx + bar_w / 2, max(y, zero_y)], fill=color)
        draw.text((cx - 45, h - margin + 10), label[:18], fill="black", font=_font(10))
    draw.rectangle([margin, margin, w - margin, h - margin], outline="black")
    draw.text((10, margin), ylabel, fill="black", font=_font(12))
    img.save(path)


def _histogram(path: Path, values: Sequence[float], title: str, xlabel: str, n_bins: int = 30) -> None:
    if Image is None or not values:
        return
    arr = np.asarray(values, dtype=np.float64)
    hist, edges = np.histogram(arr, bins=n_bins)
    w, h = 900, 520
    margin = 70
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), title, fill="black", font=_font(18))
    y_max = max(hist.max(), 1)
    plot_w = w - 2 * margin
    plot_h = h - 2 * margin
    bar_w = plot_w / n_bins
    zero_y = h - margin
    for i, count in enumerate(hist):
        x0 = margin + i * bar_w
        x1 = margin + (i + 1) * bar_w
        y0 = zero_y - plot_h * count / y_max
        draw.rectangle([x0, y0, x1, zero_y], fill="#2b6cb0", outline="white")
    draw.rectangle([margin, margin, w - margin, h - margin], outline="black")
    draw.text((10, margin), "count", fill="black", font=_font(12))
    draw.text((margin, h - 45), xlabel, fill="black", font=_font(12))
    img.save(path)


def _scatter(path: Path, xs: Sequence[float], ys: Sequence[float], title: str, xlabel: str, ylabel: str) -> None:
    if Image is None or not xs:
        return
    w, h = 720, 720
    margin = 80
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), title, fill="black", font=_font(16))
    xa = np.asarray(xs, dtype=np.float64)
    ya = np.asarray(ys, dtype=np.float64)
    x_min, x_max = float(xa.min()), float(xa.max())
    y_min, y_max = float(ya.min()), float(ya.max())
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0
    for x, y in zip(xa, ya):
        px = margin + (w - 2 * margin) * (x - x_min) / (x_max - x_min)
        py = h - margin - (h - 2 * margin) * (y - y_min) / (y_max - y_min)
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill="#2b6cb0")
    draw.rectangle([margin, margin, w - margin, h - margin], outline="black")
    draw.text((margin, h - 45), xlabel, fill="black", font=_font(12))
    draw.text((10, margin), ylabel, fill="black", font=_font(12))
    img.save(path)


def _line_plot(path: Path, xs: Sequence[float], ys: Sequence[float], title: str, xlabel: str, ylabel: str) -> None:
    if Image is None or not xs:
        return
    w, h = 900, 520
    margin = 70
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), title, fill="black", font=_font(18))
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    x_min, x_max = float(x_arr.min()), float(x_arr.max())
    y_min, y_max = float(min(y_arr.min(), 0)), float(max(y_arr.max(), 0))
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0
    zero_y = h - margin - (h - 2 * margin) * (0.0 - y_min) / (y_max - y_min)
    draw.line([(margin, zero_y), (w - margin, zero_y)], fill="#888888", width=1)
    prev = None
    for x, y in zip(x_arr, y_arr):
        px = margin + (w - 2 * margin) * (x - x_min) / (x_max - x_min)
        py = h - margin - (h - 2 * margin) * (y - y_min) / (y_max - y_min)
        if prev is not None:
            draw.line([prev, (px, py)], fill="#c53030", width=3)
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill="#c53030")
        prev = (px, py)
    draw.rectangle([margin, margin, w - margin, h - margin], outline="black")
    draw.text((margin, h - 45), xlabel, fill="black", font=_font(12))
    draw.text((10, margin), ylabel, fill="black", font=_font(12))
    img.save(path)


# ---------------------------------------------------------------------------
# Feature summary
# ---------------------------------------------------------------------------

def feature_summary(rows: Sequence[PairRow]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in STATE_FEATURES:
        s_vals = np.asarray([r.state_s[name] for r in rows], dtype=np.float64)
        t_vals = np.asarray([r.state_t[name] for r in rows], dtype=np.float64)
        out[name] = {
            "s_mean": float(s_vals.mean()),
            "s_std": float(s_vals.std()),
            "t_mean": float(t_vals.mean()),
            "t_std": float(t_vals.std()),
        }
    out["xi"] = {
        "mean": float(np.mean([r.xi for r in rows])),
        "std": float(np.std([r.xi for r in rows])),
        "p_positive": float(np.mean([r.xi > 0.0 for r in rows])),
    }
    out["g_pair"] = {
        "mean": float(np.mean([r.g_pair for r in rows])),
        "std": float(np.std([r.g_pair for r in rows])),
    }
    a = np.asarray([r.a_pair for r in rows])
    x = np.asarray([r.xi for r in rows])
    out["corr_apair_xi"] = float(np.corrcoef(a, x)[0, 1])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_out_dir(prefix: Path, git_sha: str) -> Path:
    return Path(f"{prefix}_{git_sha}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xi-file", type=Path, default=DEFAULT_XI_FILE)
    parser.add_argument("--protocol-a-dir", type=Path, default=DEFAULT_PROTOCOL_A_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--results-prefix", type=Path, default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1729)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--debug", action="store_true",
        help="Run on the first 3 seeds with 2 folds; do not write to default dir.",
    )
    args = parser.parse_args(argv)

    git_sha = git_short_hash()
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.debug:
        out_dir = Path(f"{args.results_prefix}_debug_{git_sha}")
    else:
        out_dir = default_out_dir(args.results_prefix, git_sha)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already exists and is non-empty; refusing to overwrite"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    feats, seed_T = load_protocol_a_features(args.protocol_a_dir)
    rows = load_pair_rows(args.xi_file, feats, seed_T)

    if args.debug:
        keep = sorted({r.seed for r in rows})[:3]
        rows = [r for r in rows if r.seed in keep]
        n_folds = min(2, len(keep))
    else:
        n_folds = args.n_folds

    prediction_rows, aggregate = run_audit(
        rows,
        n_folds=n_folds,
        fold_seed=args.fold_seed,
        ridge_alpha=args.ridge_alpha,
    )
    feature_meta = feature_summary(rows)

    write_json(out_dir / "config.json", {
        "xi_file": str(args.xi_file),
        "protocol_a_dir": str(args.protocol_a_dir),
        "out_dir": str(out_dir),
        "git_sha": git_sha,
        "n_folds": n_folds,
        "fold_seed": args.fold_seed,
        "ridge_alpha": args.ridge_alpha,
        "debug": bool(args.debug),
        "state_features": list(STATE_FEATURES),
        "leakage_excluded_fields": list(LEAKAGE_FIELDS),
    })
    write_json(out_dir / "aggregate_stats.json", aggregate)
    write_json(out_dir / "feature_summary.json", feature_meta)
    write_json(out_dir / "fold_metrics.json", {
        "folds": aggregate["meta"]["folds"],
        "fold_assignment": aggregate["meta"]["fold_assignment"],
    })
    write_json(out_dir / "prediction_rows.json", prediction_rows)
    write_interpretation(out_dir, aggregate)
    write_plots(out_dir, prediction_rows, aggregate)

    print(json.dumps(aggregate["verdict"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
