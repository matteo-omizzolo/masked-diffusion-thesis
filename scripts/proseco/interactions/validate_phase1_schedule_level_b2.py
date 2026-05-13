#!/usr/bin/env python3
"""Gate 3b B=2 schedule-level validation for Phase 1 interaction diagnostics.

This script uses the already measured Gate 3a sparse pair rows as the fixed
B=2 candidate pool C_2. It evaluates whether an estimated pairwise surrogate

    Q_hat({t,t'}) = A_pair + xi_hat(t,t')

predicts held-out pair gains G({t,t'}) better than the additive surrogate
A_pair. To avoid the tautology Q = A + observed xi = G, every deployable
xi_hat model is fitted on training seeds and evaluated on held-out seeds.

The observed-xi oracle is reported only as a non-deployable diagnostic ceiling.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one xi_hat model."""

    name: str
    shrinkage_strength: float = 10.0
    ridge_alpha: float = 1.0


@dataclass
class FittedXiModel:
    """Fitted state for a training-seed-only xi_hat model."""

    spec: ModelSpec
    global_mean: float
    phase_means: Dict[str, float]
    phase_distance_means: Dict[str, float]
    phase_distance_counts: Dict[str, int]
    phase_distance_shrunk: Dict[str, float]
    ridge_beta: Optional[np.ndarray] = None
    ridge_phase_levels: Optional[List[str]] = None
    ridge_distance_levels: Optional[List[str]] = None
    ridge_a_mean: float = 0.0
    ridge_a_std: float = 1.0


def phase_pair(row: Dict[str, Any]) -> str:
    return f"{row['phase_t']}-{row['phase_tp']}"


def distance_bucket(distance: int) -> str:
    if distance <= 5:
        return "short (1-5)"
    if distance <= 20:
        return "medium (6-20)"
    return "long (21+)"


def phase_distance_key(row: Dict[str, Any]) -> str:
    return f"{phase_pair(row)}|{distance_bucket(int(row['distance']))}"


def load_xi_rows(path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    required = {
        "seed",
        "t",
        "t_prime",
        "phase_t",
        "phase_tp",
        "distance",
        "A_pair",
        "G_pair",
        "xi",
    }
    for idx, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {idx} missing required keys: {sorted(missing)}")
    return rows


def validate_complete_seed_pair_grid(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    seeds = sorted({int(r["seed"]) for r in rows})
    pairs = sorted({(int(r["t"]), int(r["t_prime"])) for r in rows})
    counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
    for row in rows:
        counts[(int(row["seed"]), int(row["t"]), int(row["t_prime"]))] += 1
    duplicate_keys = sum(1 for v in counts.values() if v != 1)
    missing = len(seeds) * len(pairs) - len(rows)
    return {
        "n_rows": len(rows),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_pairs": len(pairs),
        "pairs": [[t, tp] for t, tp in pairs],
        "expected_rows": len(seeds) * len(pairs),
        "missing_seed_pair_rows": missing,
        "duplicate_seed_pair_keys": duplicate_keys,
    }


def build_splits(
    rows: Sequence[Dict[str, Any]],
    *,
    mode: str = "leave_seed_out",
    k_folds: int = 5,
    random_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Build seed-level splits; rows are never split within a seed."""
    seeds = sorted({int(r["seed"]) for r in rows})
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for held-out validation")

    splits: List[Dict[str, Any]] = []
    if mode == "leave_seed_out":
        for fold_id, seed in enumerate(seeds):
            heldout = [seed]
            train = [s for s in seeds if s != seed]
            splits.append({"fold_id": fold_id, "train_seeds": train, "heldout_seeds": heldout})
        return splits

    if mode != "kfold":
        raise ValueError(f"Unknown seed split mode: {mode}")
    if k_folds < 2:
        raise ValueError("kfold requires k_folds >= 2")
    k = min(k_folds, len(seeds))
    rng = np.random.default_rng(random_seed)
    shuffled = list(seeds)
    rng.shuffle(shuffled)
    folds = [sorted(int(x) for x in chunk) for chunk in np.array_split(shuffled, k)]
    for fold_id, heldout in enumerate(folds):
        train = [s for s in seeds if s not in heldout]
        splits.append({"fold_id": fold_id, "train_seeds": train, "heldout_seeds": heldout})
    return splits


def _mean(values: Iterable[float], fallback: float = 0.0) -> float:
    vals = list(values)
    if not vals:
        return fallback
    return float(np.mean(vals))


def _group_mean(rows: Sequence[Dict[str, Any]], key_fn: Any) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(float(row["xi"]))
    return {k: _mean(v) for k, v in grouped.items()}


def _group_count(rows: Sequence[Dict[str, Any]], key_fn: Any) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[key_fn(row)] += 1
    return dict(counts)


def _ridge_features(
    rows: Sequence[Dict[str, Any]],
    *,
    phase_levels: Sequence[str],
    distance_levels: Sequence[str],
    a_mean: float,
    a_std: float,
) -> np.ndarray:
    cols: List[List[float]] = []
    for row in rows:
        values = [1.0]
        pp = phase_pair(row)
        db = distance_bucket(int(row["distance"]))
        values.extend(1.0 if pp == level else 0.0 for level in phase_levels[1:])
        values.extend(1.0 if db == level else 0.0 for level in distance_levels[1:])
        values.append((float(row["A_pair"]) - a_mean) / a_std)
        cols.append(values)
    return np.asarray(cols, dtype=float)


def fit_xi_model(spec: ModelSpec, train_rows: Sequence[Dict[str, Any]]) -> FittedXiModel:
    """Fit one xi_hat model using training rows only."""
    if not train_rows:
        raise ValueError("Cannot fit xi model on empty training rows")

    global_mean = _mean(float(r["xi"]) for r in train_rows)
    phase_means = _group_mean(train_rows, phase_pair)
    phase_distance_means = _group_mean(train_rows, phase_distance_key)
    phase_distance_counts = _group_count(train_rows, phase_distance_key)
    phase_distance_shrunk: Dict[str, float] = {}
    lam = float(spec.shrinkage_strength)
    for key, mean in phase_distance_means.items():
        n = phase_distance_counts[key]
        phase_key = key.split("|", 1)[0]
        parent = phase_means.get(phase_key, global_mean)
        phase_distance_shrunk[key] = float((n * mean + lam * parent) / (n + lam))

    fitted = FittedXiModel(
        spec=spec,
        global_mean=global_mean,
        phase_means=phase_means,
        phase_distance_means=phase_distance_means,
        phase_distance_counts=phase_distance_counts,
        phase_distance_shrunk=phase_distance_shrunk,
    )

    if spec.name == "phase_distance_a_ridge":
        phase_levels = sorted({phase_pair(r) for r in train_rows})
        distance_levels = sorted({distance_bucket(int(r["distance"])) for r in train_rows})
        a_vals = np.asarray([float(r["A_pair"]) for r in train_rows], dtype=float)
        a_mean = float(np.mean(a_vals))
        a_std = float(np.std(a_vals))
        if a_std < 1e-12:
            a_std = 1.0
        x = _ridge_features(
            train_rows,
            phase_levels=phase_levels,
            distance_levels=distance_levels,
            a_mean=a_mean,
            a_std=a_std,
        )
        y = np.asarray([float(r["xi"]) for r in train_rows], dtype=float)
        penalty = np.eye(x.shape[1], dtype=float) * float(spec.ridge_alpha)
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
        fitted.ridge_beta = beta
        fitted.ridge_phase_levels = list(phase_levels)
        fitted.ridge_distance_levels = list(distance_levels)
        fitted.ridge_a_mean = a_mean
        fitted.ridge_a_std = a_std

    return fitted


def predict_xi(fitted: FittedXiModel, row: Dict[str, Any]) -> float:
    """Predict xi for one held-out row from training-only fitted state."""
    name = fitted.spec.name
    if name == "additive":
        return 0.0
    if name == "oracle_observed_xi":
        return float(row["xi"])
    if name == "train_global_mean":
        return fitted.global_mean
    if name == "phase_pair_mean":
        return fitted.phase_means.get(phase_pair(row), fitted.global_mean)
    if name == "phase_distance_mean":
        key = phase_distance_key(row)
        return fitted.phase_distance_means.get(
            key,
            fitted.phase_means.get(phase_pair(row), fitted.global_mean),
        )
    if name == "phase_distance_shrinkage":
        key = phase_distance_key(row)
        return fitted.phase_distance_shrunk.get(
            key,
            fitted.phase_means.get(phase_pair(row), fitted.global_mean),
        )
    if name == "phase_distance_a_ridge":
        if fitted.ridge_beta is None or fitted.ridge_phase_levels is None or fitted.ridge_distance_levels is None:
            return fitted.global_mean
        x = _ridge_features(
            [row],
            phase_levels=fitted.ridge_phase_levels,
            distance_levels=fitted.ridge_distance_levels,
            a_mean=fitted.ridge_a_mean,
            a_std=fitted.ridge_a_std,
        )
        return float(x[0] @ fitted.ridge_beta)
    raise ValueError(f"Unknown xi model: {name}")


def model_specs_for(requested: str, *, shrinkage_strength: float, ridge_alpha: float) -> List[ModelSpec]:
    specs = [
        ModelSpec("additive", shrinkage_strength, ridge_alpha),
        ModelSpec("oracle_observed_xi", shrinkage_strength, ridge_alpha),
        ModelSpec("train_global_mean", shrinkage_strength, ridge_alpha),
        ModelSpec("phase_pair_mean", shrinkage_strength, ridge_alpha),
        ModelSpec("phase_distance_mean", shrinkage_strength, ridge_alpha),
        ModelSpec("phase_distance_shrinkage", shrinkage_strength, ridge_alpha),
    ]
    if requested in {"phase_distance_a", "all"}:
        specs.append(ModelSpec("phase_distance_a_ridge", shrinkage_strength, ridge_alpha))
    return specs


def model_metadata(name: str) -> Dict[str, Any]:
    return {
        "additive": {
            "display_name": "Additive A",
            "non_deployable": False,
            "role": "baseline",
        },
        "oracle_observed_xi": {
            "display_name": "Observed-xi oracle",
            "non_deployable": True,
            "role": "diagnostic_ceiling",
        },
        "train_global_mean": {
            "display_name": "Train global mean xi",
            "non_deployable": False,
            "role": "heldout_q",
        },
        "phase_pair_mean": {
            "display_name": "Phase-pair mean xi",
            "non_deployable": False,
            "role": "heldout_q",
        },
        "phase_distance_mean": {
            "display_name": "Phase + distance mean xi",
            "non_deployable": False,
            "role": "heldout_q",
        },
        "phase_distance_shrinkage": {
            "display_name": "Phase + distance shrinkage xi",
            "non_deployable": False,
            "role": "heldout_q",
        },
        "phase_distance_a_ridge": {
            "display_name": "Ridge xi: phase + distance + A",
            "non_deployable": False,
            "role": "heldout_q",
        },
    }[name]


def run_crossfit_predictions(
    rows: Sequence[Dict[str, Any]],
    splits: Sequence[Dict[str, Any]],
    model_specs: Sequence[ModelSpec],
) -> List[Dict[str, Any]]:
    """Generate one held-out validation row per original row and model."""
    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)

    output: List[Dict[str, Any]] = []
    for split in splits:
        train_seeds = set(int(s) for s in split["train_seeds"])
        heldout_seeds = set(int(s) for s in split["heldout_seeds"])
        train_rows = [r for s in train_seeds for r in by_seed[s]]
        heldout_rows = [r for s in heldout_seeds for r in by_seed[s]]

        fitted_by_model: Dict[str, FittedXiModel] = {}
        for spec in model_specs:
            if spec.name in {"additive", "oracle_observed_xi"}:
                fitted_by_model[spec.name] = FittedXiModel(
                    spec=spec,
                    global_mean=0.0,
                    phase_means={},
                    phase_distance_means={},
                    phase_distance_counts={},
                    phase_distance_shrunk={},
                )
            else:
                fitted_by_model[spec.name] = fit_xi_model(spec, train_rows)

        for row in heldout_rows:
            g = float(row["G_pair"])
            a = float(row["A_pair"])
            for spec in model_specs:
                meta = model_metadata(spec.name)
                fitted = fitted_by_model[spec.name]
                xi_hat = predict_xi(fitted, row)
                pred = a + xi_hat
                abs_err_a = abs(g - a)
                abs_err_q = abs(g - pred)
                output.append(
                    {
                        "model": spec.name,
                        "display_name": meta["display_name"],
                        "role": meta["role"],
                        "non_deployable": bool(meta["non_deployable"]),
                        "fold_id": int(split["fold_id"]),
                        "train_seeds": sorted(train_seeds),
                        "heldout_seeds": sorted(heldout_seeds),
                        "train_seed_count": len(train_seeds),
                        "heldout_seed_count": len(heldout_seeds),
                        "seed": int(row["seed"]),
                        "t": int(row["t"]),
                        "t_prime": int(row["t_prime"]),
                        "phase_t": row["phase_t"],
                        "phase_tp": row["phase_tp"],
                        "phase_pair": phase_pair(row),
                        "distance": int(row["distance"]),
                        "distance_bucket": distance_bucket(int(row["distance"])),
                        "source": row.get("source", ""),
                        "G_pair": g,
                        "A_pair": a,
                        "xi": float(row["xi"]),
                        "xi_hat": float(xi_hat),
                        "prediction": float(pred),
                        "abs_err_A": float(abs_err_a),
                        "abs_err_Q": float(abs_err_q),
                        "sq_err_A": float((g - a) ** 2),
                        "sq_err_Q": float((g - pred) ** 2),
                    }
                )
    return output


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks, 1-indexed, implemented without scipy."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if len(xx) < 2 or float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson_corr(rankdata_average(np.asarray(x, dtype=float)), rankdata_average(np.asarray(y, dtype=float)))


def compute_model_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("Cannot compute metrics for empty rows")
    g = np.asarray([float(r["G_pair"]) for r in rows], dtype=float)
    a = np.asarray([float(r["A_pair"]) for r in rows], dtype=float)
    q = np.asarray([float(r["prediction"]) for r in rows], dtype=float)
    abs_a = np.asarray([float(r["abs_err_A"]) for r in rows], dtype=float)
    abs_q = np.asarray([float(r["abs_err_Q"]) for r in rows], dtype=float)
    sq_a = np.asarray([float(r["sq_err_A"]) for r in rows], dtype=float)
    sq_q = np.asarray([float(r["sq_err_Q"]) for r in rows], dtype=float)
    eta_mean = float(np.mean(abs_a))
    zeta_mean = float(np.mean(abs_q))
    r_s = spearman_corr(a, g)
    p_s = spearman_corr(q, g)
    r_p = pearson_corr(a, g)
    p_p = pearson_corr(q, g)
    return {
        "n_rows": float(len(rows)),
        "n_seeds": float(len({int(r["seed"]) for r in rows})),
        "eta_mean": eta_mean,
        "zeta_mean": zeta_mean,
        "eta_median": float(np.median(abs_a)),
        "zeta_median": float(np.median(abs_q)),
        "eta_p90": float(np.quantile(abs_a, 0.9)),
        "zeta_p90": float(np.quantile(abs_q, 0.9)),
        "RMSE_A": float(math.sqrt(np.mean(sq_a))),
        "RMSE_Q": float(math.sqrt(np.mean(sq_q))),
        "R_spearman": r_s,
        "P_spearman": p_s,
        "R_pearson": r_p,
        "P_pearson": p_p,
        "delta_abs_error": eta_mean - zeta_mean,
        "delta_spearman": p_s - r_s if not (math.isnan(p_s) or math.isnan(r_s)) else float("nan"),
        "delta_pearson": p_p - r_p if not (math.isnan(p_p) or math.isnan(r_p)) else float("nan"),
    }


def _json_float(value: float) -> Optional[float]:
    if value is None or math.isnan(float(value)) or math.isinf(float(value)):
        return None
    return float(value)


def seed_cluster_bootstrap(
    rows: Sequence[Dict[str, Any]],
    *,
    n_boot: int,
    random_seed: int = 0,
) -> Dict[str, Any]:
    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    seeds = sorted(by_seed)
    rng = np.random.default_rng(random_seed)
    keys = [
        "eta_mean",
        "zeta_mean",
        "delta_abs_error",
        "R_spearman",
        "P_spearman",
        "delta_spearman",
    ]
    samples: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        sampled_seeds = list(rng.choice(seeds, size=len(seeds), replace=True))
        boot_rows: List[Dict[str, Any]] = []
        for seed in sampled_seeds:
            boot_rows.extend(by_seed[int(seed)])
        metrics = compute_model_metrics(boot_rows)
        for key in keys:
            value = float(metrics[key])
            if not math.isnan(value):
                samples[key].append(value)

    out: Dict[str, Any] = {"n_boot": n_boot, "cluster_unit": "seed", "metrics": {}}
    for key, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        if len(arr) == 0:
            out["metrics"][key] = {"ci_lo_95": None, "ci_hi_95": None, "p_gt_0": None}
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out["metrics"][key] = {
                "ci_lo_95": float(lo),
                "ci_hi_95": float(hi),
                "p_gt_0": float(np.mean(arr > 0)),
            }
    return out


def summarize_by_model(
    validation_rows: Sequence[Dict[str, Any]],
    *,
    n_boot: int,
    random_seed: int,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        grouped[row["model"]].append(row)

    per_model: Dict[str, Any] = {}
    for i, (name, model_rows) in enumerate(sorted(grouped.items())):
        point = compute_model_metrics(model_rows)
        bootstrap = seed_cluster_bootstrap(model_rows, n_boot=n_boot, random_seed=random_seed + i)
        meta = model_metadata(name)
        per_model[name] = {
            "display_name": meta["display_name"],
            "role": meta["role"],
            "non_deployable": meta["non_deployable"],
            "point": {k: _json_float(v) for k, v in point.items()},
            "bootstrap": bootstrap,
        }

    deployable = {
        name: block
        for name, block in per_model.items()
        if block["role"] == "heldout_q" and not block["non_deployable"]
    }
    best_model = None
    if deployable:
        best_model = max(
            deployable,
            key=lambda name: (
                deployable[name]["point"]["delta_abs_error"],
                deployable[name]["point"]["delta_spearman"],
            ),
        )
    return {"per_model": per_model, "best_heldout_q_model": best_model}


def git_sha() -> Tuple[str, str]:
    try:
        full = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        short = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Could not resolve git SHA; refusing to write untagged output") from exc
    if not full or not short or full == "nogit" or short == "nogit":
        raise RuntimeError("Could not resolve real git SHA; refusing to write untagged output")
    return full, short


def validate_out_dir(out_dir: Path, short_sha: str) -> None:
    name = out_dir.name
    if "nogit" in name:
        raise ValueError("Output directory must not contain 'nogit'")
    if not name.startswith("phase1_schedule_validation_b2_"):
        raise ValueError("Output directory must be named phase1_schedule_validation_b2_<git-short-sha>")
    if short_sha not in name:
        raise ValueError(f"Output directory name must include current git short SHA {short_sha}")


def savefig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def rows_for_model(validation_rows: Sequence[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    return [r for r in validation_rows if r["model"] == model]


def make_figures(
    validation_rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    figures_dir: Path,
) -> List[str]:
    if not HAS_MPL:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    best = metrics["best_heldout_q_model"] or "train_global_mean"
    additive_rows = rows_for_model(validation_rows, "additive")
    best_rows = rows_for_model(validation_rows, best)

    def _scatter_identity(ax: Any, x: np.ndarray, y: np.ndarray) -> None:
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        pad = max((hi - lo) * 0.05, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    a = np.asarray([r["A_pair"] for r in additive_rows], dtype=float)
    g = np.asarray([r["G_pair"] for r in additive_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(a, g, s=14, alpha=0.35)
    _scatter_identity(ax, a, g)
    ax.set_xlabel("A_pair")
    ax.set_ylabel("G_pair")
    ax.set_title("A vs G on held-out B=2 rows")
    savefig(fig, figures_dir / "A_vs_G_scatter")
    written.extend(["A_vs_G_scatter.png", "A_vs_G_scatter.pdf"])

    q = np.asarray([r["prediction"] for r in best_rows], dtype=float)
    g_best = np.asarray([r["G_pair"] for r in best_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(q, g_best, s=14, alpha=0.35)
    _scatter_identity(ax, q, g_best)
    ax.set_xlabel(f"Q_hat prediction ({best})")
    ax.set_ylabel("G_pair")
    ax.set_title("Best held-out Q_hat vs G")
    savefig(fig, figures_dir / "Q_vs_G_scatter")
    written.extend(["Q_vs_G_scatter.png", "Q_vs_G_scatter.pdf"])

    model_names = [name for name, block in metrics["per_model"].items() if block["role"] != "baseline"]
    display = [metrics["per_model"][m]["display_name"] for m in model_names]
    zeta = [metrics["per_model"][m]["point"]["zeta_mean"] for m in model_names]
    eta = metrics["per_model"]["additive"]["point"]["eta_mean"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(model_names)), zeta, color="#4c78a8", alpha=0.85)
    ax.axhline(eta, color="black", ls="--", label=f"Additive eta={eta:.4f}")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(display, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Held-out absolute error comparison")
    ax.legend(fontsize=8)
    savefig(fig, figures_dir / "abs_error_comparison")
    written.extend(["abs_error_comparison.png", "abs_error_comparison.pdf"])

    p_vals = [metrics["per_model"][m]["point"]["P_spearman"] for m in model_names]
    r_val = metrics["per_model"]["additive"]["point"]["R_spearman"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(model_names)), p_vals, color="#59a14f", alpha=0.85)
    ax.axhline(r_val, color="black", ls="--", label=f"Additive R={r_val:.3f}")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(display, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Held-out rank correlation comparison")
    ax.legend(fontsize=8)
    savefig(fig, figures_dir / "spearman_comparison")
    written.extend(["spearman_comparison.png", "spearman_comparison.pdf"])

    res_a = np.asarray([r["G_pair"] - r["A_pair"] for r in best_rows], dtype=float)
    res_q = np.asarray([r["G_pair"] - r["prediction"] for r in best_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(res_a, res_q, s=14, alpha=0.35)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Residual G - A")
    ax.set_ylabel("Residual G - Q_hat")
    ax.set_title(f"Residual comparison for {best}")
    savefig(fig, figures_dir / "residual_A_vs_Q")
    written.extend(["residual_A_vs_Q.png", "residual_A_vs_Q.pdf"])

    xi = np.asarray([r["xi"] for r in best_rows], dtype=float)
    xi_hat = np.asarray([r["xi_hat"] for r in best_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xi_hat, xi, s=14, alpha=0.35)
    _scatter_identity(ax, xi_hat, xi)
    ax.set_xlabel("xi_hat")
    ax.set_ylabel("observed held-out xi")
    ax.set_title(f"xi_hat vs observed xi for {best}")
    savefig(fig, figures_dir / "xi_hat_vs_xi")
    written.extend(["xi_hat_vs_xi.png", "xi_hat_vs_xi.pdf"])

    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        by_seed[int(row["seed"])].append(row)
    seeds = sorted(by_seed)
    deltas = [compute_model_metrics(by_seed[s])["delta_abs_error"] for s in seeds]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(seeds)), deltas, color=["#59a14f" if d > 0 else "#e15759" for d in deltas])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], rotation=90, fontsize=7)
    ax.set_ylabel("eta_mean - zeta_mean")
    ax.set_title(f"Per-held-out-seed error improvement for {best}")
    savefig(fig, figures_dir / "seed_fold_metrics")
    written.extend(["seed_fold_metrics.png", "seed_fold_metrics.pdf"])

    rows_table = []
    for name, block in metrics["per_model"].items():
        point = block["point"]
        rows_table.append([
            name,
            f"{point['zeta_mean']:.4f}",
            f"{point['delta_abs_error']:.4f}",
            f"{point['P_spearman']:.3f}",
            f"{point['delta_spearman']:.3f}",
        ])
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.45 * len(rows_table) + 1.0)))
    ax.axis("off")
    table = ax.table(
        cellText=rows_table,
        colLabels=["model", "zeta", "eta-zeta", "P", "P-R"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("Model comparison summary")
    savefig(fig, figures_dir / "model_comparison_table")
    written.extend(["model_comparison_table.png", "model_comparison_table.pdf"])

    return written


def _ci_text(block: Dict[str, Any], key: str) -> str:
    ci = block["bootstrap"]["metrics"][key]
    lo = ci["ci_lo_95"]
    hi = ci["ci_hi_95"]
    if lo is None or hi is None:
        return "[NA, NA]"
    return f"[{lo:.4f}, {hi:.4f}]"


def write_interpretation(
    out_dir: Path,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    best = metrics["best_heldout_q_model"]
    best_block = metrics["per_model"][best] if best else None
    additive = metrics["per_model"]["additive"]
    lines = [
        "# Gate 3b B=2 Schedule-Level Validation",
        "",
        "## What This Tests",
        "",
        "For B=2, each candidate schedule is one measured Gate 3a pair S={t,t'}.",
        "The validation asks whether a training-seed estimate xi_hat(t,t') makes",
        "Q_hat(S)=A(S)+xi_hat(t,t') predict held-out G(S) better than A(S).",
        "",
        "The candidate pool C_2 is fixed to the 73 measured Gate 3a pairs. No",
        "held-out G_pair or held-out xi is used to fit deployable xi_hat models.",
        "",
        "Gate 3a found mostly negative / redundant xi. In this B=2 validation,",
        "a useful Q_hat model is therefore expected to help primarily by subtracting",
        "a learned redundancy penalty from A_pair, not by adding complementarity.",
        "",
        "The observed-xi oracle is included only as a non-deployable ceiling:",
        "A_pair + observed xi equals G_pair by definition and is not evidence for",
        "a usable Q_hat model.",
        "",
        "## Provenance",
        "",
        f"- Validation git SHA: `{config['git_sha']}`",
        f"- Input rows: `{config['xi_raw']}`",
        "- Gate 3a caveat: the input folder was `phase1_interaction_diag_nogit`",
        "  and its manifest had `git_hash: unknown`; this validation output is",
        "  explicitly SHA-tagged.",
        f"- Split mode: `{config['seed_split_mode']}`",
        f"- Bootstrap resamples: {config['n_boot']} clustered by seed",
        "",
        "## Model Comparison",
        "",
        "| Model | Role | zeta mean | eta-zeta | 95% CI eta-zeta | P Spearman | P-R | 95% CI P-R |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for name, block in metrics["per_model"].items():
        point = block["point"]
        lines.append(
            f"| {name} | {block['role']} | {point['zeta_mean']:.4f} | "
            f"{point['delta_abs_error']:.4f} | {_ci_text(block, 'delta_abs_error')} | "
            f"{point['P_spearman']:.4f} | {point['delta_spearman']:.4f} | "
            f"{_ci_text(block, 'delta_spearman')} |"
        )

    lines.extend(["", "## Interpretation", ""])
    if best and best_block:
        point = best_block["point"]
        delta_ci = best_block["bootstrap"]["metrics"]["delta_abs_error"]
        spear_ci = best_block["bootstrap"]["metrics"]["delta_spearman"]
        lines.extend(
            [
                f"- Best held-out Q_hat model by mean absolute-error improvement: `{best}`.",
                f"- Additive eta_mean: {additive['point']['eta_mean']:.4f}.",
                f"- Best-model zeta_mean: {point['zeta_mean']:.4f}.",
                f"- delta_abs_error = eta - zeta: {point['delta_abs_error']:.4f} "
                f"with 95% CI {_ci_text(best_block, 'delta_abs_error')} "
                f"and bootstrap P(delta>0)={delta_ci['p_gt_0']:.3f}.",
                f"- Additive R Spearman: {additive['point']['R_spearman']:.4f}.",
                f"- Best-model P Spearman: {point['P_spearman']:.4f}.",
                f"- delta_spearman = P - R: {point['delta_spearman']:.4f} "
                f"with 95% CI {_ci_text(best_block, 'delta_spearman')} "
                f"and bootstrap P(delta>0)={spear_ci['p_gt_0']:.3f}.",
                "",
            ]
        )
        error_support = (
            delta_ci["ci_lo_95"] is not None
            and delta_ci["ci_lo_95"] > 0
        )
        rank_support = (
            spear_ci["ci_lo_95"] is not None
            and spear_ci["ci_lo_95"] > 0
        )
        if error_support or rank_support:
            lines.append(
                "Result: B=2 Gate 3b supports the tested pairwise surrogate on "
                "held-out seeds for prediction and/or ranking, but this is still "
                "only B=2 evidence and does not validate Theorem B for B=3/4."
            )
            lines.append(
                "Next: proceed to a pre-registered B=3/4 schedule-level validation "
                "with fixed candidate pools and no held-out-G leakage."
            )
        else:
            lines.append(
                "Result: the tested low-dimensional Q_hat models do not clear the "
                "strong B=2 decision rule. Pairwise structure exists at Gate 3a, "
                "but these estimators do not yet improve held-out prediction/ranking "
                "over A with decisive uncertainty."
            )
            lines.append(
                "Next: inspect failure mode before extending to B=3/4; options are "
                "a better pre-registered xi estimator or additional targeted pair "
                "measurements, not a pairwise scheduler."
            )
    else:
        lines.append("No deployable held-out Q_hat model was evaluated.")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is B=2 only. It does not prove Theorem B for B=3/4.",
            "- It does not support a deployable scheduler; this is Phase 1",
            "  population / seed-split validation, not Level-3 feature-conditioned",
            "  scheduling.",
            "- No K=3 smoke numbers are used.",
            "- No model-agnostic claim is implied.",
            "",
        ]
    )
    (out_dir / "interpretation.md").write_text("\n".join(lines))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Gate 3b B=2 held-out Q_hat vs A")
    p.add_argument("--xi_raw", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed_split_mode", choices=["leave_seed_out", "kfold"], default="leave_seed_out")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--model", choices=["phase_distance", "phase_distance_a", "all"], default="phase_distance_a")
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--shrinkage_strength", type=float, default=10.0)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    full_sha, short_sha = git_sha()
    validate_out_dir(args.out_dir, short_sha)

    rows = load_xi_rows(args.xi_raw)
    grid = validate_complete_seed_pair_grid(rows)
    if grid["missing_seed_pair_rows"] != 0 or grid["duplicate_seed_pair_keys"] != 0:
        raise SystemExit(f"Input rows are not a complete seed-pair grid: {grid}")

    specs = model_specs_for(
        args.model,
        shrinkage_strength=args.shrinkage_strength,
        ridge_alpha=args.ridge_alpha,
    )
    splits = build_splits(
        rows,
        mode=args.seed_split_mode,
        k_folds=args.k_folds,
        random_seed=args.random_seed,
    )
    validation_rows = run_crossfit_predictions(rows, splits, specs)
    metrics = summarize_by_model(validation_rows, n_boot=args.n_boot, random_seed=args.random_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = make_figures(validation_rows, metrics, args.out_dir / "figures")

    config = {
        "script": "scripts/proseco/interactions/validate_phase1_schedule_level_b2.py",
        "command": " ".join(sys.argv),
        "started_at": started,
        "git_sha": full_sha,
        "git_short_sha": short_sha,
        "xi_raw": str(args.xi_raw),
        "out_dir": str(args.out_dir),
        "n_boot": args.n_boot,
        "seed_split_mode": args.seed_split_mode,
        "k_folds": args.k_folds,
        "model_request": args.model,
        "random_seed": args.random_seed,
        "shrinkage_strength": args.shrinkage_strength,
        "ridge_alpha": args.ridge_alpha,
        "input_summary": grid,
        "split_summary": {
            "n_splits": len(splits),
            "splits": splits,
        },
        "models": [
            {
                "name": spec.name,
                **model_metadata(spec.name),
            }
            for spec in specs
        ],
        "figures": figures,
        "gate3a_provenance_caveat": (
            "Input Gate 3a folder was phase1_interaction_diag_nogit with unknown git hash; "
            "this validation output is tagged with the current git SHA."
        ),
    }

    write_json(args.out_dir / "config.json", config)
    write_json(args.out_dir / "validation_rows.json", validation_rows)
    write_json(args.out_dir / "metrics.json", metrics)
    (args.out_dir / "command_log.txt").write_text(config["command"] + "\n")
    write_interpretation(args.out_dir, metrics, config)

    best = metrics["best_heldout_q_model"]
    print(f"Wrote Gate 3b B=2 validation to {args.out_dir}")
    print(f"Rows: {len(validation_rows)} held-out model rows; best held-out Q model: {best}")
    if best:
        block = metrics["per_model"][best]
        print(
            "Best model delta_abs_error="
            f"{block['point']['delta_abs_error']:.4f} "
            f"CI={_ci_text(block, 'delta_abs_error')} "
            "delta_spearman="
            f"{block['point']['delta_spearman']:.4f} "
            f"CI={_ci_text(block, 'delta_spearman')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
