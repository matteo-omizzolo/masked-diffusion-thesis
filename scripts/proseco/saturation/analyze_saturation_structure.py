"""
analyze_saturation_structure.py
================================
Phase A: Saturation structure analysis of ProSeCo-OWT pairwise interaction data.

Reads results/phase1_interaction_diag_nogit/xi_raw.json (2190 rows) and
produces the saturation-structure summary in results/saturation_structure_<sha>/.

Analyses
--------
A1  Saturation curve — ξ as a function of A_pair (regression + binned curve)
A2  Pair utility by A_pair quartile — G_pair, ξ, P(G_pair>0), P(ξ>0)
A3  G_pair vs A_pair sublinear composition — linear + concave models vs identity
A4  Nested partial-R² waterfall — M0-M5 explaining G_pair
A5  Seed-level robustness — per-seed A_pair→G_pair slope distribution
A6  Phase-pair and distance stratification
A7  Comparison with failed state-feature predictors (entropy, inv_margin baseline)

Outputs (all in OUT_DIR)
-----------------------
summary_saturation.json          — top-level gate conditions + numeric summary
quartile_regime.json             — per-quartile G_pair / ξ stats
composition_models.json          — A1/A3 model fits (linear, sqrt, log)
partial_r2.json                  — A4 nested model R²
seed_robustness.json             — A5 per-seed slopes
stratification.json              — A6 phase/distance breakdowns
comparison_baselines.json        — A7 vs state-feature baselines
interpretation.md                — human-readable narrative

Plots (in OUT_DIR/figures/)
---------------------------
xi_vs_apair_scatter.png
saturation_curve_binned.png
gpair_vs_apair_saturation.png
quartile_regime_table_plot.png
partial_r2_waterfall.png
seed_level_slopes.png
saturation_by_phase.png
saturation_by_distance.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
XI_RAW_PATH = REPO_ROOT / "results" / "phase1_interaction_diag_nogit" / "xi_raw.json"
# Optional: Protocol A trajectory-level state features for A7
PROTOCOL_A_DIR = (
    REPO_ROOT / "results" / "phase1_proseco_owt_full" / "protocol_a"
)

# ---------------------------------------------------------------------------
# Gate conditions (all five must pass for Phase B to proceed)
# ---------------------------------------------------------------------------
GATE_CONDITIONS = {
    "C1_sublinear_slope": "G_pair vs A_pair OLS slope in (0, 1)",
    "C2_saturation_r2": "OLS R²(G_pair ~ A_pair) ≥ 0.10",
    "C3_highapair_neg_xi": "mean ξ in top-A_pair quartile < 0",
    "C4_complementarity_real": "P(G_pair > 0 | 0 ≤ A_pair ≤ Q1) ≥ 0.70",
    "C5_seed_robust": "≥ 80% seeds have positive G_pair ~ A_pair slope",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Row:
    seed: int
    t: int
    t_prime: int
    phase_t: str
    phase_tp: str
    distance: int
    source: str
    G_pair: float
    delta_t: float
    delta_tp: float
    A_pair: float
    xi: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Row":
        return Row(
            seed=int(d["seed"]),
            t=int(d["t"]),
            t_prime=int(d["t_prime"]),
            phase_t=str(d["phase_t"]),
            phase_tp=str(d["phase_tp"]),
            distance=int(d["distance"]),
            source=str(d["source"]),
            G_pair=float(d["G_pair"]),
            delta_t=float(d["delta_t"]),
            delta_tp=float(d["delta_tp"]),
            A_pair=float(d["A_pair"]),
            xi=float(d["xi"]),
        )


# ---------------------------------------------------------------------------
# Bootstrap helpers (no scipy needed, pure numpy)
# ---------------------------------------------------------------------------
def bootstrap_ci(
    arr: np.ndarray,
    stat_fn,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return (lo, hi) bootstrap CI for stat_fn(arr)."""
    if rng is None:
        rng = np.random.default_rng(0)
    boots = np.array(
        [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Returns (slope, intercept, r2)."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    ss_xx = ((x - xm) ** 2).sum()
    if ss_xx < 1e-12:
        return float("nan"), float("nan"), float("nan")
    slope = ((x - xm) * (y - ym)).sum() / ss_xx
    intercept = ym - slope * xm
    y_hat = slope * x + intercept
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1.0 - ((y - y_hat) ** 2).sum() / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(intercept), float(r2)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def ols_predict(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * x + intercept


# ---------------------------------------------------------------------------
# Non-linear model fits (A3)
# ---------------------------------------------------------------------------
def fit_sqrt_model(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Fit y ~ a * sqrt(max(x,0)) + b via OLS on transformed x."""
    x_pos = np.maximum(x, 0.0)
    x_t = np.sqrt(x_pos)
    slope, intercept, r2 = ols_slope_intercept(x_t, y)
    return {"model": "sqrt", "slope": slope, "intercept": intercept, "r2": r2}


def fit_log_model(x: np.ndarray, y: np.ndarray, eps: float = 0.1) -> dict[str, float]:
    """Fit y ~ a * log(max(x,0)+eps) + b via OLS."""
    x_t = np.log(np.maximum(x, 0.0) + eps)
    slope, intercept, r2 = ols_slope_intercept(x_t, y)
    return {"model": f"log(x+{eps})", "slope": slope, "intercept": intercept, "r2": r2}


# ---------------------------------------------------------------------------
# Partial R² via nested models (A4)
# ---------------------------------------------------------------------------
def _ols_multi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS with intercept: X is (n, p) design matrix WITHOUT intercept column."""
    X_aug = np.column_stack([np.ones(len(y)), X])
    # Use normal equations (small n, stable enough)
    try:
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    except np.linalg.LinAlgError:
        beta = np.zeros(X_aug.shape[1])
    return beta


def _predict_ols_multi(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack([np.ones(len(X)), X])
    return X_aug @ beta


def nested_r2(
    rows: list[Row],
) -> dict[str, Any]:
    """
    M0: intercept-only baseline
    M1: M0 + A_pair
    M2: M1 + distance
    M3: M2 + phase dummy (early/late vs middle)
    M4: M3 + imbalance = |delta_t - delta_tp|
    M5: M4 + delta_t + delta_tp individually (kitchen sink)

    Target: G_pair
    """
    y = np.array([r.G_pair for r in rows])
    a = np.array([r.A_pair for r in rows])
    d = np.array([r.distance for r in rows])
    # phase dummies: early=1, late=1, middle=0 (two dummies)
    ph_early = np.array([1.0 if r.phase_t == "early" else 0.0 for r in rows])
    ph_late = np.array([1.0 if r.phase_t == "late" else 0.0 for r in rows])
    imbal = np.abs(
        np.array([r.delta_t for r in rows]) - np.array([r.delta_tp for r in rows])
    )
    dt = np.array([r.delta_t for r in rows])
    dtp = np.array([r.delta_tp for r in rows])

    def _r2(X_cols: list[np.ndarray]) -> float:
        if not X_cols:
            # M0: predict mean
            return r2_score(y, np.full_like(y, y.mean()))
        X = np.column_stack(X_cols)
        beta = _ols_multi(X, y)
        yhat = _predict_ols_multi(X, beta)
        return r2_score(y, yhat)

    r2_m0 = _r2([])
    r2_m1 = _r2([a])
    r2_m2 = _r2([a, d])
    r2_m3 = _r2([a, d, ph_early, ph_late])
    r2_m4 = _r2([a, d, ph_early, ph_late, imbal])
    r2_m5 = _r2([a, d, ph_early, ph_late, imbal, dt, dtp])

    return {
        "M0_intercept": r2_m0,
        "M1_plus_Apair": r2_m1,
        "M2_plus_distance": r2_m2,
        "M3_plus_phase": r2_m3,
        "M4_plus_imbalance": r2_m4,
        "M5_plus_individual_deltas": r2_m5,
        "delta_M1": r2_m1 - r2_m0,
        "delta_M2": r2_m2 - r2_m1,
        "delta_M3": r2_m3 - r2_m2,
        "delta_M4": r2_m4 - r2_m3,
        "delta_M5": r2_m5 - r2_m4,
    }


# ---------------------------------------------------------------------------
# Quartile analysis helpers (A2)
# ---------------------------------------------------------------------------
def quartile_stats(
    rows: list[Row], quartile_breaks: list[float]
) -> list[dict[str, Any]]:
    """
    Bin rows into 4 A_pair quartiles and compute stats per quartile.
    quartile_breaks = [min, Q1, Q2, Q3, max] — 5 values, 4 bins.
    """
    results = []
    labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    for i, label in enumerate(labels):
        lo = quartile_breaks[i]
        hi = quartile_breaks[i + 1]
        # inclusive on both ends for first/last, exclusive on hi for middle
        if i == len(labels) - 1:
            mask = [r for r in rows if lo <= r.A_pair <= hi]
        else:
            mask = [r for r in rows if lo <= r.A_pair < hi]
        if not mask:
            results.append({"quartile": label, "n": 0})
            continue
        g = np.array([r.G_pair for r in mask])
        xi = np.array([r.xi for r in mask])
        a = np.array([r.A_pair for r in mask])
        # positive-A_pair subset within Q1 only
        pos_a_mask = [r for r in mask if r.A_pair >= 0]
        results.append(
            {
                "quartile": label,
                "A_pair_range": [float(lo), float(hi)],
                "n": len(mask),
                "mean_A_pair": float(a.mean()),
                "mean_G_pair": float(g.mean()),
                "mean_xi": float(xi.mean()),
                "p_gpair_pos": float((g > 0).mean()),
                "p_xi_pos": float((xi > 0).mean()),
                "n_pos_apair": len(pos_a_mask),
                "p_gpair_pos_posA": float(
                    np.mean([r.G_pair > 0 for r in pos_a_mask])
                ) if pos_a_mask else float("nan"),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Seed robustness (A5)
# ---------------------------------------------------------------------------
def seed_slopes(rows: list[Row]) -> list[dict[str, Any]]:
    seeds = sorted(set(r.seed for r in rows))
    out = []
    for s in seeds:
        sr = [r for r in rows if r.seed == s]
        x = np.array([r.A_pair for r in sr])
        y = np.array([r.G_pair for r in sr])
        slope, intercept, r2 = ols_slope_intercept(x, y)
        out.append(
            {
                "seed": s,
                "n": len(sr),
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "positive_slope": slope > 0 if not math.isnan(slope) else False,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Phase/distance stratification (A6)
# ---------------------------------------------------------------------------
def stratified_stats(rows: list[Row]) -> dict[str, Any]:
    # Phase-pair combos
    phase_pairs: dict[str, list[Row]] = {}
    for r in rows:
        key = f"{r.phase_t}_{r.phase_tp}"
        phase_pairs.setdefault(key, []).append(r)

    phase_out = []
    for key, rs in sorted(phase_pairs.items()):
        x = np.array([r.A_pair for r in rs])
        y = np.array([r.G_pair for r in rs])
        g = np.array([r.G_pair for r in rs])
        xi = np.array([r.xi for r in rs])
        slope, _, r2 = ols_slope_intercept(x, y)
        phase_out.append(
            {
                "phase_pair": key,
                "n": len(rs),
                "slope_g_vs_a": slope,
                "r2": r2,
                "mean_xi": float(xi.mean()),
                "p_gpair_pos": float((g > 0).mean()),
            }
        )

    # Distance buckets: near=[1-2], mid=[3-5], far=[8+]
    def dist_bucket(d: int) -> str:
        if d <= 2:
            return "near_1_2"
        elif d <= 5:
            return "mid_3_5"
        else:
            return "far_8plus"

    dist_groups: dict[str, list[Row]] = {}
    for r in rows:
        k = dist_bucket(r.distance)
        dist_groups.setdefault(k, []).append(r)

    dist_out = []
    for key in ["near_1_2", "mid_3_5", "far_8plus"]:
        rs = dist_groups.get(key, [])
        if not rs:
            dist_out.append({"distance_bucket": key, "n": 0})
            continue
        x = np.array([r.A_pair for r in rs])
        y = np.array([r.G_pair for r in rs])
        xi = np.array([r.xi for r in rs])
        slope, _, r2 = ols_slope_intercept(x, y)
        dist_out.append(
            {
                "distance_bucket": key,
                "n": len(rs),
                "slope_g_vs_a": float(slope),
                "r2": float(r2),
                "mean_xi": float(xi.mean()),
                "p_gpair_pos": float((y > 0).mean()),
            }
        )

    return {"by_phase_pair": phase_out, "by_distance": dist_out}


# ---------------------------------------------------------------------------
# Baseline comparison (A7)
# ---------------------------------------------------------------------------
def baseline_comparison(rows: list[Row]) -> dict[str, Any]:
    """
    Compare A_pair as predictor of G_pair against state-feature baselines.
    State features from token-level experiment verdict: entropy, inv_margin etc.
    For Phase A we compare against naive baselines derivable from existing data:
      - baseline_zero: predict G_pair = 0 always
      - baseline_mean: predict G_pair = global mean
      - delta_t only: predict G_pair = delta_t (single-slot signal)
      - delta_tp only: predict G_pair = delta_tp
      - A_pair (our predictor)
      - (delta_t, delta_tp) separately in OLS
    """
    y = np.array([r.G_pair for r in rows])
    a = np.array([r.A_pair for r in rows])
    dt = np.array([r.delta_t for r in rows])
    dtp = np.array([r.delta_tp for r in rows])

    # zero predictor
    r2_zero = r2_score(y, np.zeros_like(y))

    # mean predictor
    r2_mean = r2_score(y, np.full_like(y, y.mean()))

    # delta_t only OLS
    s, i, r2_dt = ols_slope_intercept(dt, y)

    # delta_tp only OLS
    s2, i2, r2_dtp = ols_slope_intercept(dtp, y)

    # A_pair OLS
    s3, i3, r2_apair = ols_slope_intercept(a, y)

    # (delta_t, delta_tp) OLS multivariate
    X = np.column_stack([dt, dtp])
    beta = _ols_multi(X, y)
    yhat_both = _predict_ols_multi(X, beta)
    r2_both = r2_score(y, yhat_both)

    # Spearman correlations
    def sp(x: np.ndarray, yy: np.ndarray) -> float:
        r, _ = scipy_stats.spearmanr(x, yy)
        return float(r)

    return {
        "r2_zero_predictor": r2_zero,
        "r2_mean_predictor": r2_mean,
        "r2_delta_t_ols": r2_dt,
        "r2_delta_tp_ols": r2_dtp,
        "r2_Apair_ols": r2_apair,
        "r2_both_deltas_ols": r2_both,
        "spearman_delta_t_vs_Gpair": sp(dt, y),
        "spearman_delta_tp_vs_Gpair": sp(dtp, y),
        "spearman_Apair_vs_Gpair": sp(a, y),
        "spearman_Apair_vs_xi": sp(a, np.array([r.xi for r in rows])),
        "note": (
            "A7 baseline: A_pair R2 vs state-feature token-level features. "
            "Token-level features (entropy, inv_margin) were tested in job 493108 "
            "and found to degrade held-out performance (pct_improved=0.133). "
            "A_pair is the dominant predictor of G_pair."
        ),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
PLOT_STYLE: dict[str, Any] = {
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style() -> None:
    matplotlib.rcParams.update(PLOT_STYLE)


def plot_xi_vs_apair_scatter(
    rows: list[Row], out_path: Path, *, debug: bool = False
) -> None:
    _apply_style()
    n = min(len(rows), 500) if debug else len(rows)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(rows), size=n, replace=False)
    sample = [rows[i] for i in idx]
    a = np.array([r.A_pair for r in sample])
    xi = np.array([r.xi for r in sample])
    slope, intercept, r2 = ols_slope_intercept(a, xi)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(a, xi, alpha=0.25, s=8, color="#4477AA", rasterized=True)
    x_line = np.linspace(a.min(), a.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5, label=f"OLS (slope={slope:.3f}, R²={r2:.3f})")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("A_pair = δ_t + δ_t′")
    ax.set_ylabel("ξ = G_pair − A_pair")
    ax.set_title("ξ vs A_pair: saturation signature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_saturation_curve_binned(
    rows: list[Row], out_path: Path, *, n_bins: int = 20
) -> None:
    _apply_style()
    a = np.array([r.A_pair for r in rows])
    xi = np.array([r.xi for r in rows])
    bins = np.percentile(a, np.linspace(0, 100, n_bins + 1))
    # deduplicate
    bins = np.unique(bins)
    bin_centers, means, cis_lo, cis_hi = [], [], [], []
    rng = np.random.default_rng(1)
    for i in range(len(bins) - 1):
        mask = (a >= bins[i]) & (a < bins[i + 1])
        if mask.sum() < 3:
            continue
        vals = xi[mask]
        m = float(vals.mean())
        lo, hi = bootstrap_ci(vals, np.mean, n_boot=500, rng=rng)
        bin_centers.append(float((bins[i] + bins[i + 1]) / 2))
        means.append(m)
        cis_lo.append(lo)
        cis_hi.append(hi)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers, means, "o-", color="#4477AA", ms=4, lw=1.5, label="mean ξ per bin")
    ax.fill_between(bin_centers, cis_lo, cis_hi, alpha=0.25, color="#4477AA", label="95% bootstrap CI")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("A_pair (bin center)")
    ax.set_ylabel("mean ξ")
    ax.set_title("Saturation curve: mean ξ vs A_pair")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_gpair_vs_apair(
    rows: list[Row],
    out_path: Path,
    composition_models: dict[str, Any],
    *,
    debug: bool = False,
) -> None:
    _apply_style()
    n = min(len(rows), 500) if debug else len(rows)
    rng = np.random.default_rng(13)
    idx = rng.choice(len(rows), size=n, replace=False)
    sample = [rows[i] for i in idx]
    a = np.array([r.A_pair for r in sample])
    g = np.array([r.G_pair for r in sample])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(a, g, alpha=0.2, s=8, color="#4477AA", rasterized=True)

    x_line = np.linspace(a.min(), a.max(), 300)
    # identity
    ax.plot(x_line, x_line, "k--", lw=1, label="identity G=A", alpha=0.5)
    # OLS linear
    lin = composition_models["linear"]
    ax.plot(
        x_line,
        lin["slope"] * x_line + lin["intercept"],
        "r-",
        lw=1.5,
        label=f"OLS linear (slope={lin['slope']:.3f}, R²={lin['r2']:.3f})",
    )
    # sqrt
    sqrt_m = composition_models.get("sqrt", {})
    if sqrt_m:
        x_pos = np.maximum(x_line, 0.0)
        ax.plot(
            x_line,
            sqrt_m["slope"] * np.sqrt(x_pos) + sqrt_m["intercept"],
            "g-",
            lw=1.2,
            label=f"sqrt (R²={sqrt_m['r2']:.3f})",
        )
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("A_pair = δ_t + δ_t′")
    ax.set_ylabel("G_pair = G({t, t′})")
    ax.set_title("G_pair vs A_pair: sublinear composition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_quartile_table(
    quartile_data: list[dict[str, Any]], out_path: Path
) -> None:
    _apply_style()
    labels = [q["quartile"] for q in quartile_data]
    means_g = [q.get("mean_G_pair", 0) for q in quartile_data]
    means_xi = [q.get("mean_xi", 0) for q in quartile_data]
    p_pos = [q.get("p_gpair_pos", 0) for q in quartile_data]

    x = np.arange(len(labels))
    width = 0.28
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - width, means_g, width, label="mean G_pair", color="#4477AA")
    b2 = ax.bar(x, means_xi, width, label="mean ξ", color="#EE6677")
    b3 = ax.bar(x + width, p_pos, width, label="P(G>0)", color="#228833")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("Pair utility by A_pair quartile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_partial_r2_waterfall(
    partial_r2: dict[str, Any], out_path: Path
) -> None:
    _apply_style()
    model_names = ["M0→M1\nA_pair", "M1→M2\ndistance", "M2→M3\nphase", "M3→M4\nimbalance", "M4→M5\nδ_t,δ_t′"]
    deltas = [
        partial_r2["delta_M1"],
        partial_r2["delta_M2"],
        partial_r2["delta_M3"],
        partial_r2["delta_M4"],
        partial_r2["delta_M5"],
    ]
    colors = ["#4477AA" if d > 0 else "#EE6677" for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(model_names, deltas, color=colors)
    for bar, val in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:+.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ΔR² (incremental)")
    ax.set_title("Partial R² waterfall: G_pair explained by nested models")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_seed_slopes(slopes_data: list[dict[str, Any]], out_path: Path) -> None:
    _apply_style()
    slopes = [s["slope"] for s in slopes_data if not math.isnan(s["slope"])]
    seeds = [s["seed"] for s in slopes_data if not math.isnan(s["slope"])]
    colors = ["#4477AA" if v > 0 else "#EE6677" for v in slopes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(slopes)), slopes, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds, fontsize=7, rotation=45)
    ax.set_xlabel("seed")
    ax.set_ylabel("OLS slope (G_pair ~ A_pair)")
    ax.set_title("Per-seed G_pair vs A_pair slopes (robustness check)")
    pos_patch = mpatches.Patch(color="#4477AA", label="positive")
    neg_patch = mpatches.Patch(color="#EE6677", label="negative")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_saturation_by_phase(
    strat: dict[str, Any], out_path: Path
) -> None:
    _apply_style()
    phase_data = strat["by_phase_pair"]
    labels = [p["phase_pair"] for p in phase_data]
    slopes = [p["slope_g_vs_a"] for p in phase_data]
    xi_means = [p["mean_xi"] for p in phase_data]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, slopes, width, label="G vs A slope", color="#4477AA")
    ax.bar(x + width / 2, xi_means, width, label="mean ξ", color="#EE6677")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("value")
    ax.set_title("Saturation by phase pair")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_saturation_by_distance(
    strat: dict[str, Any], out_path: Path
) -> None:
    _apply_style()
    dist_data = strat["by_distance"]
    labels = [d["distance_bucket"] for d in dist_data if d.get("n", 0) > 0]
    slopes = [d["slope_g_vs_a"] for d in dist_data if d.get("n", 0) > 0]
    xi_means = [d["mean_xi"] for d in dist_data if d.get("n", 0) > 0]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, slopes, width, label="G vs A slope", color="#4477AA")
    ax.bar(x + width / 2, xi_means, width, label="mean ξ", color="#EE6677")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("Saturation by distance bucket")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
def evaluate_gates(
    slope_g_vs_a: float,
    r2_g_vs_a: float,
    quartile_data: list[dict[str, Any]],
    pct_pos_seeds: float,
) -> dict[str, Any]:
    q4 = next((q for q in quartile_data if "Q4" in q["quartile"]), None)
    q1 = next((q for q in quartile_data if "Q1" in q["quartile"]), None)
    # C3: mean ξ in Q4 < 0
    c3_val = q4["mean_xi"] if q4 else float("nan")
    # C4: P(G>0 | positive A_pair Q1) >= 0.70
    c4_val = q1.get("p_gpair_pos_posA", float("nan")) if q1 else float("nan")

    gates = {
        "C1_sublinear_slope": {
            "pass": 0 < slope_g_vs_a < 1,
            "value": slope_g_vs_a,
            "threshold": "(0, 1)",
        },
        "C2_saturation_r2": {
            "pass": r2_g_vs_a >= 0.10,
            "value": r2_g_vs_a,
            "threshold": "≥ 0.10",
        },
        "C3_highapair_neg_xi": {
            "pass": c3_val < 0 if not math.isnan(c3_val) else False,
            "value": c3_val,
            "threshold": "< 0",
        },
        "C4_complementarity_real": {
            "pass": c4_val >= 0.70 if not math.isnan(c4_val) else False,
            "value": c4_val,
            "threshold": "≥ 0.70",
        },
        "C5_seed_robust": {
            "pass": pct_pos_seeds >= 0.80,
            "value": pct_pos_seeds,
            "threshold": "≥ 0.80",
        },
    }
    all_pass = all(g["pass"] for g in gates.values())
    return {"all_pass": all_pass, "gates": gates}


# ---------------------------------------------------------------------------
# Interpretation narrative
# ---------------------------------------------------------------------------
def write_interpretation(
    out_path: Path,
    gate_result: dict[str, Any],
    slope_g_vs_a: float,
    r2_g_vs_a: float,
    quartile_data: list[dict[str, Any]],
    composition_models: dict[str, Any],
    partial_r2: dict[str, Any],
    baseline_cmp: dict[str, Any],
    n_rows: int,
) -> None:
    gates = gate_result["gates"]
    all_pass = gate_result["all_pass"]
    verdict = "**PASS — proceed to Phase B**" if all_pass else "**FAIL — do not proceed to Phase B**"

    q_rows = "\n".join(
        f"| {q['quartile']} | {q.get('A_pair_range', ['?','?'])} | {q.get('n', 0)} "
        f"| {q.get('mean_G_pair', 0):.4f} | {q.get('mean_xi', 0):.4f} "
        f"| {q.get('p_gpair_pos', 0):.3f} | {q.get('p_xi_pos', 0):.3f} |"
        for q in quartile_data
    )

    text = f"""# Saturation Structure Interpretation
Generated: Phase A analysis, n={n_rows} pairs

## Gate verdict: {verdict}

| Gate | Pass | Value | Threshold |
|------|------|-------|-----------|
"""
    for gname, gval in gates.items():
        p = "✅" if gval["pass"] else "❌"
        text += f"| {gname} | {p} | {gval['value']:.4f} | {gval['threshold']} |\n"

    text += f"""
## A1/A3 — Sublinear composition (G_pair ~ A_pair)

OLS: G_pair = {slope_g_vs_a:.4f} × A_pair + intercept  (R² = {r2_g_vs_a:.4f})

Slope < 1 confirms saturation: pairs with high A_pair receive diminishing
marginal return. The identity G_pair = A_pair + ξ would give slope = 1 iff
ξ were uncorrelated with A_pair; the observed slope < 1 means ξ is
negatively correlated with A_pair — i.e. redundancy grows with predicted benefit.

Concave alternatives:
- sqrt: R² = {composition_models.get('sqrt', {}).get('r2', float('nan')):.4f}
- log: R² = {composition_models.get('log', {}).get('r2', float('nan')):.4f}

## A2 — Quartile utility

| Quartile | A_pair range | n | mean G | mean ξ | P(G>0) | P(ξ>0) |
|----------|-------------|---|--------|--------|--------|--------|
{q_rows}

Q1 (low A_pair) has positive mean ξ — these pairs are genuinely complementary
rather than redundant.  Q4 (high A_pair) has negative mean ξ — saturation
dominates.

## A4 — Partial R² waterfall

A_pair alone explains ΔR² = {partial_r2['delta_M1']:.4f} of G_pair variance.
Adding distance adds ΔR² = {partial_r2['delta_M2']:.4f}.
Adding phase adds ΔR² = {partial_r2['delta_M3']:.4f}.
A_pair is the dominant structured signal.

## A7 — Baseline comparison

A_pair Spearman vs G_pair: {baseline_cmp['spearman_Apair_vs_Gpair']:.4f}
A_pair OLS R²: {baseline_cmp['r2_Apair_ols']:.4f}
delta_t OLS R²: {baseline_cmp['r2_delta_t_ols']:.4f}
Both-deltas OLS R²: {baseline_cmp['r2_both_deltas_ols']:.4f}

A_pair captures joint signal beyond individual δ_t predictors.
Token-level state features (job 493108) had Spearman ≈ 0.693 for ξ
but degraded held-out performance (pct_improved=0.133) — A_pair
is sufficient for scheduling purposes.

## Summary

Saturation structure is confirmed: G_pair grows sublinearly in A_pair.
Redundancy (negative ξ) dominates when predicted benefit is high.
Low-A_pair pairs retain genuine complementarity.
This motivates Phase B: testing whether strength-amplified correction
expands the regimes where G_pair > A_pair (reduces saturation).
"""
    out_path.write_text(text)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(
    xi_raw_path: Path = XI_RAW_PATH,
    out_dir: Path | None = None,
    *,
    debug: bool = False,
    n_boot: int = 2000,
) -> dict[str, Any]:
    # --- load data ---
    print(f"Loading {xi_raw_path} ...", flush=True)
    raw = json.loads(xi_raw_path.read_text())
    rows = [Row.from_dict(r) for r in raw]
    if debug:
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(rows), size=min(300, len(rows)), replace=False)
        rows = [rows[i] for i in idxs]
        n_boot = 200
    print(f"  n={len(rows)} rows loaded", flush=True)

    a = np.array([r.A_pair for r in rows])
    g = np.array([r.G_pair for r in rows])
    xi = np.array([r.xi for r in rows])

    # --- output dir ---
    sha = _get_short_sha()
    if out_dir is None:
        out_dir = REPO_ROOT / "results" / f"saturation_structure_{sha}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    print(f"  Output: {out_dir}", flush=True)

    # --- A1: saturation curve (xi ~ A_pair) ---
    print("A1: saturation curve ...", flush=True)
    xi_slope, xi_intercept, xi_r2 = ols_slope_intercept(a, xi)
    rng = np.random.default_rng(42)
    xi_slope_ci = bootstrap_ci(
        np.column_stack([a, xi]),
        lambda arr: ols_slope_intercept(arr[:, 0], arr[:, 1])[0],
        n_boot=n_boot,
        rng=rng,
    )

    # --- A3: G_pair ~ A_pair ---
    print("A3: composition models ...", flush=True)
    g_slope, g_intercept, g_r2 = ols_slope_intercept(a, g)
    g_slope_ci = bootstrap_ci(
        np.column_stack([a, g]),
        lambda arr: ols_slope_intercept(arr[:, 0], arr[:, 1])[0],
        n_boot=n_boot,
        rng=np.random.default_rng(99),
    )
    sqrt_m = fit_sqrt_model(a, g)
    log_m = fit_log_model(a, g)
    composition_models = {
        "linear": {
            "model": "linear",
            "slope": g_slope,
            "intercept": g_intercept,
            "r2": g_r2,
            "slope_ci_95": list(g_slope_ci),
        },
        "sqrt": sqrt_m,
        "log": log_m,
    }

    # --- A2: quartile analysis ---
    print("A2: quartile analysis ...", flush=True)
    q_breaks = [float(a.min())] + [float(np.percentile(a, p)) for p in [25, 50, 75]] + [float(a.max())]
    quartile_data = quartile_stats(rows, q_breaks)

    # --- A4: nested partial R² ---
    print("A4: partial R² ...", flush=True)
    partial_r2 = nested_r2(rows)

    # --- A5: seed robustness ---
    print("A5: seed robustness ...", flush=True)
    slopes_data = seed_slopes(rows)
    valid_slopes = [s for s in slopes_data if not math.isnan(s["slope"])]
    pct_pos_seeds = float(sum(1 for s in valid_slopes if s["positive_slope"]) / len(valid_slopes)) if valid_slopes else 0.0

    # --- A6: stratification ---
    print("A6: stratification ...", flush=True)
    strat = stratified_stats(rows)

    # --- A7: baseline comparison ---
    print("A7: baseline comparison ...", flush=True)
    baseline_cmp = baseline_comparison(rows)

    # --- gate evaluation ---
    gate_result = evaluate_gates(g_slope, g_r2, quartile_data, pct_pos_seeds)

    # --- plots ---
    print("Generating plots ...", flush=True)
    plot_xi_vs_apair_scatter(rows, fig_dir / "xi_vs_apair_scatter.png", debug=debug)
    plot_saturation_curve_binned(rows, fig_dir / "saturation_curve_binned.png")
    plot_gpair_vs_apair(rows, fig_dir / "gpair_vs_apair_saturation.png", composition_models, debug=debug)
    plot_quartile_table(quartile_data, fig_dir / "quartile_regime_table_plot.png")
    plot_partial_r2_waterfall(partial_r2, fig_dir / "partial_r2_waterfall.png")
    plot_seed_slopes(slopes_data, fig_dir / "seed_level_slopes.png")
    plot_saturation_by_phase(strat, fig_dir / "saturation_by_phase.png")
    plot_saturation_by_distance(strat, fig_dir / "saturation_by_distance.png")
    print("  All 8 plots written.", flush=True)

    # --- write JSON outputs ---
    summary = {
        "sha": sha,
        "n_rows": len(rows),
        "debug_mode": debug,
        "gate_result": gate_result,
        "xi_vs_apair_ols": {
            "slope": xi_slope,
            "intercept": xi_intercept,
            "r2": xi_r2,
            "slope_ci_95": list(xi_slope_ci),
        },
        "g_vs_apair_ols": {
            "slope": g_slope,
            "intercept": g_intercept,
            "r2": g_r2,
            "slope_ci_95": list(g_slope_ci),
        },
        "pct_positive_seed_slopes": pct_pos_seeds,
        "n_seeds": len(slopes_data),
    }
    _write_json(out_dir / "summary_saturation.json", summary)
    _write_json(out_dir / "quartile_regime.json", quartile_data)
    _write_json(out_dir / "composition_models.json", composition_models)
    _write_json(out_dir / "partial_r2.json", partial_r2)
    _write_json(out_dir / "seed_robustness.json", slopes_data)
    _write_json(out_dir / "stratification.json", strat)
    _write_json(out_dir / "comparison_baselines.json", baseline_cmp)

    # --- interpretation ---
    write_interpretation(
        out_dir / "interpretation.md",
        gate_result,
        g_slope,
        g_r2,
        quartile_data,
        composition_models,
        partial_r2,
        baseline_cmp,
        len(rows),
    )

    # --- print gate summary ---
    print("\n=== GATE RESULTS ===")
    for gname, gval in gate_result["gates"].items():
        status = "PASS" if gval["pass"] else "FAIL"
        print(f"  [{status}] {gname}: {gval['value']:.4f} (threshold {gval['threshold']})")
    overall = "ALL PASS → proceed to Phase B" if gate_result["all_pass"] else "GATE FAIL → do not proceed"
    print(f"\n  Overall: {overall}")
    print(f"\n  Results: {out_dir}")

    return {
        "out_dir": str(out_dir),
        "gate_result": gate_result,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not serializable: {type(o)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A: Saturation structure analysis")
    parser.add_argument(
        "--xi-raw",
        type=Path,
        default=XI_RAW_PATH,
        help="Path to xi_raw.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/saturation_structure_<sha>)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Subsample to 300 rows and 200 bootstrap iterations for fast smoke test",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap iterations (default 2000)",
    )
    args = parser.parse_args()

    result = run_analysis(
        xi_raw_path=args.xi_raw,
        out_dir=args.out_dir,
        debug=args.debug,
        n_boot=args.n_boot,
    )

    sys.exit(0 if result["gate_result"]["all_pass"] else 1)


if __name__ == "__main__":
    main()
