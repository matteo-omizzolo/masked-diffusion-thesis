"""Statistical utilities for Phase 2 analysis.

Implements the numeric primitives used by Stage 2a / 2b / 2c analysis scripts:

- `paired_bootstrap_ci` — paired bootstrap CI for policy-difference means.
- `spearman_bootstrap_ci` — Spearman rank correlation with bootstrap CI.
- `cohen_d` — paired effect size.
- `paired_t_test` — textbook paired-t statistic and p-value (two-sided).
- `mean_se` — mean ± standard error tuple.
- `classify_tier` — assign T1/T2/T3/T4 confidence tiers per
  ANALYSIS_SPEC §6.

See ANALYSIS_SPEC.md for the formal contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core summary statistics
# ---------------------------------------------------------------------------


def mean_se(x: Sequence[float]) -> Dict[str, float]:
    """Return mean, SE of mean, n for a 1D sample.

    SE is the unbiased estimate: sample_std(ddof=1) / sqrt(n).
    """
    arr = np.asarray(list(x), dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": 0.0, "se": 0.0, "n": 0, "std": 0.0}
    mean = float(arr.mean())
    if n < 2:
        return {"mean": mean, "se": 0.0, "n": n, "std": 0.0}
    std = float(arr.std(ddof=1))
    se = std / math.sqrt(n)
    return {"mean": mean, "se": se, "n": n, "std": std}


def cohen_d(diff: Sequence[float]) -> float:
    """Paired Cohen's d = mean(diff) / std(diff, ddof=1).

    For paired differences. Zero std ⇒ returns 0.0 (degenerate case).
    """
    arr = np.asarray(list(diff), dtype=float)
    if arr.size < 2:
        return 0.0
    std = float(arr.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(arr.mean() / std)


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-sided paired t-test (no scipy dependency).

    Returns {t_stat, p_value, df, mean_diff, se_diff, n}.
    """
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired arrays must share shape: {a_arr.shape} vs {b_arr.shape}")
    diff = a_arr - b_arr
    n = int(diff.size)
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "df": 0, "mean_diff": float(diff.mean()) if n else 0.0, "se_diff": 0.0, "n": n}
    mean_d = float(diff.mean())
    std_d = float(diff.std(ddof=1))
    se_d = std_d / math.sqrt(n) if std_d > 0 else 0.0
    if se_d == 0.0:
        return {"t_stat": math.inf if mean_d != 0 else 0.0, "p_value": 0.0 if mean_d != 0 else 1.0, "df": n - 1, "mean_diff": mean_d, "se_diff": 0.0, "n": n}
    t = mean_d / se_d
    # Two-sided p-value via Student's t CDF approximation.
    # We use the survival-function identity with scipy if available, else a
    # normal approximation for large n (df >= 30).
    try:
        from scipy import stats as _stats  # type: ignore
        p = 2.0 * (1.0 - _stats.t.cdf(abs(t), df=n - 1))
    except Exception:
        # Fallback: normal approximation (safe for n ≥ 30).
        z = abs(t)
        p = 2.0 * (1.0 - _phi(z))
    return {
        "t_stat": float(t),
        "p_value": float(p),
        "df": int(n - 1),
        "mean_diff": mean_d,
        "se_diff": float(se_d),
        "n": n,
    }


def _phi(z: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap for the mean difference (a_i - b_i).

    Returns:
      {mean_diff, se_diff, ci_lo, ci_hi, n, n_resamples, alpha}.

    Resamples paired indices with replacement `n_resamples` times and reports
    the (alpha/2, 1-alpha/2) percentile interval of the bootstrap mean-diff
    distribution.
    """
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired arrays must share shape: {a_arr.shape} vs {b_arr.shape}")
    n = int(a_arr.size)
    if n < 2:
        mean_d = float((a_arr - b_arr).mean()) if n else 0.0
        return {"mean_diff": mean_d, "se_diff": 0.0, "ci_lo": mean_d, "ci_hi": mean_d, "n": n, "n_resamples": 0, "alpha": alpha}
    rng = np.random.default_rng(seed)
    diffs = a_arr - b_arr
    means = np.empty(n_resamples, dtype=float)
    idx = rng.integers(0, n, size=(n_resamples, n))
    # Vectorised resample.
    for i in range(n_resamples):
        means[i] = diffs[idx[i]].mean()
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "mean_diff": float(diffs.mean()),
        "se_diff": float(diffs.std(ddof=1) / math.sqrt(n)),
        "ci_lo": lo,
        "ci_hi": hi,
        "n": n,
        "n_resamples": int(n_resamples),
        "alpha": float(alpha),
        "bootstrap_std": float(means.std(ddof=1)),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient (ties ⇒ average rank)."""
    if x.size < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = math.sqrt(float((rx_c * rx_c).sum() * (ry_c * ry_c).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx_c * ry_c).sum() / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Return ranks with ties broken by average (like scipy.stats.rankdata)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(x, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # Handle ties: average the ranks of equal values.
    x_sorted = x[order]
    i = 0
    n = x.size
    while i < n:
        j = i
        while j + 1 < n and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j) + 1.0  # 1-based ranks
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Spearman ρ with paired-bootstrap (x, y) CI."""
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"paired arrays must share shape: {x_arr.shape} vs {y_arr.shape}")
    n = int(x_arr.size)
    if n < 3:
        return {"rho": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": n, "n_resamples": 0}
    point = _spearman(x_arr, y_arr)
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_resamples, dtype=float)
    idx = rng.integers(0, n, size=(n_resamples, n))
    for i in range(n_resamples):
        rhos[i] = _spearman(x_arr[idx[i]], y_arr[idx[i]])
    rhos = np.nan_to_num(rhos, nan=0.0)
    return {
        "rho": float(point),
        "ci_lo": float(np.quantile(rhos, alpha / 2.0)),
        "ci_hi": float(np.quantile(rhos, 1.0 - alpha / 2.0)),
        "n": n,
        "n_resamples": int(n_resamples),
        "alpha": float(alpha),
        "bootstrap_std": float(rhos.std(ddof=1)),
    }


# ---------------------------------------------------------------------------
# Evidence tier classification (ANALYSIS_SPEC §6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierClassification:
    tier: str  # "T1" | "T2" | "T3" | "T4"
    reason: str


def classify_tier(
    mean: float,
    se: float,
    n: int,
    ci_lo: Optional[float] = None,
    ci_hi: Optional[float] = None,
    p_value: Optional[float] = None,
    alpha_bonf: float = 0.005,
    alpha_nominal: float = 0.05,
    is_paired_diff: bool = True,
    is_single_seed: bool = False,
) -> Dict[str, str]:
    """Classify a measurement into evidentiary tiers T1/T2/T3/T4.

    The rules follow ANALYSIS_SPEC.md §6:

    - T1 confirmed: n≥30, paired diff, CI excludes zero under Bonferroni α.
    - T2 suggestive: n≥30, CI excludes zero at nominal α but not Bonferroni.
    - T3 exploratory: rank/Spearman-only, no paired test available, or n<30.
    - T4 inadmissible: single-seed or ε_rms-style uninformative metric.

    Parameters
    ----------
    mean, se, n : summary statistics
    ci_lo, ci_hi : optional bootstrap CI bounds at alpha_nominal
    p_value : optional p-value from a paired test (two-sided)
    alpha_bonf : Bonferroni-corrected α (default 0.005 per spec)
    alpha_nominal : nominal α (default 0.05)
    is_paired_diff : True if the measurement is a paired difference
    is_single_seed : True if it comes from a single seed per evaluation
    """
    if is_single_seed:
        return {"tier": "T4", "reason": "single-seed evaluation, no SE possible"}

    if not is_paired_diff:
        return {"tier": "T3", "reason": "not a paired-difference measurement (rank/Spearman only)"}

    if n < 30:
        return {"tier": "T3", "reason": f"n={n} < 30; exploratory only"}

    # Prefer CI-based decision if supplied, else fall back to p-value.
    ci_excludes_zero = None
    if ci_lo is not None and ci_hi is not None:
        ci_excludes_zero = (ci_lo > 0.0) or (ci_hi < 0.0)

    if p_value is not None:
        if p_value <= alpha_bonf and (ci_excludes_zero is not False):
            return {"tier": "T1", "reason": f"p={p_value:.4g} ≤ α_bonf={alpha_bonf}"}
        if p_value <= alpha_nominal and (ci_excludes_zero is not False):
            return {"tier": "T2", "reason": f"p={p_value:.4g} ≤ α_nominal={alpha_nominal}; not Bonferroni-significant"}
        return {"tier": "T3", "reason": f"p={p_value:.4g} > α_nominal={alpha_nominal}"}

    if ci_excludes_zero is True:
        width = max(abs(ci_hi - mean), abs(mean - ci_lo))
        if width < 0.5 * abs(mean):
            return {"tier": "T1", "reason": "CI excludes 0 and is tight vs mean"}
        return {"tier": "T2", "reason": "CI excludes 0 but wide"}
    return {"tier": "T3", "reason": "CI does not exclude 0 at nominal α"}


# ---------------------------------------------------------------------------
# Friendly wrappers that produce the spec-canonical JSON dict
# ---------------------------------------------------------------------------


def paired_summary(
    policy_G: Sequence[float],
    uniform_G: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    alpha_bonf: float = 0.005,
    seed: int = 0,
) -> Dict[str, object]:
    """Produce the `paired_vs_uniform` block used in Stage 2b outputs."""
    a = np.asarray(list(policy_G), dtype=float)
    b = np.asarray(list(uniform_G), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must share shape")
    diff = (a - b).tolist()
    ms = mean_se(diff)
    boot = paired_bootstrap_ci(a, b, n_resamples=n_resamples, alpha=alpha, seed=seed)
    t = paired_t_test(a, b)
    d = cohen_d(diff)
    tier = classify_tier(
        mean=ms["mean"],
        se=ms["se"],
        n=ms["n"],
        ci_lo=boot["ci_lo"],
        ci_hi=boot["ci_hi"],
        p_value=t["p_value"],
        alpha_bonf=alpha_bonf,
        alpha_nominal=alpha,
        is_paired_diff=True,
        is_single_seed=False,
    )
    return {
        "diff_per_seed": diff,
        "mean": ms["mean"],
        "se": ms["se"],
        "n": ms["n"],
        "paired_t_stat": t["t_stat"],
        "p_value": t["p_value"],
        "bootstrap_95_ci": [boot["ci_lo"], boot["ci_hi"]],
        "cohens_d": d,
        "tier": tier["tier"],
        "tier_reason": tier["reason"],
    }
