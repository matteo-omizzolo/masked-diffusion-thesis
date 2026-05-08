"""Theorem A constant estimators for Phase 2b artefacts.

The estimators here turn Phase 2b raw MC rows into the four Theorem-A
constants that parameterise the proxy-regret bound

    G(S_B*) − G(Ŝ_B)  ≤  2 B ε  +  2 η_B

and its refinements A′ (variance form of η_B) and A″ (rank-based ε).

References
----------
- `docs/thesis/theory/THEORY_STATUS.md` — open-loop theorem status.
- `research/candidate_theorems.md` — Theorem A, Refinement A′, A″,
  Proposition B (low-gain-region exclusion), Proposition C (pairwise γ).
- `docs/thesis/theory/MDM_THEORY_LANDSCAPE_POSITIONING.md` §3.1–3.3 —
  how these constants enter ch6.

Constants
---------
σ_ξ        : per-seed std of the additivity residual ξ = G − A,
             where A = Σ_{t ∈ S} Δ̂_t is the additive proxy sum.
             Refinement A′:  η_B ≤ σ_ξ · √B / √2.

ρ          : Spearman rank correlation between A and G, per (seed, B).
             Refinement A″:  ε_R = (1 − |ρ|) · σ_Δ.

σ_Δ        : pooled std of G per B; scale of ε_R.

γ          : pairwise-interaction upper bound per B, estimated as
             q_α( 2 |residual| / (B(B−1)) ) for α ∈ (0, 1).
             Proposition C:  η_B ≤ γ · B(B−1)/2.

low_gain_share : per-seed ratio of max G over top-k MC schedules (ranked
             by the additive proxy A) to the max G over all MC schedules.
             Proposition B anchor: if this ratio is close to 1, the
             ranker class captures most of the oracle gain.

All estimators are pure functions over JSON-compatible dicts.

Input row convention
--------------------
    {"seed": int,
     "B": int,
     "mc_idx": int,
     "G": float,
     "A": float,
     "residual": float,
     "schedule_steps": [int, ...]}

Output convention
-----------------
Every estimator returns a `{"per_B": {str(B): {...}}}` bundle, mirroring
`combinatorial_diagnostics` for easy concatenation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .stats import _spearman, mean_se, spearman_bootstrap_ci

# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def _group_by_seed_B(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(int(r["seed"]), int(r["B"]))].append(r)
    return grouped


def _group_by_B(
    rows: Sequence[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["B"])].append(r)
    return grouped


def _budgets(rows: Sequence[dict[str, Any]]) -> list[int]:
    return sorted({int(r["B"]) for r in rows})


# ---------------------------------------------------------------------------
# σ_ξ — additivity-residual std per B
# ---------------------------------------------------------------------------


def residual_sigma_xi(mc_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Estimate σ_ξ = std of additivity residual per budget B.

    For each budget B, compute the per-seed residual std, then report the
    mean and SE across seeds (seed-level variability is the outer CI).
    Also reports the pooled σ_ξ across all points.

    Refinement A′:  η_B ≤ σ_ξ · √B / √2.
    """
    by_B_seed: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in mc_rows:
        if "residual" not in r:
            continue
        by_B_seed[int(r["B"])][int(r["seed"])].append(float(r["residual"]))

    per_B: dict[str, dict[str, Any]] = {}
    for B, per_seed in sorted(by_B_seed.items()):
        per_seed_std = np.asarray(
            [
                float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0
                for v in per_seed.values()
            ],
            dtype=float,
        )
        pooled = np.asarray(
            [x for vals in per_seed.values() for x in vals], dtype=float
        )
        pooled_std = float(pooled.std(ddof=1)) if pooled.size >= 2 else 0.0
        seed_stats = mean_se(per_seed_std.tolist())
        per_B[str(B)] = {
            "B": int(B),
            "n_seeds": int(len(per_seed)),
            "n_points": int(pooled.size),
            "sigma_xi_pooled": pooled_std,
            "sigma_xi_per_seed_mean": seed_stats["mean"],
            "sigma_xi_per_seed_se": seed_stats["se"],
            "sigma_xi_per_seed_std": seed_stats["std"],
        }
    return {"per_B": per_B}


# ---------------------------------------------------------------------------
# ρ — Spearman rank correlation between proxy A and true G
# ---------------------------------------------------------------------------


def proxy_rank_rho(
    mc_rows: Sequence[dict[str, Any]],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Spearman ρ between the additive proxy A and the true gain G.

    For each budget B, compute ρ per seed (across MC samples), and report
    the mean ± seed-level SE as well as a pooled (cross-seed) ρ with
    paired bootstrap CI.

    Refinement A″:  ε_R = (1 − |ρ|) · σ_Δ.
    """
    by_B_seed = _group_by_seed_B(mc_rows)
    by_B: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in mc_rows:
        if "A" not in r or "G" not in r:
            continue
        by_B[int(r["B"])].append((float(r["A"]), float(r["G"])))

    per_B: dict[str, dict[str, Any]] = {}
    budgets = sorted(by_B.keys())
    for B in budgets:
        # Per-seed ρ
        per_seed_rho: list[float] = []
        for (s, b), rows in by_B_seed.items():
            if b != B:
                continue
            A_arr = np.asarray([float(r["A"]) for r in rows], dtype=float)
            G_arr = np.asarray([float(r["G"]) for r in rows], dtype=float)
            if A_arr.size < 3:
                continue
            per_seed_rho.append(_spearman(A_arr, G_arr))

        seed_stats = mean_se(per_seed_rho) if per_seed_rho else {"mean": 0.0, "se": 0.0, "n": 0, "std": 0.0}

        # Pooled ρ with bootstrap
        A_pool = np.asarray([a for a, _ in by_B[B]], dtype=float)
        G_pool = np.asarray([g for _, g in by_B[B]], dtype=float)
        pooled = spearman_bootstrap_ci(
            A_pool.tolist(),
            G_pool.tolist(),
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed + B,
        )

        per_B[str(B)] = {
            "B": int(B),
            "n_seeds": int(len(per_seed_rho)),
            "n_points": int(A_pool.size),
            "rho_per_seed_mean": seed_stats["mean"],
            "rho_per_seed_se": seed_stats["se"],
            "rho_per_seed_std": seed_stats["std"],
            "rho_pooled": float(pooled["rho"]),
            "rho_pooled_ci_lo": float(pooled["ci_lo"]),
            "rho_pooled_ci_hi": float(pooled["ci_hi"]),
        }
    return {"per_B": per_B}


# ---------------------------------------------------------------------------
# σ_Δ — pooled std of G per B (scale for ε_R)
# ---------------------------------------------------------------------------


def gain_scale_sigma_delta(mc_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pooled std of G per budget B."""
    by_B = _group_by_B(mc_rows)
    per_B: dict[str, dict[str, Any]] = {}
    for B, rows in sorted(by_B.items()):
        vals = np.asarray([float(r["G"]) for r in rows], dtype=float)
        sigma = float(vals.std(ddof=1)) if vals.size >= 2 else 0.0
        per_B[str(B)] = {
            "B": int(B),
            "n_points": int(vals.size),
            "sigma_delta_pooled": sigma,
            "G_mean": float(vals.mean()) if vals.size else 0.0,
        }
    return {"per_B": per_B}


# ---------------------------------------------------------------------------
# γ — pairwise-interaction upper bound per B (Proposition C)
# ---------------------------------------------------------------------------


def interaction_gamma_upper(
    mc_rows: Sequence[dict[str, Any]],
    quantile: float = 0.95,
) -> dict[str, Any]:
    """Upper-bound estimator for γ in Proposition C.

    Rationale: if G − A = ξ and Proposition C's pairwise structure gives
    |ξ| ≤ γ · B(B−1)/2, then a per-schedule implied γ is

        γ_sched = 2 · |residual| / (B · (B−1)).

    The reported γ_upper(B) is the `quantile`-th sample quantile of
    γ_sched over all mc rows at that B. B=1 is skipped (no pairwise
    interaction).

    η_B plug-in via Proposition C:  η_B ≤ γ · B(B−1)/2.
    """
    by_B = _group_by_B(mc_rows)
    per_B: dict[str, dict[str, Any]] = {}
    for B, rows in sorted(by_B.items()):
        if B < 2:
            per_B[str(B)] = {
                "B": int(B),
                "n_points": int(len(rows)),
                "gamma_upper": 0.0,
                "quantile": float(quantile),
                "note": "B<2: no pairwise interaction",
            }
            continue
        residuals = np.asarray(
            [abs(float(r["residual"])) for r in rows if "residual" in r],
            dtype=float,
        )
        if residuals.size == 0:
            per_B[str(B)] = {
                "B": int(B),
                "n_points": 0,
                "gamma_upper": 0.0,
                "quantile": float(quantile),
                "note": "no residual data",
            }
            continue
        scale = 2.0 / (B * (B - 1))
        gamma_sched = residuals * scale
        q = float(np.quantile(gamma_sched, quantile))
        per_B[str(B)] = {
            "B": int(B),
            "n_points": int(residuals.size),
            "gamma_upper": q,
            "quantile": float(quantile),
            "gamma_max": float(gamma_sched.max()),
            "gamma_median": float(np.median(gamma_sched)),
        }
    return {"per_B": per_B}


# ---------------------------------------------------------------------------
# Prop B — low-gain-share anchor
# ---------------------------------------------------------------------------


def low_gain_share(
    mc_rows: Sequence[dict[str, Any]],
    top_k: int = 10,
) -> dict[str, Any]:
    """Per-seed ratio of max-G-in-top-k-by-A over max-G-across-all-MC.

    Proposition B anchor: a ratio close to 1 means the ranker class (which
    scores schedules by the additive proxy A) captures most of the
    oracle-achievable gain; a ratio near 0 means ranker failure on that
    seed.

    Returns per B: mean ± SE of that ratio across seeds.
    """
    grouped = _group_by_seed_B(mc_rows)
    by_B_ratios: dict[int, list[float]] = defaultdict(list)
    for (s, B), rows in grouped.items():
        if len(rows) < 2:
            continue
        if not all("A" in r and "G" in r for r in rows):
            continue
        A_arr = np.asarray([float(r["A"]) for r in rows], dtype=float)
        G_arr = np.asarray([float(r["G"]) for r in rows], dtype=float)
        k = min(top_k, A_arr.size)
        # Top-k by A; np.argpartition + slice for efficiency.
        topk_idx = np.argpartition(-A_arr, kth=k - 1)[:k]
        G_topk = G_arr[topk_idx]
        G_max_all = float(G_arr.max())
        G_max_topk = float(G_topk.max())
        if G_max_all == 0.0:
            continue
        ratio = G_max_topk / G_max_all
        by_B_ratios[B].append(float(ratio))

    per_B: dict[str, dict[str, Any]] = {}
    for B, vals in sorted(by_B_ratios.items()):
        stats = mean_se(vals)
        per_B[str(B)] = {
            "B": int(B),
            "top_k": int(top_k),
            "n_seeds": int(len(vals)),
            "share_mean": stats["mean"],
            "share_se": stats["se"],
            "share_std": stats["std"],
            "share_min": float(min(vals)),
            "share_max": float(max(vals)),
        }
    return {"per_B": per_B}


# ---------------------------------------------------------------------------
# Theorem-A plug-in bound
# ---------------------------------------------------------------------------


def theorem_a_plugin_bound(
    sigma_xi: dict[str, Any],
    rho: dict[str, Any],
    sigma_delta: dict[str, Any],
    gamma: dict[str, Any],
) -> dict[str, Any]:
    """Compute the plug-in proxy-regret bound per B.

    Returns three variants:

    - `bound_A_prime`:  2 B ε_R + 2 · (σ_ξ √B / √2)   (A′ + A″ plug-in)
    - `bound_prop_C`:   2 B ε_R + 2 · (γ · B(B−1)/2)  (A″ + Prop C plug-in)
    - `bound_min`:      min of the two η_B forms, conservative plug-in.
    """
    per_B: dict[str, dict[str, Any]] = {}
    # Intersect budgets present in all four dicts.
    Bs = (
        set(sigma_xi["per_B"].keys())
        & set(rho["per_B"].keys())
        & set(sigma_delta["per_B"].keys())
        & set(gamma["per_B"].keys())
    )
    for Bk in sorted(Bs, key=lambda s: int(s)):
        B = int(Bk)
        sx = float(sigma_xi["per_B"][Bk]["sigma_xi_pooled"])
        rho_val = float(rho["per_B"][Bk]["rho_pooled"])
        sd = float(sigma_delta["per_B"][Bk]["sigma_delta_pooled"])
        g = float(gamma["per_B"][Bk]["gamma_upper"])
        eps_R = (1.0 - abs(rho_val)) * sd
        eta_B_variance = sx * math.sqrt(B) / math.sqrt(2.0)
        eta_B_pairwise = g * B * (B - 1) / 2.0 if B >= 2 else 0.0
        bound_A_prime = 2.0 * B * eps_R + 2.0 * eta_B_variance
        bound_prop_C = 2.0 * B * eps_R + 2.0 * eta_B_pairwise
        bound_min = 2.0 * B * eps_R + 2.0 * min(eta_B_variance, eta_B_pairwise or eta_B_variance)
        per_B[Bk] = {
            "B": int(B),
            "epsilon_R": float(eps_R),
            "eta_B_variance_form": float(eta_B_variance),
            "eta_B_pairwise_form": float(eta_B_pairwise),
            "bound_A_prime_plus_A_double_prime": float(bound_A_prime),
            "bound_prop_C_plus_A_double_prime": float(bound_prop_C),
            "bound_tighter_eta": float(bound_min),
            "sigma_xi_used": sx,
            "rho_used": rho_val,
            "sigma_delta_used": sd,
            "gamma_used": g,
        }
    return {"per_B": per_B}


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_theorem_a_constants(
    mc_rows: Sequence[dict[str, Any]],
    *,
    top_k: int = 10,
    gamma_quantile: float = 0.95,
    rho_n_resamples: int = 1000,
    rho_alpha: float = 0.05,
    rho_seed: int = 0,
) -> dict[str, Any]:
    """Build the full Theorem-A constants bundle from Phase 2b raw rows."""
    sigma_xi = residual_sigma_xi(mc_rows)
    rho = proxy_rank_rho(
        mc_rows,
        n_resamples=rho_n_resamples,
        alpha=rho_alpha,
        seed=rho_seed,
    )
    sigma_delta = gain_scale_sigma_delta(mc_rows)
    gamma = interaction_gamma_upper(mc_rows, quantile=gamma_quantile)
    low_gain = low_gain_share(mc_rows, top_k=top_k)
    bound = theorem_a_plugin_bound(
        sigma_xi=sigma_xi, rho=rho, sigma_delta=sigma_delta, gamma=gamma
    )
    return {
        "sigma_xi": sigma_xi,
        "rho": rho,
        "sigma_delta": sigma_delta,
        "gamma": gamma,
        "low_gain_share": low_gain,
        "plugin_bound": bound,
    }
