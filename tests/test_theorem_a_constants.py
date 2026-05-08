"""Tests for the Theorem A constant estimators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mdm_playground.analysis.theorem_a_constants import (
    build_theorem_a_constants,
    gain_scale_sigma_delta,
    interaction_gamma_upper,
    low_gain_share,
    proxy_rank_rho,
    residual_sigma_xi,
    theorem_a_plugin_bound,
)


def _synth_rows(
    *,
    n_seeds: int,
    n_mc: int,
    B: int,
    T: int,
    rho_target: float,
    sigma_xi: float,
    seed_base: int = 0,
) -> list[dict]:
    """Generate synthetic MC rows with controlled rank-fidelity ρ and residual σ_ξ.

    Construction
    ------------
    For each seed, sample `n_mc` schedules uniformly without replacement over
    [0, T). Build a per-site gain vector δ ~ 𝒩(0.1, 0.05). The additive proxy
    A = Σ δ_t. The true G is a convex combination of A and pure noise:

        G = α·A_z  +  √(1-α²) · noise    where A_z is z-standardized A,

    then shifted and re-scaled so std(G) is comparable to std(A). α controls
    Spearman ρ (population). Residuals are independent Gaussian with std σ_xi
    added to G − A to stress the additivity slack.
    """
    rng = np.random.default_rng(seed_base)
    rows: list[dict] = []
    for s in range(n_seeds):
        delta = rng.normal(loc=0.1, scale=0.05, size=T)
        for m in range(n_mc):
            sched = rng.choice(T, size=B, replace=False)
            A = float(delta[sched].sum())
            # Generate correlated G with residual noise.
            noise = rng.normal(loc=0.0, scale=1.0)
            G_core = rho_target * A + math.sqrt(max(0.0, 1 - rho_target**2)) * noise * max(
                np.std(delta) * math.sqrt(B), 1e-6
            )
            residual = rng.normal(loc=0.0, scale=sigma_xi)
            G = G_core + residual
            rows.append(
                {
                    "seed": int(s + 1),
                    "B": int(B),
                    "mc_idx": int(m),
                    "A": A,
                    "G": float(G),
                    "residual": float(G - A),
                    "schedule_steps": sorted(int(x) for x in sched),
                }
            )
    return rows


def test_residual_sigma_xi_per_B_shape() -> None:
    rows = _synth_rows(n_seeds=4, n_mc=50, B=3, T=20, rho_target=0.5, sigma_xi=0.2)
    out = residual_sigma_xi(rows)["per_B"]
    assert "3" in out
    block = out["3"]
    assert block["B"] == 3
    assert block["n_seeds"] == 4
    assert block["n_points"] == 200
    assert block["sigma_xi_pooled"] > 0.0
    assert block["sigma_xi_per_seed_mean"] > 0.0


def test_sigma_xi_tracks_injected_noise() -> None:
    """σ_xi estimator should rank two regimes with different injected scale."""
    low = _synth_rows(n_seeds=4, n_mc=50, B=3, T=20, rho_target=0.5, sigma_xi=0.05, seed_base=1)
    hi = _synth_rows(n_seeds=4, n_mc=50, B=3, T=20, rho_target=0.5, sigma_xi=0.5, seed_base=2)
    low_sigma = residual_sigma_xi(low)["per_B"]["3"]["sigma_xi_pooled"]
    hi_sigma = residual_sigma_xi(hi)["per_B"]["3"]["sigma_xi_pooled"]
    assert hi_sigma > low_sigma * 2.0


def test_proxy_rank_rho_recovers_positive_rho() -> None:
    rows = _synth_rows(n_seeds=5, n_mc=80, B=4, T=30, rho_target=0.9, sigma_xi=0.01, seed_base=42)
    out = proxy_rank_rho(rows, n_resamples=200, seed=7)["per_B"]["4"]
    assert out["rho_pooled"] > 0.5
    assert out["rho_pooled_ci_lo"] > 0.0
    assert out["rho_per_seed_mean"] > 0.3


def test_proxy_rank_rho_low_correlation_regime() -> None:
    rows = _synth_rows(n_seeds=5, n_mc=80, B=4, T=30, rho_target=0.0, sigma_xi=0.01, seed_base=9)
    out = proxy_rank_rho(rows, n_resamples=200, seed=3)["per_B"]["4"]
    # Pooled rho should be near zero; allow generous slack for small n.
    assert abs(out["rho_pooled"]) < 0.35


def test_gain_scale_sigma_delta_is_positive() -> None:
    rows = _synth_rows(n_seeds=3, n_mc=40, B=2, T=15, rho_target=0.6, sigma_xi=0.1)
    out = gain_scale_sigma_delta(rows)["per_B"]["2"]
    assert out["sigma_delta_pooled"] > 0.0
    assert out["n_points"] == 120


def test_interaction_gamma_upper_shape_and_monotonicity_in_B() -> None:
    rows_B2 = _synth_rows(n_seeds=3, n_mc=50, B=2, T=20, rho_target=0.5, sigma_xi=0.2, seed_base=11)
    rows_B3 = _synth_rows(n_seeds=3, n_mc=50, B=3, T=20, rho_target=0.5, sigma_xi=0.2, seed_base=12)
    out2 = interaction_gamma_upper(rows_B2, quantile=0.9)["per_B"]["2"]
    out3 = interaction_gamma_upper(rows_B3, quantile=0.9)["per_B"]["3"]
    assert out2["gamma_upper"] > 0.0
    assert out3["gamma_upper"] > 0.0
    # With the same underlying σ_xi, γ_upper ∝ 1/(B(B-1)); B=2 produces a strictly
    # larger γ_upper than B=3 because the same |residual| is spread over more
    # pairwise terms at B=3.
    assert out2["gamma_upper"] > out3["gamma_upper"]


def test_interaction_gamma_upper_B1_note() -> None:
    rows = [
        {"seed": 1, "B": 1, "mc_idx": 0, "G": 0.1, "A": 0.1, "residual": 0.0, "schedule_steps": [0]},
    ]
    out = interaction_gamma_upper(rows)["per_B"]["1"]
    assert out["gamma_upper"] == 0.0
    assert "B<2" in out.get("note", "")


def test_low_gain_share_bounds_and_monotonicity() -> None:
    """Strong proxy ⇒ high share; weak proxy ⇒ lower share."""
    rows_strong = _synth_rows(
        n_seeds=6, n_mc=40, B=3, T=20, rho_target=0.95, sigma_xi=0.02, seed_base=100
    )
    rows_weak = _synth_rows(
        n_seeds=6, n_mc=40, B=3, T=20, rho_target=0.0, sigma_xi=0.5, seed_base=101
    )
    s_strong = low_gain_share(rows_strong, top_k=5)["per_B"]["3"]
    s_weak = low_gain_share(rows_weak, top_k=5)["per_B"]["3"]
    assert 0.0 <= s_strong["share_mean"] <= 1.0 + 1e-9
    assert 0.0 <= s_weak["share_mean"] <= 1.0 + 1e-9
    # Note: share is max_G-over-top-k / max_G-over-all, so strong proxy
    # should lift the numerator meaningfully.
    assert s_strong["share_mean"] > s_weak["share_mean"]


def test_theorem_a_plugin_bound_is_non_negative() -> None:
    rows = _synth_rows(n_seeds=4, n_mc=60, B=3, T=20, rho_target=0.7, sigma_xi=0.1, seed_base=200)
    sigma_xi = residual_sigma_xi(rows)
    rho = proxy_rank_rho(rows, n_resamples=200, seed=1)
    sigma_delta = gain_scale_sigma_delta(rows)
    gamma = interaction_gamma_upper(rows, quantile=0.95)
    out = theorem_a_plugin_bound(
        sigma_xi=sigma_xi, rho=rho, sigma_delta=sigma_delta, gamma=gamma
    )["per_B"]["3"]
    assert out["bound_A_prime_plus_A_double_prime"] >= 0.0
    assert out["bound_prop_C_plus_A_double_prime"] >= 0.0
    assert out["bound_tighter_eta"] >= 0.0
    assert out["epsilon_R"] >= 0.0


def test_theorem_a_plugin_bound_scales_with_B() -> None:
    """The bound must grow with B for fixed ε, σ_xi (2Bε dominates for B ≥ 2)."""
    rows = []
    for B in (2, 4, 8):
        rows.extend(
            _synth_rows(
                n_seeds=3, n_mc=40, B=B, T=20, rho_target=0.6, sigma_xi=0.1, seed_base=300 + B
            )
        )
    out = build_theorem_a_constants(
        rows, top_k=5, gamma_quantile=0.9, rho_n_resamples=200
    )
    b2 = out["plugin_bound"]["per_B"]["2"]["bound_A_prime_plus_A_double_prime"]
    b4 = out["plugin_bound"]["per_B"]["4"]["bound_A_prime_plus_A_double_prime"]
    b8 = out["plugin_bound"]["per_B"]["8"]["bound_A_prime_plus_A_double_prime"]
    assert b2 < b4 < b8


def test_build_theorem_a_constants_smoke() -> None:
    rows = _synth_rows(n_seeds=3, n_mc=30, B=2, T=12, rho_target=0.5, sigma_xi=0.1, seed_base=500)
    out = build_theorem_a_constants(
        rows, top_k=5, gamma_quantile=0.9, rho_n_resamples=100
    )
    assert "sigma_xi" in out
    assert "rho" in out
    assert "sigma_delta" in out
    assert "gamma" in out
    assert "low_gain_share" in out
    assert "plugin_bound" in out
    assert "2" in out["plugin_bound"]["per_B"]


def test_missing_fields_are_tolerated() -> None:
    """Rows without A/residual should not crash the estimators."""
    rows = [
        {"seed": 1, "B": 2, "mc_idx": 0, "G": 0.5, "schedule_steps": [0, 1]},
        {"seed": 1, "B": 2, "mc_idx": 1, "G": 0.7, "schedule_steps": [0, 2]},
    ]
    # σ_ξ should gracefully emit an empty per_B (no residual field present).
    out_sigma = residual_sigma_xi(rows)
    assert out_sigma == {"per_B": {}}
    # σ_Δ should still compute from G.
    out_sigma_delta = gain_scale_sigma_delta(rows)["per_B"]["2"]
    assert out_sigma_delta["sigma_delta_pooled"] > 0.0
    # γ should emit a 'no residual data' note for B>=2 with no residuals.
    out_gamma = interaction_gamma_upper(rows)["per_B"]["2"]
    assert out_gamma["gamma_upper"] == 0.0
    assert "no residual" in out_gamma.get("note", "")
