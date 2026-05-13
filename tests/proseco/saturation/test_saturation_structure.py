"""
tests/test_saturation_structure.py
====================================
Unit tests for scripts/proseco/saturation/analyze_saturation_structure.py.

Tests cover:
- Data loading and Row parsing
- OLS helpers (slope, R², prediction)
- Bootstrap CI (width, coverage)
- Quartile stats correctness
- Nested partial-R² monotonicity
- Gate evaluation logic
- Seed robustness fraction
- Stratification bucketing
- End-to-end debug smoke run (generates outputs, checks gate JSON schema)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.proseco.saturation import analyze_saturation_structure as sat  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_rows() -> list[sat.Row]:
    """30 synthetic rows with A_pair in [0, 1] and G_pair = 0.4*A_pair + noise."""
    rng = np.random.default_rng(7)
    n = 60
    a = rng.uniform(-0.2, 1.0, n)
    g = 0.4 * a + rng.normal(0, 0.05, n)
    xi = g - a
    rows = [
        sat.Row(
            seed=rng.integers(42, 72),
            t=int(rng.integers(1, 32)),
            t_prime=int(rng.integers(1, 32)),
            phase_t=rng.choice(["early", "middle", "late"]),
            phase_tp=rng.choice(["early", "middle", "late"]),
            distance=int(rng.integers(1, 12)),
            source="test",
            G_pair=float(g[i]),
            delta_t=float(a[i] / 2 + rng.normal(0, 0.02)),
            delta_tp=float(a[i] / 2 + rng.normal(0, 0.02)),
            A_pair=float(a[i]),
            xi=float(xi[i]),
        )
        for i in range(n)
    ]
    return rows


@pytest.fixture
def perfect_rows() -> list[sat.Row]:
    """Rows where G_pair = 0.5*A_pair exactly (slope=0.5, no noise)."""
    n = 50
    a = np.linspace(-0.5, 1.0, n)
    g = 0.5 * a
    return [
        sat.Row(
            seed=42,
            t=1, t_prime=2, phase_t="early", phase_tp="late",
            distance=3, source="test",
            G_pair=float(g[i]),
            delta_t=float(a[i] / 2),
            delta_tp=float(a[i] / 2),
            A_pair=float(a[i]),
            xi=float(g[i] - a[i]),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------

class TestOLSHelpers:
    def test_slope_perfect_linear(self, perfect_rows):
        a = np.array([r.A_pair for r in perfect_rows])
        g = np.array([r.G_pair for r in perfect_rows])
        slope, intercept, r2 = sat.ols_slope_intercept(a, g)
        assert abs(slope - 0.5) < 1e-6
        assert abs(intercept) < 1e-6
        assert abs(r2 - 1.0) < 1e-6

    def test_slope_constant_x_returns_nan(self):
        x = np.ones(10)
        y = np.arange(10, dtype=float)
        slope, intercept, r2 = sat.ols_slope_intercept(x, y)
        assert math.isnan(slope)

    def test_r2_score_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert abs(sat.r2_score(y, y) - 1.0) < 1e-9

    def test_r2_score_mean_predictor(self):
        y = np.array([1.0, 2.0, 3.0])
        # predicting mean → R²=0
        assert abs(sat.r2_score(y, np.full_like(y, y.mean()))) < 1e-9

    def test_r2_score_worse_than_mean_negative(self):
        y = np.array([1.0, 2.0, 3.0])
        bad = np.array([3.0, 2.0, 1.0])
        assert sat.r2_score(y, bad) < 0


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_ci_contains_true_mean(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=5.0, scale=1.0, size=200)
        lo, hi = sat.bootstrap_ci(data, np.mean, n_boot=1000, rng=rng)
        assert lo < 5.0 < hi, f"CI [{lo:.3f}, {hi:.3f}] does not contain 5.0"

    def test_ci_lo_lt_hi(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, 100)
        lo, hi = sat.bootstrap_ci(data, np.mean, n_boot=500, rng=rng)
        assert lo < hi

    def test_ci_width_shrinks_with_larger_n(self):
        rng = np.random.default_rng(2)
        small = rng.normal(0, 1, 30)
        large = rng.normal(0, 1, 300)
        lo_s, hi_s = sat.bootstrap_ci(small, np.mean, n_boot=500, rng=rng)
        lo_l, hi_l = sat.bootstrap_ci(large, np.mean, n_boot=500, rng=rng)
        assert (hi_s - lo_s) > (hi_l - lo_l)


# ---------------------------------------------------------------------------
# Quartile stats
# ---------------------------------------------------------------------------

class TestQuartileStats:
    def test_four_quartiles_returned(self, simple_rows):
        a = np.array([r.A_pair for r in simple_rows])
        q_breaks = [float(a.min())] + [float(np.percentile(a, p)) for p in [25, 50, 75]] + [float(a.max())]
        result = sat.quartile_stats(simple_rows, q_breaks)
        assert len(result) == 4

    def test_total_rows_conserved(self, simple_rows):
        a = np.array([r.A_pair for r in simple_rows])
        q_breaks = [float(a.min())] + [float(np.percentile(a, p)) for p in [25, 50, 75]] + [float(a.max())]
        result = sat.quartile_stats(simple_rows, q_breaks)
        total = sum(q["n"] for q in result)
        # All rows covered (last bin inclusive on both ends)
        assert total == len(simple_rows)

    def test_p_gpair_pos_in_unit_interval(self, simple_rows):
        a = np.array([r.A_pair for r in simple_rows])
        q_breaks = [float(a.min())] + [float(np.percentile(a, p)) for p in [25, 50, 75]] + [float(a.max())]
        result = sat.quartile_stats(simple_rows, q_breaks)
        for q in result:
            if q["n"] > 0:
                assert 0.0 <= q["p_gpair_pos"] <= 1.0

    def test_q4_lower_xi_than_q1(self, simple_rows):
        """Q4 (high A_pair) should have lower mean ξ than Q1 for our synthetic data."""
        a = np.array([r.A_pair for r in simple_rows])
        q_breaks = [float(a.min())] + [float(np.percentile(a, p)) for p in [25, 50, 75]] + [float(a.max())]
        result = sat.quartile_stats(simple_rows, q_breaks)
        q1_xi = result[0]["mean_xi"]
        q4_xi = result[3]["mean_xi"]
        # With G = 0.4*A, xi = G - A = -0.6*A → high A → lower xi
        assert q4_xi < q1_xi


# ---------------------------------------------------------------------------
# Nested R²
# ---------------------------------------------------------------------------

class TestNestedR2:
    def test_monotonicity(self, simple_rows):
        """Each nested model R² should be ≥ previous model R²."""
        result = sat.nested_r2(simple_rows)
        assert result["M1_plus_Apair"] >= result["M0_intercept"] - 1e-9
        assert result["M2_plus_distance"] >= result["M1_plus_Apair"] - 1e-9
        assert result["M3_plus_phase"] >= result["M2_plus_distance"] - 1e-9
        assert result["M4_plus_imbalance"] >= result["M3_plus_phase"] - 1e-9
        assert result["M5_plus_individual_deltas"] >= result["M4_plus_imbalance"] - 1e-9

    def test_m0_r2_is_zero_or_negative(self, simple_rows):
        result = sat.nested_r2(simple_rows)
        # M0 predicts mean → R² = 0 by definition
        assert abs(result["M0_intercept"]) < 1e-9

    def test_all_keys_present(self, simple_rows):
        result = sat.nested_r2(simple_rows)
        for key in ["M0_intercept", "M1_plus_Apair", "M2_plus_distance",
                    "M3_plus_phase", "M4_plus_imbalance", "M5_plus_individual_deltas",
                    "delta_M1", "delta_M2", "delta_M3", "delta_M4", "delta_M5"]:
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Seed robustness
# ---------------------------------------------------------------------------

class TestSeedSlopes:
    def test_returns_one_per_seed(self, simple_rows):
        result = sat.seed_slopes(simple_rows)
        seeds_in = sorted(set(r.seed for r in simple_rows))
        seeds_out = sorted(s["seed"] for s in result)
        assert seeds_in == seeds_out

    def test_positive_slope_flag(self, perfect_rows):
        """perfect_rows all have slope 0.5 → all should be positive."""
        # give each row a unique seed
        for i, r in enumerate(perfect_rows):
            object.__setattr__(r, "seed", 42 + i)
        result = sat.seed_slopes(perfect_rows)
        assert all(s["positive_slope"] for s in result if not math.isnan(s["slope"]))


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

class TestGateEvaluation:
    def test_all_pass_scenario(self):
        slope_g = 0.4
        r2_g = 0.25
        quartile_data = [
            {"quartile": "Q1 (low)", "mean_xi": 0.05, "p_gpair_pos": 0.5, "p_gpair_pos_posA": 0.82, "n": 50},
            {"quartile": "Q2", "mean_xi": 0.02, "p_gpair_pos": 0.55, "p_gpair_pos_posA": float("nan"), "n": 50},
            {"quartile": "Q3", "mean_xi": -0.05, "p_gpair_pos": 0.48, "p_gpair_pos_posA": float("nan"), "n": 50},
            {"quartile": "Q4 (high)", "mean_xi": -0.15, "p_gpair_pos": 0.38, "p_gpair_pos_posA": float("nan"), "n": 50},
        ]
        pct_pos = 0.90
        result = sat.evaluate_gates(slope_g, r2_g, quartile_data, pct_pos)
        assert result["all_pass"] is True
        for g in result["gates"].values():
            assert g["pass"] is True

    def test_slope_over_1_fails_c1(self):
        result = sat.evaluate_gates(1.5, 0.3, [
            {"quartile": "Q1 (low)", "mean_xi": 0.05, "p_gpair_pos_posA": 0.82, "n": 10},
            {"quartile": "Q2", "n": 10},
            {"quartile": "Q3", "n": 10},
            {"quartile": "Q4 (high)", "mean_xi": -0.1, "n": 10},
        ], 0.9)
        assert result["gates"]["C1_sublinear_slope"]["pass"] is False

    def test_low_r2_fails_c2(self):
        result = sat.evaluate_gates(0.4, 0.05, [
            {"quartile": "Q1 (low)", "mean_xi": 0.05, "p_gpair_pos_posA": 0.82, "n": 10},
            {"quartile": "Q2", "n": 10},
            {"quartile": "Q3", "n": 10},
            {"quartile": "Q4 (high)", "mean_xi": -0.1, "n": 10},
        ], 0.9)
        assert result["gates"]["C2_saturation_r2"]["pass"] is False
        assert result["all_pass"] is False

    def test_positive_q4_xi_fails_c3(self):
        result = sat.evaluate_gates(0.4, 0.3, [
            {"quartile": "Q1 (low)", "mean_xi": 0.05, "p_gpair_pos_posA": 0.82, "n": 10},
            {"quartile": "Q2", "n": 10},
            {"quartile": "Q3", "n": 10},
            {"quartile": "Q4 (high)", "mean_xi": 0.05, "n": 10},  # positive → fail C3
        ], 0.9)
        assert result["gates"]["C3_highapair_neg_xi"]["pass"] is False

    def test_low_complementarity_fails_c4(self):
        result = sat.evaluate_gates(0.4, 0.3, [
            {"quartile": "Q1 (low)", "mean_xi": 0.05, "p_gpair_pos_posA": 0.60, "n": 10},  # < 0.70
            {"quartile": "Q2", "n": 10},
            {"quartile": "Q3", "n": 10},
            {"quartile": "Q4 (high)", "mean_xi": -0.1, "n": 10},
        ], 0.9)
        assert result["gates"]["C4_complementarity_real"]["pass"] is False

    def test_low_seed_pct_fails_c5(self):
        result = sat.evaluate_gates(0.4, 0.3, [
            {"quartile": "Q1 (low)", "mean_xi": 0.05, "p_gpair_pos_posA": 0.82, "n": 10},
            {"quartile": "Q2", "n": 10},
            {"quartile": "Q3", "n": 10},
            {"quartile": "Q4 (high)", "mean_xi": -0.1, "n": 10},
        ], 0.70)  # < 0.80
        assert result["gates"]["C5_seed_robust"]["pass"] is False


# ---------------------------------------------------------------------------
# Stratification bucketing
# ---------------------------------------------------------------------------

class TestStratification:
    def test_phase_pairs_cover_all(self, simple_rows):
        result = sat.stratified_stats(simple_rows)
        total_phase = sum(p["n"] for p in result["by_phase_pair"])
        assert total_phase == len(simple_rows)

    def test_distance_buckets_cover_all(self, simple_rows):
        result = sat.stratified_stats(simple_rows)
        total_dist = sum(d["n"] for d in result["by_distance"] if d.get("n", 0) > 0)
        assert total_dist == len(simple_rows)

    def test_three_distance_buckets(self, simple_rows):
        result = sat.stratified_stats(simple_rows)
        # Rows have distances 1-11, so all three buckets should appear
        nonempty = [d for d in result["by_distance"] if d.get("n", 0) > 0]
        assert len(nonempty) >= 2  # at minimum 2 of 3 buckets populated


# ---------------------------------------------------------------------------
# End-to-end smoke run (debug mode)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO_ROOT / "results" / "phase1_interaction_diag_nogit" / "xi_raw.json").exists(),
    reason="xi_raw.json not present; skip e2e test",
)
def test_e2e_debug_run(tmp_path):
    """Run the full analysis in debug mode and check output schema."""
    result = sat.run_analysis(
        xi_raw_path=REPO_ROOT / "results" / "phase1_interaction_diag_nogit" / "xi_raw.json",
        out_dir=tmp_path / "saturation_debug",
        debug=True,
        n_boot=100,
    )
    out_dir = Path(result["out_dir"])

    # Check all required JSON files exist
    for fname in [
        "summary_saturation.json",
        "quartile_regime.json",
        "composition_models.json",
        "partial_r2.json",
        "seed_robustness.json",
        "stratification.json",
        "comparison_baselines.json",
    ]:
        assert (out_dir / fname).exists(), f"Missing: {fname}"

    # Check interpretation.md
    assert (out_dir / "interpretation.md").exists()

    # Check all 8 figures
    fig_dir = out_dir / "figures"
    for fig in [
        "xi_vs_apair_scatter.png",
        "saturation_curve_binned.png",
        "gpair_vs_apair_saturation.png",
        "quartile_regime_table_plot.png",
        "partial_r2_waterfall.png",
        "seed_level_slopes.png",
        "saturation_by_phase.png",
        "saturation_by_distance.png",
    ]:
        assert (fig_dir / fig).exists(), f"Missing figure: {fig}"

    # Check gate result schema
    summary = json.loads((out_dir / "summary_saturation.json").read_text())
    assert "gate_result" in summary
    gr = summary["gate_result"]
    assert "all_pass" in gr
    assert "gates" in gr
    for gate_key in ["C1_sublinear_slope", "C2_saturation_r2", "C3_highapair_neg_xi",
                     "C4_complementarity_real", "C5_seed_robust"]:
        assert gate_key in gr["gates"], f"Missing gate: {gate_key}"
        assert "pass" in gr["gates"][gate_key]
        assert "value" in gr["gates"][gate_key]

    # Check partial_r2 monotonicity on real data
    pr2 = json.loads((out_dir / "partial_r2.json").read_text())
    assert pr2["M1_plus_Apair"] >= pr2["M0_intercept"] - 1e-9

    # Check quartile_regime has 4 entries
    qr = json.loads((out_dir / "quartile_regime.json").read_text())
    assert len(qr) == 4

    # Check composition_models has linear, sqrt, log
    cm = json.loads((out_dir / "composition_models.json").read_text())
    for key in ["linear", "sqrt", "log"]:
        assert key in cm, f"Missing model: {key}"

    # Result dict has expected keys
    assert "gate_result" in result
    assert "summary" in result
    assert "out_dir" in result
