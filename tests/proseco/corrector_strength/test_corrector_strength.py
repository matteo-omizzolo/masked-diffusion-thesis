"""
tests/test_corrector_strength.py
==================================
Unit and integration tests for the Phase B corrector-strength preflight scripts.

Covers:
- xi = G_pair - A_pair identity
- A_pair = delta_t + delta_tp identity
- Strength config parsing and corrector_steps mapping
- no_correction gives Delta_t ≈ 0 (in surrogate or patched form)
- select_preflight_pairs is deterministic and size-consistent
- Pair subset spans low/mid/high A_pair strata
- no_correction entries have corrector_n_changed = 0
- Output schema: required keys present in raw_deltas and raw_pairs
- Gate evaluation logic (G1-G5)
- Surrogate end-to-end smoke run (no model required)
- Analysis smoke run (reads surrogate output, writes 7 plots)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.proseco.corrector_strength import run_corrector_strength_preflight as rfp  # noqa: E402
from scripts.proseco.corrector_strength import analyze_corrector_strength_preflight as afp  # noqa: E402

XI_RAW_PATH = REPO_ROOT / "results" / "phase1_interaction_diag_nogit" / "xi_raw.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def xi_raw_data():
    if not XI_RAW_PATH.exists():
        pytest.skip("xi_raw.json not available")
    return json.loads(XI_RAW_PATH.read_text())


@pytest.fixture
def synthetic_raw_deltas():
    """Synthetic raw_deltas as the run script would produce."""
    rows = []
    for sl, k in rfp.STRENGTH_CORRECTOR_STEPS.items():
        for seed in [42, 43]:
            for t in [5, 15, 30, 50]:
                f_base = -4.5
                delta = 0.0 if k is None else (k + 1) * 0.05
                f_branch = f_base + delta
                rows.append({
                    "seed": seed, "t": t, "strength": sl, "corrector_steps": k,
                    "f_base": f_base, "f_branch": f_branch,
                    "delta_t": delta,
                    "corrector_n_changed": 0 if k is None else (k + 1) * 50,
                    "final_n_changed": 0 if k is None else (k + 1) * 80,
                    "wall_s": 0.01,
                })
    return rows


@pytest.fixture
def synthetic_raw_pairs(synthetic_raw_deltas):
    """Synthetic raw_pairs consistent with raw_deltas."""
    delta_map = {(r["seed"], r["t"], r["strength"]): r["delta_t"]
                 for r in synthetic_raw_deltas}
    rows = []
    pairs = [{"t": 5, "t_prime": 30, "distance": 25, "phase_t": "early",
              "phase_tp": "middle", "pair_id": 0, "A_pair": 0.2, "xi": -0.05,
              "G_pair": 0.15},
             {"t": 15, "t_prime": 50, "distance": 35, "phase_t": "early",
              "phase_tp": "late", "pair_id": 1, "A_pair": 0.4, "xi": -0.15,
              "G_pair": 0.25}]
    for sl, k in rfp.STRENGTH_CORRECTOR_STEPS.items():
        for seed in [42, 43]:
            for pair in pairs:
                dt = delta_map.get((seed, pair["t"], sl), 0.0)
                dtp = delta_map.get((seed, pair["t_prime"], sl), 0.0)
                A = dt + dtp
                G = A * 0.7  # sublinear
                xi = G - A
                rows.append({
                    "seed": seed, "pair_id": pair["pair_id"],
                    "t": pair["t"], "t_prime": pair["t_prime"],
                    "phase_t": pair["phase_t"], "phase_tp": pair["phase_tp"],
                    "distance": pair["distance"],
                    "strength": sl, "corrector_steps": k,
                    "f_base": -4.5, "f_pair": -4.5 + G,
                    "G_pair": G, "delta_t": dt, "delta_tp": dtp,
                    "A_pair": A, "xi": xi,
                    "corrector_n_changed_total": 0 if k is None else (k + 1) * 50,
                    "corrector_n_changed_list": [] if k is None else [(k + 1) * 25] * 2,
                    "final_n_changed": 0 if k is None else (k + 1) * 80,
                    "canonical_G_pair": pair["G_pair"],
                    "canonical_A_pair": pair["A_pair"],
                    "canonical_xi": pair["xi"],
                    "wall_s": 0.01,
                })
    return rows


# ---------------------------------------------------------------------------
# Identity checks
# ---------------------------------------------------------------------------

class TestIdentities:
    def test_xi_identity(self, synthetic_raw_pairs):
        for r in synthetic_raw_pairs:
            if math.isnan(r.get("A_pair", float("nan"))):
                continue
            xi_computed = r["G_pair"] - r["A_pair"]
            assert abs(xi_computed - r["xi"]) < 1e-9, (
                f"xi identity failed: G={r['G_pair']}, A={r['A_pair']}, xi={r['xi']}")

    def test_apair_identity(self, synthetic_raw_pairs):
        for r in synthetic_raw_pairs:
            if math.isnan(r.get("delta_t", float("nan"))):
                continue
            a_computed = r["delta_t"] + r["delta_tp"]
            assert abs(a_computed - r["A_pair"]) < 1e-9

    def test_g_pair_equals_f_pair_minus_f_base(self, synthetic_raw_pairs):
        for r in synthetic_raw_pairs:
            g_computed = r["f_pair"] - r["f_base"]
            assert abs(g_computed - r["G_pair"]) < 1e-9


# ---------------------------------------------------------------------------
# Strength config
# ---------------------------------------------------------------------------

class TestStrengthConfig:
    def test_all_levels_in_mapping(self):
        for sl in rfp.STRENGTH_LEVELS:
            assert sl in rfp.STRENGTH_CORRECTOR_STEPS

    def test_no_correction_is_none(self):
        assert rfp.STRENGTH_CORRECTOR_STEPS["no_correction"] is None

    def test_strength_0_is_zero(self):
        assert rfp.STRENGTH_CORRECTOR_STEPS["strength_0"] == 0

    def test_strength_1_is_one(self):
        assert rfp.STRENGTH_CORRECTOR_STEPS["strength_1"] == 1

    def test_strength_2_is_two(self):
        assert rfp.STRENGTH_CORRECTOR_STEPS["strength_2"] == 2

    def test_strengths_ordered(self):
        real_ks = [v for v in rfp.STRENGTH_CORRECTOR_STEPS.values() if v is not None]
        assert real_ks == sorted(real_ks), "corrector_steps should be in ascending order"


# ---------------------------------------------------------------------------
# no_correction properties
# ---------------------------------------------------------------------------

class TestNoCorrection:
    def test_no_correction_delta_zero(self, synthetic_raw_deltas):
        no_corr = [r for r in synthetic_raw_deltas if r["strength"] == "no_correction"]
        assert len(no_corr) > 0
        for r in no_corr:
            assert abs(r["delta_t"]) < 1e-6, f"no_correction delta should be ~0, got {r['delta_t']}"

    def test_no_correction_corrector_n_changed_zero(self, synthetic_raw_deltas):
        no_corr = [r for r in synthetic_raw_deltas if r["strength"] == "no_correction"]
        for r in no_corr:
            assert r["corrector_n_changed"] == 0

    def test_no_correction_xi_zero(self, synthetic_raw_pairs):
        no_corr = [r for r in synthetic_raw_pairs if r["strength"] == "no_correction"]
        assert len(no_corr) > 0
        for r in no_corr:
            assert abs(r["G_pair"]) < 1e-6
            assert abs(r["xi"]) < 1e-6


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

class TestPairSelection:
    def test_deterministic(self, xi_raw_data):
        p1 = rfp.select_preflight_pairs(42, xi_raw_data, n_per_stratum=2)
        p2 = rfp.select_preflight_pairs(42, xi_raw_data, n_per_stratum=2)
        assert [r["pair_id"] for r in p1] == [r["pair_id"] for r in p2]

    def test_size_is_3x_n_per_stratum(self, xi_raw_data):
        for n in [1, 2, 3]:
            pairs = rfp.select_preflight_pairs(42, xi_raw_data, n_per_stratum=n)
            assert len(pairs) == 3 * n

    def test_spans_low_mid_high_apair(self, xi_raw_data):
        pairs = rfp.select_preflight_pairs(42, xi_raw_data, n_per_stratum=3)
        a = [r["A_pair"] for r in pairs]
        # Should span the A_pair distribution
        assert max(a) - min(a) > 0.2, "Selected pairs should span A_pair range"

    def test_pair_ids_unique(self, xi_raw_data):
        pairs = rfp.select_preflight_pairs(42, xi_raw_data, n_per_stratum=2)
        ids = [r["pair_id"] for r in pairs]
        assert len(ids) == len(set(ids))

    def test_needed_timesteps(self, xi_raw_data):
        pairs = rfp.select_preflight_pairs(42, xi_raw_data, n_per_stratum=1)
        ts = rfp.get_needed_timesteps(pairs)
        expected = sorted(set([p["t"] for p in pairs] + [p["t_prime"] for p in pairs]))
        assert ts == expected


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    REQUIRED_DELTA_KEYS = {"seed", "t", "strength", "corrector_steps", "f_base",
                            "f_branch", "delta_t", "corrector_n_changed",
                            "final_n_changed", "wall_s"}
    REQUIRED_PAIR_KEYS = {"seed", "pair_id", "t", "t_prime", "strength",
                           "corrector_steps", "f_base", "f_pair",
                           "G_pair", "delta_t", "delta_tp", "A_pair", "xi",
                           "corrector_n_changed_total", "final_n_changed", "wall_s"}

    def test_delta_schema(self, synthetic_raw_deltas):
        for r in synthetic_raw_deltas:
            missing = self.REQUIRED_DELTA_KEYS - set(r.keys())
            assert not missing, f"Missing delta keys: {missing}"

    def test_pair_schema(self, synthetic_raw_pairs):
        for r in synthetic_raw_pairs:
            missing = self.REQUIRED_PAIR_KEYS - set(r.keys())
            assert not missing, f"Missing pair keys: {missing}"

    def test_no_leakage_fields(self, synthetic_raw_deltas):
        leakage_keys = {"tcr", "logits", "hidden_states", "post_correction_state"}
        for r in synthetic_raw_deltas:
            found = leakage_keys & set(r.keys())
            assert not found, f"Leakage fields found: {found}"


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

class TestGateEvaluation:
    def test_g1_pass_when_canonical_matches(self, synthetic_raw_deltas):
        # Build canonical that matches strength_1
        canonical = {}
        for r in synthetic_raw_deltas:
            if r["strength"] == "strength_1":
                s, t = r["seed"], r["t"]
                if s not in canonical:
                    canonical[s] = {}
                canonical[s][t] = r["delta_t"]  # exact match → error = 0

        summary = rfp._compute_summary(
            synthetic_raw_deltas, [], canonical, rfp.STRENGTH_LEVELS
        )
        g1 = summary["gates"]["G1_canonical_reproduction"]
        assert g1["pass"] is True
        assert g1["max_abs_error"] < rfp.CANONICAL_TOL

    def test_g1_fail_when_canonical_mismatches(self, synthetic_raw_deltas):
        canonical = {}
        for r in synthetic_raw_deltas:
            if r["strength"] == "strength_1":
                s, t = r["seed"], r["t"]
                if s not in canonical:
                    canonical[s] = {}
                canonical[s][t] = r["delta_t"] + 0.1  # big mismatch

        summary = rfp._compute_summary(
            synthetic_raw_deltas, [], canonical, rfp.STRENGTH_LEVELS
        )
        g1 = summary["gates"]["G1_canonical_reproduction"]
        assert g1["pass"] is False

    def test_g4_pass_no_correction(self, synthetic_raw_deltas):
        summary = rfp._compute_summary(
            synthetic_raw_deltas, [], {}, rfp.STRENGTH_LEVELS
        )
        g4 = summary["gates"]["G4_no_correction_noop"]
        assert g4["pass"] is True

    def test_g5_crn_pass_when_f_base_consistent(self, synthetic_raw_deltas):
        summary = rfp._compute_summary(
            synthetic_raw_deltas, [], {}, rfp.STRENGTH_LEVELS
        )
        # In synthetic data, f_base is the same for all rows of a seed
        g5 = summary["gates"]["G5_crn_consistent"]
        assert g5["pass"] is True

    def test_g5_fail_when_f_base_inconsistent(self, synthetic_raw_deltas):
        # Corrupt f_base for one row
        corrupted = [dict(r) for r in synthetic_raw_deltas]
        corrupted[0]["f_base"] = corrupted[0]["f_base"] + 1.0
        summary = rfp._compute_summary(corrupted, [], {}, rfp.STRENGTH_LEVELS)
        g5 = summary["gates"]["G5_crn_consistent"]
        assert g5["pass"] is False

    def test_all_gates_present(self, synthetic_raw_deltas):
        summary = rfp._compute_summary(
            synthetic_raw_deltas, [], {}, rfp.STRENGTH_LEVELS
        )
        for g in ["G1_canonical_reproduction", "G2_standard_nontrivial",
                  "G3_strength_variants_differ", "G4_no_correction_noop",
                  "G5_crn_consistent"]:
            assert g in summary["gates"]


# ---------------------------------------------------------------------------
# Surrogate end-to-end smoke run
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not XI_RAW_PATH.exists(),
    reason="xi_raw.json required for surrogate run pair selection"
)
def test_surrogate_e2e_run(tmp_path):
    """Full surrogate run produces correct outputs and schema."""
    out_dir = tmp_path / "corrector_strength_preflight_test"
    result = rfp._run_surrogate_preflight(out_dir, seeds=[42, 43], debug=True)

    assert out_dir.exists()
    for fname in ["raw_deltas.json", "raw_pairs.json", "summary.json", "manifest.json"]:
        assert (out_dir / fname).exists(), f"Missing: {fname}"

    # Schema checks
    rd = json.loads((out_dir / "raw_deltas.json").read_text())
    rp = json.loads((out_dir / "raw_pairs.json").read_text())
    assert len(rd) > 0
    assert len(rp) > 0

    # Identity check on surrogate data
    for r in rp:
        if math.isnan(r.get("A_pair", float("nan"))):
            continue
        xi_computed = r["G_pair"] - r["A_pair"]
        assert abs(xi_computed - r["xi"]) < 1e-9

    # Gate structure
    s = json.loads((out_dir / "summary.json").read_text())
    assert "gates" in s
    assert "gate_pass" in s
    for g in ["G1_canonical_reproduction", "G2_standard_nontrivial",
              "G3_strength_variants_differ", "G4_no_correction_noop",
              "G5_crn_consistent"]:
        assert g in s["gates"]

    # Result structure
    assert "out_dir" in result
    assert "gate_pass" in result


@pytest.mark.skipif(
    not XI_RAW_PATH.exists(),
    reason="xi_raw.json required"
)
def test_analysis_smoke_run(tmp_path):
    """Surrogate run + analysis produces all 7 plots."""
    out_dir = tmp_path / "cs_preflight"
    rfp._run_surrogate_preflight(out_dir, seeds=[42, 43], debug=True)
    afp.run_analysis(out_dir)

    plots_dir = out_dir / "plots"
    for plot_name in [
        "delta_distribution_by_strength.png",
        "apair_distribution_by_strength.png",
        "xi_distribution_by_strength.png",
        "changed_tokens_by_strength.png",
        "gpair_vs_apair_by_strength_preflight.png",
        "xi_vs_apair_by_strength_preflight.png",
        "pass_increment_effects.png",
    ]:
        assert (plots_dir / plot_name).exists(), f"Missing plot: {plot_name}"

    assert (out_dir / "interpretation.md").exists()
