"""Tests for the enriched-feature state-predictability audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.proseco.state_predictability.analyze_state_predictability_enriched import (
    DERIVATIVE_BASE,
    FEATURE_AUDIT,
    LEAKAGE_FIELDS,
    SEED_CONSTANT_FIELDS,
    STATE_FEATURES,
    TrajectoryEnriched,
    _rolling_mean,
    _shift,
    bootstrap_seed_diff,
    build_marginal_feature_blocks,
    compute_derived,
    enriched_feature_names,
    fit_ridge,
    load_trajectories,
    main,
    make_seed_folds,
    run_marginal_audit,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_shift_past_pads_with_first_value() -> None:
    arr = np.asarray([10.0, 20.0, 30.0, 40.0])
    shifted = _shift(arr, 2)
    np.testing.assert_array_equal(shifted, np.asarray([10.0, 10.0, 10.0, 20.0]))


def test_shift_future_pads_with_last_value() -> None:
    arr = np.asarray([10.0, 20.0, 30.0, 40.0])
    shifted = _shift(arr, -2)
    np.testing.assert_array_equal(shifted, np.asarray([30.0, 40.0, 40.0, 40.0]))


def test_rolling_mean_window_4() -> None:
    arr = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    got = _rolling_mean(arr, 4)
    # Padding repeats arr[0]=1, so:
    # t=0: mean(1,1,1,1)=1; t=1: mean(1,1,1,2)=1.25;
    # t=2: mean(1,1,2,3)=1.75; t=3: mean(1,2,3,4)=2.5; t=4: mean(2,3,4,5)=3.5
    expected = np.asarray([1.0, 1.25, 1.75, 2.5, 3.5])
    np.testing.assert_allclose(got, expected)


# ---------------------------------------------------------------------------
# Feature audit & leakage hygiene
# ---------------------------------------------------------------------------

def test_leakage_fields_not_in_state_features() -> None:
    for f in LEAKAGE_FIELDS:
        assert f not in STATE_FEATURES


def test_seed_constant_fields_not_in_state_features() -> None:
    for f in SEED_CONSTANT_FIELDS:
        assert f not in STATE_FEATURES


def test_feature_audit_classifies_all_protocol_a_fields() -> None:
    required = {
        "t", "delta", "entropy", "inverse_margin", "quality_mass_proxy",
        "unmasked_fraction", "n_revisable", "n_masked",
        "tcr", "n_changed", "f_base", "f_branch",
    }
    # All non-time fields should appear in the audit (t is not classified — geometric).
    audited = set(FEATURE_AUDIT.keys())
    missing = (required - {"t"}) - audited
    assert not missing, f"missing audit entries: {missing}"


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def _synthetic_trajectory(seed: int = 0, T: int = 8) -> TrajectoryEnriched:
    raw = {
        "delta": np.linspace(0.0, 0.2, T),
        "entropy": np.linspace(7.0, 1.0, T),
        "inverse_margin": np.linspace(1.0, 0.2, T),
        "quality_mass_proxy": np.linspace(0.9, 0.3, T),
        "unmasked_fraction": np.linspace(0.0, 1.0, T),
        "n_revisable": np.linspace(20, 500, T),
        "n_masked": np.linspace(1000, 1, T),
    }
    return TrajectoryEnriched(seed=seed, T=T, raw=raw)


def test_compute_derived_populates_expected_families() -> None:
    traj = _synthetic_trajectory()
    compute_derived(traj)
    names = enriched_feature_names()
    for n in names["S1"]:
        assert n in traj.derived, n
    for n in names["S2"]:
        assert n in traj.derived, n
    for n in names["S3"]:
        assert n in traj.derived, n


def test_derivative_correctness_on_synthetic_trajectory() -> None:
    traj = _synthetic_trajectory()
    compute_derived(traj)
    entropy = traj.raw["entropy"]
    diff1_back = traj.derived["S2__entropy_diff1_back"]
    # diff1_back[t] = entropy[t] - entropy[t-1] with t=0 -> 0 (since shift returns arr[0]).
    expected = entropy - np.concatenate([[entropy[0]], entropy[:-1]])
    np.testing.assert_allclose(diff1_back, expected)
    diff1_fwd = traj.derived["S3__entropy_diff1_fwd"]
    expected_fwd = np.concatenate([entropy[1:], [entropy[-1]]]) - entropy
    np.testing.assert_allclose(diff1_fwd, expected_fwd)


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def test_load_trajectories_reads_per_step_features(tmp_path: Path) -> None:
    payload = {
        "seed": 42, "T": 2,
        "per_t": [
            {"t": 0, "delta": 0.0,
             "entropy": 7.5, "inverse_margin": 1.0,
             "quality_mass_proxy": 0.9, "unmasked_fraction": 0.01,
             "n_revisable": 17, "n_masked": 1000},
            {"t": 1, "delta": 0.1,
             "entropy": 3.0, "inverse_margin": 0.5,
             "quality_mass_proxy": 0.4, "unmasked_fraction": 0.03,
             "n_revisable": 30, "n_masked": 980},
        ],
    }
    (tmp_path / "trajectory_0.json").write_text(json.dumps(payload))
    trajs = load_trajectories(tmp_path)
    assert len(trajs) == 1
    assert trajs[0].T == 2
    assert trajs[0].raw["entropy"][1] == pytest.approx(3.0)


def test_load_trajectories_does_not_keep_leakage_in_enriched(tmp_path: Path) -> None:
    payload = {
        "seed": 42, "T": 2,
        "per_t": [
            {"t": 0, "delta": 0.0, "tcr": 0.5, "n_changed": 7,
             "f_base": -4.0, "f_branch": -5.0,
             "entropy": 7.5, "inverse_margin": 1.0,
             "quality_mass_proxy": 0.9, "unmasked_fraction": 0.01,
             "n_revisable": 17, "n_masked": 1000},
            {"t": 1, "delta": 0.1, "tcr": 0.5, "n_changed": 7,
             "f_base": -4.0, "f_branch": -5.0,
             "entropy": 3.0, "inverse_margin": 0.5,
             "quality_mass_proxy": 0.4, "unmasked_fraction": 0.03,
             "n_revisable": 30, "n_masked": 980},
        ],
    }
    (tmp_path / "trajectory_0.json").write_text(json.dumps(payload))
    trajs = load_trajectories(tmp_path)
    compute_derived(trajs[0])
    enriched = trajs[0].enriched(0)
    for bad in LEAKAGE_FIELDS:
        assert bad not in enriched, f"leakage field {bad} leaked into enriched()"
    for bad in SEED_CONSTANT_FIELDS:
        assert bad not in enriched, f"seed-constant field {bad} leaked into enriched()"


# ---------------------------------------------------------------------------
# Splits and standardization
# ---------------------------------------------------------------------------

def test_make_seed_folds_no_leakage() -> None:
    folds = make_seed_folds(list(range(30)), n_folds=5)
    flat = [s for f in folds for s in f]
    assert sorted(flat) == list(range(30))
    assert len(flat) == len(set(flat))


def test_fit_ridge_uses_train_only_standardization() -> None:
    X = np.random.default_rng(0).normal(size=(50, 4))
    y = X @ np.asarray([1.0, -2.0, 0.5, 0.0]) + 0.1
    fit = fit_ridge(X[:30], y[:30], columns=["a", "b", "c", "d"], alpha=1.0)
    np.testing.assert_allclose(fit.mean, X[:30].mean(axis=0))


# ---------------------------------------------------------------------------
# Marginal audit assembles correct shape & detects synthetic state signal
# ---------------------------------------------------------------------------

def test_build_marginal_feature_blocks_shapes() -> None:
    trajs = [_synthetic_trajectory(seed=s, T=8) for s in range(6)]
    for t in trajs:
        compute_derived(t)
    X, y, seeds, _, groups = build_marginal_feature_blocks(trajs)
    assert X.shape[0] == 6 * 8 == len(y) == len(seeds)
    assert X.shape[1] == len(groups["all_columns"])
    for name in ("B", "S0", "S1", "S2", "S3"):
        assert all(0 <= i < X.shape[1] for i in groups[name])


def test_run_marginal_audit_runs_on_synthetic() -> None:
    trajs = [_synthetic_trajectory(seed=s, T=12) for s in range(10)]
    for t in trajs:
        compute_derived(t)
    out = run_marginal_audit(trajs, n_folds=5, fold_seed=1)
    assert "M0_geom" in out["predictor_names"]
    assert "M4_full_linear" in out["predictor_names"]
    assert out["metrics"]["M4_full_linear"]["mse"] >= 0.0


def test_bootstrap_seed_diff_basic() -> None:
    seeds = np.asarray([1, 1, 2, 2, 3, 3])
    y = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    p_base = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    p_cand = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = bootstrap_seed_diff(seeds, y, p_base, p_cand, n_resamples=200)
    assert out["mean_mse_reduction"] == pytest.approx(0.25)
    assert out["pct_seeds_improved"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Debug-mode end-to-end smoke
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_main_debug_smoke(tmp_path: Path) -> None:
    out = tmp_path / "out"
    rc = main(["--debug", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "aggregate_stats.json").exists()
    assert (out / "feature_audit.md").exists()
    assert (out / "interpretation.md").exists()
