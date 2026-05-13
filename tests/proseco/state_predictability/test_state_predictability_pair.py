"""Tests for the pair-level state-predictability audit."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.proseco.state_predictability.analyze_state_predictability_pair import (
    LEAKAGE_FIELDS,
    PairRow,
    STATE_FEATURES,
    build_feature_vector,
    feature_names,
    feature_summary,
    fit_ridge,
    geometry_features,
    load_pair_rows,
    load_protocol_a_features,
    main,
    make_seed_folds,
    run_audit,
    spearman,
)


def _state(seed: int, step: int) -> dict[str, float]:
    return {
        "entropy": float(0.1 * step + 0.01 * seed),
        "inverse_margin": float(0.05 + 0.001 * step),
        "quality_mass_proxy": float(0.3 + 0.001 * step),
        "unmasked_fraction": float(step / 10.0),
        "n_revisable": float(step + 1),
        "n_masked": float(20 - step),
    }


def _make_synthetic_pair_rows(
    n_seeds: int = 10,
    pairs: tuple[tuple[int, int], ...] = ((0, 5), (2, 7), (1, 9), (3, 4), (0, 9)),
    T: int = 10,
    *,
    state_signal: bool = True,
    rng_seed: int = 7,
) -> list[PairRow]:
    rng = np.random.default_rng(rng_seed)
    rows: list[PairRow] = []
    for sd in range(n_seeds):
        for s, t in pairs:
            state_s = _state(sd, s)
            state_t = _state(sd, t)
            delta_s = 0.1 + 0.01 * s + rng.normal(scale=0.005)
            delta_t = 0.1 + 0.01 * t + rng.normal(scale=0.005)
            a_pair = delta_s + delta_t
            # xi has a small geometry contribution and a much larger state
            # contribution if state_signal=True.
            xi_base = -0.05 - 0.001 * abs(t - s)
            if state_signal:
                xi_base += 0.7 * (state_s["entropy"] - state_t["entropy"]) + 0.05 * state_s["unmasked_fraction"]
            xi_base += rng.normal(scale=0.01)
            g_pair = a_pair + xi_base
            rows.append(
                PairRow(
                    seed=sd, s=s, t=t, T=T,
                    distance=abs(t - s),
                    phase_s="early" if s < T / 3 else ("middle" if s < 2 * T / 3 else "late"),
                    phase_t="early" if t < T / 3 else ("middle" if t < 2 * T / 3 else "late"),
                    delta_s=delta_s, delta_t=delta_t,
                    a_pair=a_pair, g_pair=g_pair, xi=xi_base,
                    state_s=state_s, state_t=state_t,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Schema and alignment
# ---------------------------------------------------------------------------

def test_state_features_do_not_contain_leakage_fields() -> None:
    for f in LEAKAGE_FIELDS:
        assert f not in STATE_FEATURES, (
            f"{f!r} must not appear in STATE_FEATURES (post-correction leakage)"
        )


def test_load_protocol_a_features_basic(tmp_path: Path) -> None:
    payload = {
        "seed": 42,
        "T": 2,
        "per_t": [
            {
                "t": 0, "delta": 0.0,
                "entropy": 7.5, "inverse_margin": 1.0,
                "quality_mass_proxy": 0.9, "unmasked_fraction": 0.01,
                "n_revisable": 17, "n_masked": 1000,
            },
            {
                "t": 1, "delta": 0.1,
                "entropy": 3.5, "inverse_margin": 0.4,
                "quality_mass_proxy": 0.4, "unmasked_fraction": 0.03,
                "n_revisable": 30, "n_masked": 980,
            },
        ],
    }
    (tmp_path / "trajectory_0.json").write_text(json.dumps(payload))
    feats, seed_T = load_protocol_a_features(tmp_path)
    assert seed_T[42] == 2
    assert feats[(42, 0)]["entropy"] == pytest.approx(7.5)
    assert feats[(42, 1)]["n_revisable"] == pytest.approx(30.0)


def test_load_pair_rows_verifies_xi_identity(tmp_path: Path) -> None:
    # Protocol A
    proto = {
        "seed": 42, "T": 4,
        "per_t": [
            {"t": ti, "delta": 0.1 * ti,
             "entropy": 1.0 * ti, "inverse_margin": 0.1 * ti,
             "quality_mass_proxy": 0.2 * ti, "unmasked_fraction": 0.1 * ti,
             "n_revisable": float(ti + 1), "n_masked": float(10 - ti)}
            for ti in range(4)
        ],
    }
    proto_dir = tmp_path / "protocol_a"
    proto_dir.mkdir()
    (proto_dir / "trajectory_0.json").write_text(json.dumps(proto))
    feats, seed_T = load_protocol_a_features(proto_dir)
    # xi_raw
    xi_payload = [{
        "seed": 42, "t": 1, "t_prime": 3,
        "phase_t": "early", "phase_tp": "late",
        "distance": 2, "source": "test",
        "G_pair": 0.5, "delta_t": 0.1, "delta_tp": 0.3,
        "A_pair": 0.4, "xi": 0.1, "wall_time": 1.0,
    }]
    xi_path = tmp_path / "xi.json"
    xi_path.write_text(json.dumps(xi_payload))
    rows = load_pair_rows(xi_path, feats, seed_T)
    assert len(rows) == 1
    assert rows[0].xi == pytest.approx(0.1)
    assert rows[0].a_pair == pytest.approx(0.4)
    assert rows[0].g_pair == pytest.approx(0.5)


def test_load_pair_rows_rejects_inconsistent_xi(tmp_path: Path) -> None:
    proto = {
        "seed": 42, "T": 4,
        "per_t": [
            {"t": ti, "delta": 0.1 * ti,
             "entropy": 1.0, "inverse_margin": 0.1,
             "quality_mass_proxy": 0.2, "unmasked_fraction": 0.1,
             "n_revisable": 1.0, "n_masked": 1.0}
            for ti in range(4)
        ],
    }
    proto_dir = tmp_path / "protocol_a"
    proto_dir.mkdir()
    (proto_dir / "trajectory_0.json").write_text(json.dumps(proto))
    feats, seed_T = load_protocol_a_features(proto_dir)
    bad = [{
        "seed": 42, "t": 1, "t_prime": 3,
        "phase_t": "early", "phase_tp": "late",
        "distance": 2, "source": "test",
        "G_pair": 0.5, "delta_t": 0.1, "delta_tp": 0.3,
        "A_pair": 0.4, "xi": 0.4,  # WRONG: should be 0.1
        "wall_time": 1.0,
    }]
    xi_path = tmp_path / "xi.json"
    xi_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="xi != G_pair - A_pair"):
        load_pair_rows(xi_path, feats, seed_T)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_make_seed_folds_partitions_without_overlap() -> None:
    folds = make_seed_folds(list(range(30)), n_folds=5, seed=7)
    flat = [s for fold in folds for s in fold]
    assert sorted(flat) == list(range(30))
    assert len(flat) == len(set(flat))  # no duplicate
    # And no test seed appears in train of its own fold.
    for i, fold in enumerate(folds):
        train_seeds = {s for j, other in enumerate(folds) if j != i for s in other}
        assert set(fold).isdisjoint(train_seeds)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def test_geometry_features_length_matches_names() -> None:
    rows = _make_synthetic_pair_rows(n_seeds=1)
    feats = geometry_features(rows[0])
    names = feature_names("geom")
    assert len(feats) == len(names)


def test_feature_kinds_lengths_consistent() -> None:
    rows = _make_synthetic_pair_rows(n_seeds=1)
    for kind in ("geom", "geom_apair", "apair", "geom_apair_state", "geom_apair_state_poly2"):
        vec = build_feature_vector(rows[0], kind)
        names = feature_names(kind)
        assert len(vec) == len(names), (kind, len(vec), len(names))


# ---------------------------------------------------------------------------
# Train-only standardization
# ---------------------------------------------------------------------------

def test_fit_ridge_uses_train_only_standardization() -> None:
    rows = _make_synthetic_pair_rows(state_signal=True)
    seeds = sorted({r.seed for r in rows})
    train_seeds = set(seeds[:7])
    train = [r for r in rows if r.seed in train_seeds]
    model = fit_ridge(train, "geom_apair_state", "xi", alpha=1.0)
    # The mean/scale should be computed only from train rows.
    from scripts.proseco.state_predictability.analyze_state_predictability_pair import _matrix
    X_train = _matrix(train, "geom_apair_state")
    np.testing.assert_allclose(model.mean, X_train.mean(axis=0))
    np.testing.assert_allclose(
        model.scale,
        np.where(X_train.std(axis=0) < 1e-12, 1.0, X_train.std(axis=0)),
    )


# ---------------------------------------------------------------------------
# Audit detects a state signal in synthetic data
# ---------------------------------------------------------------------------

def test_run_audit_detects_synthetic_pair_state_signal() -> None:
    rows = _make_synthetic_pair_rows(state_signal=True, n_seeds=10)
    pred_rows, agg = run_audit(rows, n_folds=5, fold_seed=1)
    # Output shape
    expected = len(rows)
    assert len(pred_rows) == expected
    # No seed leakage between train and test in any fold.
    folds = agg["meta"]["folds"]
    fold_assignment = agg["meta"]["fold_assignment"]
    assert sum(f["n_test_seeds"] for f in folds) == 10
    for r in pred_rows:
        assert fold_assignment[str(int(r["seed"]))] == int(r["fold_id"])
    # On the synthetic generator, state should improve over geom + A.
    imp = agg["mse_improvements"]["xi"][
        "ridge_geom_apair_state_vs_ridge_geom_apair"
    ]
    assert imp["mean_mse_reduction"] > 0.0
    # And aggregate MSE for state model should be lower than geom+A for xi.
    mse_state = agg["predictive_metrics"]["xi"]["ridge_geom_apair_state"]["mse"]
    mse_geom_a = agg["predictive_metrics"]["xi"]["ridge_geom_apair"]["mse"]
    assert mse_state < mse_geom_a


def test_run_audit_negative_when_no_synthetic_signal() -> None:
    rows = _make_synthetic_pair_rows(state_signal=False, n_seeds=10, rng_seed=99)
    pred_rows, agg = run_audit(rows, n_folds=5, fold_seed=2)
    label = agg["verdict"]["label_xi"]
    assert label in {"state_predictability_not_supported", "state_predictability_weak"}


# ---------------------------------------------------------------------------
# Output keys and IO
# ---------------------------------------------------------------------------

def test_aggregate_has_required_keys() -> None:
    rows = _make_synthetic_pair_rows(n_seeds=10)
    _, agg = run_audit(rows, n_folds=5, fold_seed=3)
    assert "meta" in agg and "predictive_metrics" in agg
    assert "mse_improvements" in agg and "verdict" in agg
    assert "b2_top5_ranking" in agg
    for target in ("xi", "g_pair"):
        assert target in agg["predictive_metrics"]


def test_feature_summary_reports_corr_apair_xi() -> None:
    rows = _make_synthetic_pair_rows(n_seeds=10)
    summ = feature_summary(rows)
    assert math.isfinite(summ["corr_apair_xi"])


# ---------------------------------------------------------------------------
# Spearman helper (re-test for stability)
# ---------------------------------------------------------------------------

def test_spearman_basic() -> None:
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isfinite(spearman([1, 1, 2, 3], [4, 5, 6, 7]))


# ---------------------------------------------------------------------------
# End-to-end: debug-mode main on real data writes expected files
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_main_debug_writes_outputs(tmp_path: Path) -> None:
    """End-to-end smoke test on real data with --debug. Skipped by default
    via the slow marker so the fast suite stays fast.
    """
    out = tmp_path / "out"
    rc = main([
        "--out-dir", str(out),
        "--n-folds", "2",
        "--debug",
    ])
    assert rc == 0
    assert (out / "aggregate_stats.json").exists()
    assert (out / "interpretation.md").exists()
