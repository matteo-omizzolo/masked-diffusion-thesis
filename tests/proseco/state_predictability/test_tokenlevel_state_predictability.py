"""Tests for token-level feature extraction and pair-level audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.proseco.state_predictability.extract_tokenlevel_features_proseco import (
    ENTROPY_THRESHOLDS,
    LEAKAGE_SUBSTRINGS,
    MARGIN_LOW_THRESHOLDS,
    QUANTILES,
    assert_no_leakage_in_names,
    cosine,
    gini_index,
    jaccard,
    positions_for_mass_share,
    quantiles,
    run_extraction_synthetic,
    sanity_check_features,
    shannon_entropy_per_position,
    step_features_from_p_x0,
    summarize_positions,
    top1_and_margin,
    write_seed_features,
)
from scripts.proseco.state_predictability.analyze_tokenlevel_state_predictability import (
    build_pair_feature_matrix,
    fit_predict_grouped,
    make_seed_folds,
    load_seed_features,
    pair_overlap_features,
    run_audit,
    spearman,
)


# ---------------------------------------------------------------------------
# Pure feature-math helpers
# ---------------------------------------------------------------------------

def test_shannon_entropy_uniform_is_log_v() -> None:
    V = 8
    p = np.full((3, V), 1.0 / V)
    H = shannon_entropy_per_position(p)
    np.testing.assert_allclose(H, np.full(3, np.log(V)), atol=1e-9)


def test_shannon_entropy_one_hot_is_zero() -> None:
    p = np.eye(4)
    H = shannon_entropy_per_position(p)
    np.testing.assert_allclose(H, np.zeros(4), atol=1e-9)


def test_top1_and_margin_correctness() -> None:
    p = np.asarray([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]])
    top1, margin = top1_and_margin(p)
    np.testing.assert_allclose(top1, np.asarray([0.7, 0.5]))
    np.testing.assert_allclose(margin, np.asarray([0.5, 0.1]))


def test_quantiles_ordering() -> None:
    arr = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    qs = quantiles(arr)
    assert list(qs) == sorted(qs.tolist())


def test_gini_uniform_is_zero() -> None:
    v = np.full(10, 1.0)
    assert gini_index(v) == pytest.approx(0.0, abs=1e-9)


def test_gini_concentrated_is_high() -> None:
    v = np.zeros(10)
    v[0] = 1.0
    assert gini_index(v) > 0.85


def test_positions_for_50pct_mass() -> None:
    v = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0])  # uniform: half-mass requires ceil(2.5)=3
    assert positions_for_mass_share(v, 0.5) == 3
    v2 = np.asarray([10.0, 1.0, 1.0])  # 10/12 in 1 position
    assert positions_for_mass_share(v2, 0.5) == 1


def test_jaccard_basic() -> None:
    a = np.asarray([1, 2, 3])
    b = np.asarray([2, 3, 4])
    assert jaccard(a, b) == pytest.approx(2.0 / 4.0)
    assert jaccard(np.asarray([1, 2]), np.asarray([3, 4])) == pytest.approx(0.0)


def test_cosine_zero_vector_is_nan() -> None:
    import math
    assert math.isnan(cosine(np.zeros(4), np.asarray([1.0, 0, 0, 0])))


def test_cosine_identical_vectors() -> None:
    v = np.asarray([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Position summarisation
# ---------------------------------------------------------------------------

def test_summarize_positions_quantile_ordering() -> None:
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(8), size=64)
    ent = shannon_entropy_per_position(p)
    top1, margin = top1_and_margin(p)
    summ = summarize_positions(p, ent, top1, margin)
    assert list(summ.entropy_qs) == sorted(summ.entropy_qs.tolist())
    assert list(summ.top1_qs) == sorted(summ.top1_qs.tolist())
    assert list(summ.margin_qs) == sorted(summ.margin_qs.tolist())


def test_summarize_positions_empty_returns_nans() -> None:
    summ = summarize_positions(np.zeros((0, 4)), np.zeros(0), np.zeros(0), np.zeros(0))
    assert summ.n_positions == 0
    assert np.isnan(summ.entropy_mean)


# ---------------------------------------------------------------------------
# Leakage hygiene
# ---------------------------------------------------------------------------

def test_assert_no_leakage_rejects_branch() -> None:
    with pytest.raises(RuntimeError):
        assert_no_leakage_in_names(["entropy", "f_branch_max"])


def test_leakage_substrings_includes_required_terms() -> None:
    required = {"branch", "corrected", "n_changed", "tcr", "target"}
    for r in required:
        assert any(r in sub or sub in r for sub in LEAKAGE_SUBSTRINGS), r


def test_step_features_scalar_names_have_no_leakage() -> None:
    rng = np.random.default_rng(1)
    L, V = 16, 6
    p = rng.dirichlet(np.ones(V), size=L)
    x = rng.integers(0, V, size=L)
    sf = step_features_from_p_x0(t_idx=0, x_tokens=x, mask_id=0, p_x0=p)
    assert_no_leakage_in_names(sf.scalar_names)


# ---------------------------------------------------------------------------
# Synthetic extraction round-trip
# ---------------------------------------------------------------------------

def test_synthetic_extraction_writes_valid_npz(tmp_path: Path) -> None:
    steps = run_extraction_synthetic(seed=42, T=6, seq_len=24, V=8)
    out = tmp_path / "per_step_features_seed042.npz"
    write_seed_features(out, seed=42, T=6, steps=steps)
    loaded = load_seed_features(out)
    assert loaded.seed == 42
    assert loaded.T == 6
    assert loaded.scalars.shape[0] == 6
    assert loaded.entropy_per_pos.shape == (6, 24)
    assert (loaded.entropy_per_pos >= -1e-6).all()
    assert ((loaded.top1_per_pos >= -1e-6) & (loaded.top1_per_pos <= 1.0 + 1e-6)).all()


def test_sanity_check_features_passes_on_synthetic() -> None:
    steps = run_extraction_synthetic(seed=42, T=6, seq_len=24, V=8)
    report = sanity_check_features(steps)
    assert report["entropy_per_pos_min"] >= -1e-6
    assert report["quantile_ordering_violations"] == 0
    assert report["finite_feature_share_min"] > 0.0


# ---------------------------------------------------------------------------
# Pair-level feature construction and audit
# ---------------------------------------------------------------------------

def _make_synthetic_features_dir(tmp_path: Path, n_seeds: int = 5, T: int = 6) -> Path:
    fdir = tmp_path / "feats"
    fdir.mkdir()
    for sd in range(42, 42 + n_seeds):
        steps = run_extraction_synthetic(seed=sd, T=T, seq_len=32, V=8)
        write_seed_features(fdir / f"per_step_features_seed{sd:03d}.npz",
                            seed=sd, T=T, steps=steps)
    return fdir


def _make_synthetic_xi_rows(n_seeds: int = 5, pairs: list[tuple[int, int]] | None = None) -> list[dict]:
    pairs = pairs or [(0, 4), (1, 5), (2, 3)]
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for sd in range(42, 42 + n_seeds):
        for s, t in pairs:
            delta_s = float(rng.normal(0.0, 0.1))
            delta_t = float(rng.normal(0.0, 0.1))
            a_pair = delta_s + delta_t
            xi = float(rng.normal(-0.05, 0.1))
            g_pair = a_pair + xi
            rows.append({
                "seed": sd, "t": s, "t_prime": t,
                "phase_t": "early", "phase_tp": "late",
                "distance": abs(t - s), "source": "test",
                "G_pair": g_pair, "delta_t": delta_s, "delta_tp": delta_t,
                "A_pair": a_pair, "xi": xi, "wall_time": 0.0,
            })
    return rows


def test_build_pair_feature_matrix_includes_all_families(tmp_path: Path) -> None:
    fdir = _make_synthetic_features_dir(tmp_path)
    features = {load_seed_features(f).seed: load_seed_features(f) for f in fdir.glob("*.npz")}
    rows = _make_synthetic_xi_rows()
    X, y_xi, y_g, seeds, groups = build_pair_feature_matrix(rows, features)
    assert X.shape[0] == len(rows)
    # Every family should be non-empty.
    for fam in ("B_pair", "ORACLE", "T0_pair", "T1_pair", "T2_pair", "T3_pair", "T4_pair"):
        assert len(groups[fam]) > 0, fam
    # Identities must be preserved (xi = G - A).
    np.testing.assert_allclose(y_g - X[:, groups["ORACLE"][0]] , y_xi, atol=1e-9)


def test_pair_overlap_features_self_pair_has_jaccard_one(tmp_path: Path) -> None:
    fdir = _make_synthetic_features_dir(tmp_path, n_seeds=1)
    sf = load_seed_features(next(fdir.glob("*.npz")))
    feats, names = pair_overlap_features(sf, 1, 1)
    name_to_val = dict(zip(names, feats))
    # Jaccard of identical sets is 1; cosine of identical vectors is 1.
    assert name_to_val["T4_jaccard_revisable"] == pytest.approx(1.0)
    assert name_to_val["T4_cos_entropy"] == pytest.approx(1.0)
    assert name_to_val["T4_cos_margin"] == pytest.approx(1.0)


def test_make_seed_folds_no_leakage() -> None:
    folds = make_seed_folds(list(range(8)), n_folds=4)
    flat = [s for f in folds for s in f]
    assert sorted(flat) == list(range(8))
    assert len(set(flat)) == len(flat)


def test_fit_predict_grouped_train_only_standardization(tmp_path: Path) -> None:
    fdir = _make_synthetic_features_dir(tmp_path)
    features = {load_seed_features(f).seed: load_seed_features(f) for f in fdir.glob("*.npz")}
    rows = _make_synthetic_xi_rows()
    X, y_xi, _, seeds, _ = build_pair_feature_matrix(rows, features)
    folds = make_seed_folds(sorted({int(s) for s in seeds}), 5)
    preds = fit_predict_grouped(X, y_xi, seeds, folds)
    assert preds.shape == y_xi.shape


def test_run_audit_returns_expected_predictors(tmp_path: Path) -> None:
    fdir = _make_synthetic_features_dir(tmp_path)
    features = {load_seed_features(f).seed: load_seed_features(f) for f in fdir.glob("*.npz")}
    rows = _make_synthetic_xi_rows()
    X, y_xi, y_g, seeds, groups = build_pair_feature_matrix(rows, features)
    out = run_audit(X, y_xi, y_g, seeds, groups, n_folds=5, fold_seed=0)
    for name in ("P0_geom", "P1_geom_apair", "P6_plus_T4_overlaps"):
        assert name in out["metrics_xi"]
    assert "P6_plus_T4_overlaps_vs_P1_geom_apair" in out["improvements_xi"]


def test_spearman_basic() -> None:
    assert spearman(np.asarray([1, 2, 3, 4]), np.asarray([4, 3, 2, 1])) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Debug smoke (whole pipeline)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_analyze_main_debug_smoke(tmp_path: Path) -> None:
    from scripts.proseco.state_predictability.analyze_tokenlevel_state_predictability import main as analyze_main
    out = tmp_path / "analyze_out"
    rc = analyze_main(["--debug", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "aggregate_stats.json").exists()
    assert (out / "interpretation.md").exists()


@pytest.mark.slow
def test_extract_main_debug_synthetic(tmp_path: Path) -> None:
    from scripts.proseco.state_predictability.extract_tokenlevel_features_proseco import main as extract_main
    out = tmp_path / "extract_out"
    rc = extract_main(["--debug", "--use-synthetic", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "sanity_checks.json").exists()
    assert (out / "feature_extraction_plan.md").exists()
    assert sorted(out.glob("per_step_features_seed*.npz"))
