"""Tests for held-out state-predictability audit."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.proseco.state_predictability.analyze_state_predictability import (
    Row,
    additive_schedule_metrics_with_uniform,
    fit_ridge,
    load_protocol_a_rows,
    make_seed_folds,
    run_audit,
    spearman,
    uniform_schedule,
)


def _rows(n_seeds: int = 12, T: int = 10, *, state_signal: bool = True) -> list[Row]:
    rows: list[Row] = []
    rng = np.random.default_rng(123)
    for seed in range(n_seeds):
        seed_shift = rng.normal(scale=0.01)
        for t in range(T):
            z = math.sin((t + 1) / T * math.pi) + rng.normal(scale=0.05)
            delta = 0.2 * (t / T)
            if state_signal:
                delta += 0.8 * z
            delta += seed_shift
            features = {
                "entropy": z,
                "inverse_margin": 1.0 - z,
                "quality_mass_proxy": z * z,
                "unmasked_fraction": t / T,
                "n_revisable": float(t + 1),
                "n_masked": float(T - t),
            }
            rows.append(Row(seed=seed, t=t, T=T, delta=delta, features=features))
    return rows


def test_make_seed_folds_partitions_without_overlap() -> None:
    folds = make_seed_folds(list(range(10)), n_folds=5, seed=7)
    flat = [s for fold in folds for s in fold]
    assert sorted(flat) == list(range(10))
    assert len(flat) == len(set(flat))


def test_load_protocol_a_rows_reads_expected_schema(tmp_path: Path) -> None:
    payload = {
        "seed": 42,
        "T": 2,
        "per_t": [
            {
                "t": 0,
                "delta": 0.1,
                "entropy": 1.0,
                "inverse_margin": 0.2,
                "quality_mass_proxy": 0.3,
                "unmasked_fraction": 0.1,
                "n_revisable": 1,
                "n_masked": 9,
            }
        ],
    }
    (tmp_path / "trajectory_0.json").write_text(json.dumps(payload))
    rows = load_protocol_a_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0].seed == 42
    assert rows[0].features["entropy"] == pytest.approx(1.0)


def test_ridge_state_predicts_synthetic_state_signal() -> None:
    rows = _rows(state_signal=True)
    model_time = fit_ridge(rows, "time")
    model_state = fit_ridge(rows, "time_state")
    y = np.asarray([r.delta for r in rows])
    mse_time = np.mean((model_time.predict(rows) - y) ** 2)
    mse_state = np.mean((model_state.predict(rows) - y) ** 2)
    assert mse_state < mse_time * 0.25


def test_run_audit_detects_state_signal() -> None:
    prediction_rows, aggregate = run_audit(_rows(state_signal=True), n_folds=4)
    assert len(prediction_rows) == 120
    imp = aggregate["mse_improvements"]["ridge_time_state_vs_ridge_time"]
    assert imp["mean_mse_reduction"] > 0.0
    assert aggregate["predictive_metrics"]["ridge_time_state"]["mse"] < aggregate[
        "predictive_metrics"
    ]["ridge_time"]["mse"]


def test_uniform_schedule_has_requested_budget() -> None:
    for B in (1, 2, 3, 4):
        sched = uniform_schedule(64, B)
        assert len(sched) == B
        assert len(set(sched)) == B


def test_additive_schedule_metrics_with_uniform_serializable() -> None:
    rows = []
    for seed in range(3):
        for t in range(6):
            delta = float(t)
            rows.append(
                {
                    "seed": seed,
                    "t": t,
                    "delta": delta,
                    "pred_time": float(t),
                }
            )
    metrics = additive_schedule_metrics_with_uniform(rows, ["time"], [2])
    assert metrics["2"]["models"]["time"]["close_ratio_from_means"] == pytest.approx(1.0)
    json.dumps(metrics)


def test_spearman_handles_ties() -> None:
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert math.isfinite(spearman([1, 1, 2], [2, 3, 4]))
