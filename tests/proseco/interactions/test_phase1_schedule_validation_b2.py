from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "proseco" / "interactions" / "validate_phase1_schedule_level_b2.py"
spec = importlib.util.spec_from_file_location("validate_phase1_schedule_level_b2", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)

ModelSpec = validator.ModelSpec
build_splits = validator.build_splits
compute_model_metrics = validator.compute_model_metrics
fit_xi_model = validator.fit_xi_model
predict_xi = validator.predict_xi
run_crossfit_predictions = validator.run_crossfit_predictions


def _row(seed: int, t: int, tp: int, phase_t: str, phase_tp: str, distance: int, a: float, xi: float) -> dict:
    return {
        "seed": seed,
        "t": t,
        "t_prime": tp,
        "phase_t": phase_t,
        "phase_tp": phase_tp,
        "distance": distance,
        "A_pair": a,
        "G_pair": a + xi,
        "xi": xi,
        "source": "test",
        "wall_time": 0.0,
    }


def test_leave_seed_out_train_mean_uses_only_training_seeds() -> None:
    rows = [
        _row(1, 0, 1, "early", "early", 1, 1.0, 0.10),
        _row(2, 0, 1, "early", "early", 1, 1.0, 0.30),
        _row(3, 0, 1, "early", "early", 1, 1.0, 100.0),
    ]
    splits = build_splits(rows, mode="leave_seed_out", k_folds=5)
    predictions = run_crossfit_predictions(rows, splits, [ModelSpec("train_global_mean")])

    heldout_seed3 = [
        r for r in predictions
        if r["seed"] == 3 and r["model"] == "train_global_mean"
    ]
    assert len(heldout_seed3) == 1
    assert heldout_seed3[0]["xi_hat"] == pytest.approx(0.20)
    assert heldout_seed3[0]["prediction"] == pytest.approx(1.20)


def test_phase_distance_model_falls_back_to_training_global_for_unseen_stratum() -> None:
    train_rows = [
        _row(1, 0, 10, "early", "early", 10, 0.5, -0.20),
        _row(2, 0, 10, "early", "early", 10, 0.5, -0.40),
    ]
    unseen_row = _row(3, 44, 61, "late", "late", 17, 0.5, 9.0)

    fitted = fit_xi_model(ModelSpec("phase_distance_mean"), train_rows)

    assert predict_xi(fitted, unseen_row) == pytest.approx(-0.30)


def test_metric_definitions_for_eta_zeta_and_delta() -> None:
    rows = [
        {
            "model": "phase_pair_mean",
            "seed": 1,
            "G_pair": 1.0,
            "A_pair": 2.0,
            "prediction": 0.75,
            "abs_err_A": 1.0,
            "abs_err_Q": 0.25,
            "sq_err_A": 1.0,
            "sq_err_Q": 0.0625,
        },
        {
            "model": "phase_pair_mean",
            "seed": 2,
            "G_pair": 2.0,
            "A_pair": 1.0,
            "prediction": 2.25,
            "abs_err_A": 1.0,
            "abs_err_Q": 0.25,
            "sq_err_A": 1.0,
            "sq_err_Q": 0.0625,
        },
    ]

    metrics = compute_model_metrics(rows)

    assert metrics["eta_mean"] == pytest.approx(1.0)
    assert metrics["zeta_mean"] == pytest.approx(0.25)
    assert metrics["delta_abs_error"] == pytest.approx(0.75)
    assert metrics["RMSE_A"] == pytest.approx(1.0)
    assert metrics["RMSE_Q"] == pytest.approx(0.25)
    assert metrics["R_spearman"] == pytest.approx(-1.0)
    assert metrics["P_spearman"] == pytest.approx(1.0)
    assert metrics["delta_spearman"] == pytest.approx(2.0)
