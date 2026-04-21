from __future__ import annotations

import numpy as np
import pytest

from mdm_playground.scheduling import allocate_budget, evaluate_schedule


def test_allocate_budget_prioritises_expected_indices() -> None:
    signal = [0.1, 0.4, 0.9, 0.2]

    assert set(allocate_budget(signal, 2, "top_B")) == {1, 2}
    assert set(allocate_budget(signal, 2, "bottom_B")) == {0, 3}
    assert set(allocate_budget(signal, 2, "uniform")) == {0, 2}
    assert set(allocate_budget(signal, 2, "middle")) == {1, 2}
    assert set(
        allocate_budget(signal, 2, "burn_in_gated", {"low_gain_threshold": 0.15})
    ) == {1, 2}


class _DummyGenerator:
    def run_base(self, seed: int):
        return {
            "tokens": np.array([1, 2, 3]),
            "neg_nll": 1.0,
            "per_step_signals": [{"t": 0}, {"t": 1}, {"t": 2}],
        }

    def run_with_schedule(self, allocation, seed: int):
        return {
            "tokens": np.array([1, 2, 4]),
            "neg_nll": 1.5,
            "per_step_signals": [{"t": 0}, {"t": 1}, {"t": 2}],
        }


def test_evaluate_schedule_returns_contract_fields() -> None:
    generator = _DummyGenerator()
    allocation = {2: 1, 0: 1}
    delta_trace = {0: 0.25, 2: -0.1}

    result = evaluate_schedule(allocation, delta_trace, generator, F="neg_nll", seed=7)

    assert result["G"] == pytest.approx(0.5)
    assert result["A"] == pytest.approx(0.15)
    assert result["residual"] == pytest.approx(0.35)
    assert result["schedule_steps"] == [0, 2]
    assert result["budget"] == 2
    assert result["f_base"] == pytest.approx(1.0)
    assert result["f_schedule"] == pytest.approx(1.5)
    assert result["wall_time"] >= 0.0


def test_proseco_owt_snapshot_error_is_actionable(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    assert torch is not None

    from mdm_playground.scheduling.backends.proseco_owt import _validate_snapshot_dir

    with pytest.raises(FileNotFoundError, match="scripts/stage_proseco_owt.py"):
        _validate_snapshot_dir(tmp_path)
