from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "proseco" / "interactions" / "validate_phase1_schedule_level_b34.py"
spec = importlib.util.spec_from_file_location("validate_phase1_schedule_level_b34", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
b34 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b34
spec.loader.exec_module(b34)


def _schedule(seed: int, b: int, idx: int, steps: list[int], a: float = 1.0, g: float = 1.2) -> dict:
    return {
        "seed": seed,
        "B": b,
        "mc_idx": idx,
        "schedule_steps": steps,
        "A": a,
        "G": g,
        "residual": g - a,
    }


def _xi(seed: int, t: int, tp: int, xi: float, a: float = 1.0) -> dict:
    return {
        "seed": seed,
        "t": t,
        "t_prime": tp,
        "phase_t": b34.phase_for_step(t),
        "phase_tp": b34.phase_for_step(tp),
        "distance": abs(tp - t),
        "A_pair": a,
        "G_pair": a + xi,
        "delta_t": a / 2,
        "delta_tp": a / 2,
        "xi": xi,
    }


def test_candidate_pool_invariants_and_determinism() -> None:
    rows = [
        _schedule(42, 3, 1, [1, 3, 5]),
        _schedule(42, 4, 2, [2, 4, 6, 8]),
        _schedule(43, 3, 0, [5, 3, 1]),
    ]
    measured = {(1, 3), (3, 5), (1, 5)}

    pool_a = b34.build_candidate_pool(rows, measured, budgets=[3, 4], coverage_mode="estimator")
    pool_b = b34.build_candidate_pool(list(reversed(rows)), measured, budgets=[3, 4], coverage_mode="estimator")

    assert pool_a == pool_b
    assert [row["candidate_id"] for row in pool_a] == sorted(row["candidate_id"] for row in pool_a)
    for row in pool_a:
        assert len(row["schedule_steps"]) == row["B"]
        assert len(set(row["schedule_steps"])) == row["B"]
        assert row["n_internal_pairs"] == row["B"] * (row["B"] - 1) // 2


def test_leave_seed_out_never_trains_on_heldout_seed() -> None:
    xi_rows = [
        _xi(1, 1, 3, -0.1),
        _xi(2, 1, 3, -0.2),
        _xi(3, 1, 3, -0.3),
    ]
    splits = b34.build_splits(xi_rows, mode="leave_seed_out")

    for split in splits:
        assert not set(split["heldout_seeds"]) & set(split["train_seeds"])


def test_pairwise_q_is_additive_plus_sum_internal_pair_xi_hat() -> None:
    candidate = b34.normalize_schedule_row(_schedule(42, 3, 7, [1, 3, 5], a=2.0, g=2.5), {(1, 3), (1, 5), (3, 5)})
    q = b34.q_from_pair_xi(candidate, [0.1, -0.2, 0.05])

    assert q == pytest.approx(1.95)


def test_missing_pair_behavior_for_measured_only_and_estimator_modes() -> None:
    rows = [
        _schedule(42, 3, 0, [1, 3, 5]),
        _schedule(42, 3, 1, [1, 3, 9]),
    ]
    measured = {(1, 3), (1, 5), (3, 5)}

    measured_only = b34.build_candidate_pool(rows, measured, budgets=[3], coverage_mode="measured_only")
    estimator = b34.build_candidate_pool(rows, measured, budgets=[3], coverage_mode="estimator")

    assert [row["schedule_steps"] for row in measured_only] == [[1, 3, 5]]
    assert len(estimator) == 2
    assert estimator[1]["missing_internal_pairs"] == [[1, 9], [3, 9]]


def test_json_writer_converts_numpy_scalars(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    b34.write_json(path, {"x": b34.np.float64(1.25), "n": b34.np.int64(3)})

    loaded = json.loads(path.read_text())
    assert loaded == {"n": 3, "x": 1.25}
