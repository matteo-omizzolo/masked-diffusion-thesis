"""Tests for Protocol C — bounded adaptive-controller pilot on OWT."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from mdm_playground.analysis.protocol_c import (
    SIGNAL_KINDS,
    additive_surrogate,
    best_mc_schedule,
    bucket_phase,
    bucket_signal,
    bucket_state,
    build_bucket_model,
    compute_eps_linear,
    compute_eps_tilde,
    compute_signal_thresholds,
    hamming,
    policy_size_at_lambda,
    protocol_c_pipeline,
    threshold_schedule,
    topB_bucket_schedule,
    tune_lambda,
    uniform_schedule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_trajectory(
    seed: int, T: int = 8, *, signal_strength: float = 1.0
) -> dict[str, Any]:
    """Construct a deterministic synthetic Phase-1-style trajectory.

    Δ_t is constructed to be high in the middle phase, low at the edges,
    with entropy positively correlated with Δ_t.
    """
    rng = np.random.default_rng(seed)
    per_t: list[dict[str, Any]] = []
    for t in range(T):
        phase_factor = math.sin(math.pi * (t + 0.5) / T)
        entropy = float(phase_factor * 5.0 + rng.normal(scale=0.1))
        delta = float(signal_strength * phase_factor + rng.normal(scale=0.05))
        per_t.append(
            {
                "t": t,
                "delta": delta,
                "entropy": entropy,
                "inverse_margin": float(0.5 + 0.5 * phase_factor),
                "quality_mass_proxy": float(0.5 + 0.4 * phase_factor),
                "unmasked_fraction": float(t / T),
                "n_revisable": int(t),
                "n_masked": int(T - t),
                "tcr": 0.0,
                "f_base": 0.0,
                "f_branch": delta,
                "n_changed": 0,
            }
        )
    return {"seed": seed, "T": T, "per_t": per_t}


@pytest.fixture
def trajectories() -> list[dict[str, Any]]:
    return [_synthetic_trajectory(seed=42 + i, T=12) for i in range(8)]


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bucket_phase_partitions_horizon() -> None:
    T = 9
    assert bucket_phase(0, T, 3) == 0
    assert bucket_phase(2, T, 3) == 0
    assert bucket_phase(3, T, 3) == 1
    assert bucket_phase(5, T, 3) == 1
    assert bucket_phase(6, T, 3) == 2
    assert bucket_phase(8, T, 3) == 2


@pytest.mark.unit
def test_bucket_phase_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        bucket_phase(-1, 4)
    with pytest.raises(ValueError):
        bucket_phase(0, 0)
    with pytest.raises(ValueError):
        bucket_phase(0, 4, n_phase_bins=0)


@pytest.mark.unit
def test_bucket_signal_uses_thresholds_inclusive_lower() -> None:
    thr = (0.0, 1.0, 2.0)
    assert bucket_signal(-0.5, thr) == 0
    assert bucket_signal(0.0, thr) == 0
    assert bucket_signal(0.5, thr) == 1
    assert bucket_signal(2.0, thr) == 2
    assert bucket_signal(2.5, thr) == 3


@pytest.mark.unit
def test_compute_signal_thresholds_returns_quantile_cuts(
    trajectories: list[dict[str, Any]],
) -> None:
    thr = compute_signal_thresholds(trajectories, "entropy", n_signal_bins=4)
    assert len(thr) == 3
    assert thr == tuple(sorted(thr))


@pytest.mark.unit
def test_compute_signal_thresholds_rejects_unknown_signal(
    trajectories: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError):
        compute_signal_thresholds(trajectories, "not_a_signal")


@pytest.mark.unit
def test_bucket_state_is_pure_function() -> None:
    thr = (0.0, 1.0, 2.0)
    assert bucket_state(0.5, 3, 12, thr, 3) == (1, 0)
    assert bucket_state(2.5, 8, 12, thr, 3) == (3, 2)


# ---------------------------------------------------------------------------
# Bucket model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_bucket_model_keys_in_expected_grid(
    trajectories: list[dict[str, Any]],
) -> None:
    model = build_bucket_model(trajectories, "entropy", n_signal_bins=4, n_phase_bins=3)
    assert model.signal_kind == "entropy"
    assert len(model.signal_thresholds) == 3
    for (sb, pb) in model.means:
        assert 0 <= sb < 4
        assert 0 <= pb < 3
    assert sum(model.bucket_counts.values()) == sum(
        len(t["per_t"]) for t in trajectories
    )


@pytest.mark.unit
def test_build_bucket_model_psi_returns_fallback_for_empty_bucket(
    trajectories: list[dict[str, Any]],
) -> None:
    model = build_bucket_model(trajectories, "entropy")
    # request a pathological state that may not be in any populated bucket
    psi = model.psi(s_value=99.0, t=0, T=trajectories[0]["T"])
    assert math.isfinite(psi)


# ---------------------------------------------------------------------------
# Calibration ε vs ε̃
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_eps_linear_zero_when_signal_perfectly_predicts(
    trajectories: list[dict[str, Any]],
) -> None:
    # Make Δ exactly = 2 · entropy + 1 so the linear fit is perfect.
    for traj in trajectories:
        for step in traj["per_t"]:
            step["delta"] = 2.0 * step["entropy"] + 1.0
    eps = compute_eps_linear(trajectories, "entropy")
    assert eps == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_compute_eps_linear_positive_when_noisy(
    trajectories: list[dict[str, Any]],
) -> None:
    eps = compute_eps_linear(trajectories, "entropy")
    assert eps > 0.0
    assert math.isfinite(eps)


@pytest.mark.unit
def test_compute_eps_tilde_returns_finite_and_nonnegative(
    trajectories: list[dict[str, Any]],
) -> None:
    model = build_bucket_model(trajectories, "entropy")
    eps_t = compute_eps_tilde(trajectories, model)
    assert eps_t >= 0.0
    assert math.isfinite(eps_t)


@pytest.mark.unit
def test_eps_tilde_at_most_eps_for_perfect_bucket_prediction() -> None:
    # If Δ_t is constant within each (signal, phase) bucket, ε̃ = 0 and
    # ε > 0 (linear fit cannot recover a step function).
    trajectories: list[dict[str, Any]] = []
    for seed in range(4):
        per_t: list[dict[str, Any]] = []
        for t in range(12):
            entropy = float(t)
            phase_b = bucket_phase(t, 12, 3)
            delta = float(phase_b)  # piecewise-constant in phase
            per_t.append(
                {
                    "t": t,
                    "delta": delta,
                    "entropy": entropy,
                    "inverse_margin": 0.5,
                    "quality_mass_proxy": 0.5,
                }
            )
        trajectories.append({"seed": seed, "T": 12, "per_t": per_t})
    model = build_bucket_model(trajectories, "entropy", n_signal_bins=4, n_phase_bins=3)
    eps_t = compute_eps_tilde(trajectories, model)
    eps = compute_eps_linear(trajectories, "entropy")
    assert eps_t < eps


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_uniform_schedule_size() -> None:
    s = uniform_schedule(64, 4)
    assert len(s) == 4
    assert s == frozenset({0, 16, 32, 48})


@pytest.mark.unit
def test_uniform_schedule_handles_zero_and_full() -> None:
    assert uniform_schedule(64, 0) == frozenset()
    assert uniform_schedule(4, 4) == frozenset({0, 1, 2, 3})


@pytest.mark.unit
def test_threshold_schedule_capped_at_B(
    trajectories: list[dict[str, Any]],
) -> None:
    model = build_bucket_model(trajectories, "entropy")
    # very low lam: every step satisfies, schedule gets capped at B
    B = 3
    for traj in trajectories:
        S = threshold_schedule(traj, model, lam=-1e9, B=B)
        assert len(S) <= B


@pytest.mark.unit
def test_threshold_schedule_empty_when_lambda_too_high(
    trajectories: list[dict[str, Any]],
) -> None:
    model = build_bucket_model(trajectories, "entropy")
    for traj in trajectories:
        S = threshold_schedule(traj, model, lam=1e9, B=4)
        assert S == frozenset()


@pytest.mark.unit
def test_topB_returns_exactly_B_indices(
    trajectories: list[dict[str, Any]],
) -> None:
    model = build_bucket_model(trajectories, "entropy")
    for traj in trajectories:
        S = topB_bucket_schedule(traj, model, B=3)
        assert len(S) == 3
        for idx in S:
            assert 0 <= idx < traj["T"]


@pytest.mark.unit
def test_tune_lambda_yields_size_close_to_target(
    trajectories: list[dict[str, Any]],
) -> None:
    model = build_bucket_model(trajectories, "entropy")
    target = 3
    lam = tune_lambda(trajectories, model, target_B=target, n_grid=512)
    achieved = policy_size_at_lambda(trajectories, model, lam)
    # search is on a discrete grid; allow small slack
    assert abs(achieved - target) <= 1.0


# ---------------------------------------------------------------------------
# Surrogate / hamming / mc helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_additive_surrogate_returns_sum_of_deltas(
    trajectories: list[dict[str, Any]],
) -> None:
    traj = trajectories[0]
    schedule = frozenset({1, 4, 7})
    expected = sum(
        float(step["delta"]) for step in traj["per_t"] if int(step["t"]) in schedule
    )
    assert additive_surrogate(traj, schedule) == pytest.approx(expected)


@pytest.mark.unit
def test_hamming_is_symmetric_difference_size() -> None:
    assert hamming({0, 1, 2}, {0, 1, 2}, T=8) == 0
    assert hamming({0, 1}, {0, 2}, T=8) == 2
    assert hamming({0, 1, 2}, {3, 4, 5}, T=8) == 6


@pytest.mark.unit
def test_hamming_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        hamming({0, 9}, {0, 1}, T=8)


@pytest.mark.unit
def test_best_mc_schedule_picks_max_G() -> None:
    rows = [
        {"seed": 1, "B": 2, "G": 0.1, "schedule_steps": [0, 1]},
        {"seed": 1, "B": 2, "G": 0.5, "schedule_steps": [2, 3]},
        {"seed": 1, "B": 2, "G": 0.3, "schedule_steps": [4, 5]},
    ]
    sched, g = best_mc_schedule(rows, B=2)
    assert g == 0.5
    assert sched == frozenset({2, 3})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_protocol_c_pipeline_returns_well_formed_summary(
    trajectories: list[dict[str, Any]],
) -> None:
    mc_rows_by_seed = {
        traj["seed"]: [
            {
                "seed": traj["seed"],
                "B": 2,
                "G": float(0.1 + i * 0.01),
                "A": 0.0,
                "residual": 0.0,
                "schedule_steps": [int(i % traj["T"]), int((i + 3) % traj["T"])],
            }
            for i in range(5)
        ]
        for traj in trajectories
    }
    delta_open = {2: 0.45, 3: 0.45, 4: 0.45}
    sigma_xi = {2: 0.17, 3: 0.24, 4: 0.31}

    summary = protocol_c_pipeline(
        phase1_trajectories=trajectories,
        phase2b_mc_rows_by_seed=mc_rows_by_seed,
        delta_open_per_B=delta_open,
        sigma_xi_per_B=sigma_xi,
        B_values=(2,),
        n_signal_bins=4,
        n_phase_bins=3,
    )
    assert summary["meta"]["protocol"] == "C"
    assert summary["meta"]["n_phase1_seeds"] == len(trajectories)
    for signal in SIGNAL_KINDS:
        assert signal in summary["data"]["eps"]
        assert signal in summary["data"]["eps_tilde"]
        assert signal in summary["data"]["eps_ratio"]
        for B in (2,):
            assert str(B) in summary["data"][
                "delta_close_threshold_per_signal_per_B"
            ][signal]
    verdict = summary["verdict"]
    assert verdict["outcome_class"] in {
        "preliminary_positive",
        "honest_negative",
        "inconclusive",
    }
    assert verdict["best_signal"] in SIGNAL_KINDS or verdict["best_signal"] is None


@pytest.mark.unit
def test_protocol_c_pipeline_uncertainty_band_matches_formula() -> None:
    # ensure the sigma_xi · sqrt(B) / sqrt(2) bookkeeping is reflected in the summary
    trajectory = _synthetic_trajectory(seed=1, T=8)
    summary = protocol_c_pipeline(
        phase1_trajectories=[trajectory],
        phase2b_mc_rows_by_seed={},
        delta_open_per_B={2: 1.0},
        sigma_xi_per_B={2: 0.5},
        B_values=(2,),
    )
    band = summary["data"]["uncertainty_band_per_B"]["2"]
    assert band == pytest.approx(0.5 * math.sqrt(2.0) / math.sqrt(2.0))
    assert band == pytest.approx(0.5)
