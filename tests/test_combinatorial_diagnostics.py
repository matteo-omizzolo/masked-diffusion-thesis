from __future__ import annotations

import pytest

from mdm_playground.analysis.combinatorial_diagnostics import (
    build_combinatorial_diagnostics,
    mean_pairwise_jaccard,
    schedule_jaccard,
    variance_decomposition,
)


def test_schedule_jaccard_basic() -> None:
    assert schedule_jaccard([1, 2], [1, 2]) == pytest.approx(1.0)
    assert schedule_jaccard([1, 2], [3, 4]) == pytest.approx(0.0)
    assert schedule_jaccard([1, 2], [2, 3]) == pytest.approx(1.0 / 3.0)


def test_mean_pairwise_jaccard() -> None:
    schedules = [[0, 1], [0, 1], [1, 2]]
    # pairs: (1.0, 1/3, 1/3) -> mean 5/9
    assert mean_pairwise_jaccard(schedules) == pytest.approx(5.0 / 9.0)


def test_variance_decomposition_nonzero_components() -> None:
    mc_rows = [
        {"seed": 1, "B": 2, "G": 0.1, "schedule_steps": [0, 1]},
        {"seed": 1, "B": 2, "G": 0.3, "schedule_steps": [0, 2]},
        {"seed": 2, "B": 2, "G": 0.4, "schedule_steps": [1, 2]},
        {"seed": 2, "B": 2, "G": 0.6, "schedule_steps": [1, 3]},
    ]
    out = variance_decomposition(mc_rows)["per_B"]["2"]
    assert out["var_total_G"] > 0
    assert out["var_between_seed_means"] > 0
    assert out["var_within_seed"] > 0
    assert out["within_share"] + out["between_share"] > 0


def test_build_combinatorial_diagnostics_smoke() -> None:
    mc_rows = [
        {
            "seed": 10,
            "B": 2,
            "mc_idx": 0,
            "G": 0.9,
            "A": 0.0,
            "residual": 0.0,
            "schedule_steps": [1, 2],
        },
        {
            "seed": 10,
            "B": 2,
            "mc_idx": 1,
            "G": 0.8,
            "A": 0.0,
            "residual": 0.0,
            "schedule_steps": [1, 3],
        },
        {
            "seed": 10,
            "B": 2,
            "mc_idx": 2,
            "G": 0.1,
            "A": 0.0,
            "residual": 0.0,
            "schedule_steps": [4, 5],
        },
        {
            "seed": 11,
            "B": 2,
            "mc_idx": 0,
            "G": 0.7,
            "A": 0.0,
            "residual": 0.0,
            "schedule_steps": [2, 3],
        },
        {
            "seed": 11,
            "B": 2,
            "mc_idx": 1,
            "G": 0.6,
            "A": 0.0,
            "residual": 0.0,
            "schedule_steps": [2, 4],
        },
        {
            "seed": 11,
            "B": 2,
            "mc_idx": 2,
            "G": 0.2,
            "A": 0.0,
            "residual": 0.0,
            "schedule_steps": [5, 6],
        },
    ]
    policy_rows = [
        {"seed": 10, "B": 2, "policy": "mean_delta_oracle", "schedule_steps": [1, 2]},
        {"seed": 11, "B": 2, "policy": "mean_delta_oracle", "schedule_steps": [2, 3]},
    ]

    out = build_combinatorial_diagnostics(
        mc_rows=mc_rows,
        policy_rows=policy_rows,
        top_k=2,
        random_baseline_samples=500,
        random_seed=123,
    )
    b2 = out["overlap"]["per_B"]["2"]
    assert b2["n_seeds"] == 2
    assert b2["mean_topk_vs_oracle_jaccard"] > 0
    assert "ratio_topk_vs_random" in b2
    assert "2" in out["variance"]["per_B"]
