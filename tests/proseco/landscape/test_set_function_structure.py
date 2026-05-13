from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.proseco.landscape.analyze_set_function_structure import (
    canonical_schedule,
    compute_bo_suitability,
    compute_higher_order_residuals,
    compute_local_smoothness,
    compute_monotonicity_and_diminishing_returns,
    compute_search_vs_mc,
    is_subset_schedule,
    schedule_distance,
    summarize_distribution,
)


def test_schedule_distance_counts_swaps() -> None:
    assert schedule_distance([1, 2, 3], [1, 2, 4]) == 1
    assert schedule_distance([1, 2, 3], [4, 5, 6]) == 3


def test_canonical_schedule_sorts_and_deduplicates() -> None:
    assert canonical_schedule([4, 1, 4, 2]) == (1, 2, 4)


def test_is_subset_schedule() -> None:
    assert is_subset_schedule([1, 2], [1, 2, 3])
    assert not is_subset_schedule([1, 4], [1, 2, 3])


def test_summarize_distribution_json_native() -> None:
    summary = summarize_distribution([1.0, 2.0, 3.0])
    assert summary["n"] == 3
    assert isinstance(summary["mean"], float)
    assert isinstance(summary["median"], float)


def test_local_smoothness_uses_actual_schedule_distances() -> None:
    rows = [
        {"seed": 1, "B": 2, "schedule_steps": [1, 2], "G": 0.0},
        {"seed": 1, "B": 2, "schedule_steps": [1, 3], "G": 1.0},
        {"seed": 1, "B": 2, "schedule_steps": [4, 5], "G": 4.0},
    ]

    smoothness = compute_local_smoothness(rows, max_pair_comparisons=100)

    assert smoothness["by_distance"]["1"]["n"] == 1
    assert smoothness["by_distance"]["1"]["mean"] == 1.0
    assert smoothness["by_distance"]["2"]["n"] == 2


def test_search_vs_mc_reports_empirical_quantiles_by_budget() -> None:
    mc_rows = [
        {"seed": 1, "B": 2, "schedule_steps": [1, 2], "G": 0.1},
        {"seed": 1, "B": 2, "schedule_steps": [2, 3], "G": 0.3},
        {"seed": 1, "B": 2, "schedule_steps": [3, 4], "G": 0.5},
    ]
    cd_rows = [{"seed": 1, "B": 2, "G_init": 0.1, "G_final": 0.5, "schedule_final": [3, 4], "n_g_calls": 4}]
    bs_rows = [{"seed": 1, "B": 2, "G_final": 0.3, "schedule_final": [2, 3], "n_g_calls": 3}]

    search = compute_search_vs_mc(mc_rows, cd_rows, bs_rows)

    assert search["B=2"]["cd_g_quantile_mean"] == 1.0
    assert search["B=2"]["bs_ag_quantile_mean"] == 2 / 3
    assert search["B=2"]["cd_g_call_mean"] == 4.0


def test_higher_order_residuals_excludes_additive_and_non_deployable_models() -> None:
    rows = [
        {"B": 3, "model": "additive", "non_deployable": False, "abs_err_A": 1.0, "abs_err_Q": 1.0},
        {
            "B": 3,
            "model": "phase_distance_mean",
            "non_deployable": False,
            "abs_err_A": 1.0,
            "abs_err_Q": 2.0,
            "pair_xi_hat_sum": -0.5,
            "residual_G_minus_A": -0.25,
        },
        {
            "B": 3,
            "model": "observed_pair_oracle",
            "non_deployable": True,
            "abs_err_A": 1.0,
            "abs_err_Q": 0.1,
            "pair_xi_hat_sum": 0.2,
            "residual_G_minus_A": 0.1,
        },
    ]

    residuals = compute_higher_order_residuals(rows)

    assert set(residuals["B=3"]) == {"phase_distance_mean"}
    cell = residuals["B=3"]["phase_distance_mean"]
    assert cell["mean_abs_err_A"] == 1.0
    assert cell["mean_abs_err_Q"] == 2.0
    assert cell["q_worse_fraction"] == 1.0
    assert cell["mean_pair_penalty"] == -0.5


def test_monotonicity_and_diminishing_returns_require_exact_comparisons() -> None:
    rows = [
        {"seed": 1, "schedule_steps": [1], "G": 1.0},
        {"seed": 1, "schedule_steps": [1, 2], "G": 1.5},
        {"seed": 1, "schedule_steps": [1, 3], "G": 2.0},
        {"seed": 1, "schedule_steps": [1, 2, 3], "G": 2.1},
    ]

    diag = compute_monotonicity_and_diminishing_returns(rows)

    assert diag["monotonicity"]["exact_inclusion_comparisons"] == 5
    assert diag["monotonicity"]["fraction_nonnegative"] == 1.0
    assert diag["diminishing_returns"]["status"] == "identified"
    assert diag["diminishing_returns"]["exact_triples"] == 2


def test_bo_suitability_is_unclear_when_distance_correlation_is_weak() -> None:
    local = {
        "locally_smooth_proxy": True,
        "spearman_distance_abs_g_gap": 0.05,
        "one_swap_mean_abs_gap": 0.1,
        "larger_distance_mean_abs_gap": 0.2,
    }
    search = {"B=2": {"cd_g_quantile_mean": 0.95, "bs_ag_quantile_mean": 0.9}}
    residuals = {"B=3": {"phase_distance_mean": {"q_worse_fraction": 0.8}}}

    bo = compute_bo_suitability(local, search, residuals)

    assert bo["verdict"] == "unclear"
    assert any("kernel" in gap for gap in bo["blocking_gaps"])
