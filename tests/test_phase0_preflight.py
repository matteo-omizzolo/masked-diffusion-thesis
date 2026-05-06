"""Phase 0 pre-flight invariants for restarting corrector-timing experiments.

These tests are intentionally CPU-only. They validate local scheduling and
signal semantics, and mark checkpoint-backed ProSeCo equivalence checks as
explicit integration blockers rather than pretending the hooks already exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdm_playground.scheduling.allocation import allocate_budget
from mdm_playground.scheduling.evaluate import estimate_single_step_gain
from mdm_playground.scheduling.signals import compute_signals
from mdm_playground.scheduling.surrogate import SurrogateGenerator


def _extra_forward_passes(allocation: dict[int, int], c_corr: int) -> int:
    return c_corr * sum(allocation.values())


def test_pf1_deterministic_base_surrogate() -> None:
    """PF1: same seed/config gives same base tokens and F score."""
    gen = SurrogateGenerator(T=16, D=24, seed_base=11)

    first = gen.run_base(seed=5)
    second = gen.run_base(seed=5)

    np.testing.assert_array_equal(first["tokens"], second["tokens"])
    assert first["neg_nll"] == pytest.approx(second["neg_nll"])
    assert first["per_step_signals"] == second["per_step_signals"]


def test_pf2_empty_schedule_equals_base_surrogate() -> None:
    """PF2: an empty schedule is exactly the base trajectory in the surrogate."""
    gen = SurrogateGenerator(T=16, D=24, seed_base=13)

    base = gen.run_base(seed=3)
    scheduled = gen.run_with_schedule({}, seed=3)

    np.testing.assert_array_equal(scheduled["tokens"], base["tokens"])
    assert scheduled["neg_nll"] == pytest.approx(base["neg_nll"])
    assert scheduled["schedule_steps"] == []


def test_pf3_singleton_schedule_matches_protocol_a_score_surrogate() -> None:
    """PF3 partial: singleton schedule score equals the Protocol A branch score."""
    gen = SurrogateGenerator(T=16, D=24, seed_base=17, gamma=0.0)
    t = 6

    branch = gen.run_branch(t, seed=2)
    scheduled = gen.run_with_schedule({t: 1}, seed=2)

    assert scheduled["neg_nll"] == pytest.approx(branch["neg_nll"])
    assert scheduled["G_true"] == pytest.approx(branch["delta_true"])


def test_pf4_budget_accounting_binary_schedules() -> None:
    """PF4: |S| = B and same-cardinality schedules have equal corrector cost."""
    signal = np.linspace(0.0, 1.0, 12)
    budget = 4
    c_corr = 3

    top = allocate_budget(signal, budget=budget, policy_name="top_B")
    uniform = allocate_budget(signal, budget=budget, policy_name="uniform")
    random = allocate_budget(
        signal, budget=budget, policy_name="random", policy_kwargs={"seed": 9}
    )

    for allocation in (top, uniform, random):
        assert len(allocation) == budget
        assert sum(allocation.values()) == budget
        assert set(allocation.values()) == {1}
        assert _extra_forward_passes(allocation, c_corr) == c_corr * budget

    assert _extra_forward_passes(top, c_corr) == _extra_forward_passes(uniform, c_corr)
    assert _extra_forward_passes(top, c_corr) == _extra_forward_passes(random, c_corr)


def test_pf5_crn_surrogate_base_trace_is_shared() -> None:
    """PF5 partial: surrogate branches share the base signal trace by seed."""
    gen = SurrogateGenerator(T=16, D=24, seed_base=19)

    base = gen.run_base(seed=4)
    branch_a = gen.run_branch(5, seed=4)
    branch_b = gen.run_branch(9, seed=4)

    assert branch_a["per_step_signals"] == base["per_step_signals"]
    assert branch_b["per_step_signals"] == base["per_step_signals"]


def test_pf6_f_scoring_consistency_same_tokens_same_score() -> None:
    """PF6: deterministic F gives identical scores on identical token sequences."""
    tokens = np.array([4, 1, 7, 7, 2])
    y = {"tokens": tokens.copy()}

    def score(row: dict) -> float:
        return float(np.mean(row["tokens"]))

    first = estimate_single_step_gain(y, y, F=score)
    second = estimate_single_step_gain({"tokens": tokens.copy()}, y, F=score)

    assert first["f_base"] == pytest.approx(second["f_base"])
    assert first["f_branch"] == pytest.approx(second["f_branch"])
    assert first["delta"] == pytest.approx(0.0)
    assert second["delta"] == pytest.approx(0.0)


def test_pf7_pf8_signals_use_explicit_revisable_action_set() -> None:
    """PF7/PF8 local: H, inverse margin, and QM are computed on R_t only."""
    state = {
        "tokens": np.array([10, 11, 12, 13]),
        "mask_id": -1,
        "revisable_mask": np.array([True, False, True, False]),
    }
    logits = np.array(
        [
            [3.0, 1.0, 0.0],
            [100.0, -100.0, -100.0],
            [0.5, 0.5, 0.5],
            [-100.0, 100.0, -100.0],
        ]
    )
    quality_scores = np.array([0.2, 1.0, 0.6, 1.0])

    signals = compute_signals(
        state, logits, meta={"quality_scores": quality_scores}
    )
    expected = compute_signals(
        {"tokens": np.array([10, 12]), "mask_id": -1},
        logits[[0, 2]],
        meta={"quality_scores": quality_scores[[0, 2]]},
    )

    assert signals["n_revisable"] == 2
    assert signals["unmasked_fraction"] == pytest.approx(0.5)
    assert signals["entropy"] == pytest.approx(expected["entropy"])
    assert signals["inverse_margin"] == pytest.approx(expected["inverse_margin"])
    assert signals["quality_mass_proxy"] == pytest.approx(
        expected["quality_mass_proxy"]
    )


@pytest.mark.skip(
    reason=(
        "PF3 real ProSeCo token-level branch equivalence needs a staged checkpoint "
        "and a public run_branch/run_with_schedule comparison hook."
    )
)
def test_pf3_real_proseco_single_correction_equivalence_pending() -> None:
    pass


@pytest.mark.skip(
    reason=(
        "PF5 real CRN trace audit needs backend instrumentation exposing predictor "
        "and corrector RNG streams."
    )
)
def test_pf5_real_proseco_crn_trace_pending() -> None:
    pass


@pytest.mark.skip(
    reason=(
        "PF7 real ProSeCo action-set audit needs checkpoint-backed corrector "
        "instrumentation returning the acted-on R_t positions."
    )
)
def test_pf7_real_proseco_action_set_pending() -> None:
    pass
