from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.proseco.landscape.analyze_schedule_neighborhood_diagnostics import compute_aggregate_stats
from scripts.proseco.landscape.run_schedule_neighborhood_diagnostics import (
    canonical_schedule,
    construct_diminishing_return_triples,
    diminishing_return_gap,
    one_swap_neighbors,
    shard_items,
)


def test_schedule_neighbor_generation_has_no_duplicates() -> None:
    neighbors = one_swap_neighbors([1, 3], T=6)

    assert len(neighbors) == len({tuple(n) for n in neighbors})
    assert canonical_schedule([1, 3]) not in {tuple(n) for n in neighbors}


def test_one_swap_neighbors_preserve_budget() -> None:
    neighbors = one_swap_neighbors([1, 3, 5], T=8)

    assert neighbors
    assert all(len(n) == 3 for n in neighbors)
    assert all(len(set(n)) == 3 for n in neighbors)
    assert all(len(set(n).symmetric_difference({1, 3, 5})) == 2 for n in neighbors)


def test_triple_construction_satisfies_subset_and_exclusion_contract() -> None:
    triples = construct_diminishing_return_triples(
        context_schedule=[1, 3, 5],
        T=9,
        anchor_type="cd_g",
        seed=42,
        budget=3,
        max_triples=6,
        rng_seed=7,
    )

    assert triples
    for triple in triples:
        a_set = set(triple["A_steps"])
        b_set = set(triple["B_steps"])
        x = int(triple["x_step"])
        assert a_set < b_set
        assert x not in b_set
        assert set(triple["A_plus_x_steps"]) == a_set | {x}
        assert set(triple["B_plus_x_steps"]) == b_set | {x}


def test_diminishing_return_gap_formula() -> None:
    gap = diminishing_return_gap(g_A=1.0, g_B=1.5, g_A_plus_x=1.4, g_B_plus_x=1.7)

    assert gap == pytest.approx(0.2)


def test_sharding_partitions_rows_without_overlap() -> None:
    rows = [{"row_id": i} for i in range(17)]
    shards = [shard_items(rows, shard_idx=i, shard_count=4) for i in range(4)]
    flattened = [row["row_id"] for shard in shards for row in shard]

    assert sorted(flattened) == list(range(17))
    assert len(flattened) == len(set(flattened))


def test_analyzer_handles_empty_optional_groups_honestly() -> None:
    stats = compute_aggregate_stats(
        neighborhood_rows=[
            {
                "seed": 1,
                "B": 2,
                "diagnostic_type": "neighborhood",
                "anchor_type": "cd_g",
                "schedule_steps": [1, 2],
                "anchor_schedule_steps": [1, 2],
                "G": 1.0,
                "anchor_G": 1.0,
                "A": 0.8,
                "F_base": 0.0,
                "F_schedule": 1.0,
                "n_g_calls": 1,
                "neighbor_relation": "anchor",
                "phase_counts": {"early": 2},
                "git_sha": "abc123",
                "shard_id": "shard0-of-1",
                "anchor_id": "1-2-cd_g",
            }
        ],
        triple_rows=[],
        known_anchor_types=("cd_g", "bs_ag"),
    )

    assert stats["local_optimality_by_anchor"]["cd_g"]["n_anchors"] == 1
    assert stats["local_optimality_by_anchor"]["bs_ag"]["status"] == "no_rows"
    assert stats["diminishing_returns"]["status"] == "no_triples"


def test_json_outputs_are_standard_serializable() -> None:
    stats = compute_aggregate_stats(neighborhood_rows=[], triple_rows=[])

    json.dumps(stats)
