from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.proseco.landscape.analyze_schedule_landscape_geometry import (
    a_score_reliability_by_region,
    nearest_top_distance,
    phase_counts,
    schedule_distance,
    top_fraction_rows,
    top_schedule_geometry,
)


def test_top_fraction_rows_uses_at_least_one_and_sorts_by_gain() -> None:
    rows = [
        {"G": 0.1, "schedule_steps": [1, 2]},
        {"G": 0.5, "schedule_steps": [3, 4]},
        {"G": 0.3, "schedule_steps": [5, 6]},
    ]

    top = top_fraction_rows(rows, fraction=0.05)

    assert len(top) == 1
    assert top[0]["G"] == 0.5


def test_top_schedule_geometry_detects_clustered_top_region() -> None:
    rows = [
        {"seed": 1, "B": 2, "G": 1.0, "schedule_steps": [1, 2]},
        {"seed": 1, "B": 2, "G": 0.9, "schedule_steps": [1, 3]},
        {"seed": 1, "B": 2, "G": 0.0, "schedule_steps": [20, 30]},
        {"seed": 1, "B": 2, "G": -0.1, "schedule_steps": [40, 50]},
    ]

    geom = top_schedule_geometry(rows, top_fraction=0.5, rng_seed=0)

    assert geom["B=2"]["top_pair_distance"]["mean"] == 1.0
    assert geom["B=2"]["random_pair_distance"]["mean"] > geom["B=2"]["top_pair_distance"]["mean"]
    assert geom["B=2"]["top_schedules_clustered"] is True


def test_nearest_top_distance_compares_search_and_random_controls() -> None:
    top_rows = [
        {"schedule_steps": [10, 20], "G": 1.0},
        {"schedule_steps": [10, 21], "G": 0.9},
    ]

    assert nearest_top_distance([10, 22], top_rows) == 1
    assert nearest_top_distance([1, 2], top_rows) == 2


def test_phase_counts_uses_project_phase_bins() -> None:
    counts = phase_counts([0, 22, 43, 63])

    assert counts == {"early": 1, "middle": 1, "late": 2}


def test_a_score_reliability_by_region_reports_region_correlations() -> None:
    rows = [
        {"B": 2, "A": 0.0, "G": 0.0},
        {"B": 2, "A": 1.0, "G": 1.0},
        {"B": 2, "A": 2.0, "G": 2.0},
        {"B": 2, "A": 3.0, "G": 3.0},
    ]

    reliability = a_score_reliability_by_region(rows, min_n=2)

    assert reliability["B=2"]["overall"]["spearman_A_G"] == pytest.approx(1.0)
    assert reliability["B=2"]["top_half"]["spearman_A_G"] == pytest.approx(1.0)
    assert reliability["B=2"]["bottom_half"]["spearman_A_G"] == pytest.approx(1.0)
    assert reliability["B=2"]["top_10pct"]["spearman_A_G"] is None
