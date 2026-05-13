#!/usr/bin/env python3
"""Set-function structure diagnostics for corrector schedules.

This script is a local analysis pass over existing artifacts. It does not
launch experiments or fit a new scheduler.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None  # type: ignore[assignment]


def canonical_schedule(steps: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(set(int(s) for s in steps)))


def schedule_distance(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> int:
    sa = set(canonical_schedule(a))
    sb = set(canonical_schedule(b))
    return len(sa.symmetric_difference(sb)) // 2


def is_subset_schedule(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> bool:
    return set(canonical_schedule(a)).issubset(set(canonical_schedule(b)))


def summarize_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    return pearson_corr(_rankdata(x), _rankdata(y))


def _schedule_key(row: dict[str, Any]) -> tuple[int, tuple[int, ...], float]:
    return (
        int(row.get("seed", -1)),
        canonical_schedule(row["schedule_steps"]),
        float(row["G"]),
    )


def group_rows_by_seed_budget(rows: Sequence[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), int(row["B"]))].append(row)
    for key in grouped:
        grouped[key].sort(key=_schedule_key)
    return dict(sorted(grouped.items()))


def compute_local_smoothness(
    mc_rows: Sequence[dict[str, Any]],
    *,
    max_pair_comparisons: int,
) -> dict[str, Any]:
    """Compare |G(S)-G(T)| by actual fixed-budget schedule distance."""
    pair_records: list[tuple[int, float]] = []
    total_candidate_pairs = 0
    for _, rows in group_rows_by_seed_budget(mc_rows).items():
        total_candidate_pairs += len(rows) * (len(rows) - 1) // 2
        for left_idx, right_idx in itertools.combinations(range(len(rows)), 2):
            left = rows[left_idx]
            right = rows[right_idx]
            dist = schedule_distance(left["schedule_steps"], right["schedule_steps"])
            if dist == 0:
                continue
            gap = abs(float(left["G"]) - float(right["G"]))
            pair_records.append((dist, gap))

    sampled = False
    if len(pair_records) > max_pair_comparisons:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(pair_records), size=int(max_pair_comparisons), replace=False)
        pair_records = [pair_records[int(i)] for i in sorted(indices)]
        sampled = True

    by_distance_values: dict[int, list[float]] = defaultdict(list)
    distances: list[float] = []
    gaps: list[float] = []
    for dist, gap in pair_records:
        by_distance_values[int(dist)].append(float(gap))
        distances.append(float(dist))
        gaps.append(float(gap))

    by_distance = {
        str(dist): summarize_distribution(vals)
        for dist, vals in sorted(by_distance_values.items())
    }
    one_swap_mean = by_distance.get("1", {}).get("mean")
    larger_vals = [gap for dist, gap in pair_records if dist > 1]
    larger_mean = _mean(larger_vals)
    rho = spearman_corr(distances, gaps)
    locally_smooth = (
        one_swap_mean is not None
        and larger_mean is not None
        and float(one_swap_mean) < float(larger_mean)
        and (rho is None or rho >= 0.0)
    )
    if locally_smooth:
        interpretation = "one-swap neighbors have smaller |G| gaps than more distant schedules"
    elif one_swap_mean is None:
        interpretation = "no one-swap comparisons are available in the current MC pool"
    else:
        interpretation = "one-swap schedules are not clearly closer in G than more distant schedules"

    return {
        "total_candidate_pairs": int(total_candidate_pairs),
        "n_compared_pairs": int(len(pair_records)),
        "sampled": sampled,
        "max_pair_comparisons": int(max_pair_comparisons),
        "by_distance": by_distance,
        "spearman_distance_abs_g_gap": rho,
        "one_swap_mean_abs_gap": one_swap_mean,
        "larger_distance_mean_abs_gap": larger_mean,
        "locally_smooth_proxy": bool(locally_smooth),
        "interpretation": interpretation,
    }


def _empirical_quantile(values: Sequence[float], score: float) -> float:
    if not values:
        return float("nan")
    return float(sum(float(v) <= float(score) for v in values) / len(values))


def _nearest_high_mc_distance(
    mc_rows: Sequence[dict[str, Any]],
    schedule: Sequence[int],
    *,
    quantile: float = 0.9,
) -> int | None:
    if not mc_rows:
        return None
    gs = np.asarray([float(row["G"]) for row in mc_rows], dtype=float)
    threshold = float(np.quantile(gs, quantile))
    high_rows = [row for row in mc_rows if float(row["G"]) >= threshold]
    if not high_rows:
        return None
    return int(min(schedule_distance(schedule, row["schedule_steps"]) for row in high_rows))


def compute_search_vs_mc(
    mc_rows: Sequence[dict[str, Any]],
    cd_rows: Sequence[dict[str, Any]],
    bs_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    mc_by_key = group_rows_by_seed_budget(mc_rows)
    by_budget: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in sorted(cd_rows, key=lambda r: (int(r["B"]), int(r["seed"]))):
        key = (int(row["seed"]), int(row["B"]))
        pool = mc_by_key.get(key, [])
        if not pool:
            continue
        mc_g = [float(r["G"]) for r in pool]
        budget = int(row["B"])
        g_final = float(row["G_final"])
        by_budget[budget]["cd_quantiles"].append(_empirical_quantile(mc_g, g_final))
        by_budget[budget]["cd_gains"].append(g_final - float(row["G_init"]))
        by_budget[budget]["cd_g_calls"].append(float(row["n_g_calls"]))
        nearest = _nearest_high_mc_distance(pool, row["schedule_final"])
        if nearest is not None:
            by_budget[budget]["cd_nearest_high_distance"].append(float(nearest))

    for row in sorted(bs_rows, key=lambda r: (int(r["B"]), int(r["seed"]))):
        key = (int(row["seed"]), int(row["B"]))
        pool = mc_by_key.get(key, [])
        if not pool:
            continue
        mc_g = [float(r["G"]) for r in pool]
        budget = int(row["B"])
        g_final = float(row["G_final"])
        by_budget[budget]["bs_quantiles"].append(_empirical_quantile(mc_g, g_final))
        by_budget[budget]["bs_g_calls"].append(float(row["n_g_calls"]))
        nearest = _nearest_high_mc_distance(pool, row["schedule_final"])
        if nearest is not None:
            by_budget[budget]["bs_nearest_high_distance"].append(float(nearest))

    out: dict[str, Any] = {}
    for budget in sorted(by_budget):
        vals = by_budget[budget]
        cd_gain_call_corr = pearson_corr(vals.get("cd_g_calls", []), vals.get("cd_gains", []))
        out[f"B={budget}"] = {
            "n_cd": int(len(vals.get("cd_quantiles", []))),
            "n_bs": int(len(vals.get("bs_quantiles", []))),
            "cd_g_quantile": summarize_distribution(list(vals.get("cd_quantiles", []))),
            "bs_ag_quantile": summarize_distribution(list(vals.get("bs_quantiles", []))),
            "cd_g_quantile_mean": _mean(vals.get("cd_quantiles", [])),
            "bs_ag_quantile_mean": _mean(vals.get("bs_quantiles", [])),
            "cd_gain": summarize_distribution(list(vals.get("cd_gains", []))),
            "cd_g_call_mean": _mean(vals.get("cd_g_calls", [])),
            "bs_ag_g_call_mean": _mean(vals.get("bs_g_calls", [])),
            "cd_gain_vs_g_calls_pearson": cd_gain_call_corr,
            "cd_nearest_high_mc_distance": summarize_distribution(list(vals.get("cd_nearest_high_distance", []))),
            "bs_ag_nearest_high_mc_distance": summarize_distribution(list(vals.get("bs_nearest_high_distance", []))),
        }
    return out


def compute_higher_order_residuals(gate3b_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gate3b_rows:
        model = str(row.get("model", ""))
        if model == "additive" or bool(row.get("non_deployable", False)):
            continue
        if int(row.get("B", -1)) not in {3, 4}:
            continue
        groups[(int(row["B"]), model)].append(row)

    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for (budget, model), rows in sorted(groups.items()):
        abs_a = [float(row["abs_err_A"]) for row in rows]
        abs_q = [float(row["abs_err_Q"]) for row in rows]
        pair_penalty = [float(row.get("pair_xi_hat_sum", 0.0)) for row in rows]
        residual_a = [float(row.get("residual_G_minus_A", 0.0)) for row in rows]
        residual_q = [float(row.get("residual_G_minus_Q", 0.0)) for row in rows]
        out[f"B={budget}"][model] = {
            "n": int(len(rows)),
            "mean_abs_err_A": float(np.mean(abs_a)),
            "mean_abs_err_Q": float(np.mean(abs_q)),
            "q_worse_fraction": float(np.mean(np.asarray(abs_q) > np.asarray(abs_a))),
            "mean_pair_penalty": float(np.mean(pair_penalty)),
            "median_pair_penalty": float(np.median(pair_penalty)),
            "fraction_negative_pair_penalty": float(np.mean(np.asarray(pair_penalty) < 0.0)),
            "mean_residual_G_minus_A": float(np.mean(residual_a)),
            "mean_residual_G_minus_Q": float(np.mean(residual_q)),
            "pair_penalty_vs_residual_A_pearson": pearson_corr(pair_penalty, residual_a),
            "residual_Q": summarize_distribution(residual_q),
        }
    return dict(out)


def _aggregate_seed_schedules(rows: Sequence[dict[str, Any]]) -> dict[int, dict[tuple[int, ...], float]]:
    grouped_values: dict[int, dict[tuple[int, ...], list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped_values[int(row["seed"])][canonical_schedule(row["schedule_steps"])].append(float(row["G"]))
    out: dict[int, dict[tuple[int, ...], float]] = {}
    for seed, schedule_map in grouped_values.items():
        out[int(seed)] = {
            schedule: float(np.mean(vals))
            for schedule, vals in sorted(schedule_map.items(), key=lambda item: (len(item[0]), item[0]))
        }
    return dict(sorted(out.items()))


def compute_monotonicity_and_diminishing_returns(
    rows: Sequence[dict[str, Any]],
    *,
    cd_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_seed = _aggregate_seed_schedules(rows)
    mono_gaps: list[float] = []
    dr_gaps: list[float] = []

    for schedule_map in by_seed.values():
        schedules = list(schedule_map)
        schedule_sets = {schedule: set(schedule) for schedule in schedules}
        for s in schedules:
            s_set = schedule_sets[s]
            for t in schedules:
                if len(t) <= len(s):
                    continue
                if s_set.issubset(schedule_sets[t]):
                    mono_gaps.append(float(schedule_map[t] - schedule_map[s]))

        all_steps = sorted({step for schedule in schedules for step in schedule})
        for context_large in schedules:
            large_set = schedule_sets[context_large]
            for x in all_steps:
                if x in large_set:
                    continue
                large_plus = tuple(sorted((*context_large, x)))
                if large_plus not in schedule_map:
                    continue
                for context_small in schedules:
                    small_set = schedule_sets[context_small]
                    if len(context_small) >= len(context_large):
                        continue
                    if x in small_set or not small_set.issubset(large_set):
                        continue
                    small_plus = tuple(sorted((*context_small, x)))
                    if small_plus not in schedule_map:
                        continue
                    small_marginal = schedule_map[small_plus] - schedule_map[context_small]
                    large_marginal = schedule_map[large_plus] - schedule_map[context_large]
                    dr_gaps.append(float(small_marginal - large_marginal))

    cd_gains = [float(row["G_final"]) - float(row["G_init"]) for row in (cd_rows or [])]
    mono_status = "identified" if mono_gaps else "not_identifiable_from_current_artifacts"
    dr_status = "identified" if dr_gaps else "not_identifiable_from_current_artifacts"
    return {
        "monotonicity": {
            "status": mono_status,
            "exact_inclusion_comparisons": int(len(mono_gaps)),
            "fraction_nonnegative": float(np.mean(np.asarray(mono_gaps) >= 0.0)) if mono_gaps else None,
            "mono_gap": summarize_distribution(mono_gaps),
            "cd_gain": summarize_distribution(cd_gains),
            "cd_gain_fraction_nonnegative": float(np.mean(np.asarray(cd_gains) >= 0.0)) if cd_gains else None,
        },
        "diminishing_returns": {
            "status": dr_status,
            "exact_triples": int(len(dr_gaps)),
            "fraction_dr_gap_nonnegative": float(np.mean(np.asarray(dr_gaps) >= 0.0)) if dr_gaps else None,
            "dr_gap": summarize_distribution(dr_gaps),
            "fallback_evidence": [
                "Gate 3a reported mostly negative pair residuals in active docs.",
                "Gate 3b B=3/4 showed deployable pairwise Q over-composes and increases held-out error.",
            ],
        },
    }


def compute_bo_suitability(
    local_smoothness: dict[str, Any],
    search_vs_mc: dict[str, Any],
    higher_order_residuals: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    gaps: list[str] = []
    local_positive = bool(local_smoothness.get("locally_smooth_proxy", False))
    distance_rho = local_smoothness.get("spearman_distance_abs_g_gap")
    one_swap_mean = local_smoothness.get("one_swap_mean_abs_gap")
    larger_mean = local_smoothness.get("larger_distance_mean_abs_gap")
    strong_distance_signal = (
        distance_rho is not None
        and float(distance_rho) >= 0.15
        and one_swap_mean is not None
        and larger_mean is not None
        and float(one_swap_mean) <= 0.85 * float(larger_mean)
    )
    if local_positive:
        reasons.append("one-swap schedules are closer in G than more distant schedules in the MC pool")
        if not strong_distance_signal:
            gaps.append("distance signal is positive but weak; a BO kernel/acquisition story is not yet identified")
    else:
        gaps.append("local smoothness is weak or not identifiable from the MC pool")

    quantile_means: list[float] = []
    for cell in search_vs_mc.values():
        for key in ("cd_g_quantile_mean", "bs_ag_quantile_mean"):
            value = cell.get(key)
            if value is not None:
                quantile_means.append(float(value))
    mean_search_quantile = float(np.mean(quantile_means)) if quantile_means else None
    if mean_search_quantile is not None and mean_search_quantile >= 0.75:
        reasons.append("CD-G / BS-AG finals land in high empirical MC-pool quantiles")
    elif mean_search_quantile is not None:
        gaps.append("search finals are not consistently in high MC-pool quantiles")
    else:
        gaps.append("search-vs-MC quantiles are unavailable")

    q_worse: list[float] = []
    for budget_cell in higher_order_residuals.values():
        for model_cell in budget_cell.values():
            q_worse.append(float(model_cell["q_worse_fraction"]))
    if q_worse and float(np.mean(q_worse)) > 0.5:
        reasons.append("tested pairwise-Q residuals often worsen held-out error, favoring black-box/search diagnostics over pairwise composition")

    if strong_distance_signal and mean_search_quantile is not None and mean_search_quantile >= 0.75:
        verdict = "promising"
        gaps.append("BO still needs a separate acquisition/evaluation-budget spec before any scheduler implementation")
    elif mean_search_quantile is not None and mean_search_quantile >= 0.75:
        verdict = "unclear"
        gaps.append("search works, but the current diagnostics do not isolate a smooth surrogate kernel for BO")
    else:
        verdict = "not_supported"
        gaps.append("current artifacts do not show enough structure for Bayesian schedule optimization")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "blocking_gaps": gaps,
        "mean_search_mc_quantile": mean_search_quantile,
    }


def _best_deployable_model_by_budget(higher_order_residuals: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for budget, models in higher_order_residuals.items():
        if not models:
            continue
        best = min(models.items(), key=lambda item: float(item[1]["mean_abs_err_Q"]))
        out[budget] = best[0]
    return out


def choose_recommended_direction(
    local_smoothness: dict[str, Any],
    search_vs_mc: dict[str, Any],
    bo_suitability: dict[str, Any],
) -> dict[str, Any]:
    quantiles = [
        float(cell[key])
        for cell in search_vs_mc.values()
        for key in ("cd_g_quantile_mean", "bs_ag_quantile_mean")
        if cell.get(key) is not None
    ]
    search_supported = bool(quantiles) and float(np.mean(quantiles)) >= 0.7
    if search_supported:
        direction = "search_based_scheduling_spec"
        rationale = "existing true-G search methods consistently reach high MC-pool quantiles; direct pairwise Q remains blocked"
    elif bo_suitability.get("verdict") == "promising":
        direction = "bayesian_schedule_optimization_spec"
        rationale = "local smoothness and high search quantiles support an uncertainty-aware black-box view"
    else:
        direction = "conservative_higher_order_black_box_writeup"
        rationale = "current artifacts do not identify enough smooth or weak-submodular structure for a new scheduler"
    return {
        "direction": direction,
        "rationale": rationale,
        "online_value_function_approximation": "premature_without_state-conditioned evidence beyond open-loop schedule artifacts",
        "direct_pairwise_scheduler": "blocked_by_Gate_3b_B3_B4_failure",
        "new_scheduler_implemented": False,
    }


def _require_list(path: Path, payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [dict(row) for row in payload]


def _validate_output_dir(out_dir: Path) -> None:
    if not out_dir.name.startswith("set_function_structure_"):
        raise ValueError("out_dir must be a SHA-tagged results/set_function_structure_<gitsha> directory")
    if out_dir.parent.name != "results":
        raise ValueError("out_dir must live directly under results/")


def _save_current_figure(figures_dir: Path, stem: str) -> None:
    if not HAS_MPL:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    assert plt is not None
    plt.tight_layout()
    plt.savefig(figures_dir / f"{stem}.png", dpi=160)
    plt.savefig(figures_dir / f"{stem}.pdf")
    plt.close()


def _plot_text(stem: str, figures_dir: Path, title: str, lines: Sequence[str]) -> None:
    if not HAS_MPL:
        return
    assert plt is not None
    plt.figure(figsize=(8, 4.5))
    plt.axis("off")
    plt.title(title, loc="left", fontsize=12)
    plt.text(0.02, 0.88, "\n".join(lines), va="top", ha="left", fontsize=10, wrap=True)
    _save_current_figure(figures_dir, stem)


def plot_monotonicity(diag: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        return
    assert plt is not None
    mono = diag["monotonicity"]
    cd = mono["cd_gain"]
    values = [
        mono["fraction_nonnegative"] if mono["fraction_nonnegative"] is not None else 0.0,
        mono["cd_gain_fraction_nonnegative"] if mono["cd_gain_fraction_nonnegative"] is not None else 0.0,
    ]
    labels = ["exact inclusion", "CD-G trace"]
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values, color=["#4C78A8", "#F58518"])
    plt.ylim(0, 1.05)
    plt.ylabel("fraction nonnegative")
    plt.title("Monotonicity proxies")
    plt.text(
        0.02,
        0.94,
        f"exact n={mono['exact_inclusion_comparisons']} | CD-G n={cd['n']}",
        transform=plt.gca().transAxes,
        va="top",
    )
    _save_current_figure(figures_dir, "monotonicity_proxy")


def plot_diminishing_returns(diag: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        return
    dr = diag["diminishing_returns"]
    if dr["dr_gap"]["n"] == 0:
        _plot_text(
            "diminishing_returns_proxy",
            figures_dir,
            "Diminishing-returns proxy",
            [
                "No exact inclusion triples were available.",
                "Submodularity ratio / curvature is not identifiable from current artifacts.",
                *dr["fallback_evidence"],
            ],
        )
        return
    assert plt is not None
    summary = dr["dr_gap"]
    plt.figure(figsize=(7, 4.5))
    plt.bar(["mean", "median", "p05", "p95"], [summary[k] for k in ("mean", "median", "p05", "p95")], color="#54A24B")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("DR gap")
    plt.title(f"Diminishing-returns exact triples (n={dr['exact_triples']})")
    _save_current_figure(figures_dir, "diminishing_returns_proxy")


def plot_local_smoothness(local_smoothness: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        return
    assert plt is not None
    by_distance = local_smoothness["by_distance"]
    if not by_distance:
        _plot_text("local_smoothness", figures_dir, "Local smoothness", ["No schedule-pair comparisons available."])
        return
    distances = sorted(by_distance, key=lambda d: int(d))
    means = [by_distance[d]["mean"] for d in distances]
    counts = [by_distance[d]["n"] for d in distances]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(distances, means, color="#4C78A8")
    plt.xlabel("schedule distance (one swap = 1)")
    plt.ylabel("mean |G(S)-G(T)|")
    plt.title("Local smoothness by actual schedule distance")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom", fontsize=8)
    _save_current_figure(figures_dir, "local_smoothness")


def plot_search_vs_mc(search_vs_mc: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        return
    assert plt is not None
    budgets = sorted(search_vs_mc, key=lambda b: int(b.split("=")[1]))
    cd = [search_vs_mc[b]["cd_g_quantile_mean"] or 0.0 for b in budgets]
    bs = [search_vs_mc[b]["bs_ag_quantile_mean"] or 0.0 for b in budgets]
    x = np.arange(len(budgets))
    width = 0.36
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, cd, width, label="CD-G", color="#F58518")
    plt.bar(x + width / 2, bs, width, label="BS-AG", color="#54A24B")
    plt.xticks(x, budgets)
    plt.ylim(0, 1.05)
    plt.ylabel("empirical MC-pool quantile")
    plt.title("Search finals relative to MC schedule pool")
    plt.legend()
    _save_current_figure(figures_dir, "search_vs_mc_schedule_quality")


def plot_higher_order_residuals(higher_order: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        return
    assert plt is not None
    labels: list[str] = []
    err_a: list[float] = []
    err_q: list[float] = []
    for budget in sorted(higher_order, key=lambda b: int(b.split("=")[1])):
        for model in sorted(higher_order[budget]):
            labels.append(f"{budget}\n{model}")
            err_a.append(float(higher_order[budget][model]["mean_abs_err_A"]))
            err_q.append(float(higher_order[budget][model]["mean_abs_err_Q"]))
    if not labels:
        _plot_text("higher_order_residuals", figures_dir, "Higher-order residuals", ["No deployable Q rows available."])
        return
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(max(8, len(labels) * 1.1), 4.8))
    plt.bar(x - width / 2, err_a, width, label="|G-A|", color="#4C78A8")
    plt.bar(x + width / 2, err_q, width, label="|G-Q|", color="#E45756")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("mean absolute error")
    plt.title("Deployable pairwise-Q residuals")
    plt.legend()
    _save_current_figure(figures_dir, "higher_order_residuals")


def plot_bo_suitability(bo: dict[str, Any], figures_dir: Path) -> None:
    lines = [
        f"Verdict: {bo['verdict']}",
        f"Mean search MC quantile: {bo['mean_search_mc_quantile']}",
        "Reasons:",
        *(f"- {reason}" for reason in bo["reasons"]),
        "Blocking gaps:",
        *(f"- {gap}" for gap in bo["blocking_gaps"]),
    ]
    _plot_text("bayesian_optimization_suitability", figures_dir, "Bayesian optimization suitability", lines)


def write_interpretation(path: Path, diagnostics: dict[str, Any]) -> None:
    local = diagnostics["local_smoothness"]
    dr = diagnostics["diminishing_returns"]
    higher = diagnostics["higher_order_residuals"]
    search = diagnostics["search_vs_mc"]
    bo = diagnostics["bo_suitability"]
    rec = diagnostics["recommended_next_direction"]
    best_q = _best_deployable_model_by_budget(higher)

    search_lines = []
    for budget in sorted(search, key=lambda b: int(b.split("=")[1])):
        cell = search[budget]
        search_lines.append(
            f"- {budget}: CD-G mean MC quantile {cell['cd_g_quantile_mean']:.3f}; "
            f"BS-AG mean MC quantile {cell['bs_ag_quantile_mean']:.3f}."
        )

    q_lines = []
    for budget in sorted(best_q, key=lambda b: int(b.split("=")[1])):
        model = best_q[budget]
        cell = higher[budget][model]
        q_lines.append(
            f"- {budget}, best deployable Q by mean |G-Q| here is `{model}`: "
            f"|G-A|={cell['mean_abs_err_A']:.4f}, |G-Q|={cell['mean_abs_err_Q']:.4f}, "
            f"Q worse fraction={cell['q_worse_fraction']:.3f}, mean pair penalty={cell['mean_pair_penalty']:.4f}."
        )

    text = f"""# Set-Function Structure Diagnostics

This is a local diagnostic over existing artifacts. It does not prove model-agnosticity,
does not authorize new HPC jobs, and does not implement a new scheduler.

## 1. Local Smoothness

Verdict: `{local['interpretation']}`.

One-swap mean |G(S)-G(T)| is `{local['one_swap_mean_abs_gap']}` and larger-distance mean
|G(S)-G(T)| is `{local['larger_distance_mean_abs_gap']}`. The Spearman correlation between
actual schedule distance and |G difference| is `{local['spearman_distance_abs_g_gap']}`.
This uses actual schedule distances, not visual proximity or phase labels.

## 2. Diminishing Returns / Weak Submodularity

Exact inclusion comparisons: `{diagnostics['monotonicity']['exact_inclusion_comparisons']}`.
Exact diminishing-returns triples: `{dr['exact_triples']}`.
Status: `{dr['status']}`.

No submodularity ratio or curvature claim is made unless the required exact comparisons
are present. These diagnostics are finite-sample probes over the observed artifacts only.

## 3. Higher-Order Residuals

Gate 3b B=3/4 deployable Q rows still look like an over-composition failure rather than
a usable scheduler:

{chr(10).join(q_lines) if q_lines else '- No deployable Q rows were available.'}

The direct pairwise-scheduler path remains blocked.

## 4. Search Trace Explanation

CD-G / BS-AG appear to succeed by using true-G feedback to find high-quality schedules in
the existing MC-pool landscape:

{chr(10).join(search_lines)}

This supports a search-based set-function interpretation more directly than a cheap
pairwise surrogate interpretation.

## 5. Bayesian Optimization Suitability

Verdict: `{bo['verdict']}`.

Reasons:
{chr(10).join(f'- {reason}' for reason in bo['reasons']) if bo['reasons'] else '- No positive BO evidence identified.'}

Blocking gaps:
{chr(10).join(f'- {gap}' for gap in bo['blocking_gaps']) if bo['blocking_gaps'] else '- None recorded.'}

Bayesian schedule optimization should be treated as a possible follow-up spec only if the
smoothness/kernel story is made explicit.

## 6. Online Value Approximation

Online value-function approximation is premature here. These artifacts evaluate open-loop
sets S and do not show that schedule quality depends on trajectory state in a way that
requires a learned V_t(b, z_t).

## 7. Recommended Next Direction

`{rec['direction']}`: {rec['rationale']}.

Do not revive a direct pairwise-Q scheduler from this diagnostic.
"""
    path.write_text(text)


def write_command_log(path: Path, argv: Sequence[str] | None) -> None:
    command = "python3 " + " ".join([sys.argv[0], *(argv if argv is not None else sys.argv[1:])])
    path.write_text(command + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2b_mc", type=Path, required=True)
    parser.add_argument("--phase3a_cd", type=Path, required=True)
    parser.add_argument("--phase3a_bs", type=Path, required=True)
    parser.add_argument("--gate3b_rows", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_pair_comparisons", type=int, default=200_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_output_dir(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"

    mc_rows = _require_list(args.phase2b_mc, load_json(args.phase2b_mc))
    cd_rows = _require_list(args.phase3a_cd, load_json(args.phase3a_cd))
    bs_rows = _require_list(args.phase3a_bs, load_json(args.phase3a_bs))
    gate3b_rows = _require_list(args.gate3b_rows, load_json(args.gate3b_rows))

    mono_dr = compute_monotonicity_and_diminishing_returns(mc_rows, cd_rows=cd_rows)
    local_smoothness = compute_local_smoothness(mc_rows, max_pair_comparisons=int(args.max_pair_comparisons))
    search_vs_mc = compute_search_vs_mc(mc_rows, cd_rows, bs_rows)
    higher_order = compute_higher_order_residuals(gate3b_rows)
    bo_suitability = compute_bo_suitability(local_smoothness, search_vs_mc, higher_order)
    recommendation = choose_recommended_direction(local_smoothness, search_vs_mc, bo_suitability)

    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": {
            "phase2b_mc": int(len(mc_rows)),
            "phase3a_cd": int(len(cd_rows)),
            "phase3a_bs": int(len(bs_rows)),
            "gate3b_rows": int(len(gate3b_rows)),
        },
        **mono_dr,
        "local_smoothness": local_smoothness,
        "search_vs_mc": search_vs_mc,
        "higher_order_residuals": higher_order,
        "bo_suitability": bo_suitability,
        "recommended_next_direction": recommendation,
        "claims_guardrails": {
            "submodularity_claimed": False,
            "new_scheduler_implemented": False,
            "hpc_job_submitted": False,
            "direct_pairwise_scheduler_revived": False,
        },
    }

    config = {
        "git_sha": git_sha(),
        "phase2b_mc": str(args.phase2b_mc),
        "phase3a_cd": str(args.phase3a_cd),
        "phase3a_bs": str(args.phase3a_bs),
        "gate3b_rows": str(args.gate3b_rows),
        "out_dir": str(args.out_dir),
        "max_pair_comparisons": int(args.max_pair_comparisons),
        "matplotlib_available": HAS_MPL,
    }
    write_json(args.out_dir / "config.json", config)

    plot_monotonicity(diagnostics, figures_dir)
    plot_diminishing_returns(diagnostics, figures_dir)
    plot_local_smoothness(local_smoothness, figures_dir)
    plot_search_vs_mc(search_vs_mc, figures_dir)
    plot_higher_order_residuals(higher_order, figures_dir)
    plot_bo_suitability(bo_suitability, figures_dir)

    write_json(args.out_dir / "diagnostics.json", diagnostics)
    write_interpretation(args.out_dir / "interpretation.md", diagnostics)
    write_command_log(args.out_dir / "command_log.txt", argv)
    print(f"Wrote {args.out_dir / 'diagnostics.json'}")
    print(f"Wrote {args.out_dir / 'interpretation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
