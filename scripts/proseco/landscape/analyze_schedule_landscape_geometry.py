#!/usr/bin/env python3.11
"""Second-pass schedule landscape geometry diagnostics.

This local analysis reads existing Phase 2b / Phase 3a / Gate 4 artifacts and
does not implement a scheduler or submit any HPC work.
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

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


PHASES = ("early", "middle", "late")


def canonical_schedule(steps: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(int(s) for s in steps)))


def schedule_distance(a: Sequence[int], b: Sequence[int]) -> int:
    sa = set(canonical_schedule(a))
    sb = set(canonical_schedule(b))
    return len(sa.symmetric_difference(sb)) // 2


def phase_for_step(step: int, *, T: int = 64) -> str:
    step = int(step)
    if step < 0 or step >= T:
        raise ValueError(f"step must be in [0, {T - 1}], got {step}")
    if step <= 21:
        return "early"
    if step <= 42:
        return "middle"
    return "late"


def phase_counts(steps: Sequence[int]) -> dict[str, int]:
    counts = {phase: 0 for phase in PHASES}
    for step in canonical_schedule(steps):
        counts[phase_for_step(step)] += 1
    return {phase: count for phase, count in counts.items() if count}


def summarize_distribution(values: Sequence[float]) -> dict[str, float | int | None]:
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


def top_fraction_rows(rows: Sequence[dict[str, Any]], *, fraction: float = 0.05) -> list[dict[str, Any]]:
    if not rows:
        return []
    n = max(1, int(np.ceil(len(rows) * float(fraction))))
    return sorted(rows, key=lambda row: (float(row["G"]), canonical_schedule(row["schedule_steps"])), reverse=True)[:n]


def bottom_fraction_rows(rows: Sequence[dict[str, Any]], *, fraction: float = 0.05) -> list[dict[str, Any]]:
    if not rows:
        return []
    n = max(1, int(np.ceil(len(rows) * float(fraction))))
    return sorted(rows, key=lambda row: (float(row["G"]), canonical_schedule(row["schedule_steps"])))[:n]


def pairwise_schedule_distances(rows: Sequence[dict[str, Any]]) -> list[float]:
    distances: list[float] = []
    for left, right in itertools.combinations(rows, 2):
        distances.append(float(schedule_distance(left["schedule_steps"], right["schedule_steps"])))
    return distances


def group_by_seed_budget(rows: Sequence[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), int(row["B"]))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: (int(row.get("mc_idx", 0)), canonical_schedule(row["schedule_steps"])))
    return dict(sorted(grouped.items()))


def top_schedule_geometry(
    mc_rows: Sequence[dict[str, Any]],
    *,
    top_fraction: float = 0.05,
    rng_seed: int = 0,
) -> dict[str, Any]:
    del rng_seed  # All-pool random baseline is deterministic; keep argument for API stability.
    by_budget: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_, budget), rows in group_by_seed_budget(mc_rows).items():
        top_rows = top_fraction_rows(rows, fraction=top_fraction)
        top_distances = pairwise_schedule_distances(top_rows)
        random_distances = pairwise_schedule_distances(rows)
        by_budget[budget]["top"].extend(top_distances)
        by_budget[budget]["random"].extend(random_distances)
        by_budget[budget]["n_top"].append(float(len(top_rows)))
        by_budget[budget]["n_pool"].append(float(len(rows)))

    out: dict[str, Any] = {}
    for budget in sorted(by_budget):
        top_summary = summarize_distribution(by_budget[budget]["top"])
        random_summary = summarize_distribution(by_budget[budget]["random"])
        top_mean = top_summary["mean"]
        random_mean = random_summary["mean"]
        clustered = top_mean is not None and random_mean is not None and float(top_mean) < float(random_mean)
        out[f"B={budget}"] = {
            "top_pair_distance": top_summary,
            "random_pair_distance": random_summary,
            "top_minus_random_mean_distance": (
                float(top_mean) - float(random_mean) if top_mean is not None and random_mean is not None else None
            ),
            "top_schedules_clustered": bool(clustered),
            "mean_top_count_per_seed": float(np.mean(by_budget[budget]["n_top"])),
            "mean_pool_count_per_seed": float(np.mean(by_budget[budget]["n_pool"])),
        }
    return out


def nearest_top_distance(schedule: Sequence[int], top_rows: Sequence[dict[str, Any]]) -> int | None:
    if not top_rows:
        return None
    return int(min(schedule_distance(schedule, row["schedule_steps"]) for row in top_rows))


def _empirical_quantile(values: Sequence[float], score: float) -> float | None:
    if not values:
        return None
    return float(sum(float(v) <= float(score) for v in values) / len(values))


def search_to_top_mc_distance(
    mc_rows: Sequence[dict[str, Any]],
    cd_rows: Sequence[dict[str, Any]],
    bs_rows: Sequence[dict[str, Any]],
    *,
    top_fraction: float = 0.05,
    rng_seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    mc_by_key = group_by_seed_budget(mc_rows)
    cd_by_key = {(int(row["seed"]), int(row["B"])): row for row in cd_rows}
    bs_by_key = {(int(row["seed"]), int(row["B"])): row for row in bs_rows}
    by_budget: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for key, rows in mc_by_key.items():
        seed, budget = key
        top_rows = top_fraction_rows(rows, fraction=top_fraction)
        random_row = rows[int(rng.integers(0, len(rows)))]
        random_dist = nearest_top_distance(random_row["schedule_steps"], top_rows)
        if random_dist is not None:
            by_budget[budget]["random"].append(float(random_dist))
        cd = cd_by_key.get((seed, budget))
        if cd is not None:
            dist = nearest_top_distance(cd["schedule_final"], top_rows)
            if dist is not None:
                by_budget[budget]["cd"].append(float(dist))
        bs = bs_by_key.get((seed, budget))
        if bs is not None:
            dist = nearest_top_distance(bs["schedule_final"], top_rows)
            if dist is not None:
                by_budget[budget]["bs"].append(float(dist))

    out: dict[str, Any] = {}
    for budget in sorted(by_budget):
        vals = by_budget[budget]
        random_mean = summarize_distribution(vals["random"])["mean"]
        cd_mean = summarize_distribution(vals["cd"])["mean"]
        bs_mean = summarize_distribution(vals["bs"])["mean"]
        out[f"B={budget}"] = {
            "cd_to_top_distance": summarize_distribution(vals["cd"]),
            "bs_to_top_distance": summarize_distribution(vals["bs"]),
            "random_to_top_distance": summarize_distribution(vals["random"]),
            "cd_closer_than_random": bool(cd_mean is not None and random_mean is not None and float(cd_mean) < float(random_mean)),
            "bs_closer_than_random": bool(bs_mean is not None and random_mean is not None and float(bs_mean) < float(random_mean)),
        }
    return out


def _composition_record(steps: Sequence[int]) -> dict[str, float]:
    counts = {phase: 0.0 for phase in PHASES}
    for phase, count in phase_counts(steps).items():
        counts[phase] = float(count)
    total = sum(counts.values()) or 1.0
    return {phase: counts[phase] / total for phase in PHASES}


def phase_composition_diagnostic(
    mc_rows: Sequence[dict[str, Any]],
    cd_rows: Sequence[dict[str, Any]],
    bs_rows: Sequence[dict[str, Any]],
    *,
    top_fraction: float = 0.05,
) -> dict[str, Any]:
    by_budget_group: dict[int, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for (_, budget), rows in group_by_seed_budget(mc_rows).items():
        for row in top_fraction_rows(rows, fraction=top_fraction):
            by_budget_group[budget]["top_5pct_mc"].append(_composition_record(row["schedule_steps"]))
        for row in bottom_fraction_rows(rows, fraction=top_fraction):
            by_budget_group[budget]["bottom_5pct_mc"].append(_composition_record(row["schedule_steps"]))
        for row in rows:
            by_budget_group[budget]["random_mc_pool"].append(_composition_record(row["schedule_steps"]))
    for row in cd_rows:
        by_budget_group[int(row["B"])]["cd_g_final"].append(_composition_record(row["schedule_final"]))
    for row in bs_rows:
        by_budget_group[int(row["B"])]["bs_ag_final"].append(_composition_record(row["schedule_final"]))

    out: dict[str, Any] = {}
    for budget in sorted(by_budget_group):
        out[f"B={budget}"] = {}
        for group, records in sorted(by_budget_group[budget].items()):
            out[f"B={budget}"][group] = {
                "n": int(len(records)),
                "mean_fraction": {
                    phase: float(np.mean([record[phase] for record in records])) if records else None
                    for phase in PHASES
                },
            }
    return out


def _region_rows(rows: Sequence[dict[str, Any]], region: str) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: float(row["G"]))
    if region == "overall":
        return list(rows)
    if region == "top_half":
        return sorted_rows[len(sorted_rows) // 2 :]
    if region == "bottom_half":
        return sorted_rows[: len(sorted_rows) // 2]
    if region == "top_10pct":
        n = max(1, int(np.ceil(len(sorted_rows) * 0.10)))
        return sorted_rows[-n:]
    raise ValueError(f"unknown region: {region}")


def a_score_reliability_by_region(
    mc_rows: Sequence[dict[str, Any]],
    *,
    min_n: int = 10,
) -> dict[str, Any]:
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in mc_rows:
        by_budget[int(row["B"])].append(row)

    out: dict[str, Any] = {}
    for budget in sorted(by_budget):
        rows = by_budget[budget]
        out[f"B={budget}"] = {}
        for region in ("overall", "top_half", "bottom_half", "top_10pct"):
            region_rows = _region_rows(rows, region)
            a_vals = [float(row["A"]) for row in region_rows]
            g_vals = [float(row["G"]) for row in region_rows]
            corr = spearman_corr(a_vals, g_vals) if len(region_rows) >= min_n else None
            out[f"B={budget}"][region] = {
                "n": int(len(region_rows)),
                "spearman_A_G": corr,
                "mean_A": float(np.mean(a_vals)) if a_vals else None,
                "mean_G": float(np.mean(g_vals)) if g_vals else None,
            }
    return out


def search_call_efficiency(
    mc_rows: Sequence[dict[str, Any]],
    cd_rows: Sequence[dict[str, Any]],
    bs_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    mc_by_key = group_by_seed_budget(mc_rows)
    by_budget: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in cd_rows:
        budget = int(row["B"])
        key = (int(row["seed"]), budget)
        mc_g = [float(r["G"]) for r in mc_by_key.get(key, [])]
        n_calls = max(1.0, float(row["n_g_calls"]))
        g_final = float(row["G_final"])
        quantile = _empirical_quantile(mc_g, g_final)
        by_budget[budget]["cd_final_per_call"].append(g_final / n_calls)
        by_budget[budget]["cd_gain_per_call"].append((g_final - float(row["G_init"])) / n_calls)
        if quantile is not None:
            by_budget[budget]["cd_quantile_per_call"].append(quantile / n_calls)
            by_budget[budget]["cd_quantile"].append(quantile)
        by_budget[budget]["cd_n_calls"].append(n_calls)

    for row in bs_rows:
        budget = int(row["B"])
        key = (int(row["seed"]), budget)
        mc_g = [float(r["G"]) for r in mc_by_key.get(key, [])]
        n_calls = max(1.0, float(row["n_g_calls"]))
        g_final = float(row["G_final"])
        quantile = _empirical_quantile(mc_g, g_final)
        by_budget[budget]["bs_final_per_call"].append(g_final / n_calls)
        if quantile is not None:
            by_budget[budget]["bs_quantile_per_call"].append(quantile / n_calls)
            by_budget[budget]["bs_quantile"].append(quantile)
        by_budget[budget]["bs_n_calls"].append(n_calls)

    out: dict[str, Any] = {}
    for budget in sorted(by_budget):
        vals = by_budget[budget]
        out[f"B={budget}"] = {
            "cd_final_G_per_call": summarize_distribution(vals["cd_final_per_call"]),
            "cd_gain_over_initial_per_call": summarize_distribution(vals["cd_gain_per_call"]),
            "cd_mc_quantile_per_call": summarize_distribution(vals["cd_quantile_per_call"]),
            "cd_mc_quantile": summarize_distribution(vals["cd_quantile"]),
            "cd_n_g_calls": summarize_distribution(vals["cd_n_calls"]),
            "bs_final_G_per_call": summarize_distribution(vals["bs_final_per_call"]),
            "bs_mc_quantile_per_call": summarize_distribution(vals["bs_quantile_per_call"]),
            "bs_mc_quantile": summarize_distribution(vals["bs_quantile"]),
            "bs_n_g_calls": summarize_distribution(vals["bs_n_calls"]),
        }
    return out


def decision_summary(
    top_geometry: dict[str, Any],
    search_distance: dict[str, Any],
    a_reliability: dict[str, Any],
    prior_set_function: dict[str, Any],
) -> dict[str, Any]:
    cluster_flags = [bool(cell["top_schedules_clustered"]) for cell in top_geometry.values()]
    cd_near_flags = [bool(cell["cd_closer_than_random"]) for cell in search_distance.values()]
    bs_near_flags = [bool(cell["bs_closer_than_random"]) for cell in search_distance.values()]
    top10_corrs = [
        cell["top_10pct"]["spearman_A_G"]
        for cell in a_reliability.values()
        if cell["top_10pct"]["spearman_A_G"] is not None
    ]
    overall_corrs = [
        cell["overall"]["spearman_A_G"]
        for cell in a_reliability.values()
        if cell["overall"]["spearman_A_G"] is not None
    ]
    local_rho = prior_set_function.get("local_smoothness", {}).get("spearman_distance_abs_g_gap")
    cluster_share = float(np.mean(cluster_flags)) if cluster_flags else 0.0
    search_near_share = float(np.mean(cd_near_flags + bs_near_flags)) if (cd_near_flags or bs_near_flags) else 0.0
    mean_top10_corr = float(np.mean(top10_corrs)) if top10_corrs else None
    mean_overall_corr = float(np.mean(overall_corrs)) if overall_corrs else None

    bo_supported = (
        cluster_share >= 0.5
        and local_rho is not None
        and float(local_rho) >= 0.15
        and search_near_share >= 0.5
    )
    hpc_neighborhood = cluster_share >= 0.5 and search_near_share >= 0.5
    a_unreliable_top = mean_top10_corr is None or mean_top10_corr < 0.35

    if hpc_neighborhood:
        next_direction = "targeted_neighborhood_run_proposal_then_search_spec"
        rationale = (
            "top MC schedules cluster, and search-to-top distances are better than random "
            "for enough method/budget cells to justify a local-neighborhood proposal; "
            "CD-G is mixed at B=3/4, so the run should test both local optima and off-pool high-G regions"
        )
    elif bo_supported:
        next_direction = "bayesian_optimization_spec_after_kernel_definition"
        rationale = "distance/cluster structure is strong enough to motivate a set kernel"
    elif a_unreliable_top:
        next_direction = "true_G_black_box_search_spec"
        rationale = "top-region A-score reliability is weak, so surrogate learning is not the immediate explanation"
    else:
        next_direction = "conservative_higher_order_search_writeup"
        rationale = "local artifacts do not clearly identify a stronger scheduling principle"

    return {
        "top_clustered_budget_fraction": cluster_share,
        "search_near_top_budget_method_fraction": search_near_share,
        "mean_top10_spearman_A_G": mean_top10_corr,
        "mean_overall_spearman_A_G": mean_overall_corr,
        "prior_local_distance_spearman": local_rho,
        "targeted_hpc_neighborhood_justified": bool(hpc_neighborhood),
        "bayesian_optimization_justified_now": bool(bo_supported),
        "online_value_approximation_premature": True,
        "llada_feasibility_gate_only": True,
        "next_direction": next_direction,
        "rationale": rationale,
    }


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _save_current_figure(figures_dir: Path, stem: str) -> None:
    if not HAS_MPL:
        return
    assert plt is not None
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{stem}.png", dpi=160)
    plt.savefig(figures_dir / f"{stem}.pdf")
    plt.close()


def _save_text_figure(figures_dir: Path, stem: str, title: str, lines: Sequence[str]) -> None:
    if not HAS_PIL:
        return
    assert Image is not None and ImageDraw is not None and ImageFont is not None
    figures_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    y = 28
    draw.text((28, y), title, fill=(0, 0, 0), font=font)
    y += 34
    for line in lines:
        if y > height - 40:
            draw.text((28, y), "...", fill=(0, 0, 0), font=font)
            break
        draw.text((28, y), str(line), fill=(0, 0, 0), font=font)
        y += 24
    image.save(figures_dir / f"{stem}.png")
    image.save(figures_dir / f"{stem}.pdf", "PDF")


def plot_top_schedule_distance(top_geometry: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        lines = [
            f"{budget}: top_mean={cell['top_pair_distance']['mean']}, "
            f"pool_mean={cell['random_pair_distance']['mean']}, clustered={cell['top_schedules_clustered']}"
            for budget, cell in sorted(top_geometry.items(), key=lambda item: int(item[0].split("=")[1]))
        ]
        _save_text_figure(figures_dir, "top_schedule_distance_distribution", "Top-schedule distance distribution", lines)
        return
    assert plt is not None
    budgets = sorted(top_geometry, key=lambda key: int(key.split("=")[1]))
    x = np.arange(len(budgets))
    top = [top_geometry[b]["top_pair_distance"]["mean"] or 0.0 for b in budgets]
    random = [top_geometry[b]["random_pair_distance"]["mean"] or 0.0 for b in budgets]
    width = 0.36
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, top, width, label="top 5% MC", color="#4C78A8")
    plt.bar(x + width / 2, random, width, label="MC pool", color="#F58518")
    plt.xticks(x, budgets)
    plt.ylabel("mean pairwise schedule distance")
    plt.title("Top-schedule distance distribution")
    plt.legend()
    _save_current_figure(figures_dir, "top_schedule_distance_distribution")


def plot_search_to_top(search_distance: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        lines = [
            f"{budget}: cd={cell['cd_to_top_distance']['mean']}, "
            f"bs={cell['bs_to_top_distance']['mean']}, random={cell['random_to_top_distance']['mean']}"
            for budget, cell in sorted(search_distance.items(), key=lambda item: int(item[0].split("=")[1]))
        ]
        _save_text_figure(figures_dir, "search_to_top_mc_distance", "Search schedules to top MC region", lines)
        return
    assert plt is not None
    budgets = sorted(search_distance, key=lambda key: int(key.split("=")[1]))
    x = np.arange(len(budgets))
    width = 0.26
    cd = [search_distance[b]["cd_to_top_distance"]["mean"] or 0.0 for b in budgets]
    bs = [search_distance[b]["bs_to_top_distance"]["mean"] or 0.0 for b in budgets]
    random = [search_distance[b]["random_to_top_distance"]["mean"] or 0.0 for b in budgets]
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width, cd, width, label="CD-G final", color="#F58518")
    plt.bar(x, bs, width, label="BS-AG final", color="#54A24B")
    plt.bar(x + width, random, width, label="random MC", color="#4C78A8")
    plt.xticks(x, budgets)
    plt.ylabel("distance to nearest top 5% MC schedule")
    plt.title("Search schedules versus top MC region")
    plt.legend()
    _save_current_figure(figures_dir, "search_to_top_mc_distance")


def plot_phase_composition(phase_comp: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        lines: list[str] = []
        for budget, groups in sorted(phase_comp.items(), key=lambda item: int(item[0].split("=")[1])):
            for group, cell in sorted(groups.items()):
                lines.append(f"{budget} {group}: {cell['mean_fraction']}")
        _save_text_figure(figures_dir, "phase_composition_top_vs_search", "Phase composition top-vs-search", lines)
        return
    assert plt is not None
    group_order = ["top_5pct_mc", "bottom_5pct_mc", "random_mc_pool", "cd_g_final", "bs_ag_final"]
    labels: list[str] = []
    data: dict[str, list[float]] = {phase: [] for phase in PHASES}
    for budget in sorted(phase_comp, key=lambda key: int(key.split("=")[1])):
        for group in group_order:
            if group not in phase_comp[budget]:
                continue
            labels.append(f"{budget}\n{group}")
            fractions = phase_comp[budget][group]["mean_fraction"]
            for phase in PHASES:
                data[phase].append(float(fractions[phase]))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    colors = {"early": "#4C78A8", "middle": "#F58518", "late": "#54A24B"}
    plt.figure(figsize=(max(10, len(labels) * 0.65), 5.0))
    for phase in PHASES:
        vals = np.asarray(data[phase])
        plt.bar(x, vals, bottom=bottom, label=phase, color=colors[phase])
        bottom += vals
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("mean schedule fraction")
    plt.title("Phase composition of high-quality and search schedules")
    plt.legend()
    _save_current_figure(figures_dir, "phase_composition_top_vs_search")


def plot_a_vs_g_by_region(a_reliability: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        lines: list[str] = []
        for budget, regions in sorted(a_reliability.items(), key=lambda item: int(item[0].split("=")[1])):
            lines.append(
                f"{budget}: overall={regions['overall']['spearman_A_G']}, "
                f"top_half={regions['top_half']['spearman_A_G']}, "
                f"top10={regions['top_10pct']['spearman_A_G']}"
            )
        _save_text_figure(figures_dir, "A_vs_G_by_region", "A-score reliability by region", lines)
        return
    assert plt is not None
    regions = ["overall", "bottom_half", "top_half", "top_10pct"]
    budgets = sorted(a_reliability, key=lambda key: int(key.split("=")[1]))
    x = np.arange(len(budgets))
    width = 0.18
    plt.figure(figsize=(8, 4.5))
    for idx, region in enumerate(regions):
        vals = [
            a_reliability[b][region]["spearman_A_G"]
            if a_reliability[b][region]["spearman_A_G"] is not None
            else 0.0
            for b in budgets
        ]
        plt.bar(x + (idx - 1.5) * width, vals, width, label=region)
    plt.xticks(x, budgets)
    plt.ylim(-1, 1)
    plt.ylabel("Spearman(A, G)")
    plt.title("A-score reliability by schedule-quality region")
    plt.legend()
    _save_current_figure(figures_dir, "A_vs_G_by_region")


def plot_true_g_call_efficiency(efficiency: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        lines: list[str] = []
        for budget, cell in sorted(efficiency.items(), key=lambda item: int(item[0].split("=")[1])):
            lines.append(
                f"{budget}: cd_q/call={cell['cd_mc_quantile_per_call']['mean']}, "
                f"bs_q/call={cell['bs_mc_quantile_per_call']['mean']}, "
                f"cd_gain/call={cell['cd_gain_over_initial_per_call']['mean']}"
            )
        _save_text_figure(figures_dir, "true_G_call_efficiency", "True-G call efficiency", lines)
        return
    assert plt is not None
    budgets = sorted(efficiency, key=lambda key: int(key.split("=")[1]))
    x = np.arange(len(budgets))
    width = 0.36
    cd = [efficiency[b]["cd_mc_quantile_per_call"]["mean"] or 0.0 for b in budgets]
    bs = [efficiency[b]["bs_mc_quantile_per_call"]["mean"] or 0.0 for b in budgets]
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, cd, width, label="CD-G", color="#F58518")
    plt.bar(x + width / 2, bs, width, label="BS-AG", color="#54A24B")
    plt.xticks(x, budgets)
    plt.ylabel("MC-pool quantile per true-G call")
    plt.title("True-G call efficiency")
    plt.legend()
    _save_current_figure(figures_dir, "true_G_call_efficiency")


def plot_decision_summary(summary: dict[str, Any], figures_dir: Path) -> None:
    if not HAS_MPL:
        lines = [
            f"Next direction: {summary['next_direction']}",
            f"Rationale: {summary['rationale']}",
            f"Top clustered fraction: {summary['top_clustered_budget_fraction']}",
            f"Search near top fraction: {summary['search_near_top_budget_method_fraction']}",
            f"Targeted HPC justified: {summary['targeted_hpc_neighborhood_justified']}",
            f"BO justified now: {summary['bayesian_optimization_justified_now']}",
            "No scheduler implemented; no HPC submitted.",
        ]
        _save_text_figure(figures_dir, "landscape_decision_summary", "Landscape decision summary", lines)
        return
    assert plt is not None
    plt.figure(figsize=(9, 4.8))
    plt.axis("off")
    lines = [
        f"Next direction: {summary['next_direction']}",
        f"Rationale: {summary['rationale']}",
        f"Top clustered budget fraction: {summary['top_clustered_budget_fraction']:.3f}",
        f"Search near top fraction: {summary['search_near_top_budget_method_fraction']:.3f}",
        f"Mean top-10% Spearman(A,G): {summary['mean_top10_spearman_A_G']}",
        f"Prior local distance Spearman: {summary['prior_local_distance_spearman']}",
        f"Targeted HPC neighborhood justified: {summary['targeted_hpc_neighborhood_justified']}",
        f"BO justified now: {summary['bayesian_optimization_justified_now']}",
        "No scheduler implemented; no HPC submitted.",
    ]
    plt.title("Landscape decision summary", loc="left")
    plt.text(0.02, 0.9, "\n".join(lines), va="top", ha="left", fontsize=10, wrap=True)
    _save_current_figure(figures_dir, "landscape_decision_summary")


def write_interpretation(path: Path, diagnostics: dict[str, Any]) -> None:
    summary = diagnostics["decision_summary"]
    top_geometry = diagnostics["top_schedule_geometry"]
    search_distance = diagnostics["search_to_top_mc_distance"]
    a_reliability = diagnostics["a_score_reliability_by_region"]
    efficiency = diagnostics["search_call_efficiency"]

    top_lines = []
    for budget in sorted(top_geometry, key=lambda key: int(key.split("=")[1])):
        cell = top_geometry[budget]
        top_lines.append(
            f"- {budget}: top mean distance {cell['top_pair_distance']['mean']:.3f}; "
            f"MC-pool mean distance {cell['random_pair_distance']['mean']:.3f}; "
            f"clustered={cell['top_schedules_clustered']}."
        )

    search_lines = []
    for budget in sorted(search_distance, key=lambda key: int(key.split("=")[1])):
        cell = search_distance[budget]
        search_lines.append(
            f"- {budget}: CD-G distance {cell['cd_to_top_distance']['mean']:.3f}; "
            f"BS-AG distance {cell['bs_to_top_distance']['mean']:.3f}; "
            f"random control {cell['random_to_top_distance']['mean']:.3f}."
        )

    a_lines = []
    for budget in sorted(a_reliability, key=lambda key: int(key.split("=")[1])):
        cell = a_reliability[budget]
        a_lines.append(
            f"- {budget}: overall {cell['overall']['spearman_A_G']:.3f}; "
            f"top-half {cell['top_half']['spearman_A_G']:.3f}; "
            f"top-10% {cell['top_10pct']['spearman_A_G']:.3f}."
        )

    def fmt(value: Any, digits: int = 3) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    eff_lines = []
    for budget in sorted(efficiency, key=lambda key: int(key.split("=")[1])):
        cell = efficiency[budget]
        eff_lines.append(
            f"- {budget}: CD-G quantile/call {fmt(cell['cd_mc_quantile_per_call']['mean'], 4)}; "
            f"BS-AG quantile/call {fmt(cell['bs_mc_quantile_per_call']['mean'], 4)}; "
            f"CD-G gain/call {fmt(cell['cd_gain_over_initial_per_call']['mean'], 4)}."
        )

    hpc_text = (
        "Yes, as a proposal only: a targeted one-swap neighborhood run around CD-G / BS-AG "
        "final schedules is justified by clustering and search-near-top evidence."
        if summary["targeted_hpc_neighborhood_justified"]
        else "Not yet: current local evidence does not justify a targeted HPC neighborhood run."
    )

    bo_text = (
        "BO is justified enough for a spec."
        if summary["bayesian_optimization_justified_now"]
        else "BO remains theoretically appealing but not yet justified as the immediate implementation path."
    )

    text = f"""# Schedule Landscape Geometry Diagnostics

This is a local second-pass analysis over existing artifacts. It does not submit HPC,
does not overwrite canonical result folders, and does not implement a new scheduler.

## Main Decision

Recommended next direction: `{summary['next_direction']}`.

Rationale: {summary['rationale']}.

## 1. Top-Schedule Geometry

{chr(10).join(top_lines)}

Interpretation: if top schedules are closer than the MC-pool baseline, high-G
schedules form a coherent local region rather than appearing fully dispersed.

## 2. Search Distance to Top MC Region

{chr(10).join(search_lines)}

Interpretation: BS-AG finals are consistently closer to the top-MC region than
random controls, while CD-G is close at B=2 but mixed at B=3/4. This suggests a
targeted neighborhood run should test both local optimality around search finals
and whether CD-G sometimes finds high-G schedules outside the random MC top pool.

## 3. Phase Composition

See `phase_composition_top_vs_search` figures and `diagnostics.json`.
The comparison covers top 5% MC, bottom 5% MC, random MC pool, CD-G finals, and
BS-AG finals under the project early/middle/late convention.

## 4. A-Score Reliability by Region

{chr(10).join(a_lines)}

Interpretation: A is moderately rank-informative overall but weak in the top
10% region, especially at B=3/4. This supports A as a pruning aid only when
paired with true-G feedback, not as a direct scheduler.

## 5. Search-Call Efficiency

{chr(10).join(eff_lines)}

Interpretation: this motivates treating true-G calls as an explicit scarce
resource in any later search spec.

## 6. Decision Memo

1. Targeted HPC neighborhood run: {hpc_text}
2. Bayesian optimization: {bo_text}
3. Online value-function approximation: still premature; these are open-loop
   set-function artifacts, not state-conditioned value data.
4. LLaDA: do not reopen as a full pivot. At most define a small feasibility
   gate that first verifies metric/protocol and positive MC-oracle headroom.

## 7. Targeted HPC Proposals Only

No HPC job was submitted. If local diagnostics are accepted, the proposals are:

**Proposal 1: Local-neighborhood run around CD-G / BS-AG schedules.**
Purpose: determine whether CD-G / BS-AG finals are local optima, evaluate
one-swap neighborhoods, and estimate local ruggedness. Design: K=30 existing
seeds if feasible or K=10 pilot; B={{2,3,4}}; for each seed/B evaluate the
CD-G final schedule, BS-AG final schedule, a random MC control, and all
one-swap neighbors if feasible or a fixed random sample. Output must be
SHA-tagged. A Codex pre-submit review is required before any submission.

**Proposal 2: Exact diminishing-returns triples.**
Purpose: test weak submodularity / diminishing returns honestly. Design: fixed
triple pool A subset B, x not in B; evaluate G(A), G(B), G(A union {{x}}), and
G(B union {{x}}); compute the DR gap. No submodularity claim unless the
confidence interval supports it.

**Proposal 3: LLaDA feasibility gate, not a pivot.**
Purpose: test whether LLaDA gives a reliable basis. First verify metric/protocol
and positive MC-oracle headroom. Do not run Phase-3a-style search unless
headroom is positive.

## 8. Bayesian Theory Note

Treat G(S) as an expensive black-box stochastic set function. A possible model:

```text
G(S) ~ GP(m(S), k(S,S'))

k(S,S') =
  alpha * |S intersect S'| / B
  + beta * exp(-d_phase(S,S') / ell_phase)
  + gamma * exp(-d_time(S,S') / ell_time)
  + delta * k_signal(S,S')
```

Acquisition could use Thompson sampling or expected improvement over a fixed
candidate pool; batch BO or successive halving is plausible if multiple true-G
evaluations can run in parallel. This is theoretically appealing, but not the
immediate implementation path unless a kernel smoothness story is established.
"""
    path.write_text(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2b_mc", type=Path, required=True)
    parser.add_argument("--phase3a_cd", type=Path, required=True)
    parser.add_argument("--phase3a_bs", type=Path, required=True)
    parser.add_argument("--set_function_diagnostics", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--top_fraction", type=float, default=0.05)
    return parser


def _validate_out_dir(out_dir: Path) -> None:
    if out_dir.parent.name != "results" or not out_dir.name.startswith("schedule_landscape_geometry_"):
        raise ValueError("out_dir must be results/schedule_landscape_geometry_<gitsha>")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_out_dir(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"

    mc_rows = load_json(args.phase2b_mc)
    cd_rows = load_json(args.phase3a_cd)
    bs_rows = load_json(args.phase3a_bs)
    prior_diag = load_json(args.set_function_diagnostics)
    if not all(isinstance(obj, list) for obj in (mc_rows, cd_rows, bs_rows)):
        raise ValueError("phase2b_mc, phase3a_cd, and phase3a_bs must be JSON lists")

    top_geom = top_schedule_geometry(mc_rows, top_fraction=args.top_fraction)
    search_dist = search_to_top_mc_distance(mc_rows, cd_rows, bs_rows, top_fraction=args.top_fraction)
    phase_comp = phase_composition_diagnostic(mc_rows, cd_rows, bs_rows, top_fraction=args.top_fraction)
    a_reliability = a_score_reliability_by_region(mc_rows)
    efficiency = search_call_efficiency(mc_rows, cd_rows, bs_rows)
    decision = decision_summary(top_geom, search_dist, a_reliability, prior_diag)

    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": {
            "phase2b_mc": int(len(mc_rows)),
            "phase3a_cd": int(len(cd_rows)),
            "phase3a_bs": int(len(bs_rows)),
        },
        "top_fraction": float(args.top_fraction),
        "top_schedule_geometry": top_geom,
        "search_to_top_mc_distance": search_dist,
        "phase_composition": phase_comp,
        "a_score_reliability_by_region": a_reliability,
        "search_call_efficiency": efficiency,
        "decision_summary": decision,
        "guardrails": {
            "new_scheduler_implemented": False,
            "hpc_job_submitted": False,
            "canonical_results_overwritten": False,
            "submodularity_claimed": False,
        },
    }
    config = {
        "git_sha": git_sha(),
        "phase2b_mc": str(args.phase2b_mc),
        "phase3a_cd": str(args.phase3a_cd),
        "phase3a_bs": str(args.phase3a_bs),
        "set_function_diagnostics": str(args.set_function_diagnostics),
        "out_dir": str(args.out_dir),
        "top_fraction": float(args.top_fraction),
        "matplotlib_available": HAS_MPL,
        "pillow_fallback_available": HAS_PIL,
    }
    write_json(args.out_dir / "config.json", config)
    write_json(args.out_dir / "diagnostics.json", diagnostics)
    write_interpretation(args.out_dir / "interpretation.md", diagnostics)
    (args.out_dir / "command_log.txt").write_text("python3.11 " + " ".join([sys.argv[0], *(argv or sys.argv[1:])]) + "\n")

    plot_top_schedule_distance(top_geom, figures_dir)
    plot_search_to_top(search_dist, figures_dir)
    plot_phase_composition(phase_comp, figures_dir)
    plot_a_vs_g_by_region(a_reliability, figures_dir)
    plot_true_g_call_efficiency(efficiency, figures_dir)
    plot_decision_summary(decision, figures_dir)

    print(f"Wrote {args.out_dir / 'diagnostics.json'}")
    print(f"Wrote {args.out_dir / 'interpretation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
