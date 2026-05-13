#!/usr/bin/env python3
"""Analyze Phase 4 schedule-neighborhood diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.proseco.landscape.run_schedule_neighborhood_diagnostics import (  # noqa: E402
    ANCHOR_TYPES,
    REQUIRED_ROW_FIELDS,
    canonical_schedule,
    diminishing_return_gap,
    schedule_distance,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_rows(results_dir: Path, stem: str) -> list[dict[str, Any]]:
    flat = results_dir / f"{stem}.json"
    if flat.exists():
        obj = _load_json(flat)
        if not isinstance(obj, list):
            raise RuntimeError(f"Expected list in {flat}")
        return obj
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob(f"{stem}.shard*-of-*.json")):
        obj = _load_json(path)
        if not isinstance(obj, list):
            raise RuntimeError(f"Expected list in {path}")
        rows.extend(obj)
    return rows


def _finite(values: Iterable[float | int | None]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        f = float(value)
        if math.isfinite(f):
            out.append(f)
    return out


def summarize_distribution(values: Sequence[float | int | None]) -> dict[str, Any]:
    vals = _finite(values)
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "ci95": [None, None],
            "p05": None,
            "p95": None,
        }
    arr = np.asarray(vals, dtype=float)
    if arr.size == 1:
        ci = [float(arr[0]), float(arr[0])]
    else:
        rng = np.random.default_rng(0)
        boot = np.asarray(
            [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(1000)],
            dtype=float,
        )
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci95": ci,
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson_corr(_rankdata(x).tolist(), _rankdata(y).tolist())


def _validate_required_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    duplicate_keys: set[tuple[Any, ...]] = set()
    seen: set[tuple[Any, ...]] = set()
    for idx, row in enumerate(rows):
        absent = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if absent:
            missing.append({"row_idx": idx, "missing": absent})
        key = (
            row.get("diagnostic_type"),
            row.get("seed"),
            row.get("B"),
            row.get("anchor_id"),
            tuple(row.get("schedule_steps", [])),
            row.get("neighbor_relation"),
            row.get("triple_id"),
            row.get("triple_role"),
        )
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)
    return {
        "n_rows": len(rows),
        "n_missing_required_field_rows": len(missing),
        "missing_required_field_examples": missing[:20],
        "n_duplicate_row_keys": len(duplicate_keys),
    }


def _group_neighborhood_anchors(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("diagnostic_type") == "neighborhood":
            grouped[str(row.get("anchor_id", "missing_anchor"))].append(row)
    return dict(grouped)


def compute_local_optimality_by_anchor(
    rows: Sequence[dict[str, Any]],
    *,
    known_anchor_types: Sequence[str] = ANCHOR_TYPES,
) -> dict[str, dict[str, Any]]:
    grouped = _group_neighborhood_anchors(rows)
    per_type: dict[str, dict[str, list[Any]]] = {
        anchor: {"local": [], "best_improvement": [], "n_neighbors": []}
        for anchor in known_anchor_types
    }
    for anchor_rows in grouped.values():
        anchor_row = next(
            (row for row in anchor_rows if row.get("neighbor_relation") == "anchor"),
            anchor_rows[0],
        )
        anchor_type = str(anchor_row.get("anchor_type", "unknown"))
        per_type.setdefault(anchor_type, {"local": [], "best_improvement": [], "n_neighbors": []})
        neighbor_rows = [row for row in anchor_rows if row.get("neighbor_relation") == "one_swap"]
        anchor_G = float(anchor_row.get("anchor_G", anchor_row.get("G", 0.0)))
        improvements = [float(row["G"]) - anchor_G for row in neighbor_rows]
        best_improvement = max(improvements) if improvements else anchor_row.get("best_one_swap_improvement")
        if best_improvement is None:
            local_optimum = None
        else:
            local_optimum = bool(float(best_improvement) <= 0.0)
        per_type[anchor_type]["local"].append(local_optimum)
        per_type[anchor_type]["best_improvement"].append(best_improvement)
        per_type[anchor_type]["n_neighbors"].append(len(neighbor_rows))

    out: dict[str, dict[str, Any]] = {}
    for anchor_type, vals in per_type.items():
        local_known = [value for value in vals["local"] if value is not None]
        if not vals["local"]:
            out[anchor_type] = {"status": "no_rows", "n_anchors": 0}
            continue
        out[anchor_type] = {
            "status": "identified" if local_known else "no_neighbor_rows",
            "n_anchors": len(vals["local"]),
            "fraction_local_optimum": (
                float(sum(bool(v) for v in local_known) / len(local_known)) if local_known else None
            ),
            "best_one_swap_improvement": summarize_distribution(vals["best_improvement"]),
            "n_neighbors_evaluated": summarize_distribution(vals["n_neighbors"]),
        }
    return out


def _neighbor_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("diagnostic_type") == "neighborhood"
        and row.get("neighbor_relation") == "one_swap"
    ]


def compute_neighbor_delta_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    neighbors = _neighbor_rows(rows)
    deltas = [float(row.get("delta_neighbor", float(row["G"]) - float(row.get("anchor_G", 0.0)))) for row in neighbors]
    by_B: dict[str, dict[str, Any]] = {}
    for budget in sorted({int(row["B"]) for row in neighbors}):
        vals = [
            float(row.get("delta_neighbor", float(row["G"]) - float(row.get("anchor_G", 0.0))))
            for row in neighbors
            if int(row["B"]) == budget
        ]
        by_B[str(budget)] = summarize_distribution(vals)
    return {
        "overall": summarize_distribution(deltas),
        "by_B": by_B,
        "probability_improvement": (
            float(sum(delta > 0 for delta in deltas) / len(deltas)) if deltas else None
        ),
    }


def compute_smoothness(
    rows: Sequence[dict[str, Any]],
    *,
    max_pairwise_comparisons: int = 20_000,
) -> dict[str, Any]:
    by_key: dict[tuple[int, int], dict[tuple[int, ...], float]] = defaultdict(dict)
    for row in rows:
        if row.get("diagnostic_type") != "neighborhood":
            continue
        key = (int(row["seed"]), int(row["B"]))
        schedule = canonical_schedule(row["schedule_steps"])
        by_key[key][schedule] = float(row["G"])

    pairs: list[tuple[int, float]] = []
    for schedules in by_key.values():
        items = sorted(schedules.items())
        for idx, (left_schedule, left_G) in enumerate(items):
            for right_schedule, right_G in items[idx + 1 :]:
                pairs.append((schedule_distance(left_schedule, right_schedule), abs(float(left_G) - float(right_G))))
    if len(pairs) > max_pairwise_comparisons:
        rng = np.random.default_rng(0)
        indices = sorted(rng.choice(len(pairs), size=max_pairwise_comparisons, replace=False).tolist())
        pairs = [pairs[idx] for idx in indices]

    distances = [float(pair[0]) for pair in pairs]
    gaps = [float(pair[1]) for pair in pairs]
    by_distance: dict[str, dict[str, Any]] = {}
    for distance in sorted(set(distances)):
        vals = [gap for d, gap in zip(distances, gaps) if d == distance]
        by_distance[str(int(distance))] = summarize_distribution(vals)
    one_swap_mean = by_distance.get("1", {}).get("mean")
    larger_vals = [gap for d, gap in zip(distances, gaps) if d > 1]
    larger_summary = summarize_distribution(larger_vals)
    return {
        "n_pairwise_comparisons": len(pairs),
        "by_distance": by_distance,
        "mean_abs_deltaG_by_larger_distance": larger_summary,
        "one_swap_mean_abs_deltaG": one_swap_mean,
        "spearman_distance_abs_deltaG": spearman_corr(distances, gaps),
    }


def _triple_gap_records(triple_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in triple_rows:
        if row.get("diagnostic_type") == "diminishing_return":
            grouped[str(row.get("triple_id", "missing_triple"))].append(row)

    records: list[dict[str, Any]] = []
    for triple_id, rows in grouped.items():
        by_role = {str(row.get("triple_role")): row for row in rows}
        if {"A", "B", "A_plus_x", "B_plus_x"} - set(by_role):
            continue
        if "DR_gap" in by_role["A"]:
            gap = float(by_role["A"]["DR_gap"])
        else:
            gap = diminishing_return_gap(
                g_A=float(by_role["A"]["G"]),
                g_B=float(by_role["B"]["G"]),
                g_A_plus_x=float(by_role["A_plus_x"]["G"]),
                g_B_plus_x=float(by_role["B_plus_x"]["G"]),
            )
        records.append(
            {
                "triple_id": triple_id,
                "seed": int(by_role["A"]["seed"]),
                "B": int(by_role["A"]["B"]),
                "anchor_type": str(by_role["A"]["anchor_type"]),
                "x_phase": str(by_role["A"].get("x_phase", "unknown")),
                "DR_gap": float(gap),
            }
        )
    return records


def compute_diminishing_returns(triple_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = _triple_gap_records(triple_rows)
    if not records:
        return {"status": "no_triples", "n_triples": 0}
    gaps = [float(row["DR_gap"]) for row in records]
    by_anchor: dict[str, Any] = {}
    by_phase: dict[str, Any] = {}
    for anchor_type in sorted({row["anchor_type"] for row in records}):
        by_anchor[anchor_type] = summarize_distribution(
            [row["DR_gap"] for row in records if row["anchor_type"] == anchor_type]
        )
    for phase in sorted({row["x_phase"] for row in records}):
        by_phase[phase] = summarize_distribution(
            [row["DR_gap"] for row in records if row["x_phase"] == phase]
        )
    return {
        "status": "identified",
        "n_triples": len(records),
        "gap": summarize_distribution(gaps),
        "probability_DR_gap_nonnegative": float(sum(gap >= 0.0 for gap in gaps) / len(gaps)),
        "by_anchor_type": by_anchor,
        "by_x_phase": by_phase,
    }


def compute_true_g_accounting(
    neighborhood_rows: Sequence[dict[str, Any]],
    triple_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(neighborhood_rows) + list(triple_rows)
    by_diag: dict[str, int] = defaultdict(int)
    by_anchor: dict[str, int] = defaultdict(int)
    for row in rows:
        calls = int(row.get("n_g_calls", 0))
        by_diag[str(row.get("diagnostic_type", "unknown"))] += calls
        by_anchor[str(row.get("anchor_type", "unknown"))] += calls
    return {
        "total_true_G_calls": int(sum(by_diag.values())),
        "by_diagnostic_type": dict(sorted(by_diag.items())),
        "by_anchor_type": dict(sorted(by_anchor.items())),
    }


def compute_bo_suitability(
    *,
    local_optimality: Mapping[str, Any],
    neighbor_delta: Mapping[str, Any],
    smoothness: Mapping[str, Any],
    diminishing_returns: Mapping[str, Any],
) -> dict[str, Any]:
    spearman = smoothness.get("spearman_distance_abs_deltaG")
    one_swap_mean = smoothness.get("one_swap_mean_abs_deltaG")
    larger_mean = smoothness.get("mean_abs_deltaG_by_larger_distance", {}).get("mean")
    local_fracs = [
        cell.get("fraction_local_optimum")
        for cell in local_optimality.values()
        if isinstance(cell, Mapping) and cell.get("fraction_local_optimum") is not None
    ]
    mean_local = float(np.mean(local_fracs)) if local_fracs else None
    dr_prob = diminishing_returns.get("probability_DR_gap_nonnegative")

    evidence: list[str] = []
    blockers: list[str] = []
    if spearman is None:
        blockers.append("distance/|DeltaG| correlation not identifiable")
    elif float(spearman) >= 0.25:
        evidence.append("schedule-distance kernel has positive rank signal")
    elif float(spearman) < 0.10:
        blockers.append("schedule-distance kernel rank signal is weak")
    else:
        blockers.append("schedule-distance kernel signal is borderline")

    if one_swap_mean is not None and larger_mean is not None and float(larger_mean) > 1.15 * float(one_swap_mean):
        evidence.append("one-swap neighborhoods are smoother than larger-distance pairs")
    else:
        blockers.append("local smoothness separation is weak or missing")

    if mean_local is not None and mean_local >= 0.5:
        evidence.append("many anchors are sampled local optima")
    elif mean_local is None:
        blockers.append("local optimality not identified")
    else:
        blockers.append("many anchors still have sampled one-swap improvements")

    if dr_prob is not None and float(dr_prob) >= 0.65:
        evidence.append("diminishing-return gaps are often nonnegative")
    elif dr_prob is not None and float(dr_prob) < 0.45:
        blockers.append("diminishing returns often fail")

    if len(evidence) >= 3 and not any("weak" in blocker or "missing" in blocker for blocker in blockers):
        verdict = "BO_supported"
    elif spearman is not None and float(spearman) < 0.10 and (
        one_swap_mean is None or larger_mean is None or float(larger_mean) <= 1.05 * float(one_swap_mean)
    ):
        verdict = "BO_not_supported"
    else:
        verdict = "BO_unclear"

    return {
        "verdict": verdict,
        "evidence": evidence,
        "blocking_gaps": blockers,
        "spearman_distance_abs_deltaG": spearman,
        "one_swap_mean_abs_deltaG": one_swap_mean,
        "larger_distance_mean_abs_deltaG": larger_mean,
        "mean_fraction_local_optimum": mean_local,
        "probability_DR_gap_nonnegative": dr_prob,
    }


def compute_aggregate_stats(
    *,
    neighborhood_rows: Sequence[dict[str, Any]],
    triple_rows: Sequence[dict[str, Any]],
    known_anchor_types: Sequence[str] = ANCHOR_TYPES,
) -> dict[str, Any]:
    local = compute_local_optimality_by_anchor(
        neighborhood_rows,
        known_anchor_types=known_anchor_types,
    )
    neighbor_delta = compute_neighbor_delta_stats(neighborhood_rows)
    smoothness = compute_smoothness(neighborhood_rows)
    dr = compute_diminishing_returns(triple_rows)
    accounting = compute_true_g_accounting(neighborhood_rows, triple_rows)
    bo = compute_bo_suitability(
        local_optimality=local,
        neighbor_delta=neighbor_delta,
        smoothness=smoothness,
        diminishing_returns=dr,
    )
    return {
        "row_validation": {
            "neighborhood": _validate_required_rows(neighborhood_rows),
            "diminishing_return": _validate_required_rows(triple_rows),
        },
        "local_optimality_by_anchor": local,
        "neighbor_delta_distribution": neighbor_delta["overall"],
        "neighbor_delta_by_B": neighbor_delta["by_B"],
        "neighbor_improvement_probability": neighbor_delta["probability_improvement"],
        "smoothness_distance_vs_deltaG": smoothness,
        "diminishing_returns": dr,
        "BO_kernel_suitability": bo,
        "true_G_call_accounting": accounting,
    }


def _new_figure(title: str) -> Any:
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_title(title)
    return fig, ax


def _save_figure(fig: Any, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if fig is None:
        for suffix in ("png", "pdf"):
            (figures_dir / f"{name}.{suffix}").write_text(f"{name}: matplotlib unavailable\n")
        return
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}.png", dpi=160)
    fig.savefig(figures_dir / f"{name}.pdf")
    plt.close(fig)


def _plot_placeholder(figures_dir: Path, name: str, message: str) -> None:
    if not HAS_MPL and HAS_PIL:
        _save_pil_summary(figures_dir, name, title=name, lines=[message])
        return
    item = _new_figure(name)
    if item is None:
        _save_figure(None, figures_dir, name)
        return
    fig, ax = item
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    _save_figure(fig, figures_dir, name)


def _save_pil_summary(figures_dir: Path, name: str, *, title: str, lines: Sequence[str]) -> None:
    """Write a valid PNG/PDF fallback when matplotlib is unavailable."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1100, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(50, 50, 50), width=2)
    draw.text((40, 32), title, fill=(0, 0, 0))
    y = 86
    for line in lines:
        text = str(line)
        while len(text) > 0:
            chunk = text[:120]
            draw.text((44, y), chunk, fill=(20, 20, 20))
            text = text[120:]
            y += 26
            if y > height - 54:
                break
        if y > height - 54:
            break
    image.save(figures_dir / f"{name}.png")
    image.save(figures_dir / f"{name}.pdf", "PDF")


def _write_pil_fallback_figures(
    *,
    neighborhood_rows: Sequence[dict[str, Any]],
    triple_rows: Sequence[dict[str, Any]],
    stats: Mapping[str, Any],
    figures_dir: Path,
) -> None:
    local = stats["local_optimality_by_anchor"]
    neighbor = stats["neighbor_delta_distribution"]
    by_B = stats["neighbor_delta_by_B"]
    smooth = stats["smoothness_distance_vs_deltaG"]
    dr = stats["diminishing_returns"]
    bo = stats["BO_kernel_suitability"]
    accounting = stats["true_G_call_accounting"]

    _save_pil_summary(
        figures_dir,
        "local_optimality_by_anchor",
        title="local_optimality_by_anchor",
        lines=[
            f"{anchor}: n={cell.get('n_anchors')} fraction_local={cell.get('fraction_local_optimum')}"
            for anchor, cell in local.items()
            if isinstance(cell, Mapping)
        ],
    )
    _save_pil_summary(
        figures_dir,
        "neighbor_delta_distribution",
        title="neighbor_delta_distribution",
        lines=[
            f"n={neighbor.get('n')}",
            f"mean={neighbor.get('mean')}",
            f"median={neighbor.get('median')}",
            f"ci95={neighbor.get('ci95')}",
            f"P(delta>0)={stats.get('neighbor_improvement_probability')}",
        ],
    )
    _save_pil_summary(
        figures_dir,
        "neighbor_delta_by_B",
        title="neighbor_delta_by_B",
        lines=[
            f"B={budget}: n={cell.get('n')} mean={cell.get('mean')} median={cell.get('median')}"
            for budget, cell in by_B.items()
            if isinstance(cell, Mapping)
        ],
    )
    _save_pil_summary(
        figures_dir,
        "smoothness_distance_vs_deltaG",
        title="smoothness_distance_vs_deltaG",
        lines=[
            f"pairwise comparisons={smooth.get('n_pairwise_comparisons')}",
            f"Spearman(distance, |DeltaG|)={smooth.get('spearman_distance_abs_deltaG')}",
            f"one-swap mean |DeltaG|={smooth.get('one_swap_mean_abs_deltaG')}",
            f"larger-distance mean |DeltaG|={smooth.get('mean_abs_deltaG_by_larger_distance', {}).get('mean')}",
        ],
    )
    _save_pil_summary(
        figures_dir,
        "diminishing_returns_gap_distribution",
        title="diminishing_returns_gap_distribution",
        lines=[
            f"status={dr.get('status')}",
            f"n_triples={dr.get('n_triples')}",
            f"mean={dr.get('gap', {}).get('mean')}",
            f"median={dr.get('gap', {}).get('median')}",
            f"ci95={dr.get('gap', {}).get('ci95')}",
            f"P(DR_gap>=0)={dr.get('probability_DR_gap_nonnegative')}",
        ],
    )
    _save_pil_summary(
        figures_dir,
        "DR_gap_by_phase_context",
        title="DR_gap_by_phase_context",
        lines=[
            f"{phase}: n={cell.get('n')} mean={cell.get('mean')} median={cell.get('median')}"
            for phase, cell in dr.get("by_x_phase", {}).items()
            if isinstance(cell, Mapping)
        ],
    )
    _save_pil_summary(
        figures_dir,
        "top_schedule_neighborhood_comparison",
        title="top_schedule_neighborhood_comparison",
        lines=[
            f"{anchor}: mean best one-swap improvement={cell.get('best_one_swap_improvement', {}).get('mean')}"
            for anchor, cell in local.items()
            if isinstance(cell, Mapping)
        ],
    )
    _save_pil_summary(
        figures_dir,
        "BO_kernel_suitability_summary",
        title=f"BO_kernel_suitability_summary: {bo.get('verdict')}",
        lines=[
            f"rho={bo.get('spearman_distance_abs_deltaG')}",
            f"one_swap_mean={bo.get('one_swap_mean_abs_deltaG')}",
            f"larger_distance_mean={bo.get('larger_distance_mean_abs_deltaG')}",
            f"mean_fraction_local={bo.get('mean_fraction_local_optimum')}",
            f"P(DR_gap>=0)={bo.get('probability_DR_gap_nonnegative')}",
            f"evidence={', '.join(bo.get('evidence', [])) or 'none'}",
            f"blocking={', '.join(bo.get('blocking_gaps', [])) or 'none'}",
        ],
    )
    _save_pil_summary(
        figures_dir,
        "true_G_call_accounting",
        title="true_G_call_accounting",
        lines=[
            f"total={accounting.get('total_true_G_calls')}",
            f"by_diagnostic_type={accounting.get('by_diagnostic_type')}",
            f"by_anchor_type={accounting.get('by_anchor_type')}",
            f"raw_rows={len(neighborhood_rows)} neighborhood + {len(triple_rows)} DR rows",
        ],
    )


def write_figures(
    *,
    neighborhood_rows: Sequence[dict[str, Any]],
    triple_rows: Sequence[dict[str, Any]],
    stats: Mapping[str, Any],
    figures_dir: Path,
) -> None:
    if not HAS_MPL and HAS_PIL:
        _write_pil_fallback_figures(
            neighborhood_rows=neighborhood_rows,
            triple_rows=triple_rows,
            stats=stats,
            figures_dir=figures_dir,
        )
        return

    local = stats["local_optimality_by_anchor"]
    item = _new_figure("local_optimality_by_anchor")
    if item is not None:
        fig, ax = item
        labels = list(local.keys())
        vals = [local[label].get("fraction_local_optimum") for label in labels]
        ax.bar(labels, [0.0 if value is None else float(value) for value in vals])
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction sampled local optimum")
        ax.tick_params(axis="x", rotation=30)
        _save_figure(fig, figures_dir, "local_optimality_by_anchor")
    else:
        _plot_placeholder(figures_dir, "local_optimality_by_anchor", "matplotlib unavailable")

    deltas = [
        float(row.get("delta_neighbor", float(row["G"]) - float(row.get("anchor_G", 0.0))))
        for row in _neighbor_rows(neighborhood_rows)
    ]
    if deltas and HAS_MPL:
        fig, ax = _new_figure("neighbor_delta_distribution")
        ax.hist(deltas, bins=30)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("G(neighbor) - G(anchor)")
        ax.set_ylabel("count")
        _save_figure(fig, figures_dir, "neighbor_delta_distribution")
    else:
        _plot_placeholder(figures_dir, "neighbor_delta_distribution", "No one-swap neighbor deltas")

    if deltas and HAS_MPL:
        fig, ax = _new_figure("neighbor_delta_by_B")
        budgets = sorted({int(row["B"]) for row in _neighbor_rows(neighborhood_rows)})
        data = [
            [
                float(row.get("delta_neighbor", float(row["G"]) - float(row.get("anchor_G", 0.0))))
                for row in _neighbor_rows(neighborhood_rows)
                if int(row["B"]) == budget
            ]
            for budget in budgets
        ]
        ax.boxplot(data, labels=[str(budget) for budget in budgets], showmeans=True)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("B")
        ax.set_ylabel("neighbor delta")
        _save_figure(fig, figures_dir, "neighbor_delta_by_B")
    else:
        _plot_placeholder(figures_dir, "neighbor_delta_by_B", "No one-swap neighbor deltas")

    smooth = stats["smoothness_distance_vs_deltaG"]
    distances: list[float] = []
    gaps: list[float] = []
    by_key: dict[tuple[int, int], dict[tuple[int, ...], float]] = defaultdict(dict)
    for row in neighborhood_rows:
        by_key[(int(row["seed"]), int(row["B"]))][canonical_schedule(row["schedule_steps"])] = float(row["G"])
    for schedules in by_key.values():
        items = sorted(schedules.items())
        for idx, (left_schedule, left_G) in enumerate(items):
            for right_schedule, right_G in items[idx + 1 :]:
                distances.append(float(schedule_distance(left_schedule, right_schedule)))
                gaps.append(abs(float(left_G) - float(right_G)))
    if distances and HAS_MPL:
        fig, ax = _new_figure("smoothness_distance_vs_deltaG")
        ax.scatter(distances, gaps, s=10, alpha=0.35)
        ax.set_xlabel("schedule distance")
        ax.set_ylabel("|Delta G|")
        ax.text(
            0.02,
            0.95,
            f"rho={smooth.get('spearman_distance_abs_deltaG')}",
            transform=ax.transAxes,
            va="top",
        )
        _save_figure(fig, figures_dir, "smoothness_distance_vs_deltaG")
    else:
        _plot_placeholder(figures_dir, "smoothness_distance_vs_deltaG", "No pairwise neighborhood comparisons")

    triple_records = _triple_gap_records(triple_rows)
    gaps_dr = [float(row["DR_gap"]) for row in triple_records]
    if gaps_dr and HAS_MPL:
        fig, ax = _new_figure("diminishing_returns_gap_distribution")
        ax.hist(gaps_dr, bins=30)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("DR_gap")
        _save_figure(fig, figures_dir, "diminishing_returns_gap_distribution")
    else:
        _plot_placeholder(figures_dir, "diminishing_returns_gap_distribution", "No exact DR triples")

    if gaps_dr and HAS_MPL:
        fig, ax = _new_figure("DR_gap_by_phase_context")
        phases = sorted({row["x_phase"] for row in triple_records})
        data = [[row["DR_gap"] for row in triple_records if row["x_phase"] == phase] for phase in phases]
        ax.boxplot(data, labels=phases, showmeans=True)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel("DR_gap")
        _save_figure(fig, figures_dir, "DR_gap_by_phase_context")
    else:
        _plot_placeholder(figures_dir, "DR_gap_by_phase_context", "No exact DR triples")

    best_by_anchor = {
        key: value.get("best_one_swap_improvement", {}).get("mean")
        for key, value in local.items()
        if isinstance(value, Mapping)
    }
    item = _new_figure("top_schedule_neighborhood_comparison")
    if item is not None:
        fig, ax = item
        labels = list(best_by_anchor.keys())
        values = [0.0 if best_by_anchor[label] is None else float(best_by_anchor[label]) for label in labels]
        ax.bar(labels, values)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel("mean best one-swap improvement")
        ax.tick_params(axis="x", rotation=30)
        _save_figure(fig, figures_dir, "top_schedule_neighborhood_comparison")
    else:
        _plot_placeholder(figures_dir, "top_schedule_neighborhood_comparison", "matplotlib unavailable")

    bo = stats["BO_kernel_suitability"]
    item = _new_figure("BO_kernel_suitability_summary")
    if item is not None:
        fig, ax = item
        labels = ["rho", "local opt.", "DR>=0"]
        values = [
            bo.get("spearman_distance_abs_deltaG"),
            bo.get("mean_fraction_local_optimum"),
            bo.get("probability_DR_gap_nonnegative"),
        ]
        ax.bar(labels, [0.0 if value is None else float(value) for value in values])
        ax.set_ylim(-0.2, 1.0)
        ax.set_title(f"BO_kernel_suitability_summary: {bo.get('verdict')}")
        _save_figure(fig, figures_dir, "BO_kernel_suitability_summary")
    else:
        _plot_placeholder(figures_dir, "BO_kernel_suitability_summary", "matplotlib unavailable")

    accounting = stats["true_G_call_accounting"]["by_diagnostic_type"]
    item = _new_figure("true_G_call_accounting")
    if item is not None:
        fig, ax = item
        labels = list(accounting.keys())
        values = [int(accounting[label]) for label in labels]
        ax.bar(labels, values)
        ax.set_ylabel("true-G calls")
        ax.tick_params(axis="x", rotation=20)
        _save_figure(fig, figures_dir, "true_G_call_accounting")
    else:
        _plot_placeholder(figures_dir, "true_G_call_accounting", "matplotlib unavailable")


def write_interpretation(out_dir: Path, stats: Mapping[str, Any]) -> None:
    local = stats["local_optimality_by_anchor"]
    neighbor = stats["neighbor_delta_distribution"]
    dr = stats["diminishing_returns"]
    bo = stats["BO_kernel_suitability"]
    lines = [
        "# Phase 4 Schedule-Neighborhood Diagnostics",
        "",
        "Scope: ProSeCo-OWT empirical diagnostics only. Model-agnostic claims remain formal/theoretical unless a cross-backbone feasibility gate passes.",
        "",
        "## Local Search Evidence",
        "",
        f"- Neighbor delta rows: {neighbor.get('n', 0)}.",
        f"- Mean neighbor delta: {neighbor.get('mean')}; median: {neighbor.get('median')}; 95% CI: {neighbor.get('ci95')}.",
        f"- Probability a sampled one-swap neighbor improves the anchor: {stats.get('neighbor_improvement_probability')}.",
    ]
    for anchor_type, cell in local.items():
        if cell.get("status") == "no_rows":
            continue
        lines.append(
            f"- {anchor_type}: n_anchors={cell.get('n_anchors')}, "
            f"fraction_local_optimum={cell.get('fraction_local_optimum')}, "
            f"mean_best_one_swap_improvement="
            f"{cell.get('best_one_swap_improvement', {}).get('mean')}."
        )
    lines.extend(
        [
            "",
            "## Diminishing Returns",
            "",
        ]
    )
    if dr.get("status") == "identified":
        lines.extend(
            [
                f"- Exact triples: {dr.get('n_triples')}.",
                f"- DR_gap mean: {dr.get('gap', {}).get('mean')}; median: {dr.get('gap', {}).get('median')}; "
                f"95% CI: {dr.get('gap', {}).get('ci95')}.",
                f"- P(DR_gap >= 0): {dr.get('probability_DR_gap_nonnegative')}.",
            ]
        )
    else:
        lines.append("- Exact diminishing-return triples were not available in this run.")
    lines.extend(
        [
            "",
            "## BO Kernel Suitability",
            "",
            f"- Verdict: {bo.get('verdict')}.",
            f"- Distance/|DeltaG| Spearman rho: {bo.get('spearman_distance_abs_deltaG')}.",
            f"- One-swap mean |DeltaG|: {bo.get('one_swap_mean_abs_deltaG')}.",
            f"- Larger-distance mean |DeltaG|: {bo.get('larger_distance_mean_abs_deltaG')}.",
            f"- Evidence: {', '.join(bo.get('evidence', [])) or 'none'}.",
            f"- Blocking gaps: {', '.join(bo.get('blocking_gaps', [])) or 'none'}.",
            "",
            "Bayesian model framing: treat G(S) as an expensive stochastic set function. A candidate GP prior is G(S) ~ GP(m(S), k(S,S')) with",
            "",
            "k(S,S') = alpha * |S cap S'| / B + beta * exp(-d_time(S,S') / ell_time) + gamma * exp(-d_phase(S,S') / ell_phase) + delta * k_signal(S,S').",
            "",
            "Possible acquisitions are Thompson sampling over a fixed candidate pool, expected improvement, and batch BO/successive halving for parallel true-G calls. BO should only become a spec if the diagnostics support kernel smoothness; otherwise it remains future work, not the thesis mainline.",
            "",
            "## LLaDA Feasibility Gate",
            "",
            "Do not pivot to LLaDA scheduling from this run. The prior LLaDA-SFT probe was blocked by metric/protocol interpretability: the F metric and corrector behavior did not establish stable positive MC-oracle headroom.",
            "",
            "Minimal authorization run: K=8 or K=10 pilot with uniform, MC-oracle pool, and corrector non-degeneracy checks. Authorize full scheduling only if MC-oracle minus uniform has a positive CI lower bound, or at least practically meaningful positive headroom with stable F and non-degenerate corrector behavior. Falsify LLaDA as thesis backbone if headroom remains nonpositive/unstable or the corrector is degenerate under the intended F metric.",
        ]
    )
    (out_dir / "interpretation.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze phase-4 schedule-neighborhood diagnostics")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    neighborhood_rows = _load_rows(results_dir, "neighborhood_raw")
    triple_rows = _load_rows(results_dir, "triples_raw")
    if not (out_dir / "neighborhood_raw.json").exists():
        _json_dump(out_dir / "neighborhood_raw.json", neighborhood_rows)
    if not (out_dir / "triples_raw.json").exists():
        _json_dump(out_dir / "triples_raw.json", triple_rows)

    stats = compute_aggregate_stats(
        neighborhood_rows=neighborhood_rows,
        triple_rows=triple_rows,
    )
    _json_dump(out_dir / "aggregate_stats.json", stats)
    write_figures(
        neighborhood_rows=neighborhood_rows,
        triple_rows=triple_rows,
        stats=stats,
        figures_dir=out_dir / "figures",
    )
    write_interpretation(out_dir, stats)
    print(
        "PHASE4_SCHEDULE_NEIGHBORHOOD_ANALYSIS_COMPLETE "
        f"neighborhood_rows={len(neighborhood_rows)} "
        f"triple_rows={len(triple_rows)} "
        f"bo_verdict={stats['BO_kernel_suitability']['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
