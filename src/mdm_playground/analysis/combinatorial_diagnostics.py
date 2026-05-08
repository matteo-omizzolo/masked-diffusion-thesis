"""Combinatorial diagnostics for schedule-structure analysis.

Utilities in this module quantify schedule-level structure from Phase 2b raw
rows (MC schedules + policy schedules). They are designed to stay backend-agnostic
and operate on JSON-compatible dictionaries.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np


def schedule_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    """Return Jaccard similarity between two schedules."""
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 1.0
    return float(len(sa & sb) / len(union))


def mean_pairwise_jaccard(schedules: Sequence[Sequence[int]]) -> float:
    """Mean pairwise Jaccard over a collection of schedules."""
    if len(schedules) < 2:
        return 0.0
    vals = [schedule_jaccard(a, b) for a, b in combinations(schedules, 2)]
    return float(np.mean(vals)) if vals else 0.0


def random_jaccard_baseline(
    T: int,
    B: int,
    n_samples: int = 5000,
    seed: int = 0,
) -> float:
    """Monte Carlo baseline E[J(S1, S2)] for random size-B schedules over [0, T)."""
    if B <= 0 or T <= 0 or B > T:
        return 0.0
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_samples):
        a = rng.choice(T, size=B, replace=False).tolist()
        b = rng.choice(T, size=B, replace=False).tolist()
        vals.append(schedule_jaccard(a, b))
    return float(np.mean(vals)) if vals else 0.0


def _group_mc_by_seed_B(
    mc_rows: Sequence[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in mc_rows:
        key = (int(row["seed"]), int(row["B"]))
        grouped[key].append(row)
    return grouped


def _oracle_schedule_by_seed_B(
    policy_rows: Sequence[dict[str, Any]],
) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    for row in policy_rows:
        if row.get("policy") != "mean_delta_oracle":
            continue
        out[(int(row["seed"]), int(row["B"]))] = [int(x) for x in row["schedule_steps"]]
    return out


def overlap_diagnostics(
    mc_rows: Sequence[dict[str, Any]],
    policy_rows: Sequence[dict[str, Any]],
    top_k: int = 10,
    random_baseline_samples: int = 5000,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Compute overlap diagnostics against mean-delta-oracle schedules.

    Returns per-B:
    - top-k MC vs oracle Jaccard (mean over seeds)
    - random Jaccard baseline
    - ratio over baseline
    - top-k and bottom-k internal Jaccard
    """
    grouped = _group_mc_by_seed_B(mc_rows)
    oracle = _oracle_schedule_by_seed_B(policy_rows)
    Bs = sorted({int(r["B"]) for r in mc_rows})

    per_B: dict[str, dict[str, Any]] = {}
    for B in Bs:
        top_vs_oracle: list[float] = []
        top_internal: list[float] = []
        bottom_internal: list[float] = []
        n_seed_used = 0

        keys = sorted(k for k in grouped if k[1] == B)
        for seed, _ in keys:
            rows = grouped[(seed, B)]
            if (seed, B) not in oracle or not rows:
                continue
            n_seed_used += 1
            rows_sorted = sorted(rows, key=lambda r: float(r["G"]), reverse=True)
            k = min(top_k, len(rows_sorted))
            top_rows = rows_sorted[:k]
            bottom_rows = rows_sorted[-k:]

            oracle_sched = oracle[(seed, B)]
            top_j = [schedule_jaccard(r["schedule_steps"], oracle_sched) for r in top_rows]
            top_vs_oracle.append(float(np.mean(top_j)) if top_j else 0.0)
            top_internal.append(mean_pairwise_jaccard([r["schedule_steps"] for r in top_rows]))
            bottom_internal.append(
                mean_pairwise_jaccard([r["schedule_steps"] for r in bottom_rows])
            )

        T = (
            max(
                (max(int(x) for x in r["schedule_steps"]) for r in mc_rows if int(r["B"]) == B),
                default=-1,
            )
            + 1
        )
        rand_base = random_jaccard_baseline(
            T=max(T, B),
            B=B,
            n_samples=random_baseline_samples,
            seed=random_seed + B,
        )
        mean_top_vs_oracle = float(np.mean(top_vs_oracle)) if top_vs_oracle else 0.0
        ratio = mean_top_vs_oracle / rand_base if rand_base > 0 else float("nan")

        per_B[str(B)] = {
            "B": int(B),
            "n_seeds": int(n_seed_used),
            "top_k": int(top_k),
            "mean_topk_vs_oracle_jaccard": mean_top_vs_oracle,
            "random_jaccard_baseline": float(rand_base),
            "ratio_topk_vs_random": float(ratio),
            "mean_topk_internal_jaccard": float(np.mean(top_internal)) if top_internal else 0.0,
            "mean_bottomk_internal_jaccard": (
                float(np.mean(bottom_internal)) if bottom_internal else 0.0
            ),
        }
    return {"per_B": per_B}


def variance_decomposition(mc_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Decompose variance of G into within-seed and between-seed components per B."""
    by_B_seed: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in mc_rows:
        by_B_seed[int(row["B"])][int(row["seed"])].append(float(row["G"]))

    out: dict[str, dict[str, float]] = {}
    for B, per_seed in sorted(by_B_seed.items()):
        seed_means = np.asarray([np.mean(v) for v in per_seed.values()], dtype=float)
        within_vars = np.asarray(
            [np.var(v, ddof=1) if len(v) > 1 else 0.0 for v in per_seed.values()],
            dtype=float,
        )
        all_vals = np.asarray([x for vals in per_seed.values() for x in vals], dtype=float)

        between_var = float(np.var(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
        within_var = float(np.mean(within_vars)) if len(within_vars) else 0.0
        total_var = float(np.var(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0

        share_within = within_var / total_var if total_var > 0 else 0.0
        share_between = between_var / total_var if total_var > 0 else 0.0
        out[str(B)] = {
            "B": int(B),
            "n_seeds": int(len(per_seed)),
            "n_points": int(len(all_vals)),
            "var_total_G": total_var,
            "var_between_seed_means": between_var,
            "var_within_seed": within_var,
            "within_share": float(share_within),
            "between_share": float(share_between),
        }
    return {"per_B": out}


def build_combinatorial_diagnostics(
    mc_rows: Sequence[dict[str, Any]],
    policy_rows: Sequence[dict[str, Any]],
    top_k: int = 10,
    random_baseline_samples: int = 5000,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Build full diagnostics bundle from phase2b raw rows."""
    return {
        "overlap": overlap_diagnostics(
            mc_rows=mc_rows,
            policy_rows=policy_rows,
            top_k=top_k,
            random_baseline_samples=random_baseline_samples,
            random_seed=random_seed,
        ),
        "variance": variance_decomposition(mc_rows=mc_rows),
    }
