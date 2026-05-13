#!/usr/bin/env python3
"""Gate 3b B=3/4 schedule-level validation.

This script extends the B=2 seed-split Gate 3b validation to true B=3 and
B=4 schedule rows that already exist locally in the Phase 2b Monte Carlo pool.
It asks whether

    Q_hat(S) = A(S) + sum_{t<t' in S} xi_hat(t,t')

predicts held-out schedule gains G(S) better than the additive surrogate A(S).

No deployable xi_hat model is fitted on held-out schedule gains. Gate 3a
pair rows are split by seed; each fitted xi_hat model sees only training-seed
pair rows and is evaluated on held-out-seed B=3/4 schedule rows. The observed
xi oracle is reported only when every internal pair in S was directly measured
for the held-out seed; with the sparse Gate 3a graph this is usually empty for
B=3/4 and is a diagnostic ceiling, not a deployable model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_phase1_schedule_level_b2 import (  # noqa: E402
    FittedXiModel,
    ModelSpec,
    build_splits,
    distance_bucket,
    fit_xi_model,
    git_sha,
    load_xi_rows,
    model_metadata,
    pearson_corr,
    predict_xi,
    savefig,
    spearman_corr,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None  # type: ignore[assignment]


DEFAULT_SCHEDULE_ROW_PATHS = [
    REPO_ROOT / "results" / "phase2b_proseco_owt" / "mc_raw.json",
    REPO_ROOT / "results" / "phase2b_k30_rep_cf89e00" / "mc_raw.json",
    REPO_ROOT / "results" / "phase2b" / "mc_raw.json",
]


def phase_for_step(step: int, *, T: int = 64) -> str:
    """Return the phase bin used by the Gate 3a interaction diagnostics."""
    if step < 0 or step >= T:
        raise ValueError(f"step must be in [0, {T - 1}], got {step}")
    if step <= 21:
        return "early"
    if step <= 42:
        return "middle"
    return "late"


def canonical_pair(t: int, tp: int) -> Tuple[int, int]:
    if t == tp:
        raise ValueError("pair timesteps must be distinct")
    return (int(t), int(tp)) if int(t) < int(tp) else (int(tp), int(t))


def internal_pairs(steps: Sequence[int]) -> List[Tuple[int, int]]:
    return [canonical_pair(t, tp) for t, tp in itertools.combinations(sorted(int(s) for s in steps), 2)]


def phase_pair_from_steps(t: int, tp: int) -> str:
    return f"{phase_for_step(t)}-{phase_for_step(tp)}"


def phase_distance_key_from_pair(t: int, tp: int) -> str:
    return f"{phase_pair_from_steps(t, tp)}|{distance_bucket(abs(int(tp) - int(t)))}"


def load_schedule_rows(path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    required = {"seed", "B", "mc_idx", "schedule_steps", "A", "G"}
    for idx, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"schedule row {idx} missing required keys: {sorted(missing)}")
    return rows


def resolve_schedule_rows_path(path: Optional[Path]) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for candidate in DEFAULT_SCHEDULE_ROW_PATHS:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in DEFAULT_SCHEDULE_ROW_PATHS)
    raise FileNotFoundError(f"No schedule row file found. Searched: {searched}")


def measured_pair_set(xi_rows: Sequence[Dict[str, Any]]) -> set[Tuple[int, int]]:
    return {canonical_pair(int(r["t"]), int(r["t_prime"])) for r in xi_rows}


def normalize_schedule_row(
    row: Dict[str, Any],
    measured_pairs: set[Tuple[int, int]],
    *,
    source: str = "",
) -> Dict[str, Any]:
    steps = sorted(int(s) for s in row["schedule_steps"])
    budget = int(row["B"])
    if len(steps) != budget:
        raise ValueError(f"schedule row has B={budget} but {len(steps)} steps: {steps}")
    if len(set(steps)) != len(steps):
        raise ValueError(f"schedule row has duplicate steps: {steps}")
    pairs = internal_pairs(steps)
    measured_internal = [list(p) for p in pairs if p in measured_pairs]
    missing_internal = [list(p) for p in pairs if p not in measured_pairs]
    n_pairs = len(pairs)
    seed = int(row["seed"])
    mc_idx = int(row.get("mc_idx", row.get("idx", -1)))
    candidate_id = f"B{budget}_seed{seed}_mc{mc_idx}_{'-'.join(str(s) for s in steps)}"
    phases = [phase_for_step(s) for s in steps]
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "B": budget,
        "mc_idx": mc_idx,
        "schedule_steps": steps,
        "phase_counts": dict(sorted(Counter(phases).items())),
        "source": source or str(row.get("source", "")),
        "A": float(row["A"]),
        "G": float(row["G"]),
        "residual": float(row.get("residual", float(row["G"]) - float(row["A"]))),
        "n_internal_pairs": n_pairs,
        "n_measured_internal_pairs": len(measured_internal),
        "n_missing_internal_pairs": len(missing_internal),
        "measured_internal_pairs": measured_internal,
        "missing_internal_pairs": missing_internal,
        "measured_pair_fraction": float(len(measured_internal) / n_pairs) if n_pairs else 1.0,
    }


def build_candidate_pool(
    schedule_rows: Sequence[Dict[str, Any]],
    measured_pairs: set[Tuple[int, int]],
    *,
    budgets: Sequence[int],
    coverage_mode: str = "estimator",
    source: str = "",
) -> List[Dict[str, Any]]:
    """Build a deterministic fixed candidate pool.

    coverage_mode="measured_only" rejects schedules with any unmeasured
    internal pair. coverage_mode="estimator" keeps schedules and records the
    missing pairs, which must be handled by pre-registered train-only xi
    estimators.
    """
    budget_set = {int(b) for b in budgets}
    if coverage_mode not in {"estimator", "measured_only"}:
        raise ValueError(f"Unknown coverage_mode: {coverage_mode}")

    pool: List[Dict[str, Any]] = []
    for row in schedule_rows:
        if int(row["B"]) not in budget_set:
            continue
        normalized = normalize_schedule_row(row, measured_pairs, source=source)
        if coverage_mode == "measured_only" and normalized["n_missing_internal_pairs"] > 0:
            continue
        pool.append(normalized)

    pool.sort(key=lambda r: (int(r["B"]), int(r["seed"]), int(r["mc_idx"]), tuple(r["schedule_steps"])))
    return pool


def candidate_pool_for_output(pool: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return candidate identities/coverage without held-out G targets."""
    output = []
    for row in pool:
        public = {k: v for k, v in row.items() if k not in {"G", "residual"}}
        public["G_target_stored_in_validation_rows"] = True
        output.append(public)
    return output


def build_pair_lookup(xi_rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    lookup: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for row in xi_rows:
        t, tp = canonical_pair(int(row["t"]), int(row["t_prime"]))
        lookup[(int(row["seed"]), t, tp)] = row
    return lookup


def build_delta_lookup(xi_rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], float]:
    grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for row in xi_rows:
        seed = int(row["seed"])
        grouped[(seed, int(row["t"]))].append(float(row["delta_t"]))
        grouped[(seed, int(row["t_prime"]))].append(float(row["delta_tp"]))
    return {key: float(np.mean(vals)) for key, vals in grouped.items()}


def pair_feature_row(
    candidate: Dict[str, Any],
    pair: Tuple[int, int],
    pair_lookup: Dict[Tuple[int, int, int], Dict[str, Any]],
    delta_lookup: Dict[Tuple[int, int], float],
) -> Tuple[Dict[str, Any], str]:
    """Create a pair-feature row for xi_hat prediction.

    If the held-out pair was directly measured for the held-out seed, reuse its
    A_pair feature. Otherwise estimate A_pair from per-step deltas observed in
    Gate 3a for that seed; if one step was never observed in Gate 3a, fall back
    to the schedule-average pair A implied by A(S). This fallback uses only the
    additive surrogate input, not held-out G(S).
    """
    seed = int(candidate["seed"])
    t, tp = canonical_pair(*pair)
    direct = pair_lookup.get((seed, t, tp))
    if direct is not None:
        return (
            {
                "seed": seed,
                "t": t,
                "t_prime": tp,
                "phase_t": direct["phase_t"],
                "phase_tp": direct["phase_tp"],
                "distance": int(direct["distance"]),
                "A_pair": float(direct["A_pair"]),
                "xi": 0.0,
            },
            "direct_pair_row",
        )

    dt = delta_lookup.get((seed, t))
    dtp = delta_lookup.get((seed, tp))
    if dt is not None and dtp is not None:
        a_pair = float(dt + dtp)
        source = "gate3a_step_delta"
    else:
        a_pair = float(2.0 * float(candidate["A"]) / float(candidate["B"]))
        source = "schedule_average_A"

    return (
        {
            "seed": seed,
            "t": t,
            "t_prime": tp,
            "phase_t": phase_for_step(t),
            "phase_tp": phase_for_step(tp),
            "distance": abs(tp - t),
            "A_pair": a_pair,
            "xi": 0.0,
        },
        source,
    )


def q_from_pair_xi(candidate: Dict[str, Any], pair_xi_hats: Sequence[float]) -> float:
    return float(candidate["A"] + float(np.sum(np.asarray(pair_xi_hats, dtype=float))))


def _empty_fitted(spec: ModelSpec) -> FittedXiModel:
    return FittedXiModel(
        spec=spec,
        global_mean=0.0,
        phase_means={},
        phase_distance_means={},
        phase_distance_counts={},
        phase_distance_shrunk={},
    )


def model_specs_for_b34(*, shrinkage_strength: float, ridge_alpha: float) -> List[ModelSpec]:
    return [
        ModelSpec("additive", shrinkage_strength, ridge_alpha),
        ModelSpec("oracle_observed_xi", shrinkage_strength, ridge_alpha),
        ModelSpec("train_global_mean", shrinkage_strength, ridge_alpha),
        ModelSpec("phase_pair_mean", shrinkage_strength, ridge_alpha),
        ModelSpec("phase_distance_mean", shrinkage_strength, ridge_alpha),
        ModelSpec("phase_distance_shrinkage", shrinkage_strength, ridge_alpha),
        ModelSpec("phase_distance_a_ridge", shrinkage_strength, ridge_alpha),
    ]


def predict_schedule(
    candidate: Dict[str, Any],
    spec: ModelSpec,
    fitted: FittedXiModel,
    pair_lookup: Dict[Tuple[int, int, int], Dict[str, Any]],
    delta_lookup: Dict[Tuple[int, int], float],
) -> Optional[Dict[str, Any]]:
    pairs = internal_pairs(candidate["schedule_steps"])
    if spec.name == "additive":
        return {
            "prediction": float(candidate["A"]),
            "pair_xi_hat_sum": 0.0,
            "pair_xi_hats": [0.0 for _ in pairs],
            "pair_feature_sources": [],
        }

    if spec.name == "oracle_observed_xi":
        xi_vals: List[float] = []
        for pair in pairs:
            row = pair_lookup.get((int(candidate["seed"]), pair[0], pair[1]))
            if row is None:
                return None
            xi_vals.append(float(row["xi"]))
        return {
            "prediction": q_from_pair_xi(candidate, xi_vals),
            "pair_xi_hat_sum": float(np.sum(xi_vals)),
            "pair_xi_hats": xi_vals,
            "pair_feature_sources": ["direct_pair_row" for _ in pairs],
        }

    xi_hats: List[float] = []
    feature_sources: List[str] = []
    for pair in pairs:
        feature_row, source = pair_feature_row(candidate, pair, pair_lookup, delta_lookup)
        xi_hats.append(float(predict_xi(fitted, feature_row)))
        feature_sources.append(source)
    return {
        "prediction": q_from_pair_xi(candidate, xi_hats),
        "pair_xi_hat_sum": float(np.sum(xi_hats)),
        "pair_xi_hats": xi_hats,
        "pair_feature_sources": feature_sources,
    }


def run_crossfit_predictions(
    xi_rows: Sequence[Dict[str, Any]],
    candidate_pool: Sequence[Dict[str, Any]],
    splits: Sequence[Dict[str, Any]],
    model_specs: Sequence[ModelSpec],
) -> List[Dict[str, Any]]:
    by_seed_xi: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in xi_rows:
        by_seed_xi[int(row["seed"])].append(row)
    by_seed_candidate: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidate_pool:
        by_seed_candidate[int(row["seed"])].append(row)

    pair_lookup = build_pair_lookup(xi_rows)
    delta_lookup = build_delta_lookup(xi_rows)
    output: List[Dict[str, Any]] = []

    for split in splits:
        train_seeds = set(int(s) for s in split["train_seeds"])
        heldout_seeds = set(int(s) for s in split["heldout_seeds"])
        train_rows = [r for seed in train_seeds for r in by_seed_xi[seed]]
        heldout_candidates = [r for seed in heldout_seeds for r in by_seed_candidate.get(seed, [])]

        fitted_by_model: Dict[str, FittedXiModel] = {}
        for spec in model_specs:
            if spec.name in {"additive", "oracle_observed_xi"}:
                fitted_by_model[spec.name] = _empty_fitted(spec)
            else:
                fitted_by_model[spec.name] = fit_xi_model(spec, train_rows)

        for candidate in heldout_candidates:
            g = float(candidate["G"])
            a = float(candidate["A"])
            for spec in model_specs:
                prediction = predict_schedule(
                    candidate,
                    spec,
                    fitted_by_model[spec.name],
                    pair_lookup,
                    delta_lookup,
                )
                if prediction is None:
                    continue
                meta = model_metadata(spec.name)
                pred = float(prediction["prediction"])
                feature_source_counts = dict(sorted(Counter(prediction["pair_feature_sources"]).items()))
                output.append(
                    {
                        "model": spec.name,
                        "display_name": meta["display_name"],
                        "role": meta["role"],
                        "non_deployable": bool(meta["non_deployable"]),
                        "fold_id": int(split["fold_id"]),
                        "train_seeds": sorted(train_seeds),
                        "heldout_seeds": sorted(heldout_seeds),
                        "train_seed_count": len(train_seeds),
                        "heldout_seed_count": len(heldout_seeds),
                        "candidate_id": candidate["candidate_id"],
                        "seed": int(candidate["seed"]),
                        "B": int(candidate["B"]),
                        "mc_idx": int(candidate["mc_idx"]),
                        "schedule_steps": list(candidate["schedule_steps"]),
                        "phase_counts": candidate["phase_counts"],
                        "source": candidate["source"],
                        "A": a,
                        "G": g,
                        "residual_G_minus_A": float(g - a),
                        "prediction": pred,
                        "residual_G_minus_Q": float(g - pred),
                        "pair_xi_hat_sum": float(prediction["pair_xi_hat_sum"]),
                        "pair_xi_hats": [float(x) for x in prediction["pair_xi_hats"]],
                        "pair_feature_source_counts": feature_source_counts,
                        "n_internal_pairs": int(candidate["n_internal_pairs"]),
                        "n_measured_internal_pairs": int(candidate["n_measured_internal_pairs"]),
                        "n_missing_internal_pairs": int(candidate["n_missing_internal_pairs"]),
                        "measured_pair_fraction": float(candidate["measured_pair_fraction"]),
                        "abs_err_A": float(abs(g - a)),
                        "abs_err_Q": float(abs(g - pred)),
                        "sq_err_A": float((g - a) ** 2),
                        "sq_err_Q": float((g - pred) ** 2),
                    }
                )
    return output


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(np.mean(vals))


def topk_overlap(rows: Sequence[Dict[str, Any]], *, score_key: str, top_k: int) -> Optional[float]:
    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)

    overlaps: List[float] = []
    for seed_rows in by_seed.values():
        if not seed_rows:
            continue
        k = min(top_k, len(seed_rows))
        true_top = {
            r["candidate_id"]
            for r in sorted(seed_rows, key=lambda r: (float(r["G"]), r["candidate_id"]), reverse=True)[:k]
        }
        pred_top = {
            r["candidate_id"]
            for r in sorted(seed_rows, key=lambda r: (float(r[score_key]), r["candidate_id"]), reverse=True)[:k]
        }
        overlaps.append(float(len(true_top & pred_top) / k))
    return _mean(overlaps)


def compute_schedule_metrics(rows: Sequence[Dict[str, Any]], *, top_k: int = 10) -> Dict[str, float]:
    if not rows:
        raise ValueError("Cannot compute metrics for empty rows")
    g = np.asarray([float(r["G"]) for r in rows], dtype=float)
    a = np.asarray([float(r["A"]) for r in rows], dtype=float)
    q = np.asarray([float(r["prediction"]) for r in rows], dtype=float)
    abs_a = np.asarray([float(r["abs_err_A"]) for r in rows], dtype=float)
    abs_q = np.asarray([float(r["abs_err_Q"]) for r in rows], dtype=float)
    sq_a = np.asarray([float(r["sq_err_A"]) for r in rows], dtype=float)
    sq_q = np.asarray([float(r["sq_err_Q"]) for r in rows], dtype=float)
    eta_mean = float(np.mean(abs_a))
    zeta_mean = float(np.mean(abs_q))
    r_s = spearman_corr(a, g)
    p_s = spearman_corr(q, g)
    r_p = pearson_corr(a, g)
    p_p = pearson_corr(q, g)
    top_a = topk_overlap(rows, score_key="A", top_k=top_k)
    top_q = topk_overlap(rows, score_key="prediction", top_k=top_k)
    return {
        "n_rows": float(len(rows)),
        "n_seeds": float(len({int(r["seed"]) for r in rows})),
        "eta_mean": eta_mean,
        "zeta_mean": zeta_mean,
        "eta_median": float(np.median(abs_a)),
        "zeta_median": float(np.median(abs_q)),
        "eta_p90": float(np.quantile(abs_a, 0.9)),
        "zeta_p90": float(np.quantile(abs_q, 0.9)),
        "RMSE_A": float(math.sqrt(np.mean(sq_a))),
        "RMSE_Q": float(math.sqrt(np.mean(sq_q))),
        "R_spearman": r_s,
        "P_spearman": p_s,
        "R_pearson": r_p,
        "P_pearson": p_p,
        "delta_abs_error": eta_mean - zeta_mean,
        "delta_spearman": p_s - r_s if not (math.isnan(p_s) or math.isnan(r_s)) else float("nan"),
        "delta_pearson": p_p - r_p if not (math.isnan(p_p) or math.isnan(r_p)) else float("nan"),
        "topk_overlap_A": float(top_a) if top_a is not None else float("nan"),
        "topk_overlap_Q": float(top_q) if top_q is not None else float("nan"),
        "delta_topk_overlap": float(top_q - top_a) if top_a is not None and top_q is not None else float("nan"),
    }


def seed_cluster_bootstrap(
    rows: Sequence[Dict[str, Any]],
    *,
    n_boot: int,
    random_seed: int,
    top_k: int,
) -> Dict[str, Any]:
    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    seeds = sorted(by_seed)
    rng = np.random.default_rng(random_seed)
    keys = [
        "eta_mean",
        "zeta_mean",
        "delta_abs_error",
        "R_spearman",
        "P_spearman",
        "delta_spearman",
        "topk_overlap_A",
        "topk_overlap_Q",
        "delta_topk_overlap",
    ]
    samples: Dict[str, List[float]] = {key: [] for key in keys}
    for _ in range(n_boot):
        sampled_seeds = list(rng.choice(seeds, size=len(seeds), replace=True))
        boot_rows: List[Dict[str, Any]] = []
        for seed in sampled_seeds:
            boot_rows.extend(by_seed[int(seed)])
        metrics = compute_schedule_metrics(boot_rows, top_k=top_k)
        for key in keys:
            value = float(metrics[key])
            if not math.isnan(value):
                samples[key].append(value)

    out: Dict[str, Any] = {"n_boot": int(n_boot), "cluster_unit": "seed", "metrics": {}}
    for key, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        if len(arr) == 0:
            out["metrics"][key] = {"ci_lo_95": None, "ci_hi_95": None, "p_gt_0": None}
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out["metrics"][key] = {
                "ci_lo_95": float(lo),
                "ci_hi_95": float(hi),
                "p_gt_0": float(np.mean(arr > 0)),
            }
    return out


def _json_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value_f = float(value)
    if math.isnan(value_f) or math.isinf(value_f):
        return None
    return value_f


def summarize_by_budget_model(
    validation_rows: Sequence[Dict[str, Any]],
    model_specs: Sequence[ModelSpec],
    *,
    budgets: Sequence[int],
    n_boot: int,
    random_seed: int,
    top_k: int,
) -> Dict[str, Any]:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        grouped[(int(row["B"]), row["model"])].append(row)

    out: Dict[str, Any] = {"per_budget": {}, "best_heldout_q_model_by_budget": {}}
    for b_idx, budget in enumerate(sorted(int(b) for b in budgets)):
        per_model: Dict[str, Any] = {}
        for m_idx, spec in enumerate(model_specs):
            name = spec.name
            meta = model_metadata(name)
            rows = grouped.get((budget, name), [])
            if not rows:
                per_model[name] = {
                    "display_name": meta["display_name"],
                    "role": meta["role"],
                    "non_deployable": meta["non_deployable"],
                    "point": None,
                    "bootstrap": None,
                    "unavailable_reason": "No held-out rows; for oracle this means at least one internal pair was unmeasured.",
                }
                continue
            point = compute_schedule_metrics(rows, top_k=top_k)
            bootstrap = seed_cluster_bootstrap(
                rows,
                n_boot=n_boot,
                random_seed=random_seed + 100 * b_idx + m_idx,
                top_k=top_k,
            )
            per_model[name] = {
                "display_name": meta["display_name"],
                "role": meta["role"],
                "non_deployable": meta["non_deployable"],
                "point": {k: _json_float(v) for k, v in point.items()},
                "bootstrap": bootstrap,
            }

        deployable = {
            name: block
            for name, block in per_model.items()
            if block["role"] == "heldout_q" and not block["non_deployable"] and block["point"] is not None
        }
        best_model = None
        if deployable:
            best_model = max(
                deployable,
                key=lambda name: (
                    deployable[name]["point"]["delta_abs_error"] or -float("inf"),
                    deployable[name]["point"]["delta_spearman"] or -float("inf"),
                ),
            )
        out["per_budget"][str(budget)] = {
            "best_heldout_q_model": best_model,
            "per_model": per_model,
        }
        out["best_heldout_q_model_by_budget"][str(budget)] = best_model
    return out


def measured_graph_summary(
    measured_pairs: set[Tuple[int, int]],
    *,
    budgets: Sequence[int],
) -> Dict[str, Any]:
    nodes = sorted({step for pair in measured_pairs for step in pair})
    summary: Dict[str, Any] = {
        "n_measured_pairs": len(measured_pairs),
        "n_measured_steps": len(nodes),
        "measured_steps": nodes,
        "cliques_by_budget": {},
    }
    for budget in budgets:
        cliques = []
        for combo in itertools.combinations(nodes, int(budget)):
            if all(canonical_pair(*pair) in measured_pairs for pair in itertools.combinations(combo, 2)):
                cliques.append(list(combo))
        summary["cliques_by_budget"][str(int(budget))] = {
            "n_cliques": len(cliques),
            "cliques": cliques,
        }
    return summary


def candidate_pool_summary(pool: Sequence[Dict[str, Any]], *, budgets: Sequence[int]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_rows": len(pool), "by_budget": {}}
    for budget in sorted(int(b) for b in budgets):
        rows = [r for r in pool if int(r["B"]) == budget]
        if not rows:
            summary["by_budget"][str(budget)] = {"n_rows": 0}
            continue
        by_seed = Counter(int(r["seed"]) for r in rows)
        coverage = np.asarray([float(r["measured_pair_fraction"]) for r in rows], dtype=float)
        missing = np.asarray([int(r["n_missing_internal_pairs"]) for r in rows], dtype=float)
        summary["by_budget"][str(budget)] = {
            "n_rows": len(rows),
            "n_seeds": len(by_seed),
            "rows_per_seed_min": min(by_seed.values()),
            "rows_per_seed_max": max(by_seed.values()),
            "n_fully_measured": int(sum(1 for r in rows if int(r["n_missing_internal_pairs"]) == 0)),
            "n_zero_measured": int(sum(1 for r in rows if int(r["n_measured_internal_pairs"]) == 0)),
            "mean_measured_pair_fraction": float(np.mean(coverage)),
            "min_measured_pair_fraction": float(np.min(coverage)),
            "max_measured_pair_fraction": float(np.max(coverage)),
            "mean_missing_internal_pairs": float(np.mean(missing)),
        }
    return summary


def rows_for(validation_rows: Sequence[Dict[str, Any]], *, budget: Optional[int] = None, model: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = []
    for row in validation_rows:
        if budget is not None and int(row["B"]) != int(budget):
            continue
        if model is not None and row["model"] != model:
            continue
        rows.append(row)
    return rows


def make_figures(
    validation_rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    pool_summary: Dict[str, Any],
    figures_dir: Path,
) -> List[str]:
    if not HAS_MPL:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    budgets = sorted(int(b) for b in metrics["per_budget"])

    def _write(fig: Any, name: str) -> None:
        savefig(fig, figures_dir / name)
        written.extend([f"{name}.png", f"{name}.pdf"])

    def _boxplot(ax: Any, data: Sequence[Sequence[float]], labels: Sequence[str]) -> None:
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)

    labels: List[str] = []
    eta_vals: List[float] = []
    zeta_vals: List[float] = []
    colors: List[str] = []
    for budget in budgets:
        block = metrics["per_budget"][str(budget)]
        additive = block["per_model"]["additive"]["point"]
        eta = additive["eta_mean"] if additive else None
        for name, model_block in block["per_model"].items():
            if name in {"additive", "oracle_observed_xi"} or model_block["point"] is None:
                continue
            labels.append(f"B{budget}\n{name.replace('_', ' ')}")
            eta_vals.append(float(eta))
            zeta_vals.append(float(model_block["point"]["zeta_mean"]))
            colors.append("#59a14f" if float(model_block["point"]["delta_abs_error"]) > 0 else "#e15759")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, eta_vals, width, label="eta |G-A|", color="#4c78a8", alpha=0.75)
    ax.bar(x + width / 2, zeta_vals, width, label="zeta |G-Q|", color=colors, alpha=0.80)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean absolute error")
    ax.set_title("B=3/4 additive error vs pairwise-Q error")
    ax.legend(fontsize=8)
    _write(fig, "B3_B4_eta_vs_zeta")

    labels = []
    r_vals = []
    p_vals = []
    colors = []
    for budget in budgets:
        block = metrics["per_budget"][str(budget)]
        additive = block["per_model"]["additive"]["point"]
        r_val = additive["R_spearman"] if additive else None
        for name, model_block in block["per_model"].items():
            if name in {"additive", "oracle_observed_xi"} or model_block["point"] is None:
                continue
            labels.append(f"B{budget}\n{name.replace('_', ' ')}")
            r_vals.append(float(r_val))
            p_vals.append(float(model_block["point"]["P_spearman"]))
            colors.append("#59a14f" if float(model_block["point"]["delta_spearman"] or 0.0) > 0 else "#e15759")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.bar(x - width / 2, r_vals, width, label="R Spearman(A,G)", color="#4c78a8", alpha=0.75)
    ax.bar(x + width / 2, p_vals, width, label="P Spearman(Q,G)", color=colors, alpha=0.80)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("B=3/4 rank correlation")
    ax.legend(fontsize=8)
    _write(fig, "B3_B4_rank_correlation")

    fig, axes = plt.subplots(1, len(budgets), figsize=(6 * len(budgets), 5), squeeze=False)
    for ax, budget in zip(axes[0], budgets):
        best = metrics["per_budget"][str(budget)]["best_heldout_q_model"]
        best_rows = rows_for(validation_rows, budget=budget, model=best)
        additive_rows = rows_for(validation_rows, budget=budget, model="additive")
        ax.scatter([r["A"] for r in additive_rows], [r["G"] for r in additive_rows], s=10, alpha=0.18, label="A")
        if best_rows:
            ax.scatter([r["prediction"] for r in best_rows], [r["G"] for r in best_rows], s=10, alpha=0.22, label=f"Q {best}")
        all_x = [r["A"] for r in additive_rows] + [r["prediction"] for r in best_rows]
        all_y = [r["G"] for r in additive_rows] + [r["G"] for r in best_rows]
        lo = float(min(all_x + all_y))
        hi = float(max(all_x + all_y))
        pad = max((hi - lo) * 0.05, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Predicted gain")
        ax.set_ylabel("Held-out G")
        ax.set_title(f"B={budget} top model scatter")
        ax.legend(fontsize=8)
    _write(fig, "B3_B4_top_schedule_scatter")

    fig, ax = plt.subplots(figsize=(11, 5))
    data = []
    labels = []
    for budget in budgets:
        best = metrics["per_budget"][str(budget)]["best_heldout_q_model"]
        best_rows = rows_for(validation_rows, budget=budget, model=best)
        if best_rows:
            data.extend([
                [r["residual_G_minus_A"] for r in best_rows],
                [r["residual_G_minus_Q"] for r in best_rows],
            ])
            labels.extend([f"B{budget} G-A", f"B{budget} G-Q"])
    _boxplot(ax, data, labels)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Residual")
    ax.set_title("Residual distribution for best B=3/4 Q estimators")
    _write(fig, "B3_B4_residual_by_estimator")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(budgets))
    mean_cov = [pool_summary["by_budget"][str(b)]["mean_measured_pair_fraction"] for b in budgets]
    full = [pool_summary["by_budget"][str(b)]["n_fully_measured"] for b in budgets]
    zero = [pool_summary["by_budget"][str(b)]["n_zero_measured"] for b in budgets]
    ax.bar(x, mean_cov, color="#4c78a8", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}\nfull={full[i]}\nzero={zero[i]}" for i, b in enumerate(budgets)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean fraction of internal pairs directly measured")
    ax.set_title("Candidate pool Gate 3a pair coverage")
    _write(fig, "B3_B4_candidate_pool_coverage")

    fig, axes = plt.subplots(1, len(budgets), figsize=(6 * len(budgets), 4), squeeze=False)
    for ax, budget in zip(axes[0], budgets):
        best = metrics["per_budget"][str(budget)]["best_heldout_q_model"]
        best_rows = rows_for(validation_rows, budget=budget, model=best)
        additive_rows = rows_for(validation_rows, budget=budget, model="additive")
        phase_labels = ["early", "middle", "late"]
        series = []
        names = []
        for score_key, rows, name in [("A", additive_rows, "top A"), ("prediction", best_rows, "top Q")]:
            counts = Counter()
            by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_seed[int(row["seed"])].append(row)
            for seed_rows in by_seed.values():
                top = sorted(seed_rows, key=lambda r: (float(r[score_key]), r["candidate_id"]), reverse=True)[:10]
                for row in top:
                    counts.update(row["phase_counts"])
            total = sum(counts.values()) or 1
            series.append([counts[p] / total for p in phase_labels])
            names.append(name)
        xx = np.arange(len(phase_labels))
        ax.bar(xx - 0.18, series[0], 0.36, label=names[0], color="#4c78a8")
        ax.bar(xx + 0.18, series[1], 0.36, label=names[1], color="#f28e2b")
        ax.set_xticks(xx)
        ax.set_xticklabels(phase_labels)
        ax.set_ylim(0, 1)
        ax.set_title(f"B={budget} top schedule phase composition")
        ax.legend(fontsize=8)
    _write(fig, "B3_B4_phase_composition")

    fig, ax = plt.subplots(figsize=(11, 5))
    data = []
    labels = []
    for budget in budgets:
        for name, model_block in metrics["per_budget"][str(budget)]["per_model"].items():
            if name in {"additive", "oracle_observed_xi"} or model_block["point"] is None:
                continue
            rows = rows_for(validation_rows, budget=budget, model=name)
            data.append([r["pair_xi_hat_sum"] for r in rows])
            labels.append(f"B{budget}\n{name.replace('_', ' ')}")
    _boxplot(ax, data, labels)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("sum xi_hat over internal pairs")
    ax.set_title("Estimated pair-redundancy contribution to Q(S)")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    _write(fig, "B3_B4_pair_redundancy_contribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = []
    top_a = []
    top_q = []
    for budget in budgets:
        best = metrics["per_budget"][str(budget)]["best_heldout_q_model"]
        block = metrics["per_budget"][str(budget)]["per_model"][best]
        labels.append(f"B{budget}\n{best.replace('_', ' ')}")
        top_a.append(float(block["point"]["topk_overlap_A"]))
        top_q.append(float(block["point"]["topk_overlap_Q"]))
    x = np.arange(len(labels))
    ax.bar(x - width / 2, top_a, width, label="A top-k overlap", color="#4c78a8")
    ax.bar(x + width / 2, top_q, width, label="Q top-k overlap", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean per-seed top-k overlap")
    ax.set_title("Top-k retrieval of high-G schedules")
    ax.legend(fontsize=8)
    _write(fig, "B3_B4_topk_overlap")

    return written


def ci_text(block: Dict[str, Any], key: str) -> str:
    if block is None or block.get("bootstrap") is None:
        return "[NA, NA]"
    ci = block["bootstrap"]["metrics"][key]
    lo = ci["ci_lo_95"]
    hi = ci["ci_hi_95"]
    if lo is None or hi is None:
        return "[NA, NA]"
    return f"[{lo:.4f}, {hi:.4f}]"


def write_interpretation(
    out_dir: Path,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    pool_summary: Dict[str, Any],
    graph_summary: Dict[str, Any],
) -> None:
    lines = [
        "# Gate 3b B=3/4 Schedule-Level Validation",
        "",
        "## What This Tests",
        "",
        "This validation tests whether a training-seed pairwise surrogate",
        "Q_hat(S)=A(S)+sum xi_hat(t,t') improves held-out prediction/ranking of",
        "true B=3 and B=4 schedule gains over the additive surrogate A(S).",
        "",
        "The candidate pool is fixed to existing Phase 2b Monte Carlo schedule",
        "rows before fitting any xi estimator. Candidate identities are not",
        "selected using held-out schedule gains. Deployable xi_hat models are",
        "fit only on Gate 3a pair rows from training seeds and evaluated on",
        "held-out schedule rows.",
        "",
        "Gate 3a found mostly negative / redundant xi. A useful B=3/4 Q_hat",
        "should therefore improve by subtracting learned redundancy penalties",
        "from schedules whose additive score overpredicts true gain.",
        "",
        "The observed-xi oracle is non-deployable. It is only available for",
        "schedules whose every internal pair was directly measured in Gate 3a.",
        "",
        "## Provenance",
        "",
        f"- Validation git SHA: `{config['git_sha']}`",
        f"- Pair rows: `{config['xi_raw']}`",
        f"- Schedule rows: `{config['schedule_rows']}`",
        "- Gate 3a caveat: the pair-row input folder was `phase1_interaction_diag_nogit`",
        "  with unknown git hash; this validation output is SHA-tagged.",
        f"- Split mode: `{config['seed_split_mode']}`",
        f"- Bootstrap resamples: {config['n_boot']} clustered by seed",
        "",
        "## Candidate Pool Coverage",
        "",
        f"- Measured Gate 3a pairs: {graph_summary['n_measured_pairs']}.",
        f"- Measured graph B=3 cliques: {graph_summary['cliques_by_budget'].get('3', {}).get('n_cliques', 0)}.",
        f"- Measured graph B=4 cliques: {graph_summary['cliques_by_budget'].get('4', {}).get('n_cliques', 0)}.",
    ]
    for budget, block in pool_summary["by_budget"].items():
        lines.append(
            f"- B={budget}: {block['n_rows']} candidate rows, "
            f"{block['n_fully_measured']} fully measured, "
            f"mean internal-pair coverage {block['mean_measured_pair_fraction']:.3f}."
        )
    lines.extend(
        [
            "",
            "Because no B=3/4 candidate row is fully measured, this is an",
            "estimator-based local validation rather than an exact measured-pair",
            "validation. Exact measured-pair B=3/4 validation would require",
            "additional targeted pair measurements for the fixed candidate pool.",
            "",
            "## Model Comparison",
            "",
            "| B | Model | Role | zeta mean | eta-zeta | 95% CI eta-zeta | P Spearman | P-R | 95% CI P-R | top-k Q-A |",
            "|---:|---|---|---:|---:|---|---:|---:|---|---:|",
        ]
    )
    for budget in sorted(metrics["per_budget"], key=int):
        for name, block in metrics["per_budget"][budget]["per_model"].items():
            if block["point"] is None:
                lines.append(f"| {budget} | {name} | {block['role']} | NA | NA | [NA, NA] | NA | NA | [NA, NA] | NA |")
                continue
            point = block["point"]
            lines.append(
                f"| {budget} | {name} | {block['role']} | {point['zeta_mean']:.4f} | "
                f"{point['delta_abs_error']:.4f} | {ci_text(block, 'delta_abs_error')} | "
                f"{point['P_spearman']:.4f} | {point['delta_spearman']:.4f} | "
                f"{ci_text(block, 'delta_spearman')} | {point['delta_topk_overlap']:.4f} |"
            )

    lines.extend(["", "## Interpretation", ""])
    for budget in sorted(metrics["per_budget"], key=int):
        best = metrics["per_budget"][budget]["best_heldout_q_model"]
        additive = metrics["per_budget"][budget]["per_model"]["additive"]
        if not best:
            lines.append(f"- B={budget}: no deployable Q_hat model was evaluated.")
            continue
        best_block = metrics["per_budget"][budget]["per_model"][best]
        point = best_block["point"]
        delta_ci = best_block["bootstrap"]["metrics"]["delta_abs_error"]
        spear_ci = best_block["bootstrap"]["metrics"]["delta_spearman"]
        error_support = delta_ci["ci_lo_95"] is not None and delta_ci["ci_lo_95"] > 0
        rank_support = spear_ci["ci_lo_95"] is not None and spear_ci["ci_lo_95"] > 0
        support_text = "supports" if (error_support or rank_support) else "does not decisively support"
        lines.extend(
            [
                f"- B={budget}: best deployable model is `{best}`. Additive eta_mean "
                f"{additive['point']['eta_mean']:.4f}; best zeta_mean {point['zeta_mean']:.4f}; "
                f"eta-zeta {point['delta_abs_error']:.4f} with 95% CI "
                f"{ci_text(best_block, 'delta_abs_error')}.",
                f"  Ranking: R={additive['point']['R_spearman']:.4f}, "
                f"P={point['P_spearman']:.4f}, P-R={point['delta_spearman']:.4f} "
                f"with 95% CI {ci_text(best_block, 'delta_spearman')}.",
                f"  Decision: {support_text} the tested estimator-based Q_hat at B={budget}.",
            ]
        )

    lines.extend(
        [
            "",
            "## What Can and Cannot Be Claimed",
            "",
            "- This is a held-out, seed-split schedule-level test of a pre-registered",
            "  pairwise estimator on an existing fixed candidate pool.",
            "- This run does not support proceeding directly to a pairwise scheduler.",
            "  A positive future B=3/4 result would still require a separate",
            "  pre-submit review before any Gate 4 HPC job.",
            "- This does not prove ProSeCo-OWT is interaction-driven, does not support",
            "  model-agnostic claims, and does not authorize HPC pairwise scheduler",
            "  jobs without a separate pre-submit review.",
            "- The sparse Gate 3a pair graph means B=3/4 conclusions are estimator-based",
            "  and should be reported with the coverage caveat.",
            "",
        ]
    )
    (out_dir / "interpretation.md").write_text("\n".join(lines))


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2, sort_keys=True))


def validate_out_dir(out_dir: Path, short_sha: str) -> None:
    name = out_dir.name
    if "nogit" in name:
        raise ValueError("Output directory must not contain 'nogit'")
    if not name.startswith("phase1_schedule_validation_b34_"):
        raise ValueError("Output directory must be named phase1_schedule_validation_b34_<git-short-sha>")
    if short_sha not in name:
        raise ValueError(f"Output directory name must include current git short SHA {short_sha}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Gate 3b B=3/4 held-out Q_hat vs A")
    parser.add_argument("--xi_raw", type=Path, required=True)
    parser.add_argument("--schedule_rows", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--coverage_mode", choices=["estimator", "measured_only"], default="estimator")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed_split_mode", choices=["leave_seed_out", "kfold"], default="leave_seed_out")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--shrinkage_strength", type=float, default=10.0)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    full_sha, short_sha = git_sha()
    validate_out_dir(args.out_dir, short_sha)

    xi_rows = load_xi_rows(args.xi_raw)
    schedule_rows_path = resolve_schedule_rows_path(args.schedule_rows)
    schedule_rows = load_schedule_rows(schedule_rows_path)
    measured_pairs = measured_pair_set(xi_rows)

    candidate_pool = build_candidate_pool(
        schedule_rows,
        measured_pairs,
        budgets=args.budgets,
        coverage_mode=args.coverage_mode,
        source=str(schedule_rows_path),
    )
    if not candidate_pool:
        raise SystemExit(
            "No B=3/4 candidate schedules are locally evaluable under the requested coverage mode. "
            "Use estimator mode or run additional targeted pair/schedule measurements."
        )

    specs = model_specs_for_b34(
        shrinkage_strength=args.shrinkage_strength,
        ridge_alpha=args.ridge_alpha,
    )
    splits = build_splits(
        xi_rows,
        mode=args.seed_split_mode,
        k_folds=args.k_folds,
        random_seed=args.random_seed,
    )
    validation_rows = run_crossfit_predictions(xi_rows, candidate_pool, splits, specs)
    metrics = summarize_by_budget_model(
        validation_rows,
        specs,
        budgets=args.budgets,
        n_boot=args.n_boot,
        random_seed=args.random_seed,
        top_k=args.top_k,
    )
    pool_summary = candidate_pool_summary(candidate_pool, budgets=args.budgets)
    graph_summary = measured_graph_summary(measured_pairs, budgets=args.budgets)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = make_figures(validation_rows, metrics, pool_summary, args.out_dir / "figures")

    config = {
        "script": "scripts/proseco/interactions/validate_phase1_schedule_level_b34.py",
        "command": " ".join(sys.argv),
        "started_at": started,
        "git_sha": full_sha,
        "git_short_sha": short_sha,
        "xi_raw": str(args.xi_raw),
        "schedule_rows": str(schedule_rows_path),
        "out_dir": str(args.out_dir),
        "budgets": [int(b) for b in args.budgets],
        "coverage_mode": args.coverage_mode,
        "n_boot": int(args.n_boot),
        "seed_split_mode": args.seed_split_mode,
        "k_folds": int(args.k_folds),
        "random_seed": int(args.random_seed),
        "shrinkage_strength": float(args.shrinkage_strength),
        "ridge_alpha": float(args.ridge_alpha),
        "top_k": int(args.top_k),
        "models": [{"name": spec.name, **model_metadata(spec.name)} for spec in specs],
        "split_summary": {"n_splits": len(splits), "splits": splits},
        "candidate_pool_summary": pool_summary,
        "measured_graph_summary": graph_summary,
        "figures": figures,
        "gate3a_provenance_caveat": (
            "Input Gate 3a folder was phase1_interaction_diag_nogit with unknown git hash; "
            "this validation output is tagged with the current git SHA."
        ),
        "no_leakage_statement": (
            "xi_hat estimators are fitted only on Gate 3a training-seed pair rows; "
            "held-out schedule G values are used only for evaluation."
        ),
    }

    write_json(args.out_dir / "config.json", config)
    write_json(args.out_dir / "candidate_pool.json", candidate_pool_for_output(candidate_pool))
    write_json(args.out_dir / "validation_rows.json", validation_rows)
    write_json(args.out_dir / "metrics.json", metrics)
    (args.out_dir / "command_log.txt").write_text(config["command"] + "\n")
    write_interpretation(args.out_dir, metrics, config, pool_summary, graph_summary)

    best = metrics["best_heldout_q_model_by_budget"]
    print(json.dumps({"out_dir": str(args.out_dir), "best_by_budget": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
