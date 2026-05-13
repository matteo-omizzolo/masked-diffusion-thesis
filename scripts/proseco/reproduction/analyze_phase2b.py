"""Phase 2b — paired K-seed policy + MC-oracle analysis.

Consumes `results/phase2b_proseco_owt/{policy_raw.json,mc_raw.json}` (aggregated
across GPU shards by hpc/proseco/reproduction/phase2b_proseco_owt.sbatch) and emits
ANALYSIS_SPEC §3 JSON + §5.1 plots.

Outputs:
  results/phase2b/policy_comparison_paired.json   (§3b.1)
  results/phase2b/mc_oracle.json                  (§3b.2)
  results/phase2b/rank_A_vs_G_phase2b.json        (§3b.3)
  figures/phase2b/policy_gains_by_B.{png,pdf}
  figures/phase2b/paired_diff_uniform_B{B}_{policy}.{png,pdf}
  figures/phase2b/mc_oracle_headroom_B{B}.{png,pdf}

Usage
-----
    python scripts/proseco/reproduction/analyze_phase2b.py \
        --results_dir results/phase2b_proseco_owt \
        --out_dir     results/phase2b \
        --figures_dir figures/phase2b

Exits 0 iff every JSON write succeeded; non-zero on missing inputs.
Prints `ANALYSIS_COMPLETE: gates=N pass=M fail=K inconclusive=L` on success.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from mdm_playground.analysis import (  # noqa: E402
    classify_tier,
    mean_se,
    paired_bootstrap_ci,
    paired_t_test,
    spearman_bootstrap_ci,
)
from mdm_playground.analysis import plots as _plots  # noqa: E402
from mdm_playground.analysis.stats import paired_summary, _spearman  # noqa: E402


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()[:12]
    except Exception:
        return "unknown"


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _meta(stage_name: str, source_paths: Sequence[Path], F: str = "neg_nll") -> Dict[str, Any]:
    return {
        "phase": "2b",
        "stage_name": stage_name,
        "generated_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "generated_by": f"scripts/proseco/reproduction/analyze_phase2b.py@{_git_hash()}",
        "source_results_paths": [str(p) for p in source_paths],
        "config": {"F": F},
    }


# ---------------------------------------------------------------------------
# Row ingestion
# ---------------------------------------------------------------------------


def _load_policy_rows(results_dir: Path) -> List[Dict[str, Any]]:
    """Load policy_raw.json (aggregator output) or concatenate shard files."""
    agg = results_dir / "policy_raw.json"
    if agg.exists():
        return _load_json(agg)
    rows: List[Dict[str, Any]] = []
    for f in sorted(results_dir.glob("policy_raw.shard*-of-*.json")):
        rows.extend(_load_json(f))
    if not rows:
        raise FileNotFoundError(f"No policy_raw rows found in {results_dir}")
    return rows


def _load_mc_rows(results_dir: Path) -> List[Dict[str, Any]]:
    agg = results_dir / "mc_raw.json"
    if agg.exists():
        return _load_json(agg)
    rows: List[Dict[str, Any]] = []
    for f in sorted(results_dir.glob("mc_raw.shard*-of-*.json")):
        rows.extend(_load_json(f))
    if not rows:
        # MC rows may legitimately be empty if mc_P=0 was used; warn, don't fail.
        print(f"WARNING: no mc_raw rows found in {results_dir}", file=sys.stderr)
    return rows


def _group_policy_rows(
    policy_rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[int, Dict[int, Dict[str, float]]]], List[str], List[int], List[int]]:
    """Return (by_policy[pol][B][seed] -> {"G": ..., "A": ...}, policies, Bs, seeds)."""
    by_policy: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    policies_seen: List[str] = []
    seen_policy_set: set = set()
    B_set: set = set()
    seed_set: set = set()
    for row in policy_rows:
        pol = str(row["policy"])
        B = int(row["B"])
        seed = int(row["seed"])
        if pol not in seen_policy_set:
            policies_seen.append(pol)
            seen_policy_set.add(pol)
        B_set.add(B)
        seed_set.add(seed)
        by_policy[pol][B][seed] = {
            "G": float(row["G"]),
            "A": float(row["A"]),
            "residual": float(row.get("residual", float("nan"))),
            "schedule_steps": list(row.get("schedule_steps", [])),
        }
    return by_policy, policies_seen, sorted(B_set), sorted(seed_set)


# ---------------------------------------------------------------------------
# §3b.1 policy_comparison_paired.json
# ---------------------------------------------------------------------------


def _build_per_policy_per_B(
    by_policy: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    seeds: Sequence[int],
    Bs: Sequence[int],
    uniform_label: str = "uniform",
    alpha_bonf: float = 0.005,
) -> Dict[str, Dict[str, Any]]:
    """Per-policy per-B G_per_seed aggregates + paired_vs_uniform block."""
    out: Dict[str, Dict[str, Any]] = {}
    if uniform_label not in by_policy:
        raise ValueError(f"policy '{uniform_label}' missing from policy_raw")
    for pol, per_B in by_policy.items():
        out[pol] = {}
        for B in Bs:
            seed_rows = per_B.get(B, {})
            # Align G values to the sorted seeds list; missing seeds produce NaN.
            G_per_seed: List[float] = []
            for s in seeds:
                if s in seed_rows:
                    G_per_seed.append(float(seed_rows[s]["G"]))
                else:
                    G_per_seed.append(float("nan"))
            arr = np.asarray(G_per_seed, dtype=float)
            valid = ~np.isnan(arr)
            ms = mean_se(arr[valid].tolist()) if valid.any() else {
                "mean": float("nan"), "se": float("nan"), "n": 0,
            }
            entry: Dict[str, Any] = {
                "G_per_seed": G_per_seed,
                "mean": ms["mean"],
                "se": ms["se"],
                "n": int(ms["n"]),
            }
            if pol != uniform_label:
                # Paired diff vs uniform — only over seeds where BOTH exist.
                uniform_rows = by_policy[uniform_label].get(B, {})
                a_vals: List[float] = []
                b_vals: List[float] = []
                for s in seeds:
                    if s in seed_rows and s in uniform_rows:
                        a_vals.append(float(seed_rows[s]["G"]))
                        b_vals.append(float(uniform_rows[s]["G"]))
                if a_vals:
                    entry["paired_vs_uniform"] = paired_summary(
                        a_vals, b_vals, alpha_bonf=alpha_bonf, seed=abs(hash((pol, B))) % (2**31),
                    )
            out[pol][str(B)] = entry
    return out


def _ranking_at_each_B(
    per_policy_per_B: Dict[str, Dict[str, Any]],
    Bs: Sequence[int],
) -> Dict[str, List[Dict[str, Any]]]:
    rankings: Dict[str, List[Dict[str, Any]]] = {}
    for B in Bs:
        rows: List[Tuple[str, float]] = []
        for pol, per_B in per_policy_per_B.items():
            entry = per_B.get(str(B))
            if entry is None or math.isnan(entry.get("mean", float("nan"))):
                continue
            rows.append((pol, float(entry["mean"])))
        rows.sort(key=lambda r: -r[1])  # descending G
        rankings[str(B)] = [
            {"policy": p, "G_mean": g, "rank": i + 1}
            for i, (p, g) in enumerate(rows)
        ]
    return rankings


def _policy_comparison_decisions(
    per_policy_per_B: Dict[str, Dict[str, Any]],
    Bs: Sequence[int],
    uniform_label: str = "uniform",
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []

    # Gate 1: signal_beats_uniform_B8
    # pass iff any per-trajectory signal policy has bootstrap_95_ci[0] > 0 at B=8
    pt_signal_policies = [p for p in per_policy_per_B if p.endswith("_pt")]
    found = None
    if 8 in Bs:
        for pol in pt_signal_policies:
            entry = per_policy_per_B[pol].get("8", {})
            pvu = entry.get("paired_vs_uniform") if isinstance(entry, dict) else None
            if pvu and isinstance(pvu.get("bootstrap_95_ci"), list):
                lo = float(pvu["bootstrap_95_ci"][0])
                if lo > 0.0:
                    found = (pol, lo)
                    break
    if found is not None:
        decisions.append({
            "gate_id": "signal_beats_uniform_B8",
            "result": "pass",
            "evidence": f"policy={found[0]} bootstrap_95_ci_lo={found[1]:.4f} > 0 at B=8",
        })
    elif 8 in Bs:
        decisions.append({
            "gate_id": "signal_beats_uniform_B8",
            "result": "fail",
            "evidence": "no per-trajectory signal policy has CI lower bound > 0 at B=8",
        })
    else:
        decisions.append({
            "gate_id": "signal_beats_uniform_B8",
            "result": "inconclusive",
            "evidence": "B=8 not evaluated",
        })

    # Gate 2: uniform_is_optimal
    # pass iff no policy's bootstrap CI lower bound exceeds uniform's SE at any B.
    # (Note: the original spec also wants MC-oracle at B=4 to overlap zero; that
    # part is evaluated in mc_oracle.json.)
    beaten = []
    for B in Bs:
        uni = per_policy_per_B.get(uniform_label, {}).get(str(B), {})
        uni_se = uni.get("se", 0.0) if uni else 0.0
        for pol, per_B in per_policy_per_B.items():
            if pol == uniform_label:
                continue
            pvu = per_B.get(str(B), {}).get("paired_vs_uniform")
            if not pvu:
                continue
            lo = pvu.get("bootstrap_95_ci", [float("nan"), float("nan")])[0]
            if not math.isnan(lo) and lo > uni_se:
                beaten.append((pol, B, lo))
    if beaten:
        decisions.append({
            "gate_id": "uniform_is_optimal",
            "result": "fail",
            "evidence": f"{len(beaten)} (policy, B) pairs beat uniform; first = {beaten[0]}",
        })
    else:
        decisions.append({
            "gate_id": "uniform_is_optimal",
            "result": "pass",
            "evidence": "no policy's CI lower bound exceeds uniform's SE at any B",
        })

    # Gate 3: small_B_non_vacuous — at B=2 or B=3, any signal policy with d ≥ 0.5
    passed = None
    for B in (2, 3):
        if B not in Bs:
            continue
        for pol, per_B in per_policy_per_B.items():
            if pol == uniform_label:
                continue
            pvu = per_B.get(str(B), {}).get("paired_vs_uniform")
            if not pvu:
                continue
            d = pvu.get("cohens_d")
            if d is not None and d >= 0.5:
                passed = (pol, B, float(d))
                break
        if passed:
            break
    if passed:
        decisions.append({
            "gate_id": "small_B_non_vacuous",
            "result": "pass",
            "evidence": f"policy={passed[0]} at B={passed[1]} has cohens_d={passed[2]:.3f} ≥ 0.5",
        })
    elif 2 in Bs or 3 in Bs:
        decisions.append({
            "gate_id": "small_B_non_vacuous",
            "result": "fail",
            "evidence": "no signal policy reaches d ≥ 0.5 at B ∈ {2,3}",
        })
    else:
        decisions.append({
            "gate_id": "small_B_non_vacuous",
            "result": "inconclusive",
            "evidence": "neither B=2 nor B=3 evaluated",
        })

    return decisions


def policy_comparison_paired(
    by_policy: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    policies: Sequence[str],
    Bs: Sequence[int],
    seeds: Sequence[int],
    source_paths: Sequence[Path],
) -> Dict[str, Any]:
    per_policy_per_B = _build_per_policy_per_B(by_policy, seeds, Bs)
    rankings = _ranking_at_each_B(per_policy_per_B, Bs)
    decisions = _policy_comparison_decisions(per_policy_per_B, Bs)
    return {
        "meta": _meta("policy_comparison_paired", source_paths),
        "data": {
            "policies": list(policies),
            "B_values": list(Bs),
            "seeds": list(seeds),
            "per_policy_per_B": per_policy_per_B,
            "ranking_at_each_B": rankings,
        },
        "decisions": decisions,
    }


# ---------------------------------------------------------------------------
# §3b.2 mc_oracle.json
# ---------------------------------------------------------------------------


def _group_mc_rows(
    mc_rows: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
    """mc_by_seed[seed][B] -> list of {G, A, schedule_steps, mc_idx}."""
    mc_by_seed: Dict[int, Dict[int, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in mc_rows:
        mc_by_seed[int(r["seed"])][int(r["B"])].append({
            "G": float(r["G"]),
            "A": float(r["A"]),
            "schedule_steps": list(r.get("schedule_steps", [])),
            "mc_idx": int(r.get("mc_idx", -1)),
        })
    return mc_by_seed


def _per_trajectory_argmax(
    mc_by_seed: Dict[int, Dict[int, List[Dict[str, Any]]]],
    seeds: Sequence[int],
    mc_Bs: Sequence[int],
) -> List[Dict[str, Any]]:
    per_traj: List[Dict[str, Any]] = []
    for s in seeds:
        B_to_argmax: Dict[str, Any] = {}
        for B in mc_Bs:
            rows = mc_by_seed.get(s, {}).get(B, [])
            if not rows:
                continue
            best = max(rows, key=lambda r: r["G"])
            B_to_argmax[str(B)] = {
                "G": best["G"],
                "schedule": best["schedule_steps"],
                "A_of_argmax": best["A"],
                "n_samples": len(rows),
            }
        if B_to_argmax:
            per_traj.append({"seed": int(s), "B_to_argmax_G": B_to_argmax})
    return per_traj


def _mc_oracle_minus(
    argmax_by_seed: Dict[int, Dict[int, float]],
    ref_by_seed: Dict[int, Dict[int, float]],
    mc_Bs: Sequence[int],
    label: str,
    alpha_bonf: float = 0.005,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for B in mc_Bs:
        a_vals, b_vals = [], []
        for s in sorted(argmax_by_seed):
            if B in argmax_by_seed[s] and s in ref_by_seed and B in ref_by_seed[s]:
                a_vals.append(argmax_by_seed[s][B])
                b_vals.append(ref_by_seed[s][B])
        if not a_vals:
            out[str(B)] = {"mean": float("nan"), "se": float("nan"), "n": 0,
                           "note": f"no paired (mc_oracle, {label}) samples at B={B}"}
            continue
        out[str(B)] = paired_summary(a_vals, b_vals, alpha_bonf=alpha_bonf,
                                     seed=abs(hash((label, B))) % (2**31))
    return out


def mc_oracle(
    mc_rows: Sequence[Dict[str, Any]],
    by_policy: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    seeds: Sequence[int],
    mc_Bs: Sequence[int],
    source_paths: Sequence[Path],
    mean_profile_oracle_label: str = "mean_delta_oracle",
    uniform_label: str = "uniform",
) -> Dict[str, Any]:
    mc_by_seed = _group_mc_rows(mc_rows)
    per_traj = _per_trajectory_argmax(mc_by_seed, seeds, mc_Bs)

    # Build (seed -> B -> G) dictionaries for MC-argmax, uniform, and MP oracle.
    argmax: Dict[int, Dict[int, float]] = defaultdict(dict)
    for rec in per_traj:
        for B_str, v in rec["B_to_argmax_G"].items():
            argmax[int(rec["seed"])][int(B_str)] = float(v["G"])

    uniform_by_seed: Dict[int, Dict[int, float]] = defaultdict(dict)
    for B, per_seed in by_policy.get(uniform_label, {}).items():
        for s, row in per_seed.items():
            uniform_by_seed[int(s)][int(B)] = float(row["G"])

    mp_oracle_by_seed: Dict[int, Dict[int, float]] = defaultdict(dict)
    for B, per_seed in by_policy.get(mean_profile_oracle_label, {}).items():
        for s, row in per_seed.items():
            mp_oracle_by_seed[int(s)][int(B)] = float(row["G"])

    minus_uniform = _mc_oracle_minus(argmax, uniform_by_seed, mc_Bs, "uniform")
    minus_mp_oracle = _mc_oracle_minus(argmax, mp_oracle_by_seed, mc_Bs, "mean_profile_oracle")

    # Decision: headroom_exists_realistic — CI[0] > 0.05 at B=4
    decisions: List[Dict[str, Any]] = []
    if 4 in mc_Bs:
        entry = minus_uniform.get("4", {})
        ci = entry.get("bootstrap_95_ci", [float("nan"), float("nan")])
        lo = ci[0] if ci else float("nan")
        if not math.isnan(lo) and lo > 0.05:
            result = "pass"
        elif not math.isnan(lo):
            result = "fail"
        else:
            result = "inconclusive"
        decisions.append({
            "gate_id": "headroom_exists_realistic",
            "result": result,
            "evidence": f"mc_oracle_minus_uniform.4.bootstrap_95_ci_lo={lo}; threshold 0.05",
        })
    else:
        decisions.append({
            "gate_id": "headroom_exists_realistic",
            "result": "inconclusive",
            "evidence": "B=4 not in mc_B_values",
        })

    return {
        "meta": _meta("mc_oracle", source_paths),
        "data": {
            "per_trajectory": per_traj,
            "mc_oracle_minus_uniform": minus_uniform,
            "mc_oracle_minus_mean_profile_oracle": minus_mp_oracle,
            "enumeration_spot_checks": [],  # not performed here; requires deterministic enumeration
        },
        "decisions": decisions,
    }


# ---------------------------------------------------------------------------
# §3b.3 rank_A_vs_G_phase2b.json — per-seed Spearman across policies
# ---------------------------------------------------------------------------


def rank_A_vs_G_phase2b(
    by_policy: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    seeds: Sequence[int],
    Bs: Sequence[int],
    source_paths: Sequence[Path],
    n_resamples: int = 1000,
) -> Dict[str, Any]:
    per_B: Dict[str, Any] = {}
    for B in Bs:
        # Collect (A,G) pairs per seed across all policies available at this B.
        per_seed_rhos: List[float] = []
        all_A: List[float] = []
        all_G: List[float] = []
        for s in seeds:
            A_vals, G_vals = [], []
            for pol, per_B_data in by_policy.items():
                row = per_B_data.get(B, {}).get(s)
                if row is None:
                    continue
                A_vals.append(float(row["A"]))
                G_vals.append(float(row["G"]))
            if len(A_vals) < 3:
                continue
            all_A.extend(A_vals)
            all_G.extend(G_vals)
            rho_s = _spearman_from_lists(A_vals, G_vals)
            if not math.isnan(rho_s):
                per_seed_rhos.append(rho_s)

        # Pooled Spearman across all (policy, seed) pairs at this B.
        if len(all_A) >= 3:
            spearman = spearman_bootstrap_ci(
                np.asarray(all_A, float), np.asarray(all_G, float),
                n_resamples=n_resamples, seed=abs(hash(("2b_rank", B))) % (2**31),
            )
        else:
            spearman = {"rho": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(all_A)}

        # Per-seed rho: mean + simple SE
        if per_seed_rhos:
            rho_arr = np.asarray(per_seed_rhos, float)
            mean_seed_rho = float(rho_arr.mean())
            se_seed_rho = float(rho_arr.std(ddof=1) / math.sqrt(rho_arr.size)) if rho_arr.size > 1 else 0.0
        else:
            mean_seed_rho = float("nan")
            se_seed_rho = float("nan")

        per_B[str(B)] = {
            "pooled_spearman": spearman,
            "per_seed_rho": per_seed_rhos,
            "per_seed_rho_mean": mean_seed_rho,
            "per_seed_rho_se": se_seed_rho,
            "n_points_pooled": len(all_A),
            "n_seeds_with_rho": len(per_seed_rhos),
            "tier": classify_tier(
                mean=mean_seed_rho,
                se=se_seed_rho,
                n=len(per_seed_rhos),
                is_paired_diff=False,
            )["tier"],
        }

    # Decision gate: A_ranks_G_cross_policy_B8 — mean Spearman ≥ 0.6 at B=8
    decisions: List[Dict[str, Any]] = []
    if 8 in Bs:
        m = per_B.get("8", {}).get("per_seed_rho_mean", float("nan"))
        if not math.isnan(m) and m >= 0.6:
            decisions.append({
                "gate_id": "A_ranks_G_cross_policy_B8",
                "result": "pass",
                "evidence": f"per_seed_rho_mean={m:.3f} ≥ 0.6 at B=8",
            })
        else:
            decisions.append({
                "gate_id": "A_ranks_G_cross_policy_B8",
                "result": "fail" if not math.isnan(m) else "inconclusive",
                "evidence": f"per_seed_rho_mean={m} at B=8; threshold 0.6",
            })
    else:
        decisions.append({
            "gate_id": "A_ranks_G_cross_policy_B8",
            "result": "inconclusive",
            "evidence": "B=8 not evaluated",
        })

    return {
        "meta": _meta("rank_A_vs_G_phase2b", source_paths),
        "data": {"per_B": per_B},
        "decisions": decisions,
    }


def _spearman_from_lists(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3:
        return float("nan")
    try:
        return float(_spearman(np.asarray(a, float), np.asarray(b, float)))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Plot dispatch
# ---------------------------------------------------------------------------


def _gen_plots(
    policy_comparison: Dict[str, Any],
    mc_oracle_json: Dict[str, Any],
    figures_dir: Path,
    Bs: Sequence[int],
) -> None:
    per_policy_per_B = policy_comparison["data"]["per_policy_per_B"]
    policies_order = policy_comparison["data"]["policies"]

    # (A) Box plot: policy_gains_by_B
    fig = _plots.policy_gains_box_by_B(
        per_policy_per_B=per_policy_per_B,
        B_values=Bs,
        policies_order=policies_order,
    )
    _plots.save_figure_with_meta(
        fig=fig,
        png_path=figures_dir / "policy_gains_by_B.png",
        meta={
            "source": "results/phase2b/policy_comparison_paired.json",
            "fields": ["data.per_policy_per_B.*.*.G_per_seed"],
            "B_values": list(Bs),
        },
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    # (B) Per-seed paired diff plots for top 4 non-uniform policies (by mean at the largest B)
    Bs_sorted = sorted(Bs)
    largest_B = Bs_sorted[-1] if Bs_sorted else None
    if largest_B is not None:
        ranked = policy_comparison["data"]["ranking_at_each_B"].get(str(largest_B), [])
        non_uniform = [r for r in ranked if r["policy"] != "uniform"][:4]
        for B in Bs_sorted:
            for rec in non_uniform:
                pol = rec["policy"]
                pvu = per_policy_per_B.get(pol, {}).get(str(B), {}).get("paired_vs_uniform")
                if not pvu or not pvu.get("diff_per_seed"):
                    continue
                fig = _plots.paired_diff_vs_uniform(
                    policy=pol,
                    B=B,
                    diffs=pvu["diff_per_seed"],
                    ci_lo=pvu["bootstrap_95_ci"][0],
                    ci_hi=pvu["bootstrap_95_ci"][1],
                    tier=pvu.get("tier", ""),
                )
                _plots.save_figure_with_meta(
                    fig=fig,
                    png_path=figures_dir / f"paired_diff_uniform_B{B}_{pol}.png",
                    meta={
                        "source": "results/phase2b/policy_comparison_paired.json",
                        "fields": [f"data.per_policy_per_B.{pol}.{B}.paired_vs_uniform"],
                    },
                )
                plt.close(fig)

    # (C) MC oracle headroom per B
    per_traj = mc_oracle_json["data"]["per_trajectory"]
    uniform_per_seed_per_B: Dict[int, Dict[int, float]] = defaultdict(dict)
    for pol_row in per_policy_per_B.get("uniform", {}).items():
        B_str, entry = pol_row
        for s, g in zip(
            policy_comparison["data"]["seeds"], entry.get("G_per_seed", [])
        ):
            if not math.isnan(g):
                uniform_per_seed_per_B[int(s)][int(B_str)] = float(g)

    for B_str_key in sorted({int(b) for rec in per_traj for b in rec["B_to_argmax_G"].keys()}):
        u_vals, m_vals = [], []
        for rec in per_traj:
            entry = rec["B_to_argmax_G"].get(str(B_str_key))
            if entry is None:
                continue
            s = int(rec["seed"])
            if B_str_key not in uniform_per_seed_per_B.get(s, {}):
                continue
            u_vals.append(uniform_per_seed_per_B[s][B_str_key])
            m_vals.append(float(entry["G"]))
        if not u_vals:
            continue
        fig = _plots.mc_oracle_headroom_lines(
            per_seed_uniform=u_vals,
            per_seed_mc_oracle=m_vals,
            B=B_str_key,
        )
        _plots.save_figure_with_meta(
            fig=fig,
            png_path=figures_dir / f"mc_oracle_headroom_B{B_str_key}.png",
            meta={
                "source": "results/phase2b/mc_oracle.json",
                "fields": ["data.per_trajectory.*.B_to_argmax_G"],
            },
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2b analysis (paired K-seed + MC oracle)")
    p.add_argument("--results_dir", type=str, default="results/phase2b_proseco_owt")
    p.add_argument("--out_dir", type=str, default="results/phase2b")
    p.add_argument("--figures_dir", type=str, default="figures/phase2b")
    p.add_argument("--F", type=str, default="neg_nll")
    p.add_argument("--alpha_bonf", type=float, default=0.005)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading rows from {results_dir}")
    policy_rows = _load_policy_rows(results_dir)
    mc_rows = _load_mc_rows(results_dir)
    print(f"  policy_rows: {len(policy_rows)}  mc_rows: {len(mc_rows)}")

    by_policy, policies, Bs, seeds = _group_policy_rows(policy_rows)
    print(f"  policies:   {policies}")
    print(f"  B_values:   {Bs}")
    print(f"  n_seeds:    {len(seeds)}")

    # Derive mc_B_values from the MC rows themselves (defensive against config drift).
    mc_Bs = sorted({int(r["B"]) for r in mc_rows})
    print(f"  mc_Bs:      {mc_Bs}")

    # (1) policy_comparison_paired.json
    pc = policy_comparison_paired(
        by_policy=by_policy,
        policies=policies,
        Bs=Bs,
        seeds=seeds,
        source_paths=[results_dir],
    )
    _write_json(out_dir / "policy_comparison_paired.json", pc)

    # (2) mc_oracle.json
    mc = mc_oracle(
        mc_rows=mc_rows,
        by_policy=by_policy,
        seeds=seeds,
        mc_Bs=mc_Bs,
        source_paths=[results_dir],
    )
    _write_json(out_dir / "mc_oracle.json", mc)

    # (3) rank_A_vs_G_phase2b.json
    rk = rank_A_vs_G_phase2b(
        by_policy=by_policy,
        seeds=seeds,
        Bs=Bs,
        source_paths=[results_dir],
        n_resamples=args.n_bootstrap,
    )
    _write_json(out_dir / "rank_A_vs_G_phase2b.json", rk)

    # (4) Plots
    try:
        _gen_plots(pc, mc, figures_dir, Bs)
    except Exception as e:
        print(f"WARNING: plot generation failed: {e!r}", file=sys.stderr)

    # Collect gate outcomes from all three JSONs
    all_decisions = []
    for doc in (pc, mc, rk):
        all_decisions.extend(doc.get("decisions", []))
    n = len(all_decisions)
    passed = sum(1 for d in all_decisions if d.get("result") == "pass")
    failed = sum(1 for d in all_decisions if d.get("result") == "fail")
    inconc = sum(1 for d in all_decisions if d.get("result") == "inconclusive")

    print(f"ANALYSIS_COMPLETE: gates={n} pass={passed} fail={failed} inconclusive={inconc}")
    for d in all_decisions:
        print(f"  {d['gate_id']}: {d['result']} — {d.get('evidence', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
