"""Phase 2a — offline re-analysis of Phase 1 ProSeCo-OWT data.

Consumes `results/phase1_proseco_owt_full/{protocol_a,protocol_b,policy_comparison}/`
and emits ANALYSIS_SPEC-compliant JSON + plots under `results/phase2a/` and
`figures/phase2a/`.

Outputs:
  results/phase2a/rank_A_vs_G.json                  (§2a.1)
  results/phase2a/eps_R.json                        (§2a.2)
  results/phase2a/mc_oracle_lowerbound.json         (§2a.3)
  results/phase2a/phase2a_corrected_policy_table.md (§2a.4)
  figures/phase2a/A_vs_G_scatter_B{B}.{png,pdf}     (§5.1)
  figures/phase2a/per_trajectory_spearman_histogram.{png,pdf}

Usage
-----
    python scripts/analyze_phase2a.py \
        --results_dir results/phase1_proseco_owt_full \
        --out_dir     results/phase2a \
        --figures_dir figures/phase2a

Exits 0 iff every JSON write succeeded; non-zero on missing inputs.
Prints `ANALYSIS_COMPLETE: gates=N pass=M fail=K inconclusive=L` on success.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running as `python scripts/analyze_phase2a.py` without install.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from mdm_playground.analysis import (  # noqa: E402
    classify_tier,
    mean_se,
    paired_bootstrap_ci,
    paired_t_test,
    spearman_bootstrap_ci,
)
from mdm_playground.analysis import stats as _stats  # noqa: E402
from mdm_playground.analysis import plots as _plots  # noqa: E402


SIGNAL_NAMES = ("entropy", "inverse_margin", "quality_mass_proxy", "unmasked_fraction")


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
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Cannot serialise {type(o)!r}")


def _meta_block(
    stage: str,
    source_dirs: Sequence[Path],
    source_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"F": "neg_nll"}
    if source_summary is not None:
        rc = source_summary.get("config", source_summary.get("run_config", {})) or {}
        for k in ("T", "N", "M", "P", "B_values", "seed", "corrector_steps"):
            if k in rc:
                cfg[k] = rc[k]
    return {
        "phase": "2a",
        "stage_name": stage,
        "generated_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "generated_by": f"scripts/analyze_phase2a.py@{_git_hash()}",
        "source_results_paths": [str(p) for p in source_dirs],
        "config": cfg,
    }


# ---------------------------------------------------------------------------
# Load Phase 1 run
# ---------------------------------------------------------------------------


def _load_trajectories(protocol_a_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(protocol_a_dir.glob("trajectory_*.json"))
    trajectories = [_load_json(f) for f in files]
    return trajectories


def _load_schedules(protocol_b_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(
        protocol_b_dir.glob("schedule_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return [_load_json(f) for f in files]


def _load_pairs(protocol_b_dir: Path) -> List[Dict[str, Any]]:
    pairs_path = protocol_b_dir / "pairs.json"
    if not pairs_path.exists():
        return []
    return _load_json(pairs_path)


def _load_policy_comparison(pc_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(pc_dir.glob("*.json"))
    if not files:
        return []
    return _load_json(files[0])


def _load_run_config(results_dir: Path) -> Dict[str, Any]:
    rc = results_dir / "run_config.json"
    if rc.exists():
        return _load_json(rc)
    return {}


# ---------------------------------------------------------------------------
# §2a.1 — rank A vs G
# ---------------------------------------------------------------------------


def _rank_A_vs_G(
    schedules: Sequence[Dict[str, Any]],
    pairs: Sequence[Dict[str, Any]],
    out_dir: Path,
    figures_dir: Path,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Build per-B (A, G) arrays and report Spearman + Pearson with CI."""
    by_B: Dict[int, Dict[str, List[float]]] = {}

    # Pairs file contains B=2 data: G_pair, delta_t + delta_tp are the
    # per-step Δ, so A = delta_t + delta_tp. We include B=2 only if the pairs
    # belong to the B=2 marginal experiment.
    if pairs:
        A2 = [float(p.get("delta_t", 0.0) + p.get("delta_tp", 0.0)) for p in pairs]
        G2 = [float(p.get("G_pair", 0.0)) for p in pairs]
        by_B[2] = {"A": A2, "G": G2}

    for s in schedules:
        B = int(s["B"])
        by_B.setdefault(B, {"A": [], "G": []})
        by_B[B]["A"].append(float(s["A"]))
        by_B[B]["G"].append(float(s["G"]))

    spearman_by_B: Dict[str, Any] = {}
    pearson_by_B: Dict[str, Any] = {}
    scatter_points: Dict[str, List[Dict[str, float]]] = {}
    for B, data in sorted(by_B.items()):
        A_arr = np.asarray(data["A"], dtype=float)
        G_arr = np.asarray(data["G"], dtype=float)
        spear = spearman_bootstrap_ci(A_arr, G_arr, n_resamples=1000)
        pear = _pearson_bootstrap_ci(A_arr, G_arr, n_resamples=1000)
        spearman_by_B[str(B)] = {
            "rho": spear["rho"],
            "ci_lo": spear["ci_lo"],
            "ci_hi": spear["ci_hi"],
            "n_schedules": int(A_arr.size),
            "tier": _tier_for_corr(spear),
        }
        pearson_by_B[str(B)] = {
            "rho": pear["rho"],
            "ci_lo": pear["ci_lo"],
            "ci_hi": pear["ci_hi"],
            "n_schedules": int(A_arr.size),
        }
        scatter_points[str(B)] = [
            {"S_id": int(i), "A": float(a), "G": float(g)}
            for i, (a, g) in enumerate(zip(A_arr, G_arr))
        ]
        # Plot
        fig = _plots.scatter_A_vs_G(
            A_arr, G_arr, B=B,
            spearman=spearman_by_B[str(B)],
            pearson=pearson_by_B[str(B)],
        )
        _plots.save_figure_with_meta(
            fig,
            png_path=figures_dir / f"A_vs_G_scatter_B{B}.png",
            meta={
                "source_json": "results/phase2a/rank_A_vs_G.json",
                "fields": [
                    f"data.spearman_by_B.{B}",
                    f"data.pearson_by_B.{B}",
                    f"data.scatter_points_by_B.{B}",
                ],
                "n_schedules": int(A_arr.size),
            },
        )
        _plots.plt.close(fig)  # type: ignore[union-attr]

    # Decision gate: Spearman CI lower bound ≥ 0.5 at B=8.
    gate_B = "8" if "8" in spearman_by_B else sorted(spearman_by_B)[-1]
    gate_ci_lo = spearman_by_B[gate_B]["ci_lo"]
    gate_result = "pass" if gate_ci_lo >= 0.5 else "fail"

    out = {
        "meta": meta,
        "data": {
            "spearman_by_B": spearman_by_B,
            "pearson_by_B": pearson_by_B,
            "scatter_points_by_B": scatter_points,
        },
        "decisions": [
            {
                "gate_id": f"A_ranks_G_at_B{gate_B}",
                "result": gate_result,
                "evidence": (
                    f"data.spearman_by_B.{gate_B}.ci_lo={gate_ci_lo:.3f}; "
                    "threshold 0.5"
                ),
            }
        ],
    }
    _write_json(out_dir / "rank_A_vs_G.json", out)
    return out


def _pearson_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    n = x.size
    if n < 3:
        return {"rho": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": n}
    point = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xi = x[idx]
        yi = y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            rhos[i] = 0.0
        else:
            rhos[i] = float(np.corrcoef(xi, yi)[0, 1])
    return {
        "rho": point,
        "ci_lo": float(np.quantile(rhos, 0.025)),
        "ci_hi": float(np.quantile(rhos, 0.975)),
        "n": n,
    }


def _tier_for_corr(spear: Dict[str, float]) -> str:
    """Correlation measurements are exploratory by default (T3)."""
    if spear["ci_lo"] > 0 or spear["ci_hi"] < 0:
        return "T2"
    return "T3"


# ---------------------------------------------------------------------------
# §2a.2 — rank-based ε vs RMS ε per signal
# ---------------------------------------------------------------------------


def _epsilon_R_and_rms(
    trajectories: Sequence[Dict[str, Any]],
    out_dir: Path,
    figures_dir: Path,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    # Per trajectory, per signal, Spearman(signal_trace, delta_trace).
    per_traj_rho: Dict[str, List[Tuple[int, float]]] = {s: [] for s in SIGNAL_NAMES}
    per_traj_std: Dict[str, List[float]] = {s: [] for s in SIGNAL_NAMES}
    per_traj_eps_rms: Dict[str, List[float]] = {s: [] for s in SIGNAL_NAMES}
    n_traj = len(trajectories)

    for i, traj in enumerate(trajectories):
        per_t = traj["per_t"]
        delta = np.asarray([row["delta"] for row in per_t], dtype=float)
        delta_std = float(delta.std(ddof=1)) if delta.size >= 2 else 0.0

        for s in SIGNAL_NAMES:
            sig = np.asarray([row.get(s, 0.0) for row in per_t], dtype=float)
            rho = _stats._spearman(sig, delta)
            per_traj_rho[s].append((i, float(rho)))
            per_traj_std[s].append(delta_std)

            # ε_rms per trajectory: best-fit-line residual RMS of Δ vs sig.
            if sig.std() > 0:
                slope = float(np.cov(sig, delta, ddof=0)[0, 1] / sig.var())
                intercept = float(delta.mean() - slope * sig.mean())
                pred = slope * sig + intercept
                rms = float(np.sqrt(((delta - pred) ** 2).mean()))
            else:
                rms = float(np.sqrt((delta ** 2).mean()))
            per_traj_eps_rms[s].append(rms)

    spearman_quantiles: Dict[str, Dict[str, float]] = {}
    epsilon_R: Dict[str, Dict[str, float]] = {}
    epsilon_rms: Dict[str, Dict[str, float]] = {}
    rhos_by_signal: Dict[str, List[float]] = {}
    for s in SIGNAL_NAMES:
        rhos = np.asarray([r for (_, r) in per_traj_rho[s]], dtype=float)
        stds = np.asarray(per_traj_std[s], dtype=float)
        rms = np.asarray(per_traj_eps_rms[s], dtype=float)
        rhos_by_signal[s] = rhos.tolist()
        pos = int((rhos > 0).sum())
        neg = int((rhos < 0).sum())
        spearman_quantiles[s] = {
            "q25": float(np.quantile(rhos, 0.25)),
            "q50": float(np.quantile(rhos, 0.5)),
            "q75": float(np.quantile(rhos, 0.75)),
            "pos_count": pos,
            "neg_count": neg,
            "mean_abs": float(np.abs(rhos).mean()),
        }
        # ε_R = (1 - |ρ|) · σ_Δ per trajectory; aggregate as mean, SE.
        eps_R_vec = (1.0 - np.abs(rhos)) * stds
        er = mean_se(eps_R_vec.tolist())
        epsilon_R[s] = {
            "mean": er["mean"],
            "se": er["se"],
            "n": er["n"],
            "definition": "(1 - |ρ|) · σ_Δ per trajectory, then mean across trajectories",
        }
        rs = mean_se(rms.tolist())
        epsilon_rms[s] = {
            "mean": rs["mean"],
            "se": rs["se"],
            "n": rs["n"],
            "definition": "per-trajectory RMS residual of Δ_t vs signal linear fit, then mean",
        }

    # Plot
    fig = _plots.histogram_per_trajectory_spearman(rhos_by_signal)
    _plots.save_figure_with_meta(
        fig,
        png_path=figures_dir / "per_trajectory_spearman_histogram.png",
        meta={
            "source_json": "results/phase2a/eps_R.json",
            "fields": [
                f"data.per_trajectory_spearman.{s}" for s in SIGNAL_NAMES
            ],
            "n_trajectories": n_traj,
        },
    )
    _plots.plt.close(fig)  # type: ignore[union-attr]

    # Decision gate: does ε_R discriminate signals >1.5× while ε_rms stays within 1.1×?
    eR_means = [epsilon_R[s]["mean"] for s in SIGNAL_NAMES]
    eRms_means = [epsilon_rms[s]["mean"] for s in SIGNAL_NAMES]
    eR_ratio = (max(eR_means) / min(eR_means)) if min(eR_means) > 0 else float("inf")
    eRms_ratio = (max(eRms_means) / min(eRms_means)) if min(eRms_means) > 0 else float("inf")
    gate_result = "pass" if (eR_ratio >= 1.5 and eRms_ratio <= 1.1) else "fail"

    out = {
        "meta": meta,
        "data": {
            "per_trajectory_spearman": {s: per_traj_rho[s] for s in SIGNAL_NAMES},
            "spearman_quantiles": spearman_quantiles,
            "epsilon_R_by_signal": epsilon_R,
            "epsilon_rms_by_signal": epsilon_rms,
            "eR_spread_ratio": eR_ratio,
            "eRms_spread_ratio": eRms_ratio,
        },
        "decisions": [
            {
                "gate_id": "eps_R_differentiates_signals",
                "result": gate_result,
                "evidence": (
                    f"eR_ratio={eR_ratio:.2f} (need ≥1.5); "
                    f"eRms_ratio={eRms_ratio:.2f} (need ≤1.1)"
                ),
            }
        ],
    }
    _write_json(out_dir / "eps_R.json", out)
    return out


# ---------------------------------------------------------------------------
# §2a.3 — MC oracle lower bound
# ---------------------------------------------------------------------------


def _mc_oracle_lowerbound(
    schedules: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    policy_rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Max G over the M=30 random schedules at each B, compared to uniform G.

    Phase 1 protocol_b schedules are drawn i.i.d., NOT paired across trajectories.
    Consequently this analysis is only a **proxy** for per-trajectory MC-oracle
    headroom; the full per-trajectory MC analysis is a Phase 2b deliverable.
    We report the max-of-30 G per B and compare to the single-seed uniform G
    in the policy comparison file, flagging the limitation.
    """
    by_B: Dict[int, List[Dict[str, Any]]] = {}
    for s in schedules:
        by_B.setdefault(int(s["B"]), []).append(s)

    # Uniform single-seed G by B from policy_comparison.
    uniform_G: Dict[int, float] = {}
    for row in policy_rows:
        if row.get("policy") == "uniform":
            uniform_G[int(row["B"])] = float(row["G"])

    max_over_30: Dict[str, Any] = {}
    fraction_beating: Dict[str, float] = {}
    tiers: Dict[str, str] = {}
    for B, rows in sorted(by_B.items()):
        Gs = np.asarray([float(r["G"]) for r in rows], dtype=float)
        As = np.asarray([float(r["A"]) for r in rows], dtype=float)
        max_G = float(Gs.max())
        argmax = int(np.argmax(Gs))
        uG = uniform_G.get(B, None)
        if uG is None:
            headroom = None
            frac = float((Gs > 0).mean())
            note = "uniform-G not available for this B; reporting fraction of schedules with G>0"
        else:
            headroom = float(max_G - uG)
            frac = float((Gs > uG).mean())
            note = (
                "uniform_G is single-seed (Tier T4). Headroom is a lower bound "
                "on the true MC-oracle-minus-uniform gap; not paired across "
                "trajectories."
            )
        max_over_30[str(B)] = {
            "max_G": max_G,
            "argmax_schedule_A": float(As[argmax]),
            "argmax_schedule_idx": argmax,
            "uniform_G_single_seed": uG,
            "headroom_lower_bound": headroom,
            "n_schedules": int(Gs.size),
            "note": note,
            "tier": "T4",
        }
        fraction_beating[str(B)] = frac
        tiers[str(B)] = "T4"

    # Decision gate: headroom at B=8 > 0.05.
    gate_B = "8" if "8" in max_over_30 else sorted(max_over_30)[-1]
    hr = max_over_30[gate_B]["headroom_lower_bound"]
    if hr is None:
        gate_result = "inconclusive"
    else:
        gate_result = "pass" if hr >= 0.05 else "fail"

    out = {
        "meta": meta,
        "data": {
            "max_over_30_minus_uniform": max_over_30,
            "fraction_beating_uniform": fraction_beating,
            "limitation": (
                "Phase 1 protocol_b schedules are drawn i.i.d. without pairing "
                "to individual trajectories; uniform_G is a single-seed row from "
                "policy_comparison.json. Results here are preliminary (Tier T4). "
                "A proper paired per-trajectory MC-oracle analysis is a Phase 2b "
                "deliverable — see mc_oracle.json schema in ANALYSIS_SPEC §3b.2."
            ),
        },
        "decisions": [
            {
                "gate_id": "headroom_exists_B" + str(gate_B),
                "result": gate_result,
                "evidence": (
                    f"max_over_30_minus_uniform.{gate_B}.headroom_lower_bound="
                    f"{hr}; threshold 0.05"
                ),
            }
        ],
    }
    _write_json(out_dir / "mc_oracle_lowerbound.json", out)
    return out


# ---------------------------------------------------------------------------
# §2a.4 — corrected policy table
# ---------------------------------------------------------------------------


def _corrected_policy_table(
    policy_rows: Sequence[Dict[str, Any]],
    schedules: Sequence[Dict[str, Any]],
    out_dir: Path,
    meta: Dict[str, Any],
) -> Path:
    """Emit a markdown table with A(S), G(S)_single_seed, and G_min_of_30
    for each policy at each B.

    The third column matches schedule_i from protocol_b where schedule_steps
    equals the policy's allocation (rare in practice; missing otherwise).
    """
    by_B_policy: Dict[Tuple[int, str], Dict[str, Any]] = {
        (int(r["B"]), r["policy"]): r for r in policy_rows
    }
    schedules_by_B: Dict[int, List[Dict[str, Any]]] = {}
    for s in schedules:
        schedules_by_B.setdefault(int(s["B"]), []).append(s)

    lines: List[str] = []
    lines.append(
        "# Phase 2a — Corrected Policy Table (A vs G)\n"
    )
    lines.append(
        "> Generated by `scripts/analyze_phase2a.py`. Source data: "
        "`results/phase1_proseco_owt_full/policy_comparison/policy_comparison.json` "
        "and `results/phase1_proseco_owt_full/protocol_b/schedule_*.json`.\n"
    )
    lines.append(
        "> Every row here is **single-seed** (Tier T4). The Phase 1 "
        "C6 inverted-policy table confused A(S) with G(S); this document "
        "separates them explicitly. Per-seed paired comparisons are a Phase 2b "
        "deliverable.\n"
    )
    lines.append("")
    lines.append(
        "| B | Policy | A(S) | G(S) single-seed | max G from 30 random @ B | Δ(G − uniform) |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for B in sorted({row["B"] for row in policy_rows}):
        uniform_row = by_B_policy.get((B, "uniform"))
        uniform_G = float(uniform_row["G"]) if uniform_row else float("nan")
        mc_max = (
            max((float(s["G"]) for s in schedules_by_B.get(B, [])), default=float("nan"))
        )
        for row in sorted(
            [r for r in policy_rows if int(r["B"]) == int(B)],
            key=lambda r: -float(r["G"]),
        ):
            policy = row["policy"]
            A = float(row["A"])
            G = float(row["G"])
            delta = G - uniform_G if uniform_row else float("nan")
            lines.append(
                f"| {B} | `{policy}` | {A:+.3f} | {G:+.3f} | {mc_max:+.3f} | "
                f"{delta:+.3f} |"
            )

    lines.append("")
    lines.append(
        "## Reading this table\n"
        "- `A(S)`: additive surrogate = Σ_{t∈S} Δ_t from Protocol A. A proxy; "
        "  does **not** equal G(S) in general.\n"
        "- `G(S) single-seed`: true pipeline-evaluated schedule gain, one seed only.\n"
        "- `max G from 30 random @ B`: best of the 30 random schedules at this B "
        "(schedules are NOT matched to this policy's step set; this is a sanity "
        "ceiling for the budget, not a paired MC oracle for this policy).\n"
        "- `Δ(G − uniform)`: single-seed, uncorrected. **Not thesis-quotable.**\n"
    )
    out_path = out_dir / "phase2a_corrected_policy_table.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/phase1_proseco_owt_full")
    ap.add_argument("--out_dir", default="results/phase2a")
    ap.add_argument("--figures_dir", default="figures/phase2a")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    protocol_a_dir = results_dir / "protocol_a"
    protocol_b_dir = results_dir / "protocol_b"
    policy_dir = results_dir / "policy_comparison"

    if not protocol_a_dir.exists() or not protocol_b_dir.exists():
        print(
            f"ERROR: missing protocol_a or protocol_b directory under {results_dir}",
            file=sys.stderr,
        )
        return 2

    run_config = _load_run_config(results_dir)
    summary_path = results_dir / "summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {"config": run_config}

    trajectories = _load_trajectories(protocol_a_dir)
    schedules = _load_schedules(protocol_b_dir)
    pairs = _load_pairs(protocol_b_dir)
    policy_rows = _load_policy_comparison(policy_dir)

    meta = _meta_block("phase2a", [results_dir], summary)

    decisions: List[Dict[str, Any]] = []
    out_ra = _rank_A_vs_G(schedules, pairs, out_dir, figures_dir, dict(meta, stage_name="rank_A_vs_G"))
    decisions.extend(out_ra["decisions"])
    out_er = _epsilon_R_and_rms(trajectories, out_dir, figures_dir, dict(meta, stage_name="eps_R"))
    decisions.extend(out_er["decisions"])
    out_mc = _mc_oracle_lowerbound(
        schedules, summary, policy_rows, out_dir,
        dict(meta, stage_name="mc_oracle_lowerbound"),
    )
    decisions.extend(out_mc["decisions"])
    _corrected_policy_table(policy_rows, schedules, out_dir, meta)

    n_pass = sum(1 for d in decisions if d["result"] == "pass")
    n_fail = sum(1 for d in decisions if d["result"] == "fail")
    n_inconc = sum(1 for d in decisions if d["result"] == "inconclusive")
    total = len(decisions)
    print(
        f"ANALYSIS_COMPLETE: gates={total} pass={n_pass} fail={n_fail} "
        f"inconclusive={n_inconc}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
