"""Phase 3a — combinatorial-baseline analysis.

Consumes
  results/phase3a_proseco_owt/{cd_raw.json, bs_raw.json}
plus the Phase 2b paired baselines
  results/phase2b_proseco_owt/policy_raw.json    (uniform G per (seed, B))
  results/phase2b/mc_oracle.json                 (MC oracle headroom per B)

Emits
  results/phase3a/cd_paired.json                 (Δ_CD vs uniform per (seed, B), BCa CI per B)
  results/phase3a/bs_paired.json                 (Δ_BS vs uniform per (seed, B), BCa CI per B)
  results/phase3a/oracle_gap_closure.json        (Δ_CD/Δ_oracle, Δ_BS/Δ_oracle per B)
  figures/phase3a/oracle_gap_closure.{png,pdf}   (bar chart per B)
  figures/phase3a/paired_diff_uniform_B{B}_{cd,bs}.{png,pdf}

Decision rules in `docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md`.

Usage
-----
    python scripts/proseco/reproduction/analyze_phase3a.py \
        --results_dir   results/phase3a_proseco_owt \
        --phase2b_dir   results/phase2b_proseco_owt \
        --phase2b_aggr  results/phase2b \
        --out_dir       results/phase3a \
        --figures_dir   figures/phase3a

Exits 0 on success; non-zero if any required input is missing.
Prints `PHASE3A_COMPLETE: cd_pass=N cd_fail=M bs_pass=K bs_fail=L`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from mdm_playground.analysis import paired_bootstrap_ci  # noqa: E402


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Per-B paired analysis
# ---------------------------------------------------------------------------


def _paired_gain_table(
    method_rows: Sequence[Dict[str, Any]],
    uniform_by_seed_B: Dict[Tuple[int, int], float],
    method_label: str,
) -> Dict[str, Any]:
    """Build per-(seed, B) paired-diff table + per-B BCa CI."""
    by_B: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    missing: List[Tuple[int, int]] = []
    for row in method_rows:
        seed = int(row["seed"])
        B = int(row["B"])
        key = (seed, B)
        if key not in uniform_by_seed_B:
            missing.append(key)
            continue
        G_method = float(row["G_final"])
        G_uniform = float(uniform_by_seed_B[key])
        by_B[B].append((seed, G_method, G_uniform))

    per_B: Dict[str, Dict[str, Any]] = {}
    pass_count = 0
    fail_count = 0
    null_count = 0
    for B in sorted(by_B):
        triples = by_B[B]
        seeds = [t[0] for t in triples]
        a = np.asarray([t[1] for t in triples], dtype=float)
        b = np.asarray([t[2] for t in triples], dtype=float)
        boot = paired_bootstrap_ci(a, b, n_resamples=2000, alpha=0.05, seed=int(B))
        verdict = "NULL"
        if boot["ci_lo"] > 0:
            verdict = "PASS"
            pass_count += 1
        elif boot["ci_hi"] < 0:
            verdict = "FAIL"
            fail_count += 1
        else:
            null_count += 1
        per_B[str(B)] = {
            "B": int(B),
            "n_pairs": len(triples),
            "seeds": seeds,
            "G_method": a.tolist(),
            "G_uniform": b.tolist(),
            "diffs": (a - b).tolist(),
            "mean_diff": boot["mean_diff"],
            "se_diff": boot["se_diff"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "verdict": verdict,
        }

    return {
        "method": method_label,
        "n_missing_pair_keys": len(missing),
        "missing_pair_keys": missing[:20],
        "per_B": per_B,
        "summary": {
            "pass": pass_count,
            "fail": fail_count,
            "null": null_count,
        },
    }


# ---------------------------------------------------------------------------
# Oracle gap closure
# ---------------------------------------------------------------------------


def _extract_mc_per_B(mc_oracle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Locate the per-B MC-oracle-vs-uniform table across known schemas.

    Phase 2b's `mc_oracle.json` nests it at
    ``data.mc_oracle_minus_uniform.{B}.{mean, bootstrap_95_ci, ...}``.
    Older snapshots may store ``per_B.{B}.mean_diff``. Handle both.
    """
    if "data" in mc_oracle and "mc_oracle_minus_uniform" in mc_oracle["data"]:
        return mc_oracle["data"]["mc_oracle_minus_uniform"]
    if "mc_oracle_minus_uniform" in mc_oracle:
        return mc_oracle["mc_oracle_minus_uniform"]
    if "per_B" in mc_oracle:
        return mc_oracle["per_B"]
    return mc_oracle


def _oracle_mean(cell: Dict[str, Any]) -> float:
    if "mean" in cell:
        return float(cell["mean"])
    if "mean_diff" in cell:
        return float(cell["mean_diff"])
    return float("nan")


def _oracle_ci(cell: Dict[str, Any]) -> Tuple[float, float]:
    ci = cell.get("bootstrap_95_ci")
    if ci and len(ci) == 2:
        return float(ci[0]), float(ci[1])
    return float(cell.get("ci_lo", float("nan"))), float(cell.get("ci_hi", float("nan")))


def _oracle_gap_closure(
    cd_table: Dict[str, Any],
    bs_table: Dict[str, Any],
    mc_oracle: Dict[str, Any],
) -> Dict[str, Any]:
    """For each B, ratio Δ_method / Δ_MC_oracle (mean-of-paired)."""
    mc_per_B = _extract_mc_per_B(mc_oracle)
    out: Dict[str, Any] = {}
    for B_str, cd_cell in cd_table["per_B"].items():
        if B_str not in mc_per_B:
            continue
        mc_cell = mc_per_B[B_str]
        bs_cell = bs_table["per_B"].get(B_str, {})
        delta_oracle = _oracle_mean(mc_cell)
        oracle_lo, oracle_hi = _oracle_ci(mc_cell)
        delta_cd = float(cd_cell["mean_diff"])
        delta_bs = float(bs_cell.get("mean_diff", 0.0)) if bs_cell else float("nan")
        out[B_str] = {
            "B": int(B_str),
            "delta_oracle": delta_oracle,
            "oracle_ci": [oracle_lo, oracle_hi],
            "delta_cd": delta_cd,
            "delta_bs": delta_bs,
            "ratio_cd": float(delta_cd / delta_oracle) if delta_oracle not in (0.0, float("nan")) else float("nan"),
            "ratio_bs": float(delta_bs / delta_oracle) if delta_oracle not in (0.0, float("nan")) else float("nan"),
            "cd_ci": [cd_cell["ci_lo"], cd_cell["ci_hi"]],
            "bs_ci": [bs_cell.get("ci_lo", float("nan")), bs_cell.get("ci_hi", float("nan"))],
            "cd_verdict": cd_cell["verdict"],
            "bs_verdict": bs_cell.get("verdict", "MISSING"),
        }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_oracle_gap(out: Dict[str, Any], figures_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figures", file=sys.stderr)
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    Bs = sorted(int(k) for k in out.keys())
    if not Bs:
        return
    width = 0.28
    x = np.arange(len(Bs), dtype=float)
    delta_oracle = [out[str(b)]["delta_oracle"] for b in Bs]
    delta_cd = [out[str(b)]["delta_cd"] for b in Bs]
    delta_bs = [out[str(b)]["delta_bs"] for b in Bs]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - width, delta_oracle, width, label="MC oracle (best-of-100)", color="#888888")
    ax.bar(x,         delta_cd,     width, label="Coordinate descent (CD-G)", color="#1f77b4")
    ax.bar(x + width, delta_bs,     width, label="Beam search (BS-AG)",      color="#ff7f0e")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in Bs])
    ax.set_ylabel("Mean paired Δ vs uniform G")
    ax.set_title("Phase 3a — oracle-gap closure (paired, K=30)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"oracle_gap_closure.{ext}", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results/phase3a_proseco_owt")
    ap.add_argument("--phase2b_dir", type=str, default="results/phase2b_proseco_owt",
                    help="Phase 2b raw dir (provides paired uniform G per (seed, B))")
    ap.add_argument("--phase2b_aggr", type=str, default="results/phase2b",
                    help="Phase 2b aggregated dir (provides mc_oracle.json)")
    ap.add_argument("--out_dir", type=str, default="results/phase3a")
    ap.add_argument("--figures_dir", type=str, default="figures/phase3a")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    p2b_dir = Path(args.phase2b_dir)
    p2b_aggr = Path(args.phase2b_aggr)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)

    cd_path = results_dir / "cd_raw.json"
    bs_path = results_dir / "bs_raw.json"
    p2b_policy_path = p2b_dir / "policy_raw.json"
    mc_path = p2b_aggr / "mc_oracle.json"

    for p in (cd_path, bs_path, p2b_policy_path, mc_path):
        if not p.exists():
            print(f"MISSING REQUIRED INPUT: {p}", file=sys.stderr)
            return 2

    cd_rows = _load(cd_path)
    bs_rows = _load(bs_path)
    p2b_rows = _load(p2b_policy_path)
    mc_oracle = _load(mc_path)

    # Build {(seed, B) -> uniform G} from Phase 2b uniform rows.
    uniform_by_seed_B: Dict[Tuple[int, int], float] = {}
    for row in p2b_rows:
        if row.get("policy") == "uniform":
            uniform_by_seed_B[(int(row["seed"]), int(row["B"]))] = float(row["G"])
    print(f"Loaded uniform baseline for {len(uniform_by_seed_B)} (seed, B) pairs from Phase 2b")

    cd_table = _paired_gain_table(cd_rows, uniform_by_seed_B, method_label="coordinate_descent_G")
    bs_table = _paired_gain_table(bs_rows, uniform_by_seed_B, method_label="beam_search_AG")

    _write(out_dir / "cd_paired.json", cd_table)
    _write(out_dir / "bs_paired.json", bs_table)

    gap_closure = _oracle_gap_closure(cd_table, bs_table, mc_oracle)
    _write(out_dir / "oracle_gap_closure.json", {
        "per_B": gap_closure,
        "notes": (
            "Δ_CD, Δ_BS are paired mean(G_method − G_uniform) across K=30 seeds; "
            "Δ_oracle is the Phase 2b MC-oracle headroom per B; ratios capture "
            "fraction of MC-oracle headroom recovered by each search procedure."
        ),
    })

    _plot_oracle_gap(gap_closure, figures_dir)

    cd_summary = cd_table["summary"]
    bs_summary = bs_table["summary"]
    print("=" * 60)
    print("Phase 3a paired-comparison summary:")
    print(f"  CD-G:  PASS={cd_summary['pass']}  FAIL={cd_summary['fail']}  NULL={cd_summary['null']}")
    print(f"  BS-AG: PASS={bs_summary['pass']}  FAIL={bs_summary['fail']}  NULL={bs_summary['null']}")
    print("Per-B oracle-gap closure:")
    for B_str in sorted(gap_closure, key=int):
        cell = gap_closure[B_str]
        print(
            f"  B={cell['B']:2d}  Δ_oracle={cell['delta_oracle']:+.3f}  "
            f"Δ_CD={cell['delta_cd']:+.3f} ({cell['ratio_cd']*100:5.1f}%, {cell['cd_verdict']})  "
            f"Δ_BS={cell['delta_bs']:+.3f} ({cell['ratio_bs']*100:5.1f}%, {cell['bs_verdict']})"
        )
    print("=" * 60)
    print(
        f"PHASE3A_COMPLETE: "
        f"cd_pass={cd_summary['pass']} cd_fail={cd_summary['fail']} "
        f"bs_pass={bs_summary['pass']} bs_fail={bs_summary['fail']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
