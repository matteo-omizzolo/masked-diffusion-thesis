#!/usr/bin/env python3
"""Phase 1 — Interaction diagnostics analyzer (Gate 3a outputs).

Consumes:
    <out_dir>/per_seed/xi_rows_seed*.json
    (and xi_raw.json if concatenated on HPC)

Produces:
    <figures_dir>/A_xi_heatmap.{png,pdf}
    <figures_dir>/B_sign_prob_heatmap.{png,pdf}
    <figures_dir>/C_phase_pair_summary.{png,pdf}
    <figures_dir>/D_distance_summary.{png,pdf}
    <figures_dir>/E_marginal_vs_pair.{png,pdf}
    <figures_dir>/F_residual.{png,pdf}
    <figures_dir>/G_seed_distribution.{png,pdf}
    <figures_dir>/H_schedule_source_overlay.{png,pdf}  (if --phase3a_dir provided)
    <figures_dir>/I_compute_accounting.{png,pdf}
    <out_dir>/aggregate_stats.json
    <out_dir>/interpretation.md

Usage
-----
    python scripts/proseco/interactions/analyze_phase1_interactions.py \\
        --out_dir results/phase1_interaction_diag_<sha> \\
        --figures_dir figures/phase1_interaction_diag \\
        [--phase3a_dir results/phase3a_proseco_owt]

Gate 3a decision: This script produces evidence for whether interactions
are structured enough to justify Gate 3b (schedule-level validation). It
does NOT claim Theorem B validation; that requires Gate 3b.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available; figures will not be produced.", file=sys.stderr)

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

try:
    from mdm_playground.analysis import paired_bootstrap_ci  # noqa: E402
    HAS_BOOTSTRAP = True
except ImportError:
    HAS_BOOTSTRAP = False


# ---------------------------------------------------------------------------
# Bootstrap CI fallback
# ---------------------------------------------------------------------------

def _seed_bootstrap_ci(
    per_seed_values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Seed-clustered percentile bootstrap CI.

    per_seed_values must contain exactly one value per seed (e.g. per-seed
    xi for a single pair, or per-seed mean xi within a stratum). Resampling
    seeds is the correct unit since observations within a seed are correlated.
    """
    rng = np.random.default_rng(0)
    n = len(per_seed_values)
    boot_means = [
        float(np.mean(rng.choice(per_seed_values, size=n, replace=True)))
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(per_seed_values)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows(out_dir: Path) -> List[Dict[str, Any]]:
    """Load all xi rows from per_seed files or xi_raw.json."""
    xi_raw = out_dir / "xi_raw.json"
    if xi_raw.exists():
        return json.loads(xi_raw.read_text())
    per_seed_dir = out_dir / "per_seed"
    if not per_seed_dir.exists():
        raise FileNotFoundError(f"No xi_raw.json and no per_seed/ in {out_dir}")
    rows: List[Dict[str, Any]] = []
    for f in sorted(per_seed_dir.glob("xi_rows_seed*.json")):
        rows.extend(json.loads(f.read_text()))
    if not rows:
        raise FileNotFoundError(f"No xi rows found in {out_dir}")
    return rows


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregate_stats(
    rows: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[Tuple[int, int], np.ndarray]]:
    """Compute per-pair and stratum statistics with seed-clustered bootstrap CIs.

    All CIs resample seeds (the correct unit), not rows. Rows within a seed
    are correlated so row-level bootstrap would underestimate variance.

    Returns
    -------
    stats : dict with per-pair entries and stratum summaries
    by_pair : dict {(t, t') -> array of xi per seed}
    """
    # Group by (pair, seed) — one xi per (pair, seed)
    by_pair_seed: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
    # Group by (stratum, seed) — list of xi per (stratum, seed)
    by_phase_pair_seed: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_distance_bin_seed: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_seed_all: Dict[int, List[float]] = defaultdict(list)

    for row in rows:
        key = (int(row["t"]), int(row["t_prime"]))
        seed = int(row["seed"])
        xi = float(row["xi"])
        by_pair_seed[key][seed] = xi
        phase_key = f"{row['phase_t']}-{row['phase_tp']}"
        by_phase_pair_seed[phase_key][seed].append(xi)
        d = int(row["distance"])
        dbin = "short (1-5)" if d <= 5 else ("medium (6-20)" if d <= 20 else "long (21+)")
        by_distance_bin_seed[dbin][seed].append(xi)
        by_seed_all[seed].append(xi)

    # Per-pair: one value per seed → CI is valid seed-level resample
    pair_stats: Dict[str, Any] = {}
    for (t, tp), seed_map in by_pair_seed.items():
        arr = np.array(list(seed_map.values()))
        mean, ci_lo, ci_hi = _seed_bootstrap_ci(arr)
        pair_stats[f"{t},{tp}"] = {
            "t": t, "t_prime": tp,
            "n_seeds": len(arr),
            "mean_xi": mean,
            "std_xi": float(np.std(arr)),
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
            "p_positive": float(np.mean(arr > 0)),
            "median_xi": float(np.median(arr)),
        }

    # Per-phase-pair: average within each seed first, then bootstrap seeds
    phase_pair_stats: Dict[str, Any] = {}
    for pp_key, seed_map in by_phase_pair_seed.items():
        seed_means = np.array([float(np.mean(vs)) for vs in seed_map.values()])
        all_xis = np.concatenate(list(seed_map.values()))
        mean, ci_lo, ci_hi = _seed_bootstrap_ci(seed_means)
        phase_pair_stats[pp_key] = {
            "mean_xi": mean,
            "std_xi": float(np.std(all_xis)),
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
            "p_positive": float(np.mean(all_xis > 0)),
            "n_seeds": len(seed_means),
            "n_obs": int(len(all_xis)),
        }

    # Per-distance-bin: same approach
    distance_stats: Dict[str, Any] = {}
    for dbin, seed_map in by_distance_bin_seed.items():
        seed_means = np.array([float(np.mean(vs)) for vs in seed_map.values()])
        all_xis = np.concatenate(list(seed_map.values()))
        mean, ci_lo, ci_hi = _seed_bootstrap_ci(seed_means)
        distance_stats[dbin] = {
            "mean_xi": mean,
            "std_xi": float(np.std(all_xis)),
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
            "p_positive": float(np.mean(all_xis > 0)),
            "n_seeds": len(seed_means),
            "n_obs": int(len(all_xis)),
        }

    # Overall: per-seed mean of per-seed means, then bootstrap seeds
    per_seed_mean_arr = np.array([float(np.mean(vs)) for vs in by_seed_all.values()])
    overall_mean, overall_ci_lo, overall_ci_hi = _seed_bootstrap_ci(per_seed_mean_arr)
    all_xi_flat = np.array([r["xi"] for r in rows])

    stats = {
        "n_rows": len(rows),
        "n_seeds": len(by_seed_all),
        "n_pairs": len(by_pair_seed),
        "ci_method": "seed_clustered_bootstrap_percentile_95",
        "overall_mean_xi": overall_mean,
        "overall_ci_lo_95": overall_ci_lo,
        "overall_ci_hi_95": overall_ci_hi,
        "overall_p_positive": float(np.mean(all_xi_flat > 0)),
        "overall_std_xi": float(np.std(per_seed_mean_arr)),
        "per_pair": pair_stats,
        "by_phase_pair": phase_pair_stats,
        "by_distance_bin": distance_stats,
    }
    by_pair_arr = {k: np.array(list(v.values())) for k, v in by_pair_seed.items()}
    return stats, by_pair_arr


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_PHASE_ORDER = ["early", "middle", "late"]
_PHASE_PAIR_ORDER = [
    f"{a}-{b}"
    for a in _PHASE_ORDER
    for b in _PHASE_ORDER
    if _PHASE_ORDER.index(a) <= _PHASE_ORDER.index(b)
]

T = 64  # fixed for ProSeCo-OWT


def _savefig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot A — pairwise interaction heatmap (mean xi)
# ---------------------------------------------------------------------------

def plot_A_xi_heatmap(
    by_pair: Dict[Tuple[int, int], np.ndarray], figures_dir: Path
) -> None:
    if not HAS_MPL:
        return
    grid = np.full((T, T), np.nan)
    for (t, tp), xis in by_pair.items():
        v = float(np.mean(xis))
        grid[tp, t] = v
        grid[t, tp] = v

    measured = grid[~np.isnan(grid)]
    vmax = max(abs(measured).max(), 1e-6) if len(measured) else 0.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, origin="lower", cmap="RdBu_r", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="mean ξ_{t,t'}")
    ax.set_xlabel("t")
    ax.set_ylabel("t'")
    ax.set_title("(A) Pairwise interaction heatmap — mean ξ_{t,t'}\n"
                 "Red = complementary, Blue = redundant, Grey = unmeasured")
    for x in [21, 43]:
        ax.axvline(x - 0.5, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.axhline(x - 0.5, color="k", lw=0.5, ls="--", alpha=0.4)
    _savefig(fig, figures_dir / "A_xi_heatmap")


# ---------------------------------------------------------------------------
# Plot B — sign probability heatmap
# ---------------------------------------------------------------------------

def plot_B_sign_prob_heatmap(
    by_pair: Dict[Tuple[int, int], np.ndarray], figures_dir: Path
) -> None:
    if not HAS_MPL:
        return
    grid = np.full((T, T), np.nan)
    for (t, tp), xis in by_pair.items():
        v = float(np.mean(xis > 0))
        grid[tp, t] = v
        grid[t, tp] = v

    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, origin="lower", cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="P(ξ > 0)")
    ax.set_xlabel("t")
    ax.set_ylabel("t'")
    ax.set_title("(B) Interaction sign-probability — P(ξ_{t,t'} > 0)\n"
                 "Red = consistently positive, Blue = consistently negative")
    for x in [21, 43]:
        ax.axvline(x - 0.5, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.axhline(x - 0.5, color="k", lw=0.5, ls="--", alpha=0.4)
    _savefig(fig, figures_dir / "B_sign_prob_heatmap")


# ---------------------------------------------------------------------------
# Plot C — phase-pair summary
# ---------------------------------------------------------------------------

def plot_C_phase_pair_summary(
    by_phase_pair: Dict[str, Any], figures_dir: Path
) -> None:
    if not HAS_MPL:
        return
    labels = [pp for pp in _PHASE_PAIR_ORDER if pp in by_phase_pair]
    means = [by_phase_pair[pp]["mean_xi"] for pp in labels]
    ci_lo = [by_phase_pair[pp]["ci_lo_95"] for pp in labels]
    ci_hi = [by_phase_pair[pp]["ci_hi_95"] for pp in labels]
    errs = [[m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)]]

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = range(len(labels))
    colors = ["#e74c3c" if m > 0 else "#3498db" for m in means]
    ax.bar(xs, means, yerr=errs, color=colors, alpha=0.7, capsize=4, ecolor="k")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean ξ (95 % CI)")
    ax.set_title("(C) Phase-pair summary — mean ξ by (phase_t, phase_t')\n"
                 "Error bars: seed-clustered 95 % bootstrap CI (seeds resampled, not rows)")
    _savefig(fig, figures_dir / "C_phase_pair_summary")


# ---------------------------------------------------------------------------
# Plot D — distance summary
# ---------------------------------------------------------------------------

def plot_D_distance_summary(
    by_distance_bin: Dict[str, Any], figures_dir: Path
) -> None:
    if not HAS_MPL:
        return
    order = ["short (1-5)", "medium (6-20)", "long (21+)"]
    labels = [d for d in order if d in by_distance_bin]
    means = [by_distance_bin[d]["mean_xi"] for d in labels]
    ci_lo = [by_distance_bin[d]["ci_lo_95"] for d in labels]
    ci_hi = [by_distance_bin[d]["ci_hi_95"] for d in labels]
    errs = [[m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)]]
    n_obs = [by_distance_bin[d]["n_obs"] for d in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(len(labels)), means, yerr=errs, capsize=5, alpha=0.8, ecolor="k")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, n_obs)])
    ax.set_ylabel("mean ξ (95 % CI)")
    ax.set_title("(D) Distance-bucket summary — mean ξ vs |t − t'|\n"
                 "Tests whether nearby or far-apart placements interact differently")
    _savefig(fig, figures_dir / "D_distance_summary")


# ---------------------------------------------------------------------------
# Plot E — marginal vs pair
# ---------------------------------------------------------------------------

def plot_E_marginal_vs_pair(
    rows: List[Dict[str, Any]], figures_dir: Path
) -> None:
    if not HAS_MPL:
        return
    A_vals = np.array([r["A_pair"] for r in rows])
    G_vals = np.array([r["G_pair"] for r in rows])
    phase_pairs = [f"{r['phase_t']}-{r['phase_tp']}" for r in rows]
    unique_pp = sorted(set(phase_pairs))
    color_map = {pp: plt.cm.tab10(i / max(len(unique_pp) - 1, 1))
                 for i, pp in enumerate(unique_pp)}

    fig, ax = plt.subplots(figsize=(6, 5))
    for pp in unique_pp:
        mask = [p == pp for p in phase_pairs]
        ax.scatter(A_vals[mask], G_vals[mask], alpha=0.35, s=12,
                   color=color_map[pp], label=pp)
    lim_lo = min(A_vals.min(), G_vals.min()) - 0.01
    lim_hi = max(A_vals.max(), G_vals.max()) + 0.01
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="y = x (additive)")
    ax.set_xlabel("A({t,t'}) = Δ_t + Δ_{t'}")
    ax.set_ylabel("G({t,t'})")
    ax.set_title("(E) Marginal vs pair gain\n"
                 "Points above y=x: pair gain exceeds additive prediction (ξ > 0)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    _savefig(fig, figures_dir / "E_marginal_vs_pair")


# ---------------------------------------------------------------------------
# Plot F — residual plot
# ---------------------------------------------------------------------------

def plot_F_residual(rows: List[Dict[str, Any]], figures_dir: Path) -> None:
    if not HAS_MPL:
        return
    A_vals = np.array([r["A_pair"] for r in rows])
    xi_vals = np.array([r["xi"] for r in rows])
    phase_pairs = [f"{r['phase_t']}-{r['phase_tp']}" for r in rows]
    unique_pp = sorted(set(phase_pairs))
    color_map = {pp: plt.cm.tab10(i / max(len(unique_pp) - 1, 1))
                 for i, pp in enumerate(unique_pp)}

    fig, ax = plt.subplots(figsize=(6, 5))
    for pp in unique_pp:
        mask = [p == pp for p in phase_pairs]
        ax.scatter(A_vals[mask], xi_vals[mask], alpha=0.35, s=12,
                   color=color_map[pp], label=pp)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("A({t,t'}) = Δ_t + Δ_{t'}")
    ax.set_ylabel("ξ_{t,t'} = G({t,t'}) − A({t,t'})")
    ax.set_title("(F) Residual ξ vs additive prediction A\n"
                 "Systematic pattern → interactions depend on corrector quality region")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    _savefig(fig, figures_dir / "F_residual")


# ---------------------------------------------------------------------------
# Plot G — seed-level distribution (violin by phase pair)
# ---------------------------------------------------------------------------

def plot_G_seed_distribution(
    rows: List[Dict[str, Any]], figures_dir: Path
) -> None:
    if not HAS_MPL:
        return
    pp_groups: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        pp_groups[f"{r['phase_t']}-{r['phase_tp']}"].append(float(r["xi"]))

    labels = [pp for pp in _PHASE_PAIR_ORDER if pp in pp_groups]
    data = [pp_groups[pp] for pp in labels]

    fig, ax = plt.subplots(figsize=(9, 4))
    parts = ax.violinplot(data, positions=range(len(labels)), showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ξ_{t,t'}")
    ax.set_title("(G) Seed-level ξ distribution by phase pair\n"
                 "Wide violins → noisy; median far from zero → systematic effect")
    _savefig(fig, figures_dir / "G_seed_distribution")


# ---------------------------------------------------------------------------
# Plot H — schedule-source overlay (optional)
# ---------------------------------------------------------------------------

def plot_H_schedule_source_overlay(
    by_pair: Dict[Tuple[int, int], np.ndarray],
    phase3a_dir: Optional[Path],
    figures_dir: Path,
) -> None:
    if not HAS_MPL or phase3a_dir is None:
        return

    # Load CD-G best schedules at B=2
    best_pairs_cd: set = set()
    best_pairs_bs: set = set()
    cd_raw = phase3a_dir / "cd_raw.json"
    bs_raw = phase3a_dir / "bs_raw.json"
    if cd_raw.exists():
        for row in json.loads(cd_raw.read_text()):
            if int(row.get("B", 0)) == 2:
                steps = sorted(int(s) for s in row.get("schedule_final", []))
                if len(steps) == 2:
                    best_pairs_cd.add((steps[0], steps[1]))
    if bs_raw.exists():
        for row in json.loads(bs_raw.read_text()):
            if int(row.get("B", 0)) == 2:
                steps = sorted(int(s) for s in row.get("schedule_final", []))
                if len(steps) == 2:
                    best_pairs_bs.add((steps[0], steps[1]))

    measured = set(by_pair.keys())
    cd_in_measured = best_pairs_cd & measured
    bs_in_measured = best_pairs_bs & measured
    # Coverage ratio: what fraction of selected pairs are in our measured set?
    cd_coverage = len(cd_in_measured) / max(len(best_pairs_cd), 1)
    bs_coverage = len(bs_in_measured) / max(len(best_pairs_bs), 1)
    print(f"  H overlay: CD-G coverage {len(cd_in_measured)}/{len(best_pairs_cd)} "
          f"({cd_coverage:.0%}); BS-AG {len(bs_in_measured)}/{len(best_pairs_bs)} "
          f"({bs_coverage:.0%})")

    grid_mean = np.full((T, T), np.nan)
    for (t, tp), xis in by_pair.items():
        v = float(np.mean(xis))
        grid_mean[tp, t] = v
        grid_mean[t, tp] = v

    measured_arr = grid_mean[~np.isnan(grid_mean)]
    vmax = max(abs(measured_arr).max(), 1e-6) if len(measured_arr) else 0.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(grid_mean)
    ax.imshow(masked, origin="lower", cmap="RdBu_r", norm=norm, aspect="auto", alpha=0.7)
    # Overlay CD-G selected pairs
    for (t, tp) in cd_in_measured:
        ax.plot(t, tp, "^", color="darkgreen", ms=8, label="CD-G B=2")
        ax.plot(tp, t, "^", color="darkgreen", ms=8)
    for (t, tp) in bs_in_measured:
        ax.plot(t, tp, "s", color="purple", ms=6, label="BS-AG B=2")
        ax.plot(tp, t, "s", color="purple", ms=6)
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.set_xlabel("t")
    ax.set_ylabel("t'")
    ax.set_title(
        f"(H) ξ heatmap with schedule-source overlay\n"
        f"CD-G coverage {len(cd_in_measured)}/{max(len(best_pairs_cd),1)} "
        f"({cd_coverage:.0%}); "
        f"BS-AG {len(bs_in_measured)}/{max(len(best_pairs_bs),1)} ({bs_coverage:.0%})"
    )
    _savefig(fig, figures_dir / "H_schedule_source_overlay")


# ---------------------------------------------------------------------------
# Plot I — compute accounting
# ---------------------------------------------------------------------------

def plot_I_compute_accounting(
    rows: List[Dict[str, Any]], out_dir: Path, figures_dir: Path,
    expected_seeds: Optional[List[int]] = None,
    expected_pairs_per_seed: Optional[int] = None,
) -> None:
    if not HAS_MPL:
        return
    # Load expected seed list from run_config if not provided
    if expected_seeds is None:
        for rc_file in sorted(out_dir.glob("run_config.shard*.json")):
            rc = json.loads(rc_file.read_text())
            seed_start = int(rc.get("seed_start", 42))
            K = int(rc.get("K_total", 30))
            expected_seeds = list(range(seed_start, seed_start + K))
            break
    if expected_pairs_per_seed is None:
        pl_path = out_dir / "pair_list.json"
        if pl_path.exists():
            expected_pairs_per_seed = len(json.loads(pl_path.read_text()))

    present_seeds = set(r["seed"] for r in rows)
    all_seeds = sorted(expected_seeds) if expected_seeds else sorted(present_seeds)
    n_per_seed = [sum(1 for r in rows if r["seed"] == s) for s in all_seeds]
    wall_per_seed = [
        float(np.mean([r["wall_time"] for r in rows if r["seed"] == s])) if any(
            r["seed"] == s for r in rows
        ) else 0.0
        for s in all_seeds
    ]
    missing_idx = [i for i, s in enumerate(all_seeds) if s not in present_seeds]
    colors_n = ["#e74c3c" if i in missing_idx else "#3498db" for i in range(len(all_seeds))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].bar(range(len(all_seeds)), n_per_seed, color=colors_n)
    if expected_pairs_per_seed:
        axes[0].axhline(expected_pairs_per_seed, color="k", ls="--",
                        label=f"expected={expected_pairs_per_seed}")
        axes[0].legend(fontsize=8)
    if missing_idx:
        axes[0].bar(missing_idx, [0] * len(missing_idx), color="#e74c3c",
                    label=f"missing ({len(missing_idx)} seeds)")
    axes[0].set_xlabel("seed index")
    axes[0].set_ylabel("pairs evaluated")
    axes[0].set_title("Pairs evaluated per seed (red = missing/failed)")

    axes[1].bar(range(len(all_seeds)), wall_per_seed)
    axes[1].set_xlabel("seed index")
    axes[1].set_ylabel("mean wall_time / pair (s)")
    axes[1].set_title("Mean per-pair wall time by seed")

    n_missing = len(missing_idx)
    fig.suptitle(
        f"(I) Compute accounting — {len(all_seeds)} expected seeds, "
        f"{n_missing} missing/failed (red bars)"
    )
    _savefig(fig, figures_dir / "I_compute_accounting")


# ---------------------------------------------------------------------------
# Interpretation markdown
# ---------------------------------------------------------------------------

def write_interpretation(
    stats: Dict[str, Any], out_dir: Path, figures_dir: Path
) -> None:
    overall_mean = stats["overall_mean_xi"]
    overall_ci_lo = stats["overall_ci_lo_95"]
    overall_ci_hi = stats["overall_ci_hi_95"]
    p_pos = stats["overall_p_positive"]
    n_seeds = stats["n_seeds"]
    n_pairs = stats["n_pairs"]

    lines = [
        "# Phase 1 Interaction Diagnostics — Gate 3a Interpretation",
        "",
        f"> Generated from {out_dir.name}",
        f"> {n_seeds} seeds × {n_pairs} pairs = {stats['n_rows']} observations",
        "",
        "## Summary statistics",
        "",
        f"- Overall mean ξ: {overall_mean:.4f}  (95 % CI [{overall_ci_lo:.4f}, {overall_ci_hi:.4f}])",
        f"- P(ξ > 0): {p_pos:.3f}",
        f"- Overall std ξ: {stats['overall_std_xi']:.4f}",
        "",
        "## How to read the plots",
        "",
        "**ξ_{t,t'} = G({t,t'}) − Δ_t − Δ_{t'}** is the discrete second difference",
        "of the schedule-value function G. It is an *operational* pairwise term, not a",
        "claim about mechanistic decomposability of the diffusion process.",
        "",
        "- **ξ > 0**: correction placements at t and t' together produce more gain",
        "  than the sum of their individual gains. The pair is complementary.",
        "- **ξ < 0**: the two placements are redundant; placing both is less valuable",
        "  than the additive approximation A({t,t'}) = Δ_t + Δ_{t'} predicts.",
        "- **ξ ≈ 0**: placements are approximately independent; Theorem A's additive",
        "  surrogate A(S) is a good approximation for this pair.",
        "",
        "**Plot A (ξ heatmap)**: Each measured pair shows its mean ξ across seeds.",
        "Unmeasured pairs are grey (NaN). A predominantly red heatmap means most",
        "pairs are complementary; blue means mostly redundant.",
        "",
        "**Plot B (sign probability)**: P(ξ > 0) per pair. Values near 0.5 indicate",
        "noisy effects with no stable sign; values near 0 or 1 indicate a consistent",
        "directional interaction.",
        "",
        "**Plot C (phase-pair summary)**: Aggregates ξ by (phase_t, phase_t').",
        "A strong phase-pair pattern suggests interactions are structured by trajectory",
        "phase, which would support a phase-aware pairwise surrogate.",
        "",
        "**Plot D (distance summary)**: Tests whether |t − t'| predicts interaction",
        "sign or magnitude. Monotone distance dependence suggests a kernel structure.",
        "",
        "**Plot E (marginal vs pair)**: Points above y=x mean the pair outperforms",
        "the additive prediction (ξ > 0). Systematic bias above the diagonal would",
        "indicate that A(S) under-predicts pair gain.",
        "",
        "**Plot F (residual)**: ξ vs A_pair. Slope ≠ 0 suggests interactions are",
        "stronger when marginal gains are larger (or smaller). Heteroscedasticity",
        "is also informative.",
        "",
        "**Plot G (seed distribution)**: Violin widths reflect seed-to-seed",
        "variability. Wide violins with median at zero → interactions are noisy,",
        "not systematic. Narrow violins with offset median → structural signal.",
        "",
        "**Plot H (schedule-source overlay)**: Shows whether CD-G / BS-AG at B=2",
        "tend to select high-ξ pairs. Overlap with red cells (high mean ξ) would",
        "suggest that successful search leverages pairwise complementarity.",
        "",
        "**Plot I (compute accounting)**: Each bar = one seed. Missing bars = failed",
        "or skipped evaluations. Uneven wall times may indicate GPU contention.",
        "",
        "## Why pair heatmaps alone do NOT validate Theorem B",
        "",
        "Gate 3a (sparse pair diagnostics) shows whether ξ_{t,t'} is non-negligible",
        "and whether it is structured by phase or distance. This is necessary evidence",
        "but not sufficient to validate Theorem B (pairwise surrogate regret) for the",
        "following reasons:",
        "",
        "1. ξ_{t,t'} is measured at Level 1 (seed-wise diagnostic). Theorem B",
        "   requires that Q(S) = A(S) + Σ ξ_{t,t'} approximates G(S) better than",
        "   A(S) on a held-out candidate schedule pool C_B with no data leakage.",
        "2. Even strongly non-zero ξ values do not guarantee that Q̂(S) — estimated",
        "   from a finite seed set — is a better surrogate than A(S) on held-out",
        "   seeds. The no-leakage caveat (Theorem B′ §2.3) requires a separate",
        "   training/evaluation split.",
        "3. The schedule-level claim ζ_{B,C} < η_{B,C} and P_B > R_B can only be",
        "   verified via Gate 3b (schedule-level validation on pool C_B).",
        "",
        "## What result would justify Gate 3b",
        "",
        "Gate 3b is justified if Gate 3a shows:",
        "",
        "- **Structured non-zero ξ**: Mean ξ significantly different from zero",
        "  (95 % CI excludes 0) in at least one phase-pair stratum, OR",
        "- **Sign consistency**: P(ξ > 0) > 0.65 or < 0.35 in at least one stratum",
        "  across the K=30 seeds, suggesting a stable directional interaction.",
        "",
        "Gate 3b is NOT justified if interactions appear purely noise-driven",
        "(P(ξ > 0) ≈ 0.5 in all strata, CIs always include 0). In that case the",
        "regime is provisionally classified as Regime IV (Diagnostic Framework C),",
        "and the search-based result (CD-G / BS-AG) remains the primary positive",
        "contribution.",
        "",
        "## Phase-pair and distance details",
        "",
    ]

    lines.append("### By phase pair")
    lines.append("")
    lines.append("| Phase pair | mean ξ | 95 % CI | P(ξ > 0) | n |")
    lines.append("|---|---|---|---|---|")
    for pp in _PHASE_PAIR_ORDER:
        if pp in stats["by_phase_pair"]:
            s = stats["by_phase_pair"][pp]
            lines.append(
                f"| {pp} | {s['mean_xi']:.4f} | [{s['ci_lo_95']:.4f}, {s['ci_hi_95']:.4f}] "
                f"| {s['p_positive']:.3f} | {s['n_obs']} |"
            )
    lines.append("")
    lines.append("### By distance bucket")
    lines.append("")
    lines.append("| Bucket | mean ξ | 95 % CI | P(ξ > 0) | n |")
    lines.append("|---|---|---|---|---|")
    for dbin in ["short (1-5)", "medium (6-20)", "long (21+)"]:
        if dbin in stats["by_distance_bin"]:
            s = stats["by_distance_bin"][dbin]
            lines.append(
                f"| {dbin} | {s['mean_xi']:.4f} | [{s['ci_lo_95']:.4f}, {s['ci_hi_95']:.4f}] "
                f"| {s['p_positive']:.3f} | {s['n_obs']} |"
            )
    lines.append("")

    (out_dir / "interpretation.md").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'interpretation.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PHASE_PAIR_ORDER = [
    "early-early", "early-middle", "early-late",
    "middle-middle", "middle-late", "late-late",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 interaction diagnostics analyzer")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Runner output directory (contains per_seed/ or xi_raw.json)")
    p.add_argument("--figures_dir", type=str, required=True,
                   help="Output directory for figures")
    p.add_argument("--phase3a_dir", type=str, default=None,
                   help="Optional: Phase 3a results dir for schedule-source overlay (plot H)")
    p.add_argument("--n_boot", type=int, default=5000,
                   help="Bootstrap resamples for CIs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    phase3a_dir = Path(args.phase3a_dir) if args.phase3a_dir else None

    print(f"Loading rows from {out_dir} ...")
    rows = load_rows(out_dir)
    print(f"  Loaded {len(rows)} rows ({len(set(r['seed'] for r in rows))} seeds, "
          f"{len(set((r['t'], r['t_prime']) for r in rows))} pairs)")

    print("Computing aggregate statistics ...")
    stats, by_pair = compute_aggregate_stats(rows)

    agg_path = out_dir / "aggregate_stats.json"
    agg_path.write_text(json.dumps(stats, indent=2))
    print(f"  Wrote {agg_path}")

    print(f"Overall mean ξ = {stats['overall_mean_xi']:.4f} "
          f"[{stats['overall_ci_lo_95']:.4f}, {stats['overall_ci_hi_95']:.4f}]  "
          f"P(ξ>0) = {stats['overall_p_positive']:.3f}")

    if HAS_MPL:
        print(f"Producing figures in {figures_dir} ...")
        plot_A_xi_heatmap(by_pair, figures_dir)
        plot_B_sign_prob_heatmap(by_pair, figures_dir)
        plot_C_phase_pair_summary(stats["by_phase_pair"], figures_dir)
        plot_D_distance_summary(stats["by_distance_bin"], figures_dir)
        plot_E_marginal_vs_pair(rows, figures_dir)
        plot_F_residual(rows, figures_dir)
        plot_G_seed_distribution(rows, figures_dir)
        plot_H_schedule_source_overlay(by_pair, phase3a_dir, figures_dir)
        plot_I_compute_accounting(rows, out_dir, figures_dir)
        print("  All figures written (PNG + PDF).")
    else:
        print("  matplotlib unavailable; skipping figure generation.")

    write_interpretation(stats, out_dir, figures_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
