"""
analyze_corrector_strength_preflight.py
========================================
Analyze the Phase B corrector-strength preflight output and produce 7 plots.

Reads:
  results/corrector_strength_preflight_<sha>/raw_deltas.json
  results/corrector_strength_preflight_<sha>/raw_pairs.json
  results/corrector_strength_preflight_<sha>/summary.json

Outputs (in same dir):
  plots/delta_distribution_by_strength.png
  plots/apair_distribution_by_strength.png
  plots/xi_distribution_by_strength.png
  plots/changed_tokens_by_strength.png
  plots/gpair_vs_apair_by_strength_preflight.png
  plots/xi_vs_apair_by_strength_preflight.png
  plots/pass_increment_effects.png
  interpretation.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

STRENGTH_ORDER = ["no_correction", "strength_0", "strength_1", "strength_2"]
STRENGTH_LABELS = {
    "no_correction": "no-correction",
    "strength_0": "weak (k=0)",
    "strength_1": "standard (k=1)",
    "strength_2": "strong (k=2)",
}
COLORS = {
    "no_correction": "#AAAAAA",
    "strength_0": "#77AADD",
    "strength_1": "#EE8866",
    "strength_2": "#44BB99",
}
PLOT_STYLE: dict[str, Any] = {
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style() -> None:
    matplotlib.rcParams.update(PLOT_STYLE)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    ss_xx = ((x - xm) ** 2).sum()
    if ss_xx < 1e-12:
        return float("nan"), float("nan"), float("nan")
    slope = ((x - xm) * (y - ym)).sum() / ss_xx
    intercept = ym - slope * xm
    y_hat = slope * x + intercept
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1.0 - ((y - y_hat) ** 2).sum() / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(intercept), float(r2)


def plot_delta_distribution(
    raw_deltas: list[dict], strengths: list[str], out_path: Path
) -> None:
    _apply_style()
    real_strengths = [s for s in strengths if s != "no_correction"]
    fig, axes = plt.subplots(1, len(real_strengths), figsize=(4 * len(real_strengths), 3.5),
                             sharey=True)
    if len(real_strengths) == 1:
        axes = [axes]
    for ax, sl in zip(axes, real_strengths):
        vals = [r["delta_t"] for r in raw_deltas if r["strength"] == sl]
        ax.hist(vals, bins=20, color=COLORS.get(sl, "steelblue"), alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(vals), color="red", lw=1.5, ls="--",
                   label=f"mean={np.mean(vals):.3f}")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(STRENGTH_LABELS.get(sl, sl))
        ax.set_xlabel("Δ_t")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("count")
    fig.suptitle("Δ_t distribution by strength level", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_apair_distribution(
    raw_pairs: list[dict], strengths: list[str], out_path: Path
) -> None:
    _apply_style()
    real_strengths = [s for s in strengths if s != "no_correction"]
    if not real_strengths:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    data = [np.array([r["A_pair"] for r in raw_pairs if r["strength"] == sl
                      and not math.isnan(r.get("A_pair", float("nan")))])
            for sl in real_strengths]
    labels = [STRENGTH_LABELS.get(sl, sl) for sl in real_strengths]
    colors = [COLORS.get(sl, "steelblue") for sl in real_strengths]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=False)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("A_pair = Δ_t + Δ_t′")
    ax.set_title("A_pair distribution by strength level")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_xi_distribution(
    raw_pairs: list[dict], strengths: list[str], out_path: Path
) -> None:
    _apply_style()
    real_strengths = [s for s in strengths if s != "no_correction"]
    if not real_strengths:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    data = [np.array([r["xi"] for r in raw_pairs if r["strength"] == sl
                      and not math.isnan(r.get("xi", float("nan")))])
            for sl in real_strengths]
    labels = [STRENGTH_LABELS.get(sl, sl) for sl in real_strengths]
    colors = [COLORS.get(sl, "steelblue") for sl in real_strengths]
    for i, (d, label, col) in enumerate(zip(data, labels, colors), 1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.1, 0.1, len(d)),
                   d, color=col, alpha=0.5, s=15, zorder=2)
        ax.plot([i - 0.25, i + 0.25], [np.mean(d)] * 2, color=col, lw=2.5, zorder=3)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(1, len(real_strengths) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("ξ = G_pair − A_pair")
    ax.set_title("ξ distribution by strength level")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_changed_tokens(
    raw_deltas: list[dict], strengths: list[str], out_path: Path
) -> None:
    _apply_style()
    real_strengths = [s for s in strengths if s != "no_correction"]
    if not real_strengths:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Corrector-level n_changed
    data_corr = [np.array([r["corrector_n_changed"] for r in raw_deltas if r["strength"] == sl])
                 for sl in real_strengths]
    labels = [STRENGTH_LABELS.get(sl, sl) for sl in real_strengths]
    colors = [COLORS.get(sl, "steelblue") for sl in real_strengths]
    bp1 = axes[0].boxplot(data_corr, tick_labels=labels, patch_artist=True)
    for patch, col in zip(bp1["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    axes[0].set_title("Corrector-call n_changed")
    axes[0].set_ylabel("tokens changed at corrector call")
    axes[0].tick_params(axis="x", rotation=15)

    # Final n_changed (Hamming base vs branch)
    data_final = [np.array([r["final_n_changed"] for r in raw_deltas if r["strength"] == sl])
                  for sl in real_strengths]
    bp2 = axes[1].boxplot(data_final, tick_labels=labels, patch_artist=True)
    for patch, col in zip(bp2["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    axes[1].set_title("Final Hamming (base vs branch)")
    axes[1].set_ylabel("tokens different in final output")
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle("Token change counts by strength level", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_gpair_vs_apair(
    raw_pairs: list[dict], strengths: list[str], out_path: Path
) -> None:
    _apply_style()
    real_strengths = [s for s in strengths if s != "no_correction"]
    if not real_strengths:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for sl in real_strengths:
        rows = [r for r in raw_pairs if r["strength"] == sl
                and not math.isnan(r.get("A_pair", float("nan")))]
        if not rows:
            continue
        a = np.array([r["A_pair"] for r in rows])
        g = np.array([r["G_pair"] for r in rows])
        col = COLORS.get(sl, "steelblue")
        ax.scatter(a, g, color=col, alpha=0.6, s=20, label=STRENGTH_LABELS.get(sl, sl))
        if len(a) >= 2:
            slope, intercept, r2 = _ols(a, g)
            x_line = np.linspace(a.min(), a.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color=col, lw=1.5, alpha=0.8,
                    label=f"  OLS slope={slope:.3f}")

    # identity line
    all_a = [r["A_pair"] for r in raw_pairs if not math.isnan(r.get("A_pair", float("nan")))]
    if all_a:
        xm = np.linspace(min(all_a), max(all_a), 100)
        ax.plot(xm, xm, "k--", lw=0.8, alpha=0.4, label="identity G=A")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("A_pair")
    ax.set_ylabel("G_pair")
    ax.set_title("G_pair vs A_pair by strength (preflight)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_xi_vs_apair(
    raw_pairs: list[dict], strengths: list[str], out_path: Path
) -> None:
    _apply_style()
    real_strengths = [s for s in strengths if s != "no_correction"]
    if not real_strengths:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for sl in real_strengths:
        rows = [r for r in raw_pairs if r["strength"] == sl
                and not math.isnan(r.get("A_pair", float("nan")))]
        if not rows:
            continue
        a = np.array([r["A_pair"] for r in rows])
        xi = np.array([r["xi"] for r in rows])
        col = COLORS.get(sl, "steelblue")
        ax.scatter(a, xi, color=col, alpha=0.6, s=20, label=STRENGTH_LABELS.get(sl, sl))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("A_pair")
    ax.set_ylabel("ξ = G_pair − A_pair")
    ax.set_title("ξ vs A_pair by strength (preflight)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pass_increment_effects(
    raw_deltas: list[dict], strengths: list[str], out_path: Path
) -> None:
    """Show per-timestep mean corrector n_changed for strength_0/1/2 side by side."""
    _apply_style()
    real_strengths = [s for s in strengths if s != "no_correction"]
    all_ts = sorted(set(r["t"] for r in raw_deltas))
    if not all_ts:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(all_ts) // 2), 3.5))
    width = 0.8 / max(len(real_strengths), 1)
    x = np.arange(len(all_ts))

    for i, sl in enumerate(real_strengths):
        means = []
        for t_step in all_ts:
            vals = [r["corrector_n_changed"] for r in raw_deltas
                    if r["strength"] == sl and r["t"] == t_step]
            means.append(np.mean(vals) if vals else 0.0)
        col = COLORS.get(sl, "steelblue")
        offset = (i - len(real_strengths) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, color=col, alpha=0.75,
               label=STRENGTH_LABELS.get(sl, sl))

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in all_ts], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("mean corrector n_changed")
    ax.set_title("Per-timestep corrector change counts (strength comparison)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(
    out_path: Path,
    summary: dict[str, Any],
    raw_deltas: list[dict],
    raw_pairs: list[dict],
) -> None:
    gates = summary.get("gates", {})
    gate_pass = summary.get("gate_pass", False)
    verdict = "**PREFLIGHT PASS**" if gate_pass else "**PREFLIGHT FAIL**"

    per_s = summary.get("per_strength", {})

    lines = [f"# Corrector-Strength Preflight Interpretation\n",
             f"SHA: {summary.get('sha', 'unknown')} | T={summary.get('T')} | "
             f"debug={summary.get('debug')} | surrogate={summary.get('surrogate', False)}\n",
             f"## Gate verdict: {verdict}\n",
             "| Gate | Pass | Notes |",
             "|------|------|-------|"]
    for gname, gval in gates.items():
        p = gval.get("pass")
        status = "✅" if p is True else ("⚠️ N/A" if p is None else "❌")
        note_keys = [k for k in gval if k != "pass"]
        note = "; ".join(f"{k}={gval[k]}" for k in note_keys[:2])
        lines.append(f"| {gname} | {status} | {note} |")

    lines.append("\n## Per-strength summary\n")
    lines.append("| Strength | mean Δ_t | mean corr_nch | mean G_pair | mean ξ | P(ξ>0) |")
    lines.append("|----------|---------|---------------|-------------|--------|--------|")
    for sl in STRENGTH_ORDER:
        if sl not in per_s or per_s[sl].get("n_delta_rows", 0) == 0:
            continue
        s = per_s[sl]
        lines.append(f"| {sl} | {s.get('mean_delta_t', float('nan')):.4f} "
                     f"| {s.get('mean_corrector_n_changed', float('nan')):.1f} "
                     f"| {s.get('mean_G_pair', float('nan')):.4f} "
                     f"| {s.get('mean_xi', float('nan')):.4f} "
                     f"| {s.get('p_xi_pos', float('nan')):.3f} |")

    lines.append("\n## Next decision\n")
    if gate_pass:
        lines.append(
            "All preflight gates passed. Proceed to main Phase B experiment (K=30).\n"
            "Strength levels confirmed: no_correction (sanity), weak (k=0), "
            "standard (k=1), strong (k=2)."
        )
    else:
        lines.append(
            "Preflight did not fully pass. Review failing gates above before "
            "proceeding to main experiment."
        )

    out_path.write_text("\n".join(lines) + "\n")


def run_analysis(preflight_dir: Path) -> dict[str, Any]:
    raw_deltas = json.loads((preflight_dir / "raw_deltas.json").read_text())
    raw_pairs = json.loads((preflight_dir / "raw_pairs.json").read_text())
    summary = json.loads((preflight_dir / "summary.json").read_text())

    strengths_present = sorted(set(r["strength"] for r in raw_deltas),
                                key=lambda s: STRENGTH_ORDER.index(s) if s in STRENGTH_ORDER else 99)

    plots_dir = preflight_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"[analyze] Writing 7 plots to {plots_dir} ...", flush=True)
    plot_delta_distribution(raw_deltas, strengths_present, plots_dir / "delta_distribution_by_strength.png")
    plot_apair_distribution(raw_pairs, strengths_present, plots_dir / "apair_distribution_by_strength.png")
    plot_xi_distribution(raw_pairs, strengths_present, plots_dir / "xi_distribution_by_strength.png")
    plot_changed_tokens(raw_deltas, strengths_present, plots_dir / "changed_tokens_by_strength.png")
    plot_gpair_vs_apair(raw_pairs, strengths_present, plots_dir / "gpair_vs_apair_by_strength_preflight.png")
    plot_xi_vs_apair(raw_pairs, strengths_present, plots_dir / "xi_vs_apair_by_strength_preflight.png")
    plot_pass_increment_effects(raw_deltas, strengths_present, plots_dir / "pass_increment_effects.png")
    print("[analyze] All 7 plots written.", flush=True)

    write_interpretation(preflight_dir / "interpretation.md", summary, raw_deltas, raw_pairs)
    print("[analyze] interpretation.md written.", flush=True)

    return {"gate_pass": summary.get("gate_pass", False),
            "plots_dir": str(plots_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze corrector-strength preflight")
    parser.add_argument("preflight_dir", type=Path,
                        help="Path to corrector_strength_preflight_<sha>/ directory")
    args = parser.parse_args()
    run_analysis(args.preflight_dir)


if __name__ == "__main__":
    main()
