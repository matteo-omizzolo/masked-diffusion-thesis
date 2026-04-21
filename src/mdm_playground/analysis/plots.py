"""Plotting helpers for Phase 2 analysis.

Every function returns the matplotlib Figure; callers decide where to save.
Saved figures must be paired with a `.meta.json` sidecar carrying the source
JSON paths and fields, per ANALYSIS_SPEC §5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as _e:  # pragma: no cover
    plt = None  # type: ignore


def _require_mpl():
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")


def save_figure_with_meta(
    fig,
    png_path: Path,
    meta: Dict[str, object],
    pdf_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Save fig to PNG + optional PDF, write meta.json sidecar."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if pdf_path is None:
        pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    sidecar = png_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2))


def scatter_A_vs_G(
    A: Sequence[float],
    G: Sequence[float],
    B: int,
    spearman: Optional[Dict[str, float]] = None,
    pearson: Optional[Dict[str, float]] = None,
    title_extra: str = "",
):
    """Scatter of A(S) vs G(S) at budget B, with identity line."""
    _require_mpl()
    A_arr = np.asarray(list(A), dtype=float)
    G_arr = np.asarray(list(G), dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(A_arr, G_arr, s=28, alpha=0.7, color="#2c7fb8")
    lo = float(min(A_arr.min(), G_arr.min()))
    hi = float(max(A_arr.max(), G_arr.max()))
    pad = 0.1 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", alpha=0.6, label="identity")
    ax.set_xlabel("A(S) = Σ_{t∈S} ⟨Δ_t⟩ (additive surrogate)")
    ax.set_ylabel("G(S) = F(y^S) − F(y_base) (true joint gain)")
    subtitle_bits = []
    if spearman is not None:
        subtitle_bits.append(
            f"Spearman ρ = {spearman['rho']:+.3f} "
            f"[{spearman.get('ci_lo', 0):+.3f}, {spearman.get('ci_hi', 0):+.3f}]"
        )
    if pearson is not None:
        subtitle_bits.append(f"Pearson r = {pearson.get('rho', 0):+.3f}")
    suffix = " — " + "; ".join(subtitle_bits) if subtitle_bits else ""
    ax.set_title(f"A(S) vs G(S) at B={B}{suffix}{title_extra}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig


def histogram_per_trajectory_spearman(
    rhos_by_signal: Dict[str, Sequence[float]],
    bins: int = 20,
):
    """Histogram of per-trajectory Spearman ρ for each signal."""
    _require_mpl()
    n = len(rhos_by_signal)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (name, rhos) in zip(axes, rhos_by_signal.items()):
        arr = np.asarray(list(rhos), dtype=float)
        neg = int((arr < 0).sum())
        pos = int((arr > 0).sum())
        ax.hist(arr, bins=bins, color="#31a354", alpha=0.8, edgecolor="white")
        ax.axvline(0.0, color="black", linestyle="--", alpha=0.6)
        ax.axvline(float(arr.mean()), color="red", linestyle=":", alpha=0.8, label=f"mean={arr.mean():+.3f}")
        ax.set_ylabel(name, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.02, 0.88,
            f"neg={neg}, pos={pos}",
            transform=ax.transAxes, fontsize=8, alpha=0.8,
        )
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Per-trajectory Spearman(ψ, Δ)")
    fig.suptitle("Per-trajectory rank correlation of signal vs Δ_t")
    fig.tight_layout()
    return fig


def policy_gains_box_by_B(
    per_policy_per_B: Dict[str, Dict[str, Sequence[float]]],
    B_values: Sequence[int],
    policies_order: Optional[Sequence[str]] = None,
):
    """Multi-panel box-whisker of G_per_seed per policy per B, with uniform highlighted."""
    _require_mpl()
    if policies_order is None:
        policies_order = list(per_policy_per_B.keys())
    B_values = list(B_values)
    n_B = len(B_values)
    fig, axes = plt.subplots(n_B, 1, figsize=(max(8.0, 1.0 * len(policies_order)), 3.2 * n_B))
    if n_B == 1:
        axes = [axes]
    for ax, B in zip(axes, B_values):
        data: List[np.ndarray] = []
        labels: List[str] = []
        for pol in policies_order:
            entry = per_policy_per_B.get(pol, {}).get(str(B)) or per_policy_per_B.get(pol, {}).get(B)
            if entry is None:
                continue
            G = entry.get("G_per_seed") if isinstance(entry, dict) else None
            if G is None:
                continue
            data.append(np.asarray(list(G), dtype=float))
            labels.append(pol)
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.65)
        for i, patch in enumerate(bp["boxes"]):
            if labels[i] == "uniform":
                patch.set_facecolor("#f0b27a")
                patch.set_edgecolor("black")
            else:
                patch.set_facecolor("#a9cce3")
                patch.set_edgecolor("#1b4f72")
        # Add scatter of individual seeds for transparency.
        for i, g in enumerate(data):
            ax.scatter(
                np.full_like(g, i + 1, dtype=float) + (np.random.default_rng(i + B * 31).uniform(-0.08, 0.08, size=g.size)),
                g, s=8, alpha=0.5, color="#154360",
            )
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel(f"G(S) at B={B}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("Phase 2b: G(S) per policy, by budget B (paired K seeds)")
    fig.tight_layout()
    return fig


def paired_diff_vs_uniform(
    policy: str,
    B: int,
    diffs: Sequence[float],
    ci_lo: float,
    ci_hi: float,
    tier: str = "",
):
    """Per-seed paired differences (policy − uniform) with CI."""
    _require_mpl()
    arr = np.asarray(list(diffs), dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    x = np.arange(1, arr.size + 1)
    ax.stem(x, arr, basefmt=" ", markerfmt="o", linefmt="-")
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.6)
    ax.axhspan(ci_lo, ci_hi, color="#82e0aa", alpha=0.3, label="95% bootstrap CI")
    ax.axhline(arr.mean(), color="red", linestyle=":", label=f"mean={arr.mean():+.3f}")
    ax.set_xlabel("seed index")
    ax.set_ylabel(f"{policy} − uniform")
    ax.set_title(f"Paired difference — policy='{policy}', B={B} — tier {tier}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def mc_oracle_headroom_lines(
    per_seed_uniform: Sequence[float],
    per_seed_mc_oracle: Sequence[float],
    B: int,
):
    """For each seed, a line from uniform G to MC-oracle G."""
    _require_mpl()
    u = np.asarray(list(per_seed_uniform), dtype=float)
    m = np.asarray(list(per_seed_mc_oracle), dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i in range(u.size):
        ax.plot([0, 1], [u[i], m[i]], "-", alpha=0.4, color="#1b4f72")
        ax.scatter([0], [u[i]], color="#f0b27a", s=20, zorder=3)
        ax.scatter([1], [m[i]], color="#196f3d", s=20, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["uniform", "MC oracle (300 samples)"])
    ax.set_ylabel(f"G(S) at B={B}")
    diff_mean = float((m - u).mean())
    ax.set_title(f"MC-oracle headroom over uniform — B={B}, Δ_mean={diff_mean:+.3f}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def f_nll_vs_f_mauve_ranking(
    rank_nll_by_policy: Dict[str, int],
    rank_mauve_by_policy: Dict[str, int],
    B: int,
):
    """Scatter of ranking under NLL vs MAUVE, one dot per policy."""
    _require_mpl()
    keys = sorted(set(rank_nll_by_policy) & set(rank_mauve_by_policy))
    xs = np.array([rank_nll_by_policy[k] for k in keys], dtype=float)
    ys = np.array([rank_mauve_by_policy[k] for k in keys], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(xs, ys, s=40, color="#1b4f72")
    for k, x, y in zip(keys, xs, ys):
        ax.annotate(k, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    lo = 0.5
    hi = max(xs.max(), ys.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5)
    ax.set_xlabel("rank under F=NLL")
    ax.set_ylabel("rank under F=MAUVE")
    ax.set_title(f"Policy ranking cross-F at B={B}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
