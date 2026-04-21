#!/usr/bin/env python3
"""Phase 1 analysis and figure generation.

Reads JSON outputs from run_phase1_pilot.py and produces:
  figures/phase1_*/
    calibration_scatter.png   — ψ vs Δ_t per signal (3-panel)
    delta_vs_t.png            — mean Δ_t and signal profiles across t
    eta_vs_B.png              — η_B scaling with B, plus Prop C bound
    pairwise_xi_hist.png      — |ξ| pairwise interaction histogram
    theorem_A_budget.png      — 2Bε + 2η_B vs G(Ŝ_B) across budgets
    tcr_vs_delta.png          — TCR_t vs Δ_t scatter (Q8 diagnostic)

Usage:
    python scripts/analyze_phase1.py --results_dir results/phase1_pilot \
        --out_dir figures/phase1_pilot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def load_results(results_dir: Path):
    """Load all JSON outputs from the pilot run."""
    # Protocol A
    traj_files = sorted((results_dir / "protocol_a").glob("trajectory_*.json"))
    all_records = []
    for f in traj_files:
        data = json.loads(f.read_text())
        all_records.append(data["per_t"])

    # Protocol B — schedules
    sched_files = sorted((results_dir / "protocol_b").glob("schedule_*.json"))
    schedule_records = [json.loads(f.read_text()) for f in sched_files]

    # Protocol B — pairs
    pairs_path = results_dir / "protocol_b" / "pairs.json"
    pairs_records = json.loads(pairs_path.read_text()) if pairs_path.exists() else []

    # Summary
    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    return all_records, schedule_records, pairs_records, summary


def fig_calibration_scatter(all_records, out_path: Path):
    """3-panel scatter: ψ vs Δ_t for entropy, margin, quality."""
    deltas = np.array([[r["delta"] for r in traj] for traj in all_records]).flatten()
    entropy = np.array([[r["entropy"] for r in traj] for traj in all_records]).flatten()
    margin = np.array([[r["inverse_margin"] for r in traj] for traj in all_records]).flatten()
    quality = np.array([[r["quality_mass_proxy"] for r in traj] for traj in all_records]).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    signals = [
        (entropy, "Entropy $H_t$", "tab:blue"),
        (margin, "Inverse Margin $\\tilde{M}_t$", "tab:orange"),
        (quality, "Quality Mass Proxy $Q_t$", "tab:green"),
    ]

    for ax, (sig, label, col) in zip(axes, signals):
        ax.scatter(sig, deltas, alpha=0.25, s=6, color=col, rasterized=True)
        # Linear fit
        coeffs = np.polyfit(sig, deltas, 1)
        xs = np.linspace(sig.min(), sig.max(), 200)
        ax.plot(xs, np.polyval(coeffs, xs), color="black", lw=1.5,
                label=f"fit: a={coeffs[0]:.3f}")
        from scipy.stats import spearmanr, pearsonr
        rho, _ = spearmanr(sig, deltas)
        r, _ = pearsonr(sig, deltas)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("$\\Delta_t$ (one-loop gain)", fontsize=11)
        ax.set_title(f"Spearman ρ = {rho:.3f}  |  Pearson r = {r:.3f}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Protocol A — Signal vs One-Loop Marginal Gain $\\Delta_t$\n"
        "(Theorem A calibration: smaller residual → smaller ε)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_delta_vs_t(all_records, out_path: Path):
    """Mean Δ_t and signal profiles across trajectory steps t."""
    T = len(all_records[0])
    deltas = np.array([[r["delta"] for r in traj] for traj in all_records])
    entropy = np.array([[r["entropy"] for r in traj] for traj in all_records])
    margin = np.array([[r["inverse_margin"] for r in traj] for traj in all_records])
    tcr = np.array([[r["tcr"] for r in traj] for traj in all_records])

    ts = np.arange(T)
    d_mean = deltas.mean(axis=0)
    d_ci = 1.96 * deltas.std(axis=0) / np.sqrt(len(all_records))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Panel 1: Δ_t
    ax = axes[0]
    ax.fill_between(ts, d_mean - d_ci, d_mean + d_ci, alpha=0.25, color="tab:red")
    ax.plot(ts, d_mean, color="tab:red", lw=2, label="Mean $\\Delta_t$")
    ax.set_ylabel("$\\Delta_t$ (one-loop gain)", fontsize=11)
    ax.set_title("One-Loop Marginal Gain Profile", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: Entropy and margin
    ax = axes[1]
    e_mean = entropy.mean(axis=0)
    m_mean = margin.mean(axis=0)
    # Normalise to [0, 1] for comparison
    ax.plot(ts, (e_mean - e_mean.min()) / (e_mean.max() - e_mean.min() + 1e-12),
            color="tab:blue", lw=2, label="Entropy $H_t$ (normalised)")
    ax.plot(ts, (m_mean - m_mean.min()) / (m_mean.max() - m_mean.min() + 1e-12),
            color="tab:orange", lw=2, label="Inv. Margin $\\tilde{M}_t$ (normalised)")
    ax.plot(ts, (d_mean - d_mean.min()) / (d_mean.max() - d_mean.min() + 1e-12),
            color="tab:red", lw=2, linestyle="--", label="$\\Delta_t$ (normalised)")
    ax.set_ylabel("Normalised value", fontsize=11)
    ax.set_title("Signal vs Gain — Normalised Comparison", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: TCR vs Δ (not the same thing)
    ax = axes[2]
    tcr_mean = tcr.mean(axis=0)
    # Normalise both to same scale
    ax2 = ax.twinx()
    ax.plot(ts, tcr_mean, color="tab:purple", lw=2, label="TCR$_t$ (left)")
    ax2.plot(ts, d_mean, color="tab:red", lw=2, linestyle="--", label="$\\Delta_t$ (right)")
    ax.set_ylabel("TCR$_t$ (token-change rate)", fontsize=11, color="tab:purple")
    ax2.set_ylabel("$\\Delta_t$", fontsize=11, color="tab:red")
    ax.set_title("TCR$_t$ ≠ $\\Delta_t$ — Q8 Diagnostic", fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Step $t$", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_eta_vs_B(schedule_records, summary, out_path: Path):
    """η_B scaling with B, plus Proposition C bound."""
    if not schedule_records:
        print(f"  Skipped (no schedule records): {out_path}")
        return

    B_vals = sorted(set(r["B"] for r in schedule_records))
    eta_95 = []
    eta_mean = []
    for B in B_vals:
        recs = [r for r in schedule_records if r["B"] == B]
        resids = [abs(r["residual"]) for r in recs]
        eta_95.append(np.percentile(resids, 95))
        eta_mean.append(np.mean(resids))

    gamma_95 = summary.get("gamma", {}).get("gamma_95", 0.01)
    B_fine = np.linspace(0, max(B_vals) * 1.2, 200)
    prop_C = gamma_95 * B_fine * (B_fine - 1) / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(B_vals, eta_95, "o-", color="tab:blue", lw=2, ms=8,
            label="$\\hat{\\eta}_B$ (95th pct, empirical)")
    ax.plot(B_vals, eta_mean, "s--", color="tab:cyan", lw=1.5, ms=6,
            label="$\\hat{\\eta}_B$ (mean, empirical)")
    ax.plot(B_fine, prop_C, color="tab:orange", lw=2, linestyle="-.",
            label=f"Prop. C bound: $\\gamma B(B-1)/2$, $\\gamma={gamma_95:.4f}$")
    ax.set_xlabel("Budget $B$", fontsize=12)
    ax.set_ylabel("$\\eta_B$ (additivity slack)", fontsize=12)
    ax.set_title(
        "Protocol B — Approximate Additivity Slack $\\eta_B$ vs Budget $B$\n"
        "(Theorem A non-vacuous when $\\eta_B$ is small)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_pairwise_xi(pairs_records, out_path: Path):
    """Histogram of |ξ_{t,t'}| pairwise interaction estimates."""
    if not pairs_records:
        print(f"  Skipped (no pairs): {out_path}")
        return
    xis = [abs(r["xi"]) for r in pairs_records]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(xis, bins=30, color="tab:green", edgecolor="white", alpha=0.85)
    p95 = np.percentile(xis, 95)
    ax.axvline(p95, color="black", lw=2, linestyle="--",
               label=f"95th pct = {p95:.5f}  (= $\\hat{{\\gamma}}$)")
    ax.set_xlabel("$|\\xi_{{t,t'}}|$  (pairwise interaction magnitude)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Protocol B — Pairwise Interaction Distribution\n"
        "(Proposition C: $\\eta_B \\leq \\hat{\\gamma}\\, B(B-1)/2$)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_theorem_A_budget(summary, out_path: Path):
    """Bar chart: 2Bε + 2η_B vs G(Ŝ_B) for each budget level."""
    checks = summary.get("theorem_A_bound_check", {})
    if not checks:
        print(f"  Skipped (no theorem_A_check in summary): {out_path}")
        return

    B_vals = sorted(int(k) for k in checks.keys())
    bounds = [checks[str(B)]["bound_2Be_2eta"] for B in B_vals]
    G_ests = [checks[str(B)]["G_top_B_estimate"] for B in B_vals]
    useful = [checks[str(B)]["bound_useful"] for B in B_vals]

    x = np.arange(len(B_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, G_ests, width, label="$G(\\hat{S}_B)$ estimate",
                   color="tab:green", alpha=0.85)
    bars2 = ax.bar(x + width / 2, bounds, width,
                   label="Theorem A bound $2B\\varepsilon + 2\\eta_B$",
                   color=["tab:red" if not u else "tab:orange" for u in useful],
                   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"B={B}" for B in B_vals], fontsize=12)
    ax.set_ylabel("Quality gain / regret bound", fontsize=11)
    ax.set_title(
        "Theorem A Check: bound vs achievable gain per budget $B$\n"
        "(green = non-vacuous, orange = vacuous)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    for i, (G, bd, u) in enumerate(zip(G_ests, bounds, useful)):
        tag = "✓" if u else "✗"
        ax.text(i, max(G, bd) + 0.001, tag, ha="center", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_tcr_vs_delta(all_records, out_path: Path):
    """Scatter: TCR_t vs Δ_t, Q8 diagnostic."""
    deltas = np.array([[r["delta"] for r in traj] for traj in all_records]).flatten()
    tcr = np.array([[r["tcr"] for r in traj] for traj in all_records]).flatten()

    from scipy.stats import pearsonr, spearmanr
    r, _ = pearsonr(tcr, deltas)
    rho, _ = spearmanr(tcr, deltas)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tcr, deltas, alpha=0.20, s=6, color="tab:purple", rasterized=True)
    coeffs = np.polyfit(tcr, deltas, 1)
    xs = np.linspace(tcr.min(), tcr.max(), 200)
    ax.plot(xs, np.polyval(coeffs, xs), color="black", lw=1.5)
    ax.set_xlabel("TCR$_t$ (token-change rate)", fontsize=12)
    ax.set_ylabel("$\\Delta_t$ (quality gain)", fontsize=12)
    ax.set_title(
        f"Q8 Diagnostic: TCR$_t$ ≠ $\\Delta_t$\n"
        f"Pearson r = {r:.3f}   Spearman ρ = {rho:.3f}",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/phase1_pilot")
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if args.out_dir is None:
        name = results_dir.name.replace("results/", "")
        out_dir = Path("figures") / results_dir.name
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HAVE_MPL:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib --break-system-packages")
        sys.exit(1)

    try:
        from scipy.stats import spearmanr
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scipy", "--break-system-packages", "-q"])

    print(f"Loading results from {results_dir} ...")
    all_records, schedule_records, pairs_records, summary = load_results(results_dir)
    N = len(all_records)
    T = len(all_records[0]) if all_records else 0
    print(f"  Loaded {N} trajectories, T={T}, {len(schedule_records)} schedules, {len(pairs_records)} pairs")

    print("\nGenerating figures ...")
    fig_calibration_scatter(all_records, out_dir / "calibration_scatter.png")
    fig_delta_vs_t(all_records, out_dir / "delta_vs_t.png")
    fig_eta_vs_B(schedule_records, summary, out_dir / "eta_vs_B.png")
    fig_pairwise_xi(pairs_records, out_dir / "pairwise_xi_hist.png")
    fig_theorem_A_budget(summary, out_dir / "theorem_A_budget.png")
    fig_tcr_vs_delta(all_records, out_dir / "tcr_vs_delta.png")

    print(f"\nAll figures written to {out_dir}/")


if __name__ == "__main__":
    main()
