#!/usr/bin/env python3
"""Plot gen_ppl and MAUVE vs diffusion steps for the step-sweep experiment.

Reads comparison.json from results/full_eval (T=128), results/sweep/T256,
results/sweep/T512, and results/t1000_eval (T=1000).

Usage:
    python scripts/plot_sweep.py [--results_dir results] [--out figures/step_sweep.pdf]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STEPS = [128, 256, 512, 1000]
RESULT_DIRS = {
    128:  "full_eval",
    256:  "sweep/T256",
    512:  "sweep/T512",
    1000: "t1000_eval",
}
STRATEGIES = ["mdlm", "remdm-conf", "remdm-loop"]
COLORS = {"mdlm": "#4C72B0", "remdm-conf": "#DD8452", "remdm-loop": "#55A868"}
MARKERS = {"mdlm": "o", "remdm-conf": "s", "remdm-loop": "^"}
LABELS = {"mdlm": "MDLM", "remdm-conf": "ReMDM-conf", "remdm-loop": "ReMDM-loop"}


def load_comparison(results_dir: Path, subdir: str) -> dict[str, dict] | None:
    """Load comparison.json for one step count. Returns {strategy: metrics} or None."""
    p = results_dir / subdir / "comparison.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    return {r["strategy"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="figures/step_sweep.pdf")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — pip install matplotlib")
        raise SystemExit(1)

    # Collect data: data[strategy][step] = {gen_ppl, MAUVE}
    data: dict[str, dict[int, dict]] = {s: {} for s in STRATEGIES}
    missing = []
    for step, subdir in RESULT_DIRS.items():
        comp = load_comparison(results_dir, subdir)
        if comp is None:
            missing.append(f"T={step} ({subdir}/comparison.json)")
            continue
        for strategy in STRATEGIES:
            if strategy in comp:
                data[strategy][step] = comp[strategy]

    if missing:
        print(f"WARNING: missing data for: {', '.join(missing)}")
        print("Available data only — plot may be incomplete.")

    # Check we have anything to plot
    all_steps_present = {s: sorted(data[s].keys()) for s in STRATEGIES}
    if not any(all_steps_present.values()):
        print("No data found. Run eval jobs first.")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for strategy in STRATEGIES:
        steps_avail = sorted(data[strategy].keys())
        if not steps_avail:
            continue
        ppl_vals  = [data[strategy][t]["gen_ppl"] for t in steps_avail]
        mauve_vals = [data[strategy][t]["MAUVE"]   for t in steps_avail]

        kw = dict(color=COLORS[strategy], marker=MARKERS[strategy],
                  linewidth=2, markersize=7, label=LABELS[strategy])
        ax1.plot(steps_avail, ppl_vals, **kw)
        ax2.plot(steps_avail, mauve_vals, **kw)

    ax1.set_xlabel("Diffusion steps (T)")
    ax1.set_ylabel("Generation perplexity (↓ better)")
    ax1.set_title("gen_ppl vs T")
    ax1.set_xscale("log")
    ax1.set_xticks(STEPS)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Diffusion steps (T)")
    ax2.set_ylabel("MAUVE (↑ better)")
    ax2.set_title("MAUVE vs T")
    ax2.set_xscale("log")
    ax2.set_xticks(STEPS)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("ReMDM strategy comparison — step sweep (OWT reference, 100 samples, seed=42)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    # Also save PNG alongside PDF
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Saved: {png_path}")

    # Print data table
    print("\n## Step sweep — full table\n")
    header = f"{'strategy':<14} | " + " | ".join(f"T={t} ppl / MAUVE" for t in STEPS)
    print(header)
    print("-" * len(header))
    for strategy in STRATEGIES:
        row = f"{strategy:<14} | "
        cells = []
        for t in STEPS:
            if t in data[strategy]:
                d = data[strategy][t]
                cells.append(f"{d['gen_ppl']:6.2f} / {d['MAUVE']:.3f}")
            else:
                cells.append("  —    /   —  ")
        print(row + " | ".join(cells))


if __name__ == "__main__":
    main()
