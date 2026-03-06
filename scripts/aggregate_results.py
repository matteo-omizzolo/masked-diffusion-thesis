#!/usr/bin/env python3
"""Aggregate ReMDM results into a comparison table.

Scans results/ for generated_sequences.json files (written by upstream ReMDM
in sample_eval mode), extracts gen_ppl / entropy / MAUVE, and prints a
markdown table. Also writes results/comparison.json.

Usage:
    python scripts/aggregate_results.py [--results_dir results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_result_files(results_dir: Path) -> list[tuple[str, Path]]:
    """Return (strategy, json_path) pairs, sorted by strategy name."""
    found = []
    for json_path in sorted(results_dir.rglob("generated_sequences.json")):
        # Path structure: results/full_eval/<strategy>/external_remdm/generated_sequences.json
        # or:             results/<run_dir>/external_remdm/generated_sequences.json
        # Infer strategy from run_meta.json sibling if available, else from path.
        run_dir = json_path.parent.parent  # one level above external_remdm/
        meta_path = run_dir / "run_meta.json"
        strategy = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                strategy = meta.get("remdm_strategy") or meta.get("strategy")
            except Exception:
                pass
        if strategy is None:
            # Fall back: use the directory name two levels up (full_eval/<strategy>)
            strategy = run_dir.name
        found.append((strategy, json_path))
    found.sort(key=lambda x: x[0])
    return found


def load_metrics(json_path: Path) -> dict:
    data = json.loads(json_path.read_text())
    return {
        "gen_ppl": data.get("gen_ppl"),
        "entropy": data.get("entropy"),
        "MAUVE": data.get("MAUVE"),
    }


def fmt(val, decimals=3) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def main():
    p = argparse.ArgumentParser(description="Aggregate ReMDM results into a comparison table.")
    p.add_argument("--results_dir", default="results", help="Root results directory.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results_dir '{results_dir}' not found.")
        raise SystemExit(1)

    pairs = find_result_files(results_dir)
    if not pairs:
        print(f"No generated_sequences.json files found under '{results_dir}'.")
        raise SystemExit(0)

    rows = []
    for strategy, json_path in pairs:
        try:
            metrics = load_metrics(json_path)
        except Exception as e:
            print(f"WARNING: could not load {json_path}: {e}")
            metrics = {"gen_ppl": None, "entropy": None, "MAUVE": None}
        rows.append({"strategy": strategy, "path": str(json_path), **metrics})

    # Print markdown table
    print("\n## ReMDM Results Comparison\n")
    print(f"| {'strategy':<14} | {'gen_ppl':>8} | {'entropy':>8} | {'MAUVE':>8} |")
    print(f"|{'-'*16}|{'-'*10}|{'-'*10}|{'-'*10}|")
    for r in rows:
        print(
            f"| {r['strategy']:<14} | {fmt(r['gen_ppl']):>8} | {fmt(r['entropy']):>8} | {fmt(r['MAUVE']):>8} |"
        )
    print()

    # Write comparison.json
    out_path = results_dir / "comparison.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
