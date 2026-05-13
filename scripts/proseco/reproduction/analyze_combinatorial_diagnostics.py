#!/usr/bin/env python3
"""Compute combinatorial schedule diagnostics from Phase 2b raw artifacts.

Inputs:
  - results/phase2b_proseco_owt/mc_raw.json
  - results/phase2b_proseco_owt/policy_raw.json

Output:
  - results/phase2b/combinatorial_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from mdm_playground.analysis import build_combinatorial_diagnostics  # noqa: E402


def _load(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _write(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase2b_raw_dir",
        type=str,
        default="results/phase2b_proseco_owt",
        help="Directory containing mc_raw.json and policy_raw.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default="results/phase2b/combinatorial_diagnostics.json",
        help="Output JSON path",
    )
    p.add_argument("--top_k", type=int, default=10, help="Top-k schedules for overlap diagnostics")
    p.add_argument(
        "--random_baseline_samples",
        type=int,
        default=5000,
        help="Samples used for random Jaccard baseline",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for baseline simulation")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.phase2b_raw_dir)
    mc_path = raw_dir / "mc_raw.json"
    policy_path = raw_dir / "policy_raw.json"

    if not mc_path.exists():
        print(f"Missing required file: {mc_path}", file=sys.stderr)
        return 2
    if not policy_path.exists():
        print(f"Missing required file: {policy_path}", file=sys.stderr)
        return 2

    mc_rows = _load(mc_path)
    policy_rows = _load(policy_path)
    if not isinstance(mc_rows, list) or not mc_rows:
        print(f"Invalid or empty mc_raw payload: {mc_path}", file=sys.stderr)
        return 2
    if not isinstance(policy_rows, list) or not policy_rows:
        print(f"Invalid or empty policy_raw payload: {policy_path}", file=sys.stderr)
        return 2

    data = build_combinatorial_diagnostics(
        mc_rows=mc_rows,
        policy_rows=policy_rows,
        top_k=int(args.top_k),
        random_baseline_samples=int(args.random_baseline_samples),
        random_seed=int(args.seed),
    )

    out = {
        "meta": {
            "source_phase2b_raw_dir": str(raw_dir),
            "top_k": int(args.top_k),
            "random_baseline_samples": int(args.random_baseline_samples),
            "seed": int(args.seed),
            "n_mc_rows": int(len(mc_rows)),
            "n_policy_rows": int(len(policy_rows)),
        },
        "data": data,
    }
    _write(Path(args.out), out)
    print(f"Wrote diagnostics to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
