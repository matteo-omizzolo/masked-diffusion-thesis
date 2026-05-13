#!/usr/bin/env python3
"""Compute Theorem A constants from Phase 2b raw artefacts.

Inputs:
  - results/phase2b_proseco_owt/mc_raw.json  (list of MC rows: seed, B, G, A,
    residual, schedule_steps)

Output:
  - results/phase2b/theorem_a_constants.json

Constants computed (see src/mdm_playground/analysis/theorem_a_constants.py):

  σ_ξ           additivity-residual std per B   (Refinement A′ plug-in)
  ρ             Spearman(A, G) per seed + pooled (Refinement A″ plug-in)
  σ_Δ           pooled std of G per B            (ε_R scale)
  γ             pairwise-interaction q_α upper   (Proposition C plug-in)
  low_gain_share  max-G(top-k by A) / max-G-all  (Proposition B anchor)
  plugin_bound    2Bε_R + 2η_B under A′ and Prop C variants

All estimators are data-only; no experiment re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from mdm_playground.analysis import build_theorem_a_constants  # noqa: E402


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
        help="Directory containing mc_raw.json",
    )
    p.add_argument(
        "--mc_raw_path",
        type=str,
        default=None,
        help="Optional explicit path to mc_raw.json (overrides --phase2b_raw_dir)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="results/phase2b/theorem_a_constants.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k schedules for the low-gain-share anchor (Proposition B)",
    )
    p.add_argument(
        "--gamma_quantile",
        type=float,
        default=0.95,
        help="Quantile used for γ upper bound (Proposition C)",
    )
    p.add_argument(
        "--rho_n_resamples",
        type=int,
        default=1000,
        help="Bootstrap resamples for the pooled Spearman ρ CI",
    )
    p.add_argument(
        "--rho_alpha",
        type=float,
        default=0.05,
        help="Alpha for Spearman bootstrap CI",
    )
    p.add_argument(
        "--rho_seed",
        type=int,
        default=0,
        help="Seed for Spearman bootstrap (per B increments from here)",
    )
    return p.parse_args()


def _resolve_mc_path(args: argparse.Namespace) -> Path:
    if args.mc_raw_path:
        return Path(args.mc_raw_path)
    return Path(args.phase2b_raw_dir) / "mc_raw.json"


def main() -> int:
    args = parse_args()
    mc_path = _resolve_mc_path(args)
    if not mc_path.exists():
        print(f"Missing required file: {mc_path}", file=sys.stderr)
        return 2

    mc_rows = _load(mc_path)
    if not isinstance(mc_rows, list) or not mc_rows:
        print(f"Invalid or empty mc_raw payload: {mc_path}", file=sys.stderr)
        return 2

    data = build_theorem_a_constants(
        mc_rows=mc_rows,
        top_k=int(args.top_k),
        gamma_quantile=float(args.gamma_quantile),
        rho_n_resamples=int(args.rho_n_resamples),
        rho_alpha=float(args.rho_alpha),
        rho_seed=int(args.rho_seed),
    )

    out = {
        "meta": {
            "source_mc_raw_path": str(mc_path),
            "top_k": int(args.top_k),
            "gamma_quantile": float(args.gamma_quantile),
            "rho_n_resamples": int(args.rho_n_resamples),
            "rho_alpha": float(args.rho_alpha),
            "rho_seed": int(args.rho_seed),
            "n_mc_rows": int(len(mc_rows)),
        },
        "data": data,
    }
    _write(Path(args.out), out)
    print(f"Wrote Theorem A constants to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
