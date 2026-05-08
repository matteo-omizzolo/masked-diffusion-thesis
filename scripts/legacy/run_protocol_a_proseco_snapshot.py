#!/usr/bin/env python3
"""Generate Protocol-A trajectories for a staged ProSeCo-family snapshot.

Outputs `trajectory_{i}.json` rows compatible with:
  - scripts/run_phase2b_proseco_owt.py
  - scripts/run_phase3a_combinatorial.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from mdm_playground.scheduling import estimate_single_step_gain  # noqa: E402
from mdm_playground.scheduling.surrogate import SurrogateGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protocol-A generation for ProSeCo snapshots")
    p.add_argument("--surrogate", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "proseco_owt", "proseco_llada_sft"),
        help="Backend type. 'auto' infers from checkpoint snapshot files.",
    )
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--K", type=int, default=8, help="Number of paired seeds")
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--corrector_steps", type=int, default=1)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--sigma_gain", type=float, default=0.005)
    p.add_argument("--gamma_surrogate", type=float, default=0.008)
    return p.parse_args()


def build_generator(args: argparse.Namespace) -> Any:
    if args.surrogate:
        return SurrogateGenerator(
            T=args.T,
            sigma_gain=args.sigma_gain,
            gamma=args.gamma_surrogate,
            seed_base=args.seed_start,
        )
    if args.checkpoint is None:
        raise SystemExit("--checkpoint required unless --surrogate")

    from mdm_playground.scheduling.backends import detect_proseco_snapshot_backend

    backend = args.backend if args.backend != "auto" else detect_proseco_snapshot_backend(
        args.checkpoint
    )
    if backend == "proseco_owt":
        from mdm_playground.scheduling.backends.proseco_owt import ProSeCoOWTGenerator

        return ProSeCoOWTGenerator(
            checkpoint=args.checkpoint,
            T=args.T,
            corrector_steps=args.corrector_steps,
        )
    if backend == "proseco_llada_sft":
        from mdm_playground.scheduling.backends.proseco_llada_sft import (
            ProSeCoLLaDASFTGenerator,
        )

        return ProSeCoLLaDASFTGenerator(
            checkpoint=args.checkpoint,
            T=args.T,
            corrector_steps=args.corrector_steps,
        )
    raise SystemExit(f"Unsupported backend: {backend}")


def run_protocol_a(gen: Any, K: int, T: int, seed_start: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(K):
        seed = int(seed_start + i)
        t0 = time.time()
        print(f"[seed {seed}] ({i + 1}/{K}) base + {T} branch steps...", flush=True)

        y_base = gen.run_base(seed=seed)
        base_signals = y_base["per_step_signals"]
        if len(base_signals) < T:
            raise RuntimeError(
                f"run_base returned {len(base_signals)} signal rows, expected T={T}."
            )

        per_t: List[Dict[str, Any]] = []
        for t in range(T):
            y_branch = gen.run_branch(t_corrected=t, seed=seed)
            gain = estimate_single_step_gain(y_base, y_branch, F="neg_nll")
            sigs = base_signals[t]
            per_t.append(
                {
                    "t": int(t),
                    "delta": float(gain["delta"]),
                    "tcr": float(gain["tcr"]),
                    "f_base": float(gain["f_base"]),
                    "f_branch": float(gain["f_branch"]),
                    "n_changed": int(gain["n_changed"]),
                    "entropy": float(sigs.get("entropy", 0.0)),
                    "inverse_margin": float(sigs.get("inverse_margin", 0.0)),
                    "quality_mass_proxy": float(sigs.get("quality_mass_proxy", 0.0)),
                    "unmasked_fraction": float(sigs.get("unmasked_fraction", 0.0)),
                    "n_revisable": int(sigs.get("n_revisable", 0)),
                    "n_masked": int(sigs.get("n_masked", 0)),
                }
            )

        record = {"seed": seed, "T": int(T), "per_t": per_t}
        (out_dir / f"trajectory_{i}.json").write_text(json.dumps(record, indent=2))
        elapsed = time.time() - t0
        dvals = [r["delta"] for r in per_t]
        print(
            f"[seed {seed}] done in {elapsed:.1f}s; "
            f"delta_range=[{min(dvals):+.4f},{max(dvals):+.4f}]",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_label = "surrogate"
    if not args.surrogate:
        from mdm_playground.scheduling.backends import detect_proseco_snapshot_backend

        backend_label = (
            args.backend
            if args.backend != "auto"
            else detect_proseco_snapshot_backend(args.checkpoint)
        )

    print("=" * 70)
    print("Protocol A — ProSeCo snapshot")
    print("=" * 70)
    print(f"  backend:        {backend_label}")
    print(f"  T, K:           {args.T}, {args.K}")
    print(f"  seed_start:     {args.seed_start}")
    print(f"  out_dir:        {out_dir}")

    gen = build_generator(args)
    run_protocol_a(
        gen=gen,
        K=args.K,
        T=args.T,
        seed_start=args.seed_start,
        out_dir=out_dir,
    )

    run_cfg = {
        "backend": backend_label,
        "T": args.T,
        "K": args.K,
        "seed_start": args.seed_start,
        "corrector_steps": args.corrector_steps,
        "checkpoint": args.checkpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    print("-" * 70)
    print(f"Wrote Protocol-A trajectories to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

