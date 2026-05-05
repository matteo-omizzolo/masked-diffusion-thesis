#!/usr/bin/env python3
"""CPU/GPU preflight for the ProSeCo-LLaDA-SFT backend."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Local proseco-llada-sft snapshot dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--corrector_steps", type=int, default=1)
    return p.parse_args()


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> None:
    args = parse_args()
    snap = Path(args.checkpoint)

    print("=" * 55)
    print("ProSeCo-LLaDA-SFT Preflight")
    print(f"  checkpoint: {snap}")
    print(f"  device:     {args.device}")
    print(f"  T:          {args.T}")
    print("=" * 55)

    expected = [
        "config.json",
        "configuration_llada.py",
        "modeling_llada.py",
        "model.safetensors.index.json",
    ]
    missing = [f for f in expected if not (snap / f).exists()]
    all_pass = check(
        "Snapshot files present",
        not missing,
        detail=f"missing: {missing}" if missing else "all present",
    )
    if not all_pass:
        sys.exit(1)

    try:
        from mdm_playground.scheduling.backends.proseco_llada_sft import (
            ProSeCoLLaDASFTGenerator,
        )

        gen = ProSeCoLLaDASFTGenerator(
            checkpoint=str(snap),
            T=args.T,
            corrector_steps=args.corrector_steps,
            device=args.device,
        )
        check("Generator instantiates", True, gen.corrector_description())
    except Exception as e:
        check("Generator instantiates", False, str(e))
        sys.exit(1)

    try:
        y_base = gen.run_base(seed=42)
        nll = float(y_base["neg_nll"])
        check("run_base neg_nll finite", math.isfinite(nll), f"neg_nll={nll:.4f}")
    except Exception as e:
        check("run_base neg_nll finite", False, str(e))
        sys.exit(1)

    t_mid = args.T // 2
    try:
        y_branch = gen.run_branch(t_corrected=t_mid, seed=42)
        nll_b = float(y_branch["neg_nll"])
        check(
            f"run_branch(t={t_mid}) neg_nll finite",
            math.isfinite(nll_b),
            f"neg_nll={nll_b:.4f}",
        )
    except Exception as e:
        check(f"run_branch(t={t_mid}) neg_nll finite", False, str(e))
        sys.exit(1)

    delta = float(y_branch["neg_nll"] - y_base["neg_nll"])
    check(f"Delta_t(t={t_mid}) finite", math.isfinite(delta), f"Delta_t={delta:+.6f}")

    print("\nPreflight complete.")


if __name__ == "__main__":
    main()

