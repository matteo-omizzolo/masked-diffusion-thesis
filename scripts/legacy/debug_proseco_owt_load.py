#!/usr/bin/env python3
"""CPU preflight for the ProSeCo-OWT backend.

Runs five checks (same structure as debug_mdlm_conf_load.py):
  1. Snapshot path exists and contains expected files
  2. ProSeCoOWTGenerator instantiates
  3. run_base produces finite neg_nll (T steps, N=1)
  4. run_branch produces finite neg_nll at t=T//2
  5. Δ_t at t=T//2 is finite (ideally non-zero)

Usage:
    conda activate remdm311
    python scripts/legacy/debug_proseco_owt_load.py \
        --checkpoint /home/3316152/mdm/checkpoints/proseco_owt \
        --device cpu --T 4
"""

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Local proseco-owt snapshot dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--corrector_steps", type=int, default=1)
    return p.parse_args()


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    args = parse_args()
    snap = Path(args.checkpoint)

    print("=" * 55)
    print("ProSeCo-OWT Preflight")
    print(f"  checkpoint: {snap}")
    print(f"  device:     {args.device}")
    print(f"  T:          {args.T}")
    print("=" * 55)

    all_pass = True

    # Check 1: snapshot files present
    expected = ["config.json", "modeling_proseco.py"]
    missing = [f for f in expected if not (snap / f).exists()]
    all_pass &= check(
        "Snapshot files present",
        len(missing) == 0,
        detail=f"missing: {missing}" if missing else "all present",
    )

    # Check 2: generator instantiates
    try:
        from mdm_playground.scheduling.backends.proseco_owt import ProSeCoOWTGenerator
        gen = ProSeCoOWTGenerator(
            checkpoint=str(snap),
            T=args.T,
            corrector_steps=args.corrector_steps,
            device=args.device,
        )
        all_pass &= check("ProSeCoOWTGenerator instantiates", True,
                          gen.corrector_description())
    except Exception as e:
        all_pass &= check("ProSeCoOWTGenerator instantiates", False, str(e))
        print("\nPreflight FAILED at instantiation — cannot continue.")
        sys.exit(1)

    # Check 3: run_base
    try:
        y_base = gen.run_base(seed=42)
        nll_finite = y_base["neg_nll"] not in (float("inf"), float("-inf"))
        import math
        nll_finite = nll_finite and not math.isnan(y_base["neg_nll"])
        all_pass &= check("run_base: neg_nll finite", nll_finite,
                          f"neg_nll={y_base['neg_nll']:.4f}")
    except Exception as e:
        all_pass &= check("run_base: neg_nll finite", False, str(e))
        sys.exit(1)

    # Check 4: run_branch
    t_mid = args.T // 2
    try:
        y_branch = gen.run_branch(t_corrected=t_mid, seed=42)
        nll_finite = (
            y_branch["neg_nll"] not in (float("inf"), float("-inf"))
            and not math.isnan(y_branch["neg_nll"])
        )
        all_pass &= check(f"run_branch(t={t_mid}): neg_nll finite", nll_finite,
                          f"neg_nll={y_branch['neg_nll']:.4f}")
    except Exception as e:
        all_pass &= check(f"run_branch(t={t_mid}): neg_nll finite", False, str(e))
        sys.exit(1)

    # Check 5: Δ_t non-trivially zero
    delta = y_branch["neg_nll"] - y_base["neg_nll"]
    non_trivial = delta != 0.0
    all_pass &= check(
        f"Δ_t(t={t_mid}) non-trivially zero",
        non_trivial,
        f"Δ_t={delta:.6f}  (zero → structural no-op; non-zero → corrector active)",
    )

    # Bonus: signal traces
    print("\n  Per-step signal trace (first 4 steps):")
    for rec in (y_base.get("per_step_signals") or [])[:4]:
        print(f"    t={rec['t']:2d}  H={rec['entropy']:.4f}  "
              f"margin={rec['inverse_margin']:.4f}  "
              f"n_rev={rec['n_revisable']}")

    print()
    if all_pass:
        print("All checks PASS. Ready to submit phase1_proseco_owt.sbatch")
    else:
        print("One or more checks FAILED. Fix before submitting HPC job.")
        sys.exit(1)


if __name__ == "__main__":
    main()
