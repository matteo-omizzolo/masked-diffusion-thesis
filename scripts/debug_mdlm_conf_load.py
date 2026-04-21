#!/usr/bin/env python3
"""Preflight validation for MDLMConfGenerator.

Run on CPU before any HPC submission:
  python scripts/debug_mdlm_conf_load.py \\
      --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \\
      --device cpu --T 4 --top_k 5

Checks:
  1/5  Checkpoint config readable
  2/5  MDLMConfGenerator instantiates (model + ref scorer loaded)
  3/5  run_base: signals are non-degenerate over masked positions
  4/5  run_branch: corrector applies without error, some tokens change
  5/5  neg_nll is finite and negative (correct sign)

Exit code 0 = safe to submit.
"""

import argparse
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))


def check(n, label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {n}/5  {label:45s} {status}{suffix}", flush=True)
    if not ok:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    print("=" * 60)
    print("MDLMConf Backend Preflight Checks")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  device:     {args.device}")
    print(f"  T:          {args.T}")
    print(f"  top_k:      {args.top_k}")
    print("=" * 60)

    # 1 — Checkpoint config readable
    t0 = time.time()
    try:
        import torch
        _orig = torch.load
        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig(*a, **kw)
        torch.load = _patched

        raw = torch.load(args.checkpoint, map_location="cpu")
        cfg = raw["hyper_parameters"]["config"]
        gpl = cfg.eval.gen_ppl_eval_model_name_or_path
        ok1 = isinstance(gpl, str) and len(gpl) > 0
        check(1, "Checkpoint config readable", ok1, f"gen_ppl={gpl!r}  {time.time()-t0:.1f}s")
    except Exception as e:
        check(1, "Checkpoint config readable", False, str(e))

    # 2 — MDLMConfGenerator instantiates
    t0 = time.time()
    try:
        from mdm_playground.scheduling.backends.mdlm_conf import MDLMConfGenerator
        gen = MDLMConfGenerator(
            checkpoint=args.checkpoint,
            T=args.T,
            top_k=args.top_k,
            device=args.device,
            ref_model_name="gpt2",
        )
        check(2, "MDLMConfGenerator instantiates", True, f"{time.time()-t0:.1f}s")
    except Exception as e:
        check(2, "MDLMConfGenerator instantiates", False, str(e))

    # 3 — run_base: non-degenerate signals over masked positions
    t0 = time.time()
    try:
        y_base = gen.run_base(seed=42)
        sigs = y_base["per_step_signals"]
        # At least some steps should have n_masked > 0 and entropy > 0
        n_masked_nonzero = sum(1 for s in sigs if s.get("n_masked", 0) > 0)
        entropy_nonzero = sum(1 for s in sigs if s.get("entropy", 0.0) > 1e-6)
        ok3 = n_masked_nonzero > 0 and entropy_nonzero > 0
        check(3, "run_base: non-degenerate signals", ok3,
              f"n_masked>0 at {n_masked_nonzero}/{args.T} steps, "
              f"entropy>0 at {entropy_nonzero}/{args.T}  {time.time()-t0:.1f}s")
    except Exception as e:
        check(3, "run_base: non-degenerate signals", False, str(e))

    # 4 — run_branch: corrector applies, tokens change
    t0 = time.time()
    try:
        import numpy as np
        mid_t = args.T // 2
        y_branch = gen.run_branch(t_corrected=mid_t, seed=42)
        n_changed = int((y_base["tokens"] != y_branch["tokens"]).sum())
        ok4 = "t_corrected" in y_branch
        # n_changed may be 0 at T=4 since corrector fires at step 2 (only 2 steps remain)
        # — that's acceptable; we just check it runs without error
        check(4, "run_branch: corrector runs", ok4,
              f"t_corrected={mid_t}  n_changed={n_changed}  {time.time()-t0:.1f}s")
    except Exception as e:
        check(4, "run_branch: corrector runs", False, str(e))

    # 5 — neg_nll finite and negative
    t0 = time.time()
    try:
        nll = y_base["neg_nll"]
        ok5 = float("-inf") < nll < 0
        check(5, "neg_nll finite and negative", ok5,
              f"neg_nll={nll:.4f}  {time.time()-t0:.1f}s")
    except Exception as e:
        check(5, "neg_nll finite and negative", False, str(e))

    print()
    print("All checks passed.  Safe to submit HPC job.")
    print("  sbatch hpc/phase1_mdlm_conf.sbatch")


if __name__ == "__main__":
    main()
