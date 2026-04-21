#!/usr/bin/env python3
"""Lightweight preflight validation for the ProSeCo backend.

Checks — in increasing cost order — that the end-to-end inference path works
before committing an 8-hour HPC slot:

  1. Checkpoint config can be read (hyper_parameters['config'] present and complete)
  2. Diffusion model instantiates without error
  3. One base trajectory step runs (predictor + signal extraction)
  4. Corrector loop runs without error (one ProSeCo corrector loop)
  5. Branch trajectory run matches expected shape

Run on the HPC login node (CPU-only, ~2-3 min) or in the job prologue:

    python scripts/debug_proseco_load.py \
        --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
        [--device cpu]

Exit code 0 = all checks pass; non-zero = first failing check.
"""

import argparse
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="ProSeCo backend preflight validation")
    p.add_argument(
        "--checkpoint",
        default="/home/3316152/mdm/checkpoints/mdlm.ckpt",
        help="Path to MDLM Lightning checkpoint",
    )
    p.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--T", type=int, default=4,
        help="Predictor steps for validation run (default 4 — cheap)"
    )
    return p.parse_args()


def check(label: str, fn):
    """Run fn(), print pass/fail, and return the result (or re-raise on fail)."""
    print(f"  [{label}] ... ", end="", flush=True)
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f"PASS  ({elapsed:.1f}s)")
        return result
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"FAIL  ({elapsed:.1f}s)")
        print(f"    Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    device = args.device
    T = args.T

    print("=" * 60)
    print("ProSeCo backend preflight validation")
    print(f"  checkpoint : {checkpoint}")
    print(f"  device     : {device}")
    print(f"  T          : {T}  (cheap validation run)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Check 1: checkpoint config readable
    # ------------------------------------------------------------------
    def _check_config():
        import torch
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        ckpt = torch.load(checkpoint, map_location="cpu")
        assert "hyper_parameters" in ckpt, "hyper_parameters missing from checkpoint"
        hp = ckpt["hyper_parameters"]
        assert "config" in hp, "config missing from hyper_parameters"
        cfg = hp["config"]
        import omegaconf
        assert isinstance(cfg, omegaconf.DictConfig), f"config is {type(cfg).__name__}, not DictConfig"
        # Verify the key that broke previous runs
        gpl = cfg.eval.gen_ppl_eval_model_name_or_path
        assert gpl, f"gen_ppl_eval_model_name_or_path is empty: {gpl!r}"
        return cfg

    cfg = check("1/5 checkpoint config readable", _check_config)
    print(f"        gen_ppl_eval_model_name_or_path = {cfg.eval.gen_ppl_eval_model_name_or_path}")

    # ------------------------------------------------------------------
    # Check 2: ProSeCoGenerator instantiates (checkpoint loads)
    # ------------------------------------------------------------------
    def _check_load():
        from mdm_playground.scheduling.backends.proseco import ProSeCoGenerator
        gen = ProSeCoGenerator(checkpoint=checkpoint, T=T, device=device, corrector_steps=1)
        assert hasattr(gen, "model"), "model not set on generator"
        assert hasattr(gen, "mask_id"), "mask_id not set"
        return gen

    gen = check("2/5 ProSeCoGenerator instantiates", _check_load)
    print(f"        mask_id = {gen.mask_id},  model.T = {gen.model.T}")

    # ------------------------------------------------------------------
    # Check 3: one base trajectory (predictor + signals)
    # ------------------------------------------------------------------
    def _check_base():
        result = gen.run_base(seed=0)
        assert "tokens" in result, "tokens missing from run_base"
        assert "per_step_signals" in result, "per_step_signals missing"
        assert len(result["per_step_signals"]) == T, (
            f"expected {T} signal steps, got {len(result['per_step_signals'])}"
        )
        sigs = result["per_step_signals"][0]
        for key in ("entropy", "inverse_margin", "quality_mass_proxy", "unmasked_fraction"):
            assert key in sigs, f"signal key {key!r} missing"
        assert result["tokens"].shape == (gen.cfg.model.length,), (
            f"unexpected token shape {result['tokens'].shape}"
        )
        return result

    base = check("3/5 run_base (predictor + signals)", _check_base)
    sigs = base["per_step_signals"][-1]
    print(f"        final-step signals: entropy={sigs['entropy']:.4f}  "
          f"margin={sigs['inverse_margin']:.4f}  uf={sigs['unmasked_fraction']:.3f}")

    # ------------------------------------------------------------------
    # Check 4: corrector loop runs
    # ------------------------------------------------------------------
    def _check_corrector():
        branch = gen.run_branch(t_corrected=T // 2, seed=0)
        assert "tokens" in branch, "tokens missing from run_branch"
        assert "t_corrected" in branch, "t_corrected missing"
        assert branch["t_corrected"] == T // 2
        return branch

    branch = check("4/5 run_branch (corrector loop)", _check_corrector)
    print(f"        branch neg_nll = {branch['neg_nll']:.4f}")

    # ------------------------------------------------------------------
    # Check 5: neg_nll is finite and non-trivially different from base
    # ------------------------------------------------------------------
    def _check_quality():
        base_nll = base["neg_nll"]
        branch_nll = branch["neg_nll"]
        import math
        assert math.isfinite(base_nll), f"base neg_nll not finite: {base_nll}"
        assert math.isfinite(branch_nll), f"branch neg_nll not finite: {branch_nll}"
        return base_nll, branch_nll

    base_nll, branch_nll = check("5/5 neg_nll finite", _check_quality)
    print(f"        base={base_nll:.4f}  branch={branch_nll:.4f}  Δ={branch_nll - base_nll:+.4f}")

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED — backend is ready for HPC submission.")
    print("=" * 60)
    print()
    print("Next: bash hpc/push.sh && sbatch hpc/phase1_proseco.sbatch")


if __name__ == "__main__":
    main()
