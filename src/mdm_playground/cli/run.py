"""Unified inference CLI for all methods.

Usage
-----
RemeDi (direct HF inference)::

    python -m mdm_playground.cli.run \\
        --method remedi \\
        --model_id maple-research-lab/RemeDi-RL \\
        --prompt "Explain masked diffusion" \\
        --strategy remedi_policy \\
        --steps 32 --max_len 256 --out_dir results/my_run

ReMDM (subprocess, dry-run on macOS)::

    python -m mdm_playground.cli.run \\
        --method remdm \\
        --model_id /path/to/remdm.ckpt \\
        --strategy remdm_conf \\
        --steps 256 --out_dir results/remdm_run \\
        --dry_run

PRISM (subprocess, toy mode for testing)::

    python -m mdm_playground.cli.run \\
        --method prism \\
        --strategy prism \\
        --toy_mode \\
        --out_dir results/prism_run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from ..core.logging import TrajectoryLogger
from ..core.utils import get_git_commit_hash, save_json, seed_everything
from ..models import PRISMAdapter, PRISMConfig, RemeDiAdapter, ReMDMAdapter, ReMDMConfig
from ..samplers import run_block_diffusion
from ..strategies import (
    BaselineUnmaskStrategy,
    ConfidenceThresholdRemaskStrategy,
    RemediPolicyStrategy,
    ScheduledRemaskStrategy,
    TopKLowConfidenceRemaskStrategy,
)


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def build_strategy(name: str, args: argparse.Namespace):
    strategies = {
        "baseline": lambda: BaselineUnmaskStrategy(),
        "remedi_policy": lambda: RemediPolicyStrategy(),
        "threshold": lambda: ConfidenceThresholdRemaskStrategy(tau=args.tau),
        "topk": lambda: TopKLowConfidenceRemaskStrategy(k_remask=args.k),
        "schedule": lambda: ScheduledRemaskStrategy(
            max_remask_prob=args.remask_prob, schedule=args.schedule
        ),
        # Aliases for clarity in experiment scripts
        "remedi": lambda: RemediPolicyStrategy(),
        "remdm_conf": lambda: BaselineUnmaskStrategy(),  # ReMDM uses its own loop
        "prism": lambda: ConfidenceThresholdRemaskStrategy(tau=args.tau),
    }
    if name not in strategies:
        raise ValueError(
            f"Unknown strategy '{name}'. Choices: {', '.join(strategies)}"
        )
    return strategies[name]()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mdm_playground.cli.run",
        description="Unified MDM inference playground.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Method
    p.add_argument(
        "--method", required=True, choices=["remedi", "remdm", "prism"],
        help="Backend method to use.",
    )
    p.add_argument(
        "--model_id", default="",
        help="HF model id (remedi) or checkpoint path (remdm/prism).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)

    # Prompt (remedi only; remdm/prism use their own data loaders)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--prompt", help="Single prompt string (remedi only).")
    g.add_argument("--prompts_file", help="File with one prompt per line (remedi only).")

    # Generation
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--no_ups", action="store_true",
                   help="Fallback to softmax-prob confidence (remedi only).")

    # Execution mode (remdm / prism)
    p.add_argument("--toy_mode", action="store_true",
                   help="Use tiny stub model instead of real checkpoint.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print command without executing (subprocess methods).")

    # Strategy
    p.add_argument(
        "--strategy", default="remedi_policy",
        choices=["baseline", "remedi_policy", "remedi", "threshold",
                 "topk", "schedule", "remdm_conf", "prism"],
    )
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--remask_prob", type=float, default=0.1)
    p.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])

    # Output
    p.add_argument("--out_dir", default="results/infer")
    p.add_argument("--save_arrays", action="store_true")

    # ReMDM-specific (only used when --method remdm)
    p.add_argument("--remdm_data", default="openwebtext-split")
    p.add_argument("--remdm_model_size", default="small")
    p.add_argument("--remdm_strategy", default="remdm-conf")
    p.add_argument("--remdm_num_batches", type=int, default=10)

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = make_parser().parse_args(argv)

    if args.seed is not None:
        seed_everything(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    save_json(out_dir / "run_meta.json", {
        "git_commit": get_git_commit_hash(),
        "method": args.method,
        "strategy": args.strategy,
        "model_id": args.model_id,
        "steps": args.steps,
        "seed": args.seed,
        "timestamp": ts,
    })

    # ------------------------------------------------------------------
    # Dispatch by method
    # ------------------------------------------------------------------
    if args.method == "remedi":
        _run_remedi(args, out_dir)
    elif args.method == "remdm":
        _run_remdm(args, out_dir)
    elif args.method == "prism":
        _run_prism(args, out_dir)


def _run_remedi(args: argparse.Namespace, out_dir: Path) -> None:
    """Load RemeDi and run interactive/batch inference."""
    if not args.prompt and not args.prompts_file:
        print("ERROR: --prompt or --prompts_file is required for --method remedi.", file=sys.stderr)
        sys.exit(1)

    prompts = (
        [args.prompt] if args.prompt
        else [l.strip() for l in open(args.prompts_file) if l.strip()]
    )

    model_id = args.model_id or "maple-research-lab/RemeDi-RL"
    print(f"Loading RemeDi: {model_id} on {args.device}")
    adapter = RemeDiAdapter.load(
        model_id=model_id,
        device=args.device,
        use_ups=not args.no_ups,
    )
    print("Model loaded.")

    strategy = build_strategy(args.strategy, args)

    for i, prompt in enumerate(prompts):
        run_id = f"prompt_{i:04d}"
        logger = TrajectoryLogger(out_dir=out_dir, run_id=run_id, method="remedi")
        messages = [{"role": "user", "content": prompt}]
        print(f"\n[{run_id}] {prompt!r}")

        result = run_block_diffusion(
            adapter=adapter,
            messages=messages,
            strategy=strategy,
            steps=args.steps,
            max_length=args.max_len,
            block_size=args.block_size,
            seed=args.seed,
        )
        print(f"[{run_id}] → {result['generated_text']!r}")
        logger.log_result(result, generated_text=result["generated_text"])
        if args.save_arrays:
            logger.save_arrays(result)
        print(f"[{run_id}] Saved to {logger.jsonl_path}")


def _run_remdm(args: argparse.Namespace, out_dir: Path) -> None:
    """Drive upstream ReMDM via subprocess."""
    cfg = ReMDMConfig(
        toy_mode=args.toy_mode,
        dry_run=args.dry_run,
        upstream_checkpoint_path=args.model_id or None,
        data=args.remdm_data,
        model_size=args.remdm_model_size,
        strategy=args.remdm_strategy,
        steps=args.steps,
        num_sample_batches=args.remdm_num_batches,
    )
    adapter = ReMDMAdapter(cfg=cfg, run_output_dir=out_dir)
    print(f"Running ReMDM (toy={cfg.toy_mode}, dry_run={cfg.dry_run})")
    result = adapter.sample()

    save_json(out_dir / "summary.json", {
        "method": "remdm",
        "strategy": cfg.strategy,
        "steps": cfg.steps,
        "dry_run": result.get("dry_run", False),
        "toy": cfg.toy_mode,
        "meta": result.get("meta", {}),
    })
    if result.get("dry_run"):
        print("DRY RUN — command:", " ".join(result.get("command", [])))
    else:
        print(f"Done. Outputs at: {result.get('external_run_dir', out_dir)}")


def _run_prism(args: argparse.Namespace, out_dir: Path) -> None:
    """Drive upstream PRISM via subprocess."""
    cfg = PRISMConfig(
        toy_mode=args.toy_mode,
        dry_run=args.dry_run,
        upstream_checkpoint_path=args.model_id or None,
        steps=args.steps,
    )
    adapter = PRISMAdapter(cfg=cfg, run_output_dir=out_dir)
    print(f"Running PRISM (toy={cfg.toy_mode}, dry_run={cfg.dry_run})")
    result = adapter.sample()

    save_json(out_dir / "summary.json", {
        "method": "prism",
        "strategy": cfg.strategy,
        "steps": cfg.steps,
        "dry_run": result.get("dry_run", False),
        "toy": cfg.toy_mode,
        "meta": result.get("meta", {}),
    })
    if result.get("dry_run"):
        print("DRY RUN — command:", " ".join(result.get("command", [])))
    else:
        print(f"Done. Outputs at: {result.get('external_run_dir', out_dir)}")


if __name__ == "__main__":
    main()
