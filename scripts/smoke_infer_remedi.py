#!/usr/bin/env python
"""smoke_infer_remedi.py — single real-model RemeDi inference run.

Requires the HuggingFace checkpoint (downloaded automatically on first run,
~2 GB).  No GPU required — runs on CPU with reduced steps.

Usage::

    # Baseline (no remasking)
    python scripts/smoke_infer_remedi.py --steps 4 --max_len 32

    # PRISM-style threshold remasking
    python scripts/smoke_infer_remedi.py \\
        --strategy threshold --tau 0.4 --steps 8

    # Run on GPU if available
    python scripts/smoke_infer_remedi.py --device cuda --steps 32 --max_len 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mdm_playground.models.remedi import RemeDiAdapter
from mdm_playground.samplers import run_block_diffusion
from mdm_playground.strategies import (
    BaselineUnmaskStrategy,
    ConfidenceThresholdRemaskStrategy,
    RemediPolicyStrategy,
    ScheduledRemaskStrategy,
    TopKLowConfidenceRemaskStrategy,
)

DEFAULT_MODEL = "maple-research-lab/RemeDi-RL"
DEFAULT_PROMPT = "Explain masked diffusion language models in two sentences."


def build_strategy(name: str, args: argparse.Namespace):
    if name == "baseline":
        return BaselineUnmaskStrategy()
    elif name in ("remedi", "remedi_policy"):
        return RemediPolicyStrategy()
    elif name == "threshold":
        return ConfidenceThresholdRemaskStrategy(tau=args.tau)
    elif name == "topk":
        return TopKLowConfidenceRemaskStrategy(k_remask=args.k)
    elif name == "schedule":
        return ScheduledRemaskStrategy(max_remask_prob=args.remask_prob, schedule=args.schedule)
    raise ValueError(f"Unknown strategy: {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--strategy", default="remedi_policy",
                   choices=["baseline", "remedi", "remedi_policy", "threshold", "topk", "schedule"])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_ups", action="store_true")
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--remask_prob", type=float, default=0.1)
    p.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    p.add_argument("--out", default=None, help="Save result JSON to this path.")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Loading RemeDi: {args.model}  device={device}")
    t0 = time.time()
    adapter = RemeDiAdapter.load(
        model_id=args.model,
        device=args.device,
        use_ups=not args.no_ups,
    )
    print(f"Loaded in {time.time() - t0:.1f}s")

    strategy = build_strategy(args.strategy, args)
    messages = [{"role": "user", "content": args.prompt}]

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Strategy: {args.strategy}  steps={args.steps}  max_len={args.max_len}\n")

    t1 = time.time()
    result = run_block_diffusion(
        adapter=adapter,
        messages=messages,
        strategy=strategy,
        steps=args.steps,
        max_length=args.max_len,
        block_size=args.block_size,
        seed=args.seed,
    )
    elapsed = time.time() - t1

    print(f"Generated ({elapsed:.1f}s):\n  {result['generated_text']!r}\n")

    # Per-step confidence summary
    for bi, block in enumerate(result["blocks"]):
        for si, step in enumerate(block["steps"]):
            avg_conf = sum(step["confidence"]) / len(step["confidence"])
            n_unmask = len(step["unmask_indices"])
            n_remask = len(step["remask_indices"])
            print(f"  block={bi} step={si}: avg_conf={avg_conf:.3f} "
                  f"unmask={n_unmask} remask={n_remask}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise tensor-safe
        out_path.write_text(json.dumps({
            "prompt": args.prompt,
            "generated": result["generated_text"],
            "strategy": args.strategy,
            "steps": args.steps,
            "elapsed_s": round(elapsed, 2),
        }, indent=2))
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
