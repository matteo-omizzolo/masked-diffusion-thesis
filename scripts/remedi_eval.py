#!/usr/bin/env python3
"""RemeDi-RL evaluation script.

Generates text samples from maple-research-lab/RemeDi-RL and computes
gen_ppl (via GPT-2 large), entropy, and MAUVE (vs OWT reference) — the
same metrics used for the ReMDM strategy comparison.

Usage:
    python scripts/remedi_eval.py \
        --steps 32 \
        --num_samples 100 \
        --out_dir results/remedi/T32 \
        --owt_ref /path/to/owt_reference_1000.json \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_remedi(ckpt_path: str, device: str):
    import transformers
    # RemeDiUPMModelLM must be importable — ensure external/remedi is on sys.path
    repo_root = Path(__file__).parent.parent
    remedi_path = repo_root / "external" / "remedi"
    if str(remedi_path) not in sys.path:
        sys.path.insert(0, str(remedi_path))

    from remedi import RemeDiUPMModelLM  # noqa: PLC0415

    print(f"Loading tokenizer from {ckpt_path} ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=True)

    print(f"Loading model from {ckpt_path} ...")
    model = RemeDiUPMModelLM.from_pretrained(
        ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval().requires_grad_(False).to(device)
    print("Model loaded.")
    return model, tokenizer


def generate_samples(
    model,
    tokenizer,
    device: str,
    num_samples: int,
    steps: int,
    block_size: int = 32,
    max_length: int = 512,
    seed: int = 42,
) -> list[str]:
    """Generate num_samples texts using RemeDi block diffusion."""
    repo_root = Path(__file__).parent.parent
    remedi_path = repo_root / "external" / "remedi"
    if str(remedi_path) not in sys.path:
        sys.path.insert(0, str(remedi_path))

    from inference import generate_block_diffusion  # noqa: PLC0415

    # Infer mask/eos token IDs from tokenizer (override hardcoded inference.py defaults)
    mask_token_id = getattr(tokenizer, "mask_token_id", None) or 126336
    eos_id = tokenizer.eos_token_id or 126081

    torch.manual_seed(seed)

    # Use an empty user prompt for unconditional-style generation
    conv = [{"role": "user", "content": ""}]

    samples: list[str] = []
    t0 = time.time()
    for i in range(num_samples):
        torch.manual_seed(seed + i)
        texts = generate_block_diffusion(
            model=model,
            conv=conv,
            tokenizer=tokenizer,
            device=device,
            num_generations=1,
            steps=steps,
            max_length=max_length,
            block_size=block_size,
            mask_token_id=mask_token_id,
            eos_id=eos_id,
        )
        samples.append(texts[0] if texts else "")
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{num_samples}] {rate:.2f} samples/s  "
                  f"eta {(num_samples - i - 1) / rate:.0f}s")

    return samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_gen_ppl(texts: list[str], device: str, batch_size: int = 8) -> float:
    """Compute generation perplexity using GPT-2 large as reference model."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print("Loading GPT-2 large for gen_ppl ...")
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2-large")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-large").eval().to(device)

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = gpt2_tok(batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = gpt2(**enc, labels=enc["input_ids"])
        # out.loss = mean NLL per token (over non-padding tokens)
        n_tokens = enc["attention_mask"].sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    ppl = math.exp(total_nll / total_tokens)
    del gpt2
    torch.cuda.empty_cache()
    return ppl


def compute_entropy(texts: list[str]) -> float:
    """Compute mean character-level entropy of generated texts."""
    from collections import Counter
    all_chars = "".join(texts)
    counts = Counter(all_chars)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy


def compute_mauve(generated: list[str], reference_path: str, device: str) -> float:
    """Compute MAUVE score against OWT reference."""
    import mauve as mauve_lib

    print(f"Loading OWT reference from {reference_path} ...")
    ref_data = json.loads(Path(reference_path).read_text())
    if isinstance(ref_data, list):
        reference = [r["text"] if isinstance(r, dict) else r for r in ref_data]
    else:
        reference = ref_data.get("texts", [])

    # Trim to same length
    n = min(len(generated), len(reference))
    gen_trim = generated[:n]
    ref_trim = reference[:n]

    print(f"Computing MAUVE (n={n}) ...")
    result = mauve_lib.compute_mauve(
        p_text=gen_trim,
        q_text=ref_trim,
        device_id=0 if device == "cuda" else -1,
        max_text_length=512,
        verbose=False,
        batch_size=16,
    )
    return float(result.mauve)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", default="maple-research-lab/RemeDi-RL")
    ap.add_argument("--steps", type=int, default=32,
                    help="Denoising steps per block (paper default=32)")
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--owt_ref", default="/home/3316152/mdm/data/owt_reference_1000.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print(f"RemeDi-RL eval  steps={args.steps}  block_size={args.block_size}")
    print(f"num_samples={args.num_samples}  seed={args.seed}  device={args.device}")
    print("=" * 50)

    # --- Generate ---
    model, tokenizer = load_remedi(args.ckpt_path, args.device)
    t_gen = time.time()
    samples = generate_samples(
        model, tokenizer,
        device=args.device,
        num_samples=args.num_samples,
        steps=args.steps,
        block_size=args.block_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    gen_time = time.time() - t_gen
    print(f"Generation done in {gen_time:.1f}s  ({len(samples)} samples)")

    # Save generated texts
    gen_path = out_dir / "generated_texts.json"
    gen_path.write_text(json.dumps({"texts": samples, "steps": args.steps,
                                    "num_samples": len(samples)}, indent=2))
    print(f"Saved generated texts → {gen_path}")

    # Free model memory before GPT-2 load
    del model
    torch.cuda.empty_cache()

    # --- Metrics ---
    print("\nComputing metrics ...")
    gen_ppl = compute_gen_ppl(samples, args.device)
    entropy = compute_entropy(samples)
    print(f"  gen_ppl  = {gen_ppl:.3f}")
    print(f"  entropy  = {entropy:.3f}")

    mauve_score = 0.0
    if Path(args.owt_ref).exists():
        mauve_score = compute_mauve(samples, args.owt_ref, args.device)
        print(f"  MAUVE    = {mauve_score:.3f}")
    else:
        print(f"  MAUVE    = skipped (ref not found: {args.owt_ref})")

    # --- Save summary ---
    summary = {
        "method": "remedi",
        "ckpt_path": args.ckpt_path,
        "steps": args.steps,
        "block_size": args.block_size,
        "num_samples": len(samples),
        "seed": args.seed,
        "gen_ppl": gen_ppl,
        "entropy": entropy,
        "MAUVE": mauve_score,
        "gen_time_s": gen_time,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary → {summary_path}")

    print("\n" + "=" * 50)
    print(f"RESULT  gen_ppl={gen_ppl:.3f}  entropy={entropy:.3f}  MAUVE={mauve_score:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
