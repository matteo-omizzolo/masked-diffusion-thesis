#!/usr/bin/env python3
"""Stage OpenWebText reference samples for MAUVE evaluation.

Streams OpenWebText from HuggingFace (no full download), tokenizes N samples
with the GPT-2 tokenizer truncated to 1024 tokens, decodes back to text, and
saves them to {cache_dir}/owt_reference_1000.json (~5 MB).

Run once on the HPC login node before submitting eval jobs:

    python scripts/stage_owt_reference.py \
        --n 1000 \
        --cache_dir /home/3316152/mdm/data
"""
import argparse
import json
import os

from transformers import AutoTokenizer
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000,
                   help="Number of reference samples to collect.")
    p.add_argument("--cache_dir", default="/home/3316152/mdm/data",
                   help="Directory to write owt_reference_1000.json.")
    p.add_argument("--skip", type=int, default=5000,
                   help="Number of OWT docs to skip before collecting (avoid overlap).")
    p.add_argument("--max_length", type=int, default=1024,
                   help="Token length to truncate each sample to.")
    p.add_argument("--tokenizer", default="gpt2")
    args = p.parse_args()

    print(f"Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Streaming OpenWebText (skip={args.skip}, n={args.n})...")
    ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

    texts = []
    skipped = 0
    for example in ds:
        if skipped < args.skip:
            skipped += 1
            continue
        # Tokenize and decode to normalise whitespace / truncate
        ids = tokenizer.encode(
            example["text"],
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        if len(ids) < 64:
            # Skip very short documents
            continue
        text = tokenizer.decode(ids, skip_special_tokens=True)
        texts.append(text)
        if len(texts) % 100 == 0:
            print(f"  collected {len(texts)}/{args.n}")
        if len(texts) >= args.n:
            break

    out_path = os.path.join(args.cache_dir, "owt_reference_1000.json")
    os.makedirs(args.cache_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"texts": texts, "n": len(texts), "skip": args.skip,
                   "max_length": args.max_length}, f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSaved N={len(texts)} samples, total size: {size_kb:.1f} KB")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
