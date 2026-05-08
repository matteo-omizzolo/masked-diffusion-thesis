#!/usr/bin/env python3
"""Stage kuleshov-group/proseco-llada-sft snapshot locally."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage ProSeCo-LLaDA-SFT checkpoint")
    p.add_argument(
        "--dest",
        default=str(Path.home() / "mdm" / "checkpoints" / "proseco_llada_sft"),
        help="Target directory for the snapshot",
    )
    p.add_argument(
        "--repo_id",
        default="kuleshov-group/proseco-llada-sft",
        help="HuggingFace repo id",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest)
    print(f"Downloading {args.repo_id} -> {dest}")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(dest),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
    )

    required = [
        "config.json",
        "configuration_llada.py",
        "modeling_llada.py",
        "model.safetensors.index.json",
    ]
    missing = [name for name in required if not (dest / name).exists()]
    if missing:
        raise SystemExit(
            f"Snapshot download incomplete at {dest}. Missing: {', '.join(missing)}"
        )
    shard_count = len(list(dest.glob("model-*.safetensors")))
    print(f"Snapshot ready: {dest} (shards={shard_count})")
    print(
        "Next: python scripts/debug_proseco_llada_sft_load.py "
        f"--checkpoint {dest} --device cpu --T 4"
    )


if __name__ == "__main__":
    main()

