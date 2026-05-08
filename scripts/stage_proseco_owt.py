#!/usr/bin/env python3
"""Stage kuleshov-group/proseco-owt checkpoint on HPC.

Downloads the HuggingFace snapshot for proseco-owt to a local directory,
then patches the downloaded modeling_proseco.py to remove the flash_attn
hard dependency (replaces with PyTorch SDPA fallback — same approach as
external/remdm/models/dit.py).

Run on HPC login node (requires internet):
    module load miniconda3
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate remdm311
    python scripts/stage_proseco_owt.py --dest ~/mdm/checkpoints/proseco_owt

Expected outputs:
    ~/mdm/checkpoints/proseco_owt/config.json
    ~/mdm/checkpoints/proseco_owt/pytorch_model.bin  (or .safetensors)
    ~/mdm/checkpoints/proseco_owt/modeling_proseco.py  (patched)
    ~/mdm/checkpoints/proseco_owt/configuration_proseco.py

Wall-clock: ~5-10 min depending on HPC outbound bandwidth.
"""

import argparse
import re
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Stage proseco-owt checkpoint")
    p.add_argument(
        "--dest",
        default=str(Path.home() / "mdm" / "checkpoints" / "proseco_owt"),
        help="Target directory for the snapshot",
    )
    p.add_argument(
        "--repo_id",
        default="kuleshov-group/proseco-owt",
        help="HuggingFace repo id",
    )
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download; only apply patch to existing snapshot",
    )
    return p.parse_args()


FLASH_ATTN_PATCH = '''\
try:
    import flash_attn
    import flash_attn.layers.rotary
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False
'''

ROTARY_FALLBACK = '''\
def apply_rotary_pos_emb(qkv, cos, sin):
    # cos/sin from Rotary.forward: shape (1, seq_len, 3, 1, head_dim)
    # qkv: (batch, seq_len, 3, n_heads, head_dim)
    if _HAS_FLASH_ATTN:
        cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
        sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
        return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
    # Pure PyTorch fallback
    d = qkv.shape[-1]
    cos_s = cos[0, :, 0, 0, :d // 2]
    sin_s = sin[0, :, 0, 0, :d // 2]
    cos_s = cos_s.unsqueeze(0).unsqueeze(2)
    sin_s = sin_s.unsqueeze(0).unsqueeze(2)
    for qi in (0, 1):
        x = qkv[:, :, qi, :, :]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        qkv[:, :, qi, :, :] = __import__('torch').cat(
            [x1 * cos_s - x2 * sin_s, x2 * cos_s + x1 * sin_s], dim=-1
        )
    return qkv
'''


def patch_modeling_file(path: Path) -> None:
    """Patch modeling_proseco.py to make flash_attn optional."""
    src = path.read_text()

    # Replace hard flash_attn imports
    src = re.sub(
        r"import flash_attn\nimport flash_attn\.layers\.rotary\n",
        FLASH_ATTN_PATCH,
        src,
        count=1,
    )

    # Replace apply_rotary_pos_emb function
    src = re.sub(
        r"def apply_rotary_pos_emb\(qkv, cos, sin\):.*?return flash_attn\.layers\.rotary\.apply_rotary_emb_qkv_\(qkv, cos, sin\)\n",
        ROTARY_FALLBACK,
        src,
        count=1,
        flags=re.DOTALL,
    )

    path.write_text(src)
    print(f"  [patch] Patched {path}")


def main():
    args = parse_args()
    dest = Path(args.dest)

    if not args.skip_download:
        print(f"Downloading {args.repo_id} → {dest}")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(dest),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        print(f"Download complete: {dest}")
    else:
        print(f"Skipping download, using existing: {dest}")

    # Patch the downloaded modeling file
    modeling_path = dest / "modeling_proseco.py"
    if modeling_path.exists():
        patch_modeling_file(modeling_path)
    else:
        print(f"  WARNING: {modeling_path} not found — patch skipped")

    # Verify the patched file loads
    print("\nVerifying patched model loads ...")
    sys.path.insert(0, str(dest))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("modeling_proseco", modeling_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("  [OK] modeling_proseco.py imports successfully (no flash_attn required)")
    except Exception as e:
        print(f"  [WARN] Import check failed: {e}")

    print(f"\nStaging complete. Checkpoint at: {dest}")
    print("Next: python scripts/debug_proseco_owt_load.py --checkpoint", dest, "--device cpu --T 4")


if __name__ == "__main__":
    main()
