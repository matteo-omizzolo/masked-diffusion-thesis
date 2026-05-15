#!/usr/bin/env python3
"""Prepare a patched informed-correctors/Text8 tree for smoke runs.

The upstream checkout is kept clean. Stage 1 copies `text8/` into a run
directory, then applies the minimal local patches needed for this thesis:

- redirect the upstream hardcoded DATA_DIR away from /root;
- make distrax-dependent ImageNet utilities optional so Text8 HollowMD4 can
  import on recent JAX stacks where distrax/TFP substrate APIs are broken.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _patch_input_pipeline(text8_root: Path, data_dir: Path) -> None:
    path = text8_root / "md4" / "input_pipeline.py"
    text = path.read_text()
    old = 'DATA_DIR = "/root/md4/data_dir"'
    new = f"DATA_DIR = {str(data_dir)!r}"
    if old not in text:
        raise SystemExit(f"expected DATA_DIR line not found in {path}")
    path.write_text(text.replace(old, new))


def _patch_distrax_import(text8_root: Path) -> None:
    path = text8_root / "md4" / "utils.py"
    text = path.read_text()

    old_import = "import distrax"
    new_import = (
        "try:\n"
        "    import distrax\n"
        "except Exception as _distrax_exc:  # informed-correctors Text8 patch\n"
        "    distrax = None\n"
    )
    if old_import not in text:
        raise SystemExit(f"expected 'import distrax' line not found in {path}")
    text = text.replace(old_import, new_import, 1)

    match = re.search(
        r"^class DiscretizedLogisticMixture\(distrax\.Distribution\):.*?(?=^class |^def |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise SystemExit(f"DiscretizedLogisticMixture class not found in {path}")

    class_body = match.group(0)
    indented = "    " + class_body.rstrip("\n").replace("\n", "\n    ") + "\n"
    guarded = (
        "if distrax is not None:  # informed-correctors Text8 patch\n"
        + indented
        + "else:\n"
        + "    DiscretizedLogisticMixture = None  # type: ignore[assignment]\n\n"
    )
    path.write_text(text[: match.start()] + guarded + text[match.end() :])


def prepare_copy(external_repo: Path, run_dir: Path, data_dir: Path) -> Path:
    source = external_repo / "text8"
    dest = run_dir / "text8"
    if not source.is_dir():
        raise SystemExit(f"missing informed-correctors text8 tree: {source}")

    if dest.exists():
        shutil.rmtree(dest)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)

    _patch_input_pipeline(dest, data_dir)
    _patch_distrax_import(dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    dest = prepare_copy(
        external_repo=args.external_repo.resolve(),
        run_dir=args.run_dir.resolve(),
        data_dir=args.data_dir.resolve(),
    )
    print(f"prepared patched Text8 tree: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
