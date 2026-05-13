#!/usr/bin/env python3
"""Check local feasibility for informed-correctors/Text8 training.

This is a non-training smoke utility. It inspects the local upstream checkout,
optional dependency imports, JAX device visibility, Text8 data layout, and the
HollowMD4 Text8 config. It never downloads data or checkpoints.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEPENDENCIES = [
    "absl",
    "clu",
    "datasets",
    "distrax",
    "flax",
    "grain",
    "jax",
    "jax.numpy",
    "ml_collections",
    "optax",
    "orbax.checkpoint",
    "tensorflow",
    "tensorflow_datasets",
    "wandb",
]


def _status(ok: bool, detail: str | None = None) -> dict[str, Any]:
    return {"ok": ok, "detail": detail or ""}


def _safe_exists(path: Path) -> bool:
    """Path.exists() that returns False on PermissionError.

    On HPC, the user typically cannot traverse the upstream-hardcoded
    /root/md4/data_dir tree; Path.exists() on a file inside it raises
    PermissionError. For feasibility reporting, "we can't see it" is
    equivalent to "not staged here" — both mean the user must stage data
    under TEXT8_DATA_DIR.
    """
    try:
        return path.exists()
    except (PermissionError, OSError):
        return False


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _import_module(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", None)
        return {"ok": True, "version": version}
    except Exception as exc:  # pragma: no cover - exercised in environment smoke
        return {"ok": False, "error": repr(exc)}


def _load_config(config_path: Path) -> dict[str, Any]:
    try:
        spec = importlib.util.spec_from_file_location("hollow_text8_config", config_path)
        if spec is None or spec.loader is None:
            return {"ok": False, "error": "could not create module spec"}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.get_config()
        keys = [
            "model_type",
            "dataset",
            "data_shape",
            "timesteps",
            "noise_schedule",
            "hidden_dim",
            "n_layers",
            "n_layers_per_mixed",
            "batch_size",
            "num_microbatches",
            "num_train_steps",
            "checkpoint_every_steps",
            "vocab_dir",
        ]
        values = {k: str(getattr(cfg, k)) for k in keys if hasattr(cfg, k)}
        missing = [k for k in keys if not hasattr(cfg, k)]
        return {"ok": True, "values": values, "missing": missing}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.external_repo.resolve()
    text8_root = repo / "text8"
    config_path = text8_root / "md4" / "configs" / "hollow_md4" / "text8.py"
    data_zip = args.data_dir.resolve() / "text8" / "text8.zip"
    upstream_hardcoded_zip = Path("/root/md4/data_dir/text8/text8.zip")

    report: dict[str, Any] = {
        "external_repo": str(repo),
        "repo_exists": _safe_exists(repo),
        "repo_commit": _git_commit(repo),
        "text8_root_exists": _safe_exists(text8_root),
        "hollow_config_path": str(config_path),
        "hollow_config_exists": _safe_exists(config_path),
        "data_dir": str(args.data_dir.resolve()),
        "expected_data_zip": str(data_zip),
        "expected_data_zip_exists": _safe_exists(data_zip),
        "upstream_hardcoded_data_zip": str(upstream_hardcoded_zip),
        "upstream_hardcoded_data_zip_exists": _safe_exists(upstream_hardcoded_zip),
        "notes": [
            "No downloads or training are performed by this script.",
            "Upstream input_pipeline.py hardcodes DATA_DIR=/root/md4/data_dir.",
            "Stage 1 should patch a copied upstream tree, not external_repos/.",
        ],
    }

    if args.check_imports:
        report["imports"] = {name: _import_module(name) for name in DEPENDENCIES}

    if args.check_jax:
        try:
            import jax

            report["jax"] = {
                "ok": True,
                "devices": [str(d) for d in jax.devices()],
                "local_device_count": jax.local_device_count(),
                "process_count": jax.process_count(),
            }
        except Exception as exc:  # pragma: no cover - environment smoke
            report["jax"] = {"ok": False, "error": repr(exc)}

    if args.check_data:
        data_ok = _safe_exists(data_zip) or _safe_exists(upstream_hardcoded_zip)
        report["data"] = {
            "ok": data_ok,
            "preferred_layout": str(data_zip),
            "upstream_layout": str(upstream_hardcoded_zip),
            "needs_manual_staging": not data_ok,
        }

    report["config"] = _load_config(config_path) if _safe_exists(config_path) else _status(False, "missing")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-repo",
        type=Path,
        default=Path("external_repos/informed-correctors"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("TEXT8_DATA_DIR", "data/text8_md4")),
    )
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--check-jax", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args(argv)

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(text + "\n")

    hard_fail = (
        not report["repo_exists"]
        or not report["hollow_config_exists"]
        or not report.get("config", {}).get("ok", False)
    )
    if args.check_data and report.get("data", {}).get("needs_manual_staging"):
        hard_fail = True
    imports = report.get("imports", {})
    if args.check_imports and any(not item["ok"] for item in imports.values()):
        hard_fail = True
    if args.check_jax and not report.get("jax", {}).get("ok", False):
        hard_fail = True
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
