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


# Required imports that must succeed cleanly. distrax is intentionally
# NOT in this list: distrax 0.1.8 (the last published version) is
# incompatible with the recent JAX shipped with `jax[cuda13]`, but distrax
# is only used by upstream `md4/utils.py::DiscretizedLogisticMixture`,
# which is an ImageNet-specific code path that Text8 HollowMD4 training
# never exercises. The Stage 1 sbatch patches the copied upstream tree to
# make `import distrax` defensive (try/except), so a missing or broken
# distrax does not block Text8 training. See the Stage 1 sbatch and
# CLAUDE.md issue #14 for the rationale.
DEPENDENCIES = [
    "absl",
    "clu",
    "datasets",
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
                "version": getattr(jax, "__version__", None),
            }
        except Exception as exc:  # pragma: no cover - environment smoke
            report["jax"] = {"ok": False, "error": repr(exc)}

    if args.check_compute:
        # End-to-end JIT-compile + allocate + execute + read-back probe.
        # jax.devices() returns a handle from the enumeration interface alone;
        # it does not exercise compiled-kernel allocation. On A100 MIG nodes
        # with a cuda-12 plugin against a cuda-13 driver this can pass at
        # enumeration time and silently crash later. The probe below catches
        # that interaction at Stage 0 time. Tensor stays tiny (64x64 fp32 =
        # 16 KB) and runs in well under one second on any working GPU.
        #
        # The probe REQUIRES a GPU device by default. CPU-only execution
        # would silently false-pass the Stage 0 gate even if CUDA failed,
        # so any non-gpu device is rejected unless --allow-cpu-probe is
        # passed (used only for local Mac CI / debugging).
        try:
            import jax
            import jax.numpy as jnp

            devices = jax.devices()
            device_kinds = sorted({d.platform for d in devices})
            x = jnp.ones((64, 64), dtype=jnp.float32)
            f = jax.jit(lambda y: (y @ y).sum())
            out_arr = f(x)
            out = float(out_arr.block_until_ready())
            expected = 64.0 * 64.0 * 64.0  # 64x64 ones @ 64x64 ones -> sum
            # In modern JAX `jax.Array.device` is a property; the legacy
            # callable was removed. Prefer `.devices()` (a set) since the
            # property behavior differs across the jax 0.4 / 0.5 / 0.10
            # bumps and across SDA vs MDA arrays.
            try:
                probe_devices = list(out_arr.devices())
            except AttributeError:
                probe_devices = [getattr(out_arr, "device", devices[0])]
            probe_device = str(probe_devices[0]) if probe_devices else "unknown"
            on_gpu = any(k in {"gpu", "cuda", "rocm"} for k in device_kinds)
            report["compute_probe"] = {
                "ok": True,
                "result": out,
                "expected": expected,
                "matches": abs(out - expected) <= 1e-3 * expected,
                "device_kinds": device_kinds,
                "probe_device": probe_device,
                "on_gpu": on_gpu,
            }
        except Exception as exc:  # pragma: no cover - environment smoke
            report["compute_probe"] = {"ok": False, "error": repr(exc)}

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
    parser.add_argument(
        "--check-compute",
        action="store_true",
        help="Run a tiny JIT-compile + matmul + readback probe on the JAX "
        "default device. Catches CUDA/driver/MIG mismatches that pass "
        "device enumeration but break compiled-kernel allocation. By "
        "default, requires a GPU device (CPU-only false-passes are "
        "rejected). Use --allow-cpu-probe to relax for local debugging.",
    )
    parser.add_argument(
        "--allow-cpu-probe",
        action="store_true",
        help="Allow the compute probe to pass on a CPU-only JAX backend. "
        "Off by default because Stage 0 on HPC must require a real GPU.",
    )
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args(argv)

    report = build_report(args)

    # Compute hard_fail BEFORE serializing so that annotated diagnostic
    # fields (e.g. compute_probe.forced_fail_reason) are persisted in
    # feasibility.json instead of only affecting the exit code.
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
    if args.check_compute:
        probe = report.get("compute_probe", {})
        if not probe.get("ok", False) or not probe.get("matches", False):
            hard_fail = True
        elif not args.allow_cpu_probe and not probe.get("on_gpu", False):
            # Probe technically ran and matched, but on a CPU backend. On
            # HPC this would be a false-pass: Stage 0 is supposed to gate
            # GPU readiness for Stage 1.
            probe["forced_fail_reason"] = (
                "compute probe ran on CPU; pass --allow-cpu-probe to permit"
            )
            hard_fail = True
    report["hard_fail"] = hard_fail

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(text + "\n")

    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
