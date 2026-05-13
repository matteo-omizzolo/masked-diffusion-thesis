#!/usr/bin/env python3
"""Phase 2b — Paired K-seed policy comparison + MC oracle on ProSeCo-OWT.

Consumes pre-computed Phase 1 Protocol A per-trajectory Δ_t + signals
(to avoid re-running the most expensive part of the pipeline), then for
each of K paired seeds runs:

  * one base trajectory (run_base)
  * one schedule per (policy, B) combination   — paired across seeds
  * P random-schedule MC samples per seed per B ∈ mc_B_values (default 2,3,4)
    — for the MC oracle headroom analysis.

Per-trajectory signal policies use THAT seed's signal trace; mean-profile
policies use the mean signal trace across the loaded trajectories. Both
variants are evaluated so the "mean-profile collapse" hypothesis is
falsifiable inside this run.

Outputs (under --out_dir, default results/phase2b_proseco_owt):

    manifest.json                                   # reproducibility
    per_seed/policy_rows_seed{seed}.json            # K files, raw per-policy
    per_seed/mc_rows_seed{seed}.json                # K files, raw MC
    policy_raw.json                                 # concatenated policy rows
    mc_raw.json                                     # concatenated MC rows
    run_config.json

Usage
-----
    # Local dry-run with surrogate
    python scripts/proseco/reproduction/run_phase2b_proseco_owt.py --surrogate \
        --K 6 --B_values 2,3,4,8,16 --mc_B_values 2,3,4 --mc_P 8 \
        --protocol_a_dir results/phase1_proseco_owt_full/protocol_a \
        --out_dir results/phase2b_dryrun

    # HPC real run (per-GPU shard)
    python scripts/proseco/reproduction/run_phase2b_proseco_owt.py \
        --checkpoint $HOME/mdm/checkpoints/proseco_owt \
        --T 64 --K 30 --B_values 2,3,4,8,16 --mc_B_values 2,3,4 --mc_P 100 \
        --seed_start 42 \
        --protocol_a_dir $HOME/mdm/masked-diffusion-thesis/results/phase1_proseco_owt_full/protocol_a \
        --out_dir results/phase2b_proseco_owt \
        --shard_idx 0 --shard_count 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from mdm_playground.scheduling import allocate_budget, evaluate_schedule  # noqa: E402
from mdm_playground.scheduling.surrogate import SurrogateGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------


# (policy_label, signal_kind, signal_name, allocation_policy, kwargs)
#   signal_kind ∈ {"mean_delta", "per_trajectory_signal", "mean_profile_signal"}
#   signal_name ∈ {"entropy", "inverse_margin", "quality_mass_proxy", None}
#   allocation_policy used by allocate_budget
POLICIES: List[Tuple[str, str, Optional[str], str, Dict[str, Any]]] = [
    ("uniform",          "none",                    None,                 "uniform",          {}),
    ("front",            "none",                    None,                 "front",            {}),
    ("back",             "none",                    None,                 "back",             {}),
    ("middle",           "none",                    None,                 "middle",           {}),
    # Per-trajectory signal policies — use THIS seed's signal trace.
    ("entropy_top_B_pt", "per_trajectory_signal",   "entropy",            "top_B",            {}),
    ("entropy_bot_B_pt", "per_trajectory_signal",   "entropy",            "bottom_B",         {}),
    ("margin_top_B_pt",  "per_trajectory_signal",   "inverse_margin",     "top_B",            {}),
    ("quality_top_B_pt", "per_trajectory_signal",   "quality_mass_proxy", "top_B",            {}),
    # Mean-profile ablation — should collapse to positional policy if signal is monotone.
    ("entropy_top_B_mp", "mean_profile_signal",     "entropy",            "top_B",            {}),
    # Mean-Δ oracle (phase-1 style "oracle") — single schedule across all seeds.
    ("mean_delta_oracle","mean_delta",              None,                 "top_B",            {}),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2b paired K-seed policy evaluation")
    p.add_argument("--surrogate", action="store_true", help="Use surrogate generator (CPU dev)")
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--K", type=int, default=30, help="Number of paired seeds")
    p.add_argument(
        "--B_values", type=str, default="2,3,4,8,16",
        help="Comma-separated B values for main policy grid",
    )
    p.add_argument(
        "--mc_B_values", type=str, default="2,3,4",
        help="Bs at which MC oracle samples are drawn (expensive)",
    )
    p.add_argument("--mc_P", type=int, default=100, help="MC samples per seed per B")
    p.add_argument("--seed_start", type=int, default=42,
                   help="First seed; seeds are seed_start + i for i=0..K-1 "
                        "(matches Phase 1 protocol_a file indexing)")
    p.add_argument("--protocol_a_dir", type=str, required=True,
                   help="Phase 1 protocol_a directory with trajectory_*.json")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="ProSeCo-OWT checkpoint (required unless --surrogate)")
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "proseco_owt", "proseco_llada_sft"),
        help="Backend type. 'auto' infers from checkpoint snapshot files.",
    )
    p.add_argument("--corrector_steps", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="results/phase2b_proseco_owt")
    p.add_argument("--shard_idx", type=int, default=0,
                   help="This shard's 0-based index (for GPU parallelism)")
    p.add_argument("--shard_count", type=int, default=1,
                   help="Total number of shards (seeds are distributed modulo shard_count)")
    p.add_argument("--resume", action="store_true",
                   help="Skip seeds with existing per_seed/*.json")
    p.add_argument("--sigma_gain", type=float, default=0.005)
    p.add_argument("--gamma_surrogate", type=float, default=0.008)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Protocol A loader
# ---------------------------------------------------------------------------


def load_protocol_a(protocol_a_dir: Path, K: int, seed_start: int) -> Dict[int, Dict[str, Any]]:
    """Load per-seed trajectory data from Phase 1 protocol_a.

    Returns
    -------
    by_seed : dict {seed -> {"delta": np.ndarray[T], "signals": dict[str -> np.ndarray[T]]}}
    """
    by_seed: Dict[int, Dict[str, Any]] = {}
    files = sorted(protocol_a_dir.glob("trajectory_*.json"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    for i, f in enumerate(files):
        if i >= K:
            break
        obj = json.loads(f.read_text())
        seed = int(obj.get("seed", seed_start + i))
        per_t = obj["per_t"]
        T = int(obj["T"])
        delta = np.asarray([row["delta"] for row in per_t], dtype=float)
        signals: Dict[str, np.ndarray] = {}
        for sig in ("entropy", "inverse_margin", "quality_mass_proxy", "unmasked_fraction"):
            signals[sig] = np.asarray([row.get(sig, 0.0) for row in per_t], dtype=float)
        by_seed[seed] = {"delta": delta, "signals": signals, "T": T}
    if not by_seed:
        raise RuntimeError(f"No protocol_a trajectories loaded from {protocol_a_dir}")
    if len(by_seed) < K:
        print(f"WARNING: only {len(by_seed)}/{K} trajectories available in {protocol_a_dir}",
              file=sys.stderr)
    return by_seed


def compute_mean_profile(by_seed: Dict[int, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Compute mean Δ and mean signal trace across all loaded seeds."""
    seeds = list(by_seed.keys())
    T = by_seed[seeds[0]]["T"]
    mean_delta = np.zeros(T, dtype=float)
    mean_signals: Dict[str, np.ndarray] = {}
    for seed in seeds:
        mean_delta += by_seed[seed]["delta"]
        for sig_name, sig in by_seed[seed]["signals"].items():
            mean_signals.setdefault(sig_name, np.zeros(T, dtype=float))
            mean_signals[sig_name] += sig
    n = float(len(seeds))
    mean_delta /= n
    for k in mean_signals:
        mean_signals[k] /= n
    return {"delta": mean_delta, "signals": mean_signals}


# ---------------------------------------------------------------------------
# Per-seed policy + MC evaluation
# ---------------------------------------------------------------------------


def _pick_trace(
    policy_label: str,
    signal_kind: str,
    signal_name: Optional[str],
    seed_data: Dict[str, Any],
    mean_profile: Dict[str, Any],
) -> np.ndarray:
    if signal_kind == "none":
        return mean_profile["delta"]  # irrelevant for positional policies
    if signal_kind == "mean_delta":
        return mean_profile["delta"]
    if signal_kind == "per_trajectory_signal":
        return seed_data["signals"][signal_name]  # type: ignore[index]
    if signal_kind == "mean_profile_signal":
        return mean_profile["signals"][signal_name]  # type: ignore[index]
    raise ValueError(f"Unknown signal_kind={signal_kind} for policy {policy_label}")


def evaluate_seed(
    gen: Any,
    seed: int,
    seed_data: Dict[str, Any],
    mean_profile: Dict[str, Any],
    B_values: Sequence[int],
    mc_B_values: Sequence[int],
    mc_P: int,
    T: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the full Phase 2b per-seed work.

    Returns
    -------
    policy_rows : list of dicts, one per (policy, B)
    mc_rows     : list of dicts, one per MC sample
    """
    delta_trace = {t: float(seed_data["delta"][t]) for t in range(T)}

    policy_rows: List[Dict[str, Any]] = []
    for B in B_values:
        for (policy_label, sig_kind, sig_name, alloc_name, kwargs) in POLICIES:
            trace = _pick_trace(policy_label, sig_kind, sig_name, seed_data, mean_profile)
            kwargs_local = dict(kwargs)
            if "seed" in kwargs_local:
                kwargs_local["seed"] = seed
            allocation = allocate_budget(trace, int(B), alloc_name, kwargs_local)
            result = evaluate_schedule(allocation, delta_trace, gen, F="neg_nll", seed=seed)
            policy_rows.append({
                "seed": int(seed),
                "B": int(B),
                "policy": policy_label,
                "signal_kind": sig_kind,
                "signal_name": sig_name,
                "allocation_policy": alloc_name,
                "G": float(result["G"]),
                "A": float(result["A"]),
                "residual": float(result["residual"]),
                "f_base": float(result["f_base"]),
                "f_schedule": float(result["f_schedule"]),
                "schedule_steps": list(result["schedule_steps"]),
                "budget": int(result["budget"]),
                "wall_time": float(result["wall_time"]),
            })

    mc_rows: List[Dict[str, Any]] = []
    if mc_P > 0:
        for B in mc_B_values:
            # Reproducible MC RNG per (seed, B)
            rng = np.random.default_rng(abs(hash((int(seed), int(B)))) % (2**32))
            for j in range(int(mc_P)):
                steps = sorted(int(s) for s in rng.choice(T, size=int(B), replace=False))
                allocation = {s: 1 for s in steps}
                # Use seed (not rng) for the pipeline call so base is paired across MC samples at this seed.
                result = evaluate_schedule(allocation, delta_trace, gen, F="neg_nll", seed=int(seed))
                mc_rows.append({
                    "seed": int(seed),
                    "B": int(B),
                    "mc_idx": int(j),
                    "G": float(result["G"]),
                    "A": float(result["A"]),
                    "residual": float(result["residual"]),
                    "schedule_steps": list(result["schedule_steps"]),
                })
    return policy_rows, mc_rows


# ---------------------------------------------------------------------------
# Backend loader
# ---------------------------------------------------------------------------


def build_generator(args: argparse.Namespace) -> Any:
    if args.surrogate:
        return SurrogateGenerator(
            T=args.T, sigma_gain=args.sigma_gain,
            gamma=args.gamma_surrogate, seed_base=args.seed_start,
        )
    if args.checkpoint is None:
        raise SystemExit("--checkpoint required unless --surrogate")
    from mdm_playground.scheduling.backends import detect_proseco_snapshot_backend

    backend = args.backend if args.backend != "auto" else detect_proseco_snapshot_backend(
        args.checkpoint
    )
    if backend == "proseco_owt":
        from mdm_playground.scheduling.backends.proseco_owt import ProSeCoOWTGenerator

        return ProSeCoOWTGenerator(
            checkpoint=args.checkpoint,
            T=args.T,
            corrector_steps=args.corrector_steps,
        )
    if backend == "proseco_llada_sft":
        from mdm_playground.scheduling.backends.proseco_llada_sft import (
            ProSeCoLLaDASFTGenerator,
        )

        return ProSeCoLLaDASFTGenerator(
            checkpoint=args.checkpoint,
            T=args.T,
            corrector_steps=args.corrector_steps,
        )
    raise SystemExit(f"Unsupported backend: {backend}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def _checkpoint_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.is_dir():
        # ProSeCo-OWT is a directory; hash a few known files for provenance.
        h = hashlib.sha256()
        for fname in sorted(p.rglob("*")):
            if fname.is_file() and fname.stat().st_size < 1_500_000_000:
                h.update(fname.name.encode())
                h.update(str(fname.stat().st_size).encode())
        return h.hexdigest()[:24]
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def write_manifest(out_dir: Path, args: argparse.Namespace, extra: Dict[str, Any]) -> None:
    import torch  # type: ignore
    manifest = {
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version.split()[0],
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(torch.version, "cuda", None) if hasattr(torch, "version") else None,
        "checkpoint_sha256_truncated": _checkpoint_sha256(args.checkpoint),
        "invocation_command": " ".join(sys.argv),
        "hpc_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        **extra,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    B_values = [int(x) for x in args.B_values.split(",") if x.strip()]
    mc_B_values = [int(x) for x in args.mc_B_values.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    (out_dir / "per_seed").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 2b — Paired K-seed policy comparison")
    print("=" * 70)
    backend_label = "SURROGATE" if args.surrogate else args.backend
    if not args.surrogate and args.backend == "auto" and args.checkpoint is not None:
        from mdm_playground.scheduling.backends import detect_proseco_snapshot_backend

        backend_label = detect_proseco_snapshot_backend(args.checkpoint)
    print(f"  Backend:      {backend_label}")
    print(f"  T, K:         {args.T}, {args.K}")
    print(f"  B_values:     {B_values}")
    print(f"  mc_B_values:  {mc_B_values} (P={args.mc_P} per seed per B)")
    print(f"  shard:        {args.shard_idx}/{args.shard_count}")
    print(f"  out_dir:      {out_dir}")

    # Load protocol A (cheap)
    by_seed = load_protocol_a(Path(args.protocol_a_dir), K=args.K, seed_start=args.seed_start)
    mean_profile = compute_mean_profile(by_seed)
    seeds_all = sorted(by_seed.keys())
    # Shard seeds by index modulo shard_count
    my_seeds = [s for i, s in enumerate(seeds_all) if i % args.shard_count == args.shard_idx]
    print(f"  my_seeds ({len(my_seeds)}/{len(seeds_all)}): {my_seeds[:5]}{'...' if len(my_seeds) > 5 else ''}")

    # Config manifest
    run_config = {
        "backend": backend_label.lower(),
        "T": args.T,
        "K_total": args.K,
        "K_shard": len(my_seeds),
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        "B_values": B_values,
        "mc_B_values": mc_B_values,
        "mc_P": args.mc_P,
        "seed_start": args.seed_start,
        "policies": [p[0] for p in POLICIES],
        "checkpoint": args.checkpoint,
        "corrector_steps": args.corrector_steps,
        "protocol_a_dir": args.protocol_a_dir,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shard_out_suffix": f"shard{args.shard_idx}-of-{args.shard_count}",
    }
    (out_dir / f"run_config.shard{args.shard_idx}.json").write_text(json.dumps(run_config, indent=2))

    # Build generator
    gen = build_generator(args)

    # Per-seed evaluation
    all_policy_rows: List[Dict[str, Any]] = []
    all_mc_rows: List[Dict[str, Any]] = []
    t_start = time.time()
    for i, seed in enumerate(my_seeds):
        seed_policy_path = out_dir / "per_seed" / f"policy_rows_seed{seed}.json"
        seed_mc_path = out_dir / "per_seed" / f"mc_rows_seed{seed}.json"
        if args.resume and seed_policy_path.exists() and seed_mc_path.exists():
            print(f"[seed {seed}] SKIP (resume; files exist)")
            all_policy_rows.extend(json.loads(seed_policy_path.read_text()))
            all_mc_rows.extend(json.loads(seed_mc_path.read_text()))
            continue

        t0 = time.time()
        print(f"[seed {seed}] ({i+1}/{len(my_seeds)}) evaluating ...", flush=True)
        policy_rows, mc_rows = evaluate_seed(
            gen=gen,
            seed=seed,
            seed_data=by_seed[seed],
            mean_profile=mean_profile,
            B_values=B_values,
            mc_B_values=mc_B_values,
            mc_P=args.mc_P,
            T=args.T,
        )
        seed_policy_path.write_text(json.dumps(policy_rows, indent=2))
        seed_mc_path.write_text(json.dumps(mc_rows, indent=2))
        all_policy_rows.extend(policy_rows)
        all_mc_rows.extend(mc_rows)
        elapsed = time.time() - t0
        print(f"[seed {seed}] done in {elapsed:.1f}s "
              f"({len(policy_rows)} policy + {len(mc_rows)} mc rows)")

    total_elapsed = time.time() - t_start
    # Shard-level concat (downstream aggregator merges across shards)
    shard_suffix = f"shard{args.shard_idx}-of-{args.shard_count}"
    (out_dir / f"policy_raw.{shard_suffix}.json").write_text(
        json.dumps(all_policy_rows, indent=2)
    )
    (out_dir / f"mc_raw.{shard_suffix}.json").write_text(
        json.dumps(all_mc_rows, indent=2)
    )

    write_manifest(out_dir, args, extra={
        "walltime_seconds": total_elapsed,
        "shard_suffix": shard_suffix,
        "n_seeds_evaluated": len(my_seeds),
        "n_policy_rows": len(all_policy_rows),
        "n_mc_rows": len(all_mc_rows),
    })

    print("-" * 70)
    print(f"Phase 2b shard {args.shard_idx}/{args.shard_count} done in {total_elapsed:.1f}s")
    print(f"  {len(all_policy_rows)} policy rows, {len(all_mc_rows)} MC rows")
    print(f"  Wrote policy_raw.{shard_suffix}.json, mc_raw.{shard_suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
