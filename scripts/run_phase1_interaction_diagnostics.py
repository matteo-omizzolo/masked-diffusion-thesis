#!/usr/bin/env python3
"""Phase 1 — Sparse pairwise interaction diagnostics (Gate 3a).

Measures xi_{t,t'} = G({t,t'}) - Delta_t - Delta_{t'} for a pre-registered
sparse stratified set of corrector-placement pairs, using paired common
random numbers (CRN).

Delta_t values are loaded from Phase 1 Protocol A trajectories so that
singletons are NOT re-evaluated. Only G({t,t'}) is newly computed per pair.

This is Gate 3a (sparse pair diagnostics). Gate 3a does NOT validate
Theorem B; Gate 3b (schedule-level validation with no-leakage candidate
pool C_B) is required for that decision.

Outputs (under --out_dir):
    manifest.json
    pair_list.json                       # pre-registered pair list (saved once)
    per_seed/xi_rows_seed{seed}.json     # K per-seed files
    xi_raw.shard{i}-of-{n}.json         # per-shard concat
    run_config.shard{i}.json

Usage
-----
    # Local dry-run (no GPU required)
    python scripts/run_phase1_interaction_diagnostics.py --surrogate \\
        --K 4 --seed_start 42 \\
        --protocol_a_dir results/phase1_proseco_owt_full/protocol_a \\
        --out_dir /tmp/phase1_test_run

    # HPC real run (one call per GPU shard)
    python scripts/run_phase1_interaction_diagnostics.py \\
        --checkpoint $HOME/mdm/checkpoints/proseco_owt \\
        --protocol_a_dir $HOME/mdm/masked-diffusion-thesis/results/phase1_proseco_owt_full/protocol_a \\
        --out_dir results/phase1_interaction_diag_<sha> \\
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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from mdm_playground.scheduling import evaluate_schedule  # noqa: E402
from mdm_playground.scheduling.surrogate import SurrogateGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# Pre-registered pair list — deterministic, must not change after submission
# ---------------------------------------------------------------------------

_PHASE_BINS: Dict[str, List[int]] = {
    "early":  list(range(0, 21)),   # t in [0, 20]
    "middle": list(range(21, 43)),  # t in [21, 42]
    "late":   list(range(43, 64)),  # t in [43, 63]
}

# Hardcoded anchor pairs: systematic positions for interpretability.
# Include extremes, phase boundaries, near-adjacent, and diagonal.
_ANCHOR_PAIRS: List[Tuple[int, int]] = [
    (0, 63), (0, 32), (0, 21), (0, 10),
    (21, 42), (21, 63), (42, 63), (32, 52),
    (5, 10), (53, 58), (20, 21), (19, 22), (41, 44),
]

# Target number of pairs per (phase_t, phase_tp) stratum.
_STRATA_TARGETS: Dict[Tuple[str, str], int] = {
    ("early",  "early"):  8,
    ("early",  "middle"): 12,
    ("early",  "late"):   12,
    ("middle", "middle"): 8,
    ("middle", "late"):   12,
    ("late",   "late"):   8,
}

# Minimum pairs in short-distance (<=5) and long-distance (>=40) buckets.
_MIN_SHORT = 8
_MIN_LONG  = 8


def _phase_of(t: int) -> str:
    for name, ts in _PHASE_BINS.items():
        if t in ts:
            return name
    raise ValueError(f"Step {t} not covered by T=64 phase bins")


def make_pair_list(T: int = 64, rng_seed: int = 42) -> List[Dict[str, Any]]:
    """Build the pre-registered stratified pair list.

    This function is deterministic given (T, rng_seed). Its output is saved
    to pair_list.json at job start so the exact pairs are logged for
    reproducibility. Do NOT regenerate this list after data collection begins.

    Stratification layers (in order, duplicates removed):
      1. Anchor pairs — hardcoded systematic positions.
      2. Phase-stratified random pairs — covers 6 (phase, phase) combinations.
      3. Short-distance top-up — ensures >= _MIN_SHORT pairs with |t-t'| <= 5.
      4. Long-distance top-up — ensures >= _MIN_LONG pairs with |t-t'| >= 40.
    """
    assert T == 64, "Pair list is hardcoded for T=64; update _PHASE_BINS if T changes."
    rng = np.random.default_rng(rng_seed)
    seen: Set[Tuple[int, int]] = set()
    pairs: List[Dict[str, Any]] = []

    def _try_add(t: int, tp: int, source: str) -> bool:
        key = (min(t, tp), max(t, tp))
        if key[0] == key[1] or key in seen:
            return False
        seen.add(key)
        pairs.append({
            "t": key[0],
            "t_prime": key[1],
            "phase_t": _phase_of(key[0]),
            "phase_tp": _phase_of(key[1]),
            "distance": key[1] - key[0],
            "source": source,
        })
        return True

    # Layer 1: anchor pairs
    for (a, b) in _ANCHOR_PAIRS:
        _try_add(a, b, "anchor")

    # Layer 2: phase-stratified random pairs
    for (ph1, ph2), target in _STRATA_TARGETS.items():
        pool1 = _PHASE_BINS[ph1]
        pool2 = _PHASE_BINS[ph2]
        added = 0
        for _ in range(10_000):
            if added >= target:
                break
            t = int(rng.choice(pool1))
            tp = int(rng.choice(pool2))
            if _try_add(t, tp, f"stratum_{ph1}_{ph2}"):
                added += 1

    # Layer 3: short-distance top-up
    for _ in range(2_000):
        if sum(1 for p in pairs if p["distance"] <= 5) >= _MIN_SHORT:
            break
        t = int(rng.integers(0, T - 1))
        d = int(rng.integers(1, 6))
        if t + d < T:
            _try_add(t, t + d, "dist_short")

    # Layer 4: long-distance top-up
    for _ in range(2_000):
        if sum(1 for p in pairs if p["distance"] >= 40) >= _MIN_LONG:
            break
        t = int(rng.integers(0, T - 40))
        max_d = min(T - 1 - t, T - 1)
        if max_d < 40:
            continue
        d = int(rng.integers(40, max_d + 1))
        _try_add(t, t + d, "dist_long")

    return pairs


# ---------------------------------------------------------------------------
# Protocol A loader
# ---------------------------------------------------------------------------


def load_protocol_a(
    protocol_a_dir: Path, K: int, seed_start: int
) -> Dict[int, Dict[str, Any]]:
    """Load per-seed delta_t arrays and signals from Phase 1 Protocol A.

    Returns
    -------
    by_seed : dict {seed -> {"delta": np.ndarray[T], "signals": dict, "T": int}}
    """
    by_seed: Dict[int, Dict[str, Any]] = {}
    files = sorted(
        protocol_a_dir.glob("trajectory_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
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
        raise RuntimeError(f"No Protocol A trajectories found in {protocol_a_dir}")
    if len(by_seed) < K:
        print(
            f"WARNING: only {len(by_seed)}/{K} trajectories in {protocol_a_dir}",
            file=sys.stderr,
        )
    # Validate T and per-t length for all loaded seeds
    for s, data in by_seed.items():
        if data["T"] != K or len(data["delta"]) != data["T"]:
            # T here is the trajectory length, not K; use first seed's T as reference
            pass
    reference_T = next(iter(by_seed.values()))["T"]
    for s, data in by_seed.items():
        if data["T"] != reference_T:
            raise RuntimeError(
                f"Seed {s} has T={data['T']} but expected T={reference_T} "
                "(inconsistent Protocol A trajectories)"
            )
        if len(data["delta"]) != reference_T:
            raise RuntimeError(
                f"Seed {s} has {len(data['delta'])} delta values but T={reference_T}"
            )
    return by_seed


# ---------------------------------------------------------------------------
# Per-seed pair evaluation
# ---------------------------------------------------------------------------


def evaluate_seed_pairs(
    gen: Any,
    seed: int,
    delta: np.ndarray,
    pairs: List[Dict[str, Any]],
    T: int,
) -> List[Dict[str, Any]]:
    """Evaluate G({t,t'}) for all pairs at this seed.

    CRN is preserved: all evaluate_schedule calls use the same seed argument,
    so run_base returns the same y^0 for every pair.

    Delta_t values come from Protocol A (not re-evaluated here). The only
    new G-calls are for G({t,t'}) = F(y^{t,t'}) - F(y^0).
    """
    delta_trace = {int(t_): float(delta[t_]) for t_ in range(T)}
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        t = pair["t"]
        tp = pair["t_prime"]
        allocation = {t: 1, tp: 1}
        t0 = time.time()
        result = evaluate_schedule(allocation, delta_trace, gen, F="neg_nll", seed=seed)
        wall_time = time.time() - t0
        G_pair = float(result["G"])
        delta_t = float(delta[t])
        delta_tp = float(delta[tp])
        xi = G_pair - delta_t - delta_tp
        rows.append({
            "seed": int(seed),
            "t": t,
            "t_prime": tp,
            "phase_t": pair["phase_t"],
            "phase_tp": pair["phase_tp"],
            "distance": pair["distance"],
            "source": pair.get("source", "unknown"),
            "G_pair": G_pair,
            "delta_t": delta_t,
            "delta_tp": delta_tp,
            "A_pair": delta_t + delta_tp,
            "xi": xi,
            "wall_time": wall_time,
        })
    return rows


# ---------------------------------------------------------------------------
# Backend construction (mirrors Phase 2b / Phase 3a)
# ---------------------------------------------------------------------------


def build_generator(args: argparse.Namespace) -> Any:
    if args.surrogate:
        return SurrogateGenerator(
            T=args.T,
            sigma_gain=args.sigma_gain,
            gamma=args.gamma_surrogate,
            seed_base=args.seed_start,
        )
    if args.checkpoint is None:
        raise SystemExit("--checkpoint required unless --surrogate")
    from mdm_playground.scheduling.backends import detect_proseco_snapshot_backend

    backend = (
        args.backend
        if args.backend != "auto"
        else detect_proseco_snapshot_backend(args.checkpoint)
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
# Manifest helpers (verbatim from Phase 2b / Phase 3a)
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
        h = hashlib.sha256()
        for fname in sorted(p.rglob("*")):
            if fname.is_file() and fname.stat().st_size < 1_500_000_000:
                h.update(fname.name.encode())
                h.update(str(fname.stat().st_size).encode())
        return h.hexdigest()[:24]
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def write_manifest(out_dir: Path, args: argparse.Namespace, extra: Dict[str, Any]) -> None:
    """Write a per-shard manifest. The sbatch merge step produces the final manifest."""
    try:
        import torch  # type: ignore
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_version = getattr(torch.version, "cuda", None)
    except ImportError:
        torch_version = "not_imported"
        cuda_version = None
    shard_suffix = extra.get("shard_suffix", f"shard{args.shard_idx}")
    manifest = {
        "gate": "Gate3a_sparse_pair_diagnostics",
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "checkpoint_sha256_truncated": _checkpoint_sha256(args.checkpoint),
        "invocation_command": " ".join(sys.argv),
        "hpc_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        **extra,
    }
    # Write shard-specific manifest; merged manifest written by sbatch after all shards.
    (out_dir / f"manifest.{shard_suffix}.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 sparse pairwise interaction diagnostics (Gate 3a)"
    )
    p.add_argument("--surrogate", action="store_true", help="Surrogate generator (CPU dev)")
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--K", type=int, default=30, help="Number of paired seeds")
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--protocol_a_dir", type=str, required=True,
                   help="Phase 1 protocol_a/ directory with trajectory_*.json")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--backend", type=str, default="auto",
                   choices=("auto", "proseco_owt", "proseco_llada_sft"))
    p.add_argument("--corrector_steps", type=int, default=1)
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory (should include git SHA for reproducibility)")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true",
                   help="Skip seeds with existing per_seed/xi_rows_seed*.json")
    p.add_argument("--pair_rng_seed", type=int, default=42,
                   help="RNG seed for make_pair_list (must match across shards)")
    # Surrogate knobs
    p.add_argument("--sigma_gain", type=float, default=0.005)
    p.add_argument("--gamma_surrogate", type=float, default=0.008)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "per_seed").mkdir(parents=True, exist_ok=True)

    # Build pair list first — must be identical across all shards.
    pairs = make_pair_list(T=args.T, rng_seed=args.pair_rng_seed)
    n_pairs = len(pairs)

    # All shards verify the pair list is consistent (shard 0 writes it).
    pair_list_path = out_dir / "pair_list.json"
    if not pair_list_path.exists():
        pair_list_path.write_text(json.dumps(pairs, indent=2))
    else:
        saved = json.loads(pair_list_path.read_text())
        saved_keys = [(p["t"], p["t_prime"]) for p in saved]
        generated_keys = [(p["t"], p["t_prime"]) for p in pairs]
        if saved_keys != generated_keys:
            raise RuntimeError(
                f"Pair list mismatch for shard {args.shard_idx}: "
                f"saved {len(saved_keys)} pairs vs generated {len(generated_keys)} pairs. "
                "Do not mix runs with different --pair_rng_seed or --T values."
            )

    print("=" * 70)
    print("Phase 1 interaction diagnostics — sparse pairwise Gate 3a")
    print("=" * 70)
    backend_label = "SURROGATE" if args.surrogate else args.backend
    if not args.surrogate and args.backend == "auto" and args.checkpoint is not None:
        from mdm_playground.scheduling.backends import detect_proseco_snapshot_backend
        backend_label = detect_proseco_snapshot_backend(args.checkpoint)
    print(f"  Backend:    {backend_label}")
    print(f"  T, K:       {args.T}, {args.K}")
    print(f"  Pairs:      {n_pairs}")
    print(f"  shard:      {args.shard_idx}/{args.shard_count}")
    print(f"  out_dir:    {out_dir}")
    print(f"  pair_list:  {pair_list_path}")

    # Load Protocol A delta_t for all seeds
    by_seed = load_protocol_a(
        Path(args.protocol_a_dir), K=args.K, seed_start=args.seed_start
    )
    seeds_all = sorted(by_seed.keys())
    my_seeds = [s for i, s in enumerate(seeds_all) if i % args.shard_count == args.shard_idx]
    print(f"  my_seeds ({len(my_seeds)}/{len(seeds_all)}): "
          f"{my_seeds[:5]}{'...' if len(my_seeds) > 5 else ''}")
    print(f"  G-calls this shard: {len(my_seeds)} seeds × {n_pairs} pairs = "
          f"{len(my_seeds) * n_pairs}")

    # Run config for this shard
    run_config = {
        "gate": "Gate3a",
        "backend": backend_label.lower(),
        "T": args.T,
        "K_total": args.K,
        "K_shard": len(my_seeds),
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        "n_pairs": n_pairs,
        "pair_rng_seed": args.pair_rng_seed,
        "seed_start": args.seed_start,
        "checkpoint": args.checkpoint,
        "corrector_steps": args.corrector_steps,
        "protocol_a_dir": args.protocol_a_dir,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shard_out_suffix": f"shard{args.shard_idx}-of-{args.shard_count}",
    }
    (out_dir / f"run_config.shard{args.shard_idx}.json").write_text(
        json.dumps(run_config, indent=2)
    )

    gen = build_generator(args)

    all_rows: List[Dict[str, Any]] = []
    t_start = time.time()
    for i, seed in enumerate(my_seeds):
        seed_path = out_dir / "per_seed" / f"xi_rows_seed{seed}.json"
        if args.resume and seed_path.exists():
            print(f"[seed {seed}] SKIP (resume; file exists)")
            all_rows.extend(json.loads(seed_path.read_text()))
            continue

        t0 = time.time()
        print(f"[seed {seed}] ({i + 1}/{len(my_seeds)}) evaluating {n_pairs} pairs ...",
              flush=True)
        rows = evaluate_seed_pairs(
            gen=gen,
            seed=seed,
            delta=by_seed[seed]["delta"],
            pairs=pairs,
            T=args.T,
        )
        seed_path.write_text(json.dumps(rows, indent=2))
        all_rows.extend(rows)
        elapsed = time.time() - t0
        xi_vals = [r["xi"] for r in rows]
        print(
            f"[seed {seed}] done in {elapsed:.1f}s  "
            f"xi: mean={np.mean(xi_vals):.4f} std={np.std(xi_vals):.4f} "
            f"P(xi>0)={np.mean(np.array(xi_vals) > 0):.2f}"
        )

    total_elapsed = time.time() - t_start
    shard_suffix = f"shard{args.shard_idx}-of-{args.shard_count}"
    shard_out = out_dir / f"xi_raw.{shard_suffix}.json"
    shard_out.write_text(json.dumps(all_rows, indent=2))

    write_manifest(out_dir, args, extra={
        "walltime_seconds": total_elapsed,
        "shard_suffix": shard_suffix,
        "n_seeds_evaluated": len(my_seeds),
        "n_rows": len(all_rows),
        "n_pairs": n_pairs,
    })

    print("-" * 70)
    print(f"Shard {args.shard_idx}/{args.shard_count} done in {total_elapsed:.1f}s")
    print(f"  {len(all_rows)} rows  ({len(my_seeds)} seeds × {n_pairs} pairs)")
    print(f"  Wrote {shard_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
