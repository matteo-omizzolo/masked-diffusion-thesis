#!/usr/bin/env python3
"""Phase 3a — Combinatorial scheduling baselines on ProSeCo-OWT.

Tests whether *non-greedy* search procedures can recover the +0.45 MC-oracle
headroom over uniform observed in Phase 2b without ground-truth Δ_t access.

Two procedures, both paired against uniform on the same K=30 seeds:

  1. **Coordinate descent (CD-G)** — start from uniform schedule of size B,
     sample (in ∈ schedule, out ∈ ¬schedule) swaps, re-evaluate true G(S),
     accept if improved. Stop when no swap accepted over a window of N=16
     attempts (or after ``max_attempts``). One true-G call per swap attempt.

  2. **Beam search (BS-AG)** — maintain a beam of W=8 partial schedules of
     size k. Expand each by every legal next position; rank by cheap A(S)
     surrogate (free, no GPU); rollout-evaluate top-W expansions with true
     G; advance to k+1. Repeat until k=B. ~W·B true-G calls per (seed, B).

Per-seed sharding mirrors Phase 2b (4 GPUs × shard_idx 0..3). Reuses the
Phase 1 protocol_a Δ_t traces and the existing
``mdm_playground.scheduling.evaluate_schedule`` pipeline so the G-metric
is identical to Phase 2b for paired comparison.

Outputs (under --out_dir, default results/phase3a_proseco_owt):

    manifest.json
    per_seed/cd_rows_seed{seed}.json                # K files
    per_seed/bs_rows_seed{seed}.json                # K files
    cd_raw.shard{i}-of-{n}.json                     # per-shard concat
    bs_raw.shard{i}-of-{n}.json                     # per-shard concat
    run_config.shard{i}.json

Decision-relevant downstream aggregator: ``scripts/proseco/reproduction/analyze_phase3a.py``
emits ``oracle_gap_closure.json`` with Δ_CD/Δ_oracle and Δ_BS/Δ_oracle
ratios per B. Decision rules in ``PHASE3_ALTERNATIVE_PLAN.md``.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from mdm_playground.scheduling import allocate_budget, evaluate_schedule  # noqa: E402
from mdm_playground.scheduling.surrogate import SurrogateGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_schedule(T: int, B: int) -> List[int]:
    """Same uniform allocation as ``allocate_budget(... policy='uniform')``.

    Re-derived locally so CD/BS initialisation does not depend on a
    signal_trace argument.
    """
    if B <= 0:
        return []
    if B >= T:
        return list(range(T))
    alloc = allocate_budget([0.0] * T, B, "uniform", {})
    return sorted(int(t) for t in alloc.keys())


def _allocation_from_steps(steps: Iterable[int]) -> Dict[int, int]:
    return {int(t): 1 for t in steps}


def _cheap_A(steps: Iterable[int], delta_trace: Dict[int, float]) -> float:
    """Additive surrogate A(S) = ∑_{t ∈ S} Δ_t. No GPU."""
    return float(sum(delta_trace.get(int(t), 0.0) for t in steps))


def _eval_G(
    steps: Sequence[int],
    delta_trace: Dict[int, float],
    gen: Any,
    seed: int,
) -> Dict[str, Any]:
    """One true-G evaluation via the same pipeline Phase 2b uses."""
    return evaluate_schedule(
        allocation=_allocation_from_steps(steps),
        delta_trace=delta_trace,
        generator=gen,
        F="neg_nll",
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# Coordinate descent (true-G feedback)
# ---------------------------------------------------------------------------


def coordinate_descent_G(
    delta_trace: Dict[int, float],
    T: int,
    B: int,
    gen: Any,
    seed: int,
    max_attempts: int,
    window: int,
    init_steps: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Run CD-G at fixed (seed, B). Returns trajectory + final schedule + G.

    Acceptance is on **true G**. Sampling is a uniform random
    (in, out) swap among (schedule × ¬schedule). Stops when ``window``
    consecutive attempts fail to improve G, or after ``max_attempts``
    total attempts.
    """
    rng = np.random.default_rng(abs(hash(("phase3a_cd", int(seed), int(B)))) % (2**32))
    schedule = (
        sorted(int(t) for t in init_steps) if init_steps is not None
        else _uniform_schedule(T, B)
    )
    res = _eval_G(schedule, delta_trace, gen, seed=seed)
    G_best = float(res["G"])
    G_init = G_best
    schedule_init = list(schedule)

    n_attempts = 0
    n_accepted = 0
    consec_failures = 0
    history: List[Dict[str, Any]] = [{
        "step": 0, "G": G_best, "A": float(res["A"]), "schedule_steps": list(schedule),
        "accepted": True, "wall_time": float(res["wall_time"]),
    }]

    while n_attempts < max_attempts and consec_failures < window:
        out_options = [t for t in range(T) if t not in schedule]
        if not out_options or not schedule:
            break
        in_pos = int(rng.choice(schedule))
        out_pos = int(rng.choice(out_options))
        cand = sorted([t for t in schedule if t != in_pos] + [out_pos])
        res_c = _eval_G(cand, delta_trace, gen, seed=seed)
        G_cand = float(res_c["G"])
        n_attempts += 1
        accepted = G_cand > G_best
        if accepted:
            schedule = cand
            G_best = G_cand
            n_accepted += 1
            consec_failures = 0
        else:
            consec_failures += 1
        history.append({
            "step": n_attempts,
            "G": G_cand,
            "A": float(res_c["A"]),
            "schedule_steps": cand,
            "accepted": accepted,
            "wall_time": float(res_c["wall_time"]),
        })

    return {
        "G_init": G_init,
        "G_final": G_best,
        "schedule_init": schedule_init,
        "schedule_final": list(schedule),
        "n_attempts": n_attempts,
        "n_accepted": n_accepted,
        "stop_reason": "window" if consec_failures >= window else "max_attempts",
        "history": history,
        "n_g_calls": n_attempts + 1,  # init + per-attempt
    }


# ---------------------------------------------------------------------------
# Beam search (cheap-A ranking, true-G rollouts)
# ---------------------------------------------------------------------------


def beam_search_AG(
    delta_trace: Dict[int, float],
    T: int,
    B: int,
    gen: Any,
    seed: int,
    beam_width: int,
) -> Dict[str, Any]:
    """BS with cheap-A ranking + true-G rollouts.

    Maintains a beam of W partial schedules of size k. At each round, every
    schedule in the beam is expanded by every legal next position, ranked
    by cheap A(S∪{t}); the top-W expansions are rollout-evaluated with
    true G; the W with highest G become the next beam. Repeats until k=B.
    """
    if B <= 0:
        res = _eval_G([], delta_trace, gen, seed=seed)
        return {
            "G_final": float(res["G"]),
            "schedule_final": [],
            "n_g_calls": 1,
            "history": [],
        }

    # Round 1: pick top-W single steps by cheap-A.
    singles = sorted(range(T), key=lambda t: -_cheap_A([t], delta_trace))
    beam_steps = [[t] for t in singles[: beam_width]]
    rollouts = [_eval_G(s, delta_trace, gen, seed=seed) for s in beam_steps]
    beam = sorted(
        ({"steps": s, "G": float(r["G"]), "A": float(r["A"])} for s, r in zip(beam_steps, rollouts)),
        key=lambda d: -d["G"],
    )
    n_g_calls = len(rollouts)
    history: List[Dict[str, Any]] = [{
        "round": 1, "k": 1,
        "beam_top_G": [b["G"] for b in beam],
        "beam_top_steps": [b["steps"] for b in beam],
    }]

    for k in range(2, B + 1):
        # Generate all (schedule, new_pos) candidates ranked by cheap A.
        candidates: List[Tuple[float, List[int]]] = []
        seen: set = set()
        for entry in beam:
            steps = entry["steps"]
            for t in range(T):
                if t in steps:
                    continue
                cand = tuple(sorted(steps + [t]))
                if cand in seen:
                    continue
                seen.add(cand)
                a_score = _cheap_A(cand, delta_trace)
                candidates.append((a_score, list(cand)))
        candidates.sort(key=lambda x: -x[0])
        top_candidates = [c[1] for c in candidates[: beam_width]]
        rollouts = [_eval_G(s, delta_trace, gen, seed=seed) for s in top_candidates]
        beam = sorted(
            ({"steps": s, "G": float(r["G"]), "A": float(r["A"])} for s, r in zip(top_candidates, rollouts)),
            key=lambda d: -d["G"],
        )
        n_g_calls += len(rollouts)
        history.append({
            "round": k, "k": k,
            "beam_top_G": [b["G"] for b in beam],
            "beam_top_steps": [b["steps"] for b in beam],
        })

    best = beam[0]
    return {
        "G_final": best["G"],
        "schedule_final": list(best["steps"]),
        "A_final": best["A"],
        "n_g_calls": int(n_g_calls),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Per-seed orchestration
# ---------------------------------------------------------------------------


def evaluate_seed(
    gen: Any,
    seed: int,
    delta: np.ndarray,
    T: int,
    B_values: Sequence[int],
    cd_max_attempts: int,
    cd_window: int,
    bs_beam_width: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run CD-G + BS-AG for every B at this seed."""
    delta_trace = {int(t): float(delta[t]) for t in range(T)}
    cd_rows: List[Dict[str, Any]] = []
    bs_rows: List[Dict[str, Any]] = []
    for B in B_values:
        # ---------------- CD-G ----------------
        t0 = time.time()
        cd = coordinate_descent_G(
            delta_trace=delta_trace, T=T, B=int(B), gen=gen, seed=int(seed),
            max_attempts=cd_max_attempts, window=cd_window,
        )
        cd_rows.append({
            "seed": int(seed),
            "B": int(B),
            "method": "coordinate_descent_G",
            "G_init": cd["G_init"],
            "G_final": cd["G_final"],
            "schedule_init": cd["schedule_init"],
            "schedule_final": cd["schedule_final"],
            "n_attempts": cd["n_attempts"],
            "n_accepted": cd["n_accepted"],
            "stop_reason": cd["stop_reason"],
            "n_g_calls": cd["n_g_calls"],
            "wall_time": time.time() - t0,
        })

        # ---------------- BS-AG ----------------
        t0 = time.time()
        bs = beam_search_AG(
            delta_trace=delta_trace, T=T, B=int(B), gen=gen, seed=int(seed),
            beam_width=bs_beam_width,
        )
        bs_rows.append({
            "seed": int(seed),
            "B": int(B),
            "method": "beam_search_AG",
            "beam_width": int(bs_beam_width),
            "G_final": bs["G_final"],
            "schedule_final": bs["schedule_final"],
            "n_g_calls": bs["n_g_calls"],
            "wall_time": time.time() - t0,
        })
    return cd_rows, bs_rows


# ---------------------------------------------------------------------------
# Protocol A loader (verbatim copy of Phase 2b interface)
# ---------------------------------------------------------------------------


def load_protocol_a(protocol_a_dir: Path, K: int, seed_start: int) -> Dict[int, Dict[str, Any]]:
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
        by_seed[seed] = {"delta": delta, "T": T}
    if not by_seed:
        raise RuntimeError(f"No protocol_a trajectories loaded from {protocol_a_dir}")
    if len(by_seed) < K:
        print(
            f"WARNING: only {len(by_seed)}/{K} trajectories available in {protocol_a_dir}",
            file=sys.stderr,
        )
    return by_seed


# ---------------------------------------------------------------------------
# Backend
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
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3a — combinatorial scheduling baselines")
    p.add_argument("--surrogate", action="store_true")
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--B_values", type=str, default="2,3,4,8")
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--protocol_a_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "proseco_owt", "proseco_llada_sft"),
        help="Backend type. 'auto' infers from checkpoint snapshot files.",
    )
    p.add_argument("--corrector_steps", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="results/phase3a_proseco_owt")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    # CD knobs
    p.add_argument("--cd_max_attempts", type=int, default=64,
                   help="Hard cap on CD-G swap attempts per (seed, B)")
    p.add_argument("--cd_window", type=int, default=16,
                   help="Stop CD-G after this many consecutive non-improving swaps")
    # BS knobs
    p.add_argument("--bs_beam_width", type=int, default=8,
                   help="Beam width W for BS-AG")
    # Surrogate knobs (mirror Phase 2b)
    p.add_argument("--sigma_gain", type=float, default=0.005)
    p.add_argument("--gamma_surrogate", type=float, default=0.008)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    B_values = [int(x) for x in args.B_values.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    (out_dir / "per_seed").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 3a — Combinatorial scheduling baselines (CD-G + BS-AG)")
    print("=" * 70)
    backend_label = "SURROGATE" if args.surrogate else args.backend
    if not args.surrogate and args.backend == "auto" and args.checkpoint is not None:
        from mdm_playground.scheduling.backends import detect_proseco_snapshot_backend

        backend_label = detect_proseco_snapshot_backend(args.checkpoint)
    print(f"  Backend:      {backend_label}")
    print(f"  T, K:         {args.T}, {args.K}")
    print(f"  B_values:     {B_values}")
    print(f"  CD:           max_attempts={args.cd_max_attempts}, window={args.cd_window}")
    print(f"  BS:           beam_width={args.bs_beam_width}")
    print(f"  shard:        {args.shard_idx}/{args.shard_count}")
    print(f"  out_dir:      {out_dir}")

    by_seed = load_protocol_a(Path(args.protocol_a_dir), K=args.K, seed_start=args.seed_start)
    seeds_all = sorted(by_seed.keys())
    my_seeds = [s for i, s in enumerate(seeds_all) if i % args.shard_count == args.shard_idx]
    print(f"  my_seeds ({len(my_seeds)}/{len(seeds_all)}): {my_seeds[:5]}{'...' if len(my_seeds) > 5 else ''}")

    run_config = {
        "backend": backend_label.lower(),
        "T": args.T,
        "K_total": args.K,
        "K_shard": len(my_seeds),
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        "B_values": B_values,
        "cd_max_attempts": args.cd_max_attempts,
        "cd_window": args.cd_window,
        "bs_beam_width": args.bs_beam_width,
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

    all_cd_rows: List[Dict[str, Any]] = []
    all_bs_rows: List[Dict[str, Any]] = []
    t_start = time.time()
    for i, seed in enumerate(my_seeds):
        cd_path = out_dir / "per_seed" / f"cd_rows_seed{seed}.json"
        bs_path = out_dir / "per_seed" / f"bs_rows_seed{seed}.json"
        if args.resume and cd_path.exists() and bs_path.exists():
            print(f"[seed {seed}] SKIP (resume; files exist)")
            all_cd_rows.extend(json.loads(cd_path.read_text()))
            all_bs_rows.extend(json.loads(bs_path.read_text()))
            continue

        t0 = time.time()
        print(f"[seed {seed}] ({i+1}/{len(my_seeds)}) running CD-G + BS-AG ...", flush=True)
        cd_rows, bs_rows = evaluate_seed(
            gen=gen,
            seed=int(seed),
            delta=by_seed[seed]["delta"],
            T=args.T,
            B_values=B_values,
            cd_max_attempts=args.cd_max_attempts,
            cd_window=args.cd_window,
            bs_beam_width=args.bs_beam_width,
        )
        cd_path.write_text(json.dumps(cd_rows, indent=2))
        bs_path.write_text(json.dumps(bs_rows, indent=2))
        all_cd_rows.extend(cd_rows)
        all_bs_rows.extend(bs_rows)
        elapsed = time.time() - t0
        n_g_cd = sum(r["n_g_calls"] for r in cd_rows)
        n_g_bs = sum(r["n_g_calls"] for r in bs_rows)
        print(
            f"[seed {seed}] done in {elapsed:.1f}s "
            f"(CD G-calls={n_g_cd}, BS G-calls={n_g_bs})"
        )

    total_elapsed = time.time() - t_start
    shard_suffix = f"shard{args.shard_idx}-of-{args.shard_count}"
    (out_dir / f"cd_raw.{shard_suffix}.json").write_text(
        json.dumps(all_cd_rows, indent=2)
    )
    (out_dir / f"bs_raw.{shard_suffix}.json").write_text(
        json.dumps(all_bs_rows, indent=2)
    )

    write_manifest(out_dir, args, extra={
        "walltime_seconds": total_elapsed,
        "shard_suffix": shard_suffix,
        "n_seeds_evaluated": len(my_seeds),
        "n_cd_rows": len(all_cd_rows),
        "n_bs_rows": len(all_bs_rows),
    })

    print("-" * 70)
    print(f"Phase 3a shard {args.shard_idx}/{args.shard_count} done in {total_elapsed:.1f}s")
    print(f"  {len(all_cd_rows)} CD rows, {len(all_bs_rows)} BS rows")
    print(f"  Wrote cd_raw.{shard_suffix}.json, bs_raw.{shard_suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
