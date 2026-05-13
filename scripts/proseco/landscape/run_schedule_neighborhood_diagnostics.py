#!/usr/bin/env python3
"""Phase 4 schedule-neighborhood diagnostics for corrector timing.

This is a diagnostic runner, not a scheduler. It evaluates one-swap
neighborhoods around existing search/MC/control schedules and exact
diminishing-return triples so the next method choice can be based on measured
local/set-function structure.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from mdm_playground.scheduling import allocate_budget, evaluate_schedule  # noqa: E402
from mdm_playground.scheduling.surrogate import SurrogateGenerator  # noqa: E402


PHASES = ("early", "middle", "late")
ANCHOR_TYPES = ("cd_g", "bs_ag", "top_mc", "random_mc", "uniform", "middle")
REQUIRED_ROW_FIELDS = (
    "seed",
    "B",
    "diagnostic_type",
    "anchor_type",
    "schedule_steps",
    "anchor_schedule_steps",
    "G",
    "A",
    "F_base",
    "F_schedule",
    "n_g_calls",
    "phase_counts",
    "git_sha",
    "shard_id",
)


def canonical_schedule(steps: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(int(step) for step in steps)))


def schedule_distance(left: Sequence[int], right: Sequence[int]) -> int:
    left_set = set(canonical_schedule(left))
    right_set = set(canonical_schedule(right))
    diff = len(left_set.symmetric_difference(right_set))
    if len(left_set) == len(right_set):
        return diff // 2
    return diff


def phase_for_step(step: int, *, T: int = 64) -> str:
    step = int(step)
    if step < 0 or step >= T:
        raise ValueError(f"step must be in [0, {T - 1}], got {step}")
    early_end = T // 3
    middle_end = (2 * T) // 3
    if step < early_end:
        return "early"
    if step < middle_end:
        return "middle"
    return "late"


def phase_counts(steps: Sequence[int], *, T: int = 64) -> dict[str, int]:
    counts = {phase: 0 for phase in PHASES}
    for step in canonical_schedule(steps):
        counts[phase_for_step(step, T=T)] += 1
    return counts


def one_swap_neighbors(
    schedule_steps: Sequence[int],
    *,
    T: int,
    max_neighbors: int | None = None,
    rng_seed: int = 0,
) -> list[list[int]]:
    """Return deterministic one-swap neighbors of a fixed-budget schedule."""
    anchor = canonical_schedule(schedule_steps)
    if not anchor:
        return []
    anchor_set = set(anchor)
    if len(anchor_set) >= T:
        return []
    neighbors: list[tuple[int, ...]] = []
    for remove_step in anchor:
        for add_step in range(T):
            if add_step in anchor_set:
                continue
            candidate = canonical_schedule((anchor_set - {remove_step}) | {add_step})
            if candidate != anchor:
                neighbors.append(candidate)
    unique = sorted(set(neighbors))
    if max_neighbors is None or len(unique) <= int(max_neighbors):
        return [list(candidate) for candidate in unique]
    rng = np.random.default_rng(int(rng_seed))
    indices = sorted(rng.choice(len(unique), size=int(max_neighbors), replace=False).tolist())
    return [list(unique[idx]) for idx in indices]


def diminishing_return_gap(
    *,
    g_A: float,
    g_B: float,
    g_A_plus_x: float,
    g_B_plus_x: float,
) -> float:
    return float((g_A_plus_x - g_A) - (g_B_plus_x - g_B))


def construct_diminishing_return_triples(
    *,
    context_schedule: Sequence[int],
    T: int,
    anchor_type: str,
    seed: int,
    budget: int,
    max_triples: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    """Create fixed exact-DR triples A subset B, x not in B.

    The context schedule is used as B. A is B with one context step removed.
    Candidate x values are stratified across early/middle/late phases, then
    deterministically sampled if the full pool is larger than max_triples.
    """
    B_steps = canonical_schedule(context_schedule)
    if len(B_steps) < 2:
        return []
    B_set = set(B_steps)
    candidates: list[dict[str, Any]] = []
    x_by_phase: dict[str, list[int]] = {phase: [] for phase in PHASES}
    for x in range(T):
        if x not in B_set:
            x_by_phase[phase_for_step(x, T=T)].append(x)

    # Phase-stratified deterministic anchors first.
    phase_xs: list[int] = []
    for phase in PHASES:
        pool = x_by_phase[phase]
        if pool:
            phase_xs.append(pool[len(pool) // 2])
    if not phase_xs:
        return []

    for remove_step in B_steps:
        A_steps = canonical_schedule(step for step in B_steps if step != remove_step)
        for x in phase_xs:
            triple_id = (
                f"seed{int(seed)}_B{int(budget)}_{anchor_type}_"
                f"rm{int(remove_step)}_x{int(x)}"
            )
            candidates.append(
                {
                    "triple_id": triple_id,
                    "seed": int(seed),
                    "B": int(budget),
                    "anchor_type": anchor_type,
                    "A_steps": list(A_steps),
                    "B_steps": list(B_steps),
                    "x_step": int(x),
                    "x_phase": phase_for_step(x, T=T),
                    "removed_step": int(remove_step),
                    "A_plus_x_steps": list(canonical_schedule((*A_steps, x))),
                    "B_plus_x_steps": list(canonical_schedule((*B_steps, x))),
                }
            )

    unique: dict[str, dict[str, Any]] = {candidate["triple_id"]: candidate for candidate in candidates}
    values = [unique[key] for key in sorted(unique)]
    if len(values) <= int(max_triples):
        return values
    rng = np.random.default_rng(int(rng_seed))
    indices = sorted(rng.choice(len(values), size=int(max_triples), replace=False).tolist())
    return [values[idx] for idx in indices]


def shard_items(items: Sequence[dict[str, Any]], *, shard_idx: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_idx < 0 or shard_idx >= shard_count:
        raise ValueError("shard_idx must satisfy 0 <= shard_idx < shard_count")
    return [item for idx, item in enumerate(items) if idx % shard_count == shard_idx]


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _git_hash(short: bool = False) -> str:
    env_sha = os.environ.get("PHASE4_GIT_SHA")
    if env_sha:
        return env_sha[:7] if short else env_sha
    args = ["git", "-C", str(repo_root), "rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
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


def _checkpoint_sha256(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.is_dir():
        h = hashlib.sha256()
        for child in sorted(p.rglob("*")):
            if child.is_file() and child.stat().st_size < 1_500_000_000:
                h.update(child.name.encode())
                h.update(str(child.stat().st_size).encode())
        return h.hexdigest()[:24]
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _allocation_from_steps(steps: Sequence[int]) -> dict[int, int]:
    return {int(step): 1 for step in canonical_schedule(steps)}


def _uniform_schedule(T: int, B: int) -> list[int]:
    return sorted(int(step) for step in allocate_budget([0.0] * T, B, "uniform", {}).keys())


def _middle_schedule(T: int, B: int) -> list[int]:
    return sorted(int(step) for step in allocate_budget([0.0] * T, B, "middle", {}).keys())


def _cheap_A(steps: Sequence[int], delta_trace: Mapping[int, float]) -> float:
    return float(sum(float(delta_trace.get(int(step), 0.0)) for step in canonical_schedule(steps)))


def load_protocol_a(protocol_a_dir: Path, K: int, seed_start: int) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    files = sorted(
        protocol_a_dir.glob("trajectory_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    for idx, file_path in enumerate(files):
        if idx >= K:
            break
        obj = json.loads(file_path.read_text())
        seed = int(obj.get("seed", seed_start + idx))
        per_t = obj["per_t"]
        T = int(obj["T"])
        by_seed[seed] = {
            "delta": np.asarray([row["delta"] for row in per_t], dtype=float),
            "T": T,
        }
    if not by_seed:
        raise RuntimeError(f"No Protocol A trajectories found in {protocol_a_dir}")
    reference_T = next(iter(by_seed.values()))["T"]
    for seed, data in by_seed.items():
        if int(data["T"]) != reference_T:
            raise RuntimeError(f"Seed {seed} has T={data['T']} but expected {reference_T}")
        if len(data["delta"]) != reference_T:
            raise RuntimeError(f"Seed {seed} has {len(data['delta'])} deltas but T={reference_T}")
    return by_seed


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    obj = json.loads(path.read_text())
    if not isinstance(obj, list):
        raise RuntimeError(f"Expected list in {path}")
    return obj


def _index_single_rows(
    rows: Sequence[dict[str, Any]],
    *,
    schedule_key: str,
    gain_key: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["B"]))
        indexed[key] = {
            "schedule_steps": list(canonical_schedule(row[schedule_key])),
            "source_G": float(row[gain_key]),
        }
    return indexed


def _index_mc_rows(rows: Sequence[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), int(row["B"]))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: (float(row["G"]), -int(row.get("mc_idx", 0))), reverse=True)
    return dict(grouped)


def _index_policy_rows(rows: Sequence[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        indexed[(int(row["seed"]), int(row["B"]), str(row["policy"]))] = row
    return indexed


def _pick_random_mc(
    mc_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    B: int,
    rng_seed: int,
) -> dict[str, Any] | None:
    if not mc_rows:
        return None
    rng = np.random.default_rng(stable_seed("random_mc", int(seed), int(B), int(rng_seed)))
    return mc_rows[int(rng.integers(0, len(mc_rows)))]


def build_anchor_specs(
    *,
    seed: int,
    B: int,
    T: int,
    cd_by_key: Mapping[tuple[int, int], dict[str, Any]],
    bs_by_key: Mapping[tuple[int, int], dict[str, Any]],
    mc_by_key: Mapping[tuple[int, int], list[dict[str, Any]]],
    policy_by_key: Mapping[tuple[int, int, str], dict[str, Any]],
    random_anchor_seed: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    key = (int(seed), int(B))
    if key in cd_by_key:
        specs.append({"anchor_type": "cd_g", **cd_by_key[key]})
    if key in bs_by_key:
        specs.append({"anchor_type": "bs_ag", **bs_by_key[key]})
    mc_rows = list(mc_by_key.get(key, []))
    if mc_rows:
        top = max(mc_rows, key=lambda row: float(row["G"]))
        specs.append(
            {
                "anchor_type": "top_mc",
                "schedule_steps": list(canonical_schedule(top["schedule_steps"])),
                "source_G": float(top["G"]),
                "source_mc_idx": int(top.get("mc_idx", -1)),
            }
        )
        random_row = _pick_random_mc(mc_rows, seed=seed, B=B, rng_seed=random_anchor_seed)
        if random_row is not None:
            specs.append(
                {
                    "anchor_type": "random_mc",
                    "schedule_steps": list(canonical_schedule(random_row["schedule_steps"])),
                    "source_G": float(random_row["G"]),
                    "source_mc_idx": int(random_row.get("mc_idx", -1)),
                }
            )
    for policy in ("uniform", "middle"):
        row = policy_by_key.get((int(seed), int(B), policy))
        if row is not None:
            schedule = list(canonical_schedule(row["schedule_steps"]))
            source_G = float(row["G"])
        elif policy == "uniform":
            schedule = _uniform_schedule(T, B)
            source_G = None
        else:
            schedule = _middle_schedule(T, B)
            source_G = None
        specs.append({"anchor_type": policy, "schedule_steps": schedule, "source_G": source_G})

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for spec in specs:
        schedule = canonical_schedule(spec["schedule_steps"])
        if len(schedule) != int(B):
            continue
        key_seen = (str(spec["anchor_type"]), schedule)
        if key_seen in seen:
            continue
        seen.add(key_seen)
        spec["schedule_steps"] = list(schedule)
        deduped.append(spec)
    return deduped


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

    backend = args.backend if args.backend != "auto" else detect_proseco_snapshot_backend(args.checkpoint)
    if backend == "proseco_owt":
        from mdm_playground.scheduling.backends.proseco_owt import ProSeCoOWTGenerator

        return ProSeCoOWTGenerator(
            checkpoint=args.checkpoint,
            T=args.T,
            corrector_steps=args.corrector_steps,
        )
    if backend == "proseco_llada_sft":
        from mdm_playground.scheduling.backends.proseco_llada_sft import ProSeCoLLaDASFTGenerator

        return ProSeCoLLaDASFTGenerator(
            checkpoint=args.checkpoint,
            T=args.T,
            corrector_steps=args.corrector_steps,
        )
    raise SystemExit(f"Unsupported backend: {backend}")


class EvaluationCache:
    def __init__(
        self,
        *,
        gen: Any,
        delta_trace: Mapping[int, float],
        seed: int,
    ) -> None:
        self.gen = gen
        self.delta_trace = delta_trace
        self.seed = int(seed)
        self.cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.n_true_g_calls = 0

    def evaluate(self, steps: Sequence[int]) -> tuple[dict[str, Any], int]:
        schedule = canonical_schedule(steps)
        if schedule in self.cache:
            return self.cache[schedule], 0
        result = evaluate_schedule(
            allocation=_allocation_from_steps(schedule),
            delta_trace=dict(self.delta_trace),
            generator=self.gen,
            F="neg_nll",
            seed=self.seed,
        )
        self.cache[schedule] = result
        self.n_true_g_calls += 1
        return result, 1


def _base_row(
    *,
    seed: int,
    budget: int,
    diagnostic_type: str,
    anchor_type: str,
    schedule_steps: Sequence[int],
    anchor_schedule_steps: Sequence[int],
    result: Mapping[str, Any],
    n_g_calls: int,
    git_sha: str,
    shard_id: str,
    T: int,
) -> dict[str, Any]:
    row = {
        "seed": int(seed),
        "B": int(budget),
        "diagnostic_type": diagnostic_type,
        "anchor_type": anchor_type,
        "schedule_steps": list(canonical_schedule(schedule_steps)),
        "anchor_schedule_steps": list(canonical_schedule(anchor_schedule_steps)),
        "G": float(result["G"]),
        "A": float(result.get("A", 0.0)),
        "F_base": float(result["f_base"]) if result.get("f_base") is not None else None,
        "F_schedule": float(result["f_schedule"]) if result.get("f_schedule") is not None else None,
        "n_g_calls": int(n_g_calls),
        "phase_counts": phase_counts(schedule_steps, T=T),
        "git_sha": git_sha,
        "shard_id": shard_id,
    }
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise RuntimeError(f"Internal row construction bug; missing {missing}")
    return row


def evaluate_seed_diagnostics(
    *,
    gen: Any,
    seed: int,
    seed_data: Mapping[str, Any],
    B_values: Sequence[int],
    T: int,
    anchors_by_budget: Mapping[int, list[dict[str, Any]]],
    neighborhood_max_neighbors: int,
    all_neighbors: bool,
    dr_triples_per_context: int,
    dr_anchor_types: set[str],
    rng_seed: int,
    git_sha: str,
    shard_id: str,
) -> dict[str, Any]:
    delta = np.asarray(seed_data["delta"], dtype=float)
    delta_trace = {int(t): float(delta[t]) for t in range(T)}
    cache = EvaluationCache(gen=gen, delta_trace=delta_trace, seed=seed)
    neighborhood_rows: list[dict[str, Any]] = []
    triple_rows: list[dict[str, Any]] = []

    for B in B_values:
        for anchor_idx, anchor in enumerate(anchors_by_budget.get(int(B), [])):
            anchor_type = str(anchor["anchor_type"])
            anchor_steps = list(canonical_schedule(anchor["schedule_steps"]))
            anchor_id = f"seed{int(seed)}_B{int(B)}_{anchor_type}_{anchor_idx}"
            anchor_result, anchor_call = cache.evaluate(anchor_steps)
            anchor_row = _base_row(
                seed=seed,
                budget=int(B),
                diagnostic_type="neighborhood",
                anchor_type=anchor_type,
                schedule_steps=anchor_steps,
                anchor_schedule_steps=anchor_steps,
                result=anchor_result,
                n_g_calls=anchor_call,
                git_sha=git_sha,
                shard_id=shard_id,
                T=T,
            )
            anchor_row.update(
                {
                    "anchor_id": anchor_id,
                    "neighbor_relation": "anchor",
                    "anchor_G": float(anchor_result["G"]),
                    "delta_neighbor": 0.0,
                    "schedule_distance_to_anchor": 0,
                    "source_G": anchor.get("source_G"),
                    "source_mc_idx": anchor.get("source_mc_idx"),
                    "A": _cheap_A(anchor_steps, delta_trace),
                }
            )
            neighborhood_rows.append(anchor_row)

            max_neighbors = None if all_neighbors else int(neighborhood_max_neighbors)
            neighbors = one_swap_neighbors(
                anchor_steps,
                T=T,
                max_neighbors=max_neighbors,
                rng_seed=stable_seed("neighbors", seed, B, anchor_type, rng_seed),
            )
            best_improvement = float("-inf")
            for neighbor_idx, neighbor_steps in enumerate(neighbors):
                result, n_call = cache.evaluate(neighbor_steps)
                delta_neighbor = float(result["G"]) - float(anchor_result["G"])
                best_improvement = max(best_improvement, delta_neighbor)
                row = _base_row(
                    seed=seed,
                    budget=int(B),
                    diagnostic_type="neighborhood",
                    anchor_type=anchor_type,
                    schedule_steps=neighbor_steps,
                    anchor_schedule_steps=anchor_steps,
                    result=result,
                    n_g_calls=n_call,
                    git_sha=git_sha,
                    shard_id=shard_id,
                    T=T,
                )
                row.update(
                    {
                        "anchor_id": anchor_id,
                        "neighbor_idx": int(neighbor_idx),
                        "neighbor_relation": "one_swap",
                        "anchor_G": float(anchor_result["G"]),
                        "delta_neighbor": delta_neighbor,
                        "schedule_distance_to_anchor": schedule_distance(neighbor_steps, anchor_steps),
                    }
                )
                neighborhood_rows.append(row)
            anchor_row["best_one_swap_improvement"] = (
                None if best_improvement == float("-inf") else float(best_improvement)
            )
            anchor_row["is_local_optimum"] = bool(best_improvement <= 0.0) if neighbors else None
            anchor_row["neighbor_evaluation_mode"] = "all" if all_neighbors else "sampled"
            anchor_row["n_neighbors_evaluated"] = int(len(neighbors))

            if anchor_type not in dr_anchor_types:
                continue
            triples = construct_diminishing_return_triples(
                context_schedule=anchor_steps,
                T=T,
                anchor_type=anchor_type,
                seed=seed,
                budget=int(B),
                max_triples=dr_triples_per_context,
                rng_seed=stable_seed("dr", seed, B, anchor_type, rng_seed),
            )
            for triple in triples:
                role_to_steps = {
                    "A": triple["A_steps"],
                    "B": triple["B_steps"],
                    "A_plus_x": triple["A_plus_x_steps"],
                    "B_plus_x": triple["B_plus_x_steps"],
                }
                role_results: dict[str, dict[str, Any]] = {}
                role_calls: dict[str, int] = {}
                for role, steps in role_to_steps.items():
                    role_result, n_call = cache.evaluate(steps)
                    role_results[role] = role_result
                    role_calls[role] = n_call
                dr_gap = diminishing_return_gap(
                    g_A=float(role_results["A"]["G"]),
                    g_B=float(role_results["B"]["G"]),
                    g_A_plus_x=float(role_results["A_plus_x"]["G"]),
                    g_B_plus_x=float(role_results["B_plus_x"]["G"]),
                )
                for role, steps in role_to_steps.items():
                    row = _base_row(
                        seed=seed,
                        budget=int(B),
                        diagnostic_type="diminishing_return",
                        anchor_type=anchor_type,
                        schedule_steps=steps,
                        anchor_schedule_steps=anchor_steps,
                        result=role_results[role],
                        n_g_calls=role_calls[role],
                        git_sha=git_sha,
                        shard_id=shard_id,
                        T=T,
                    )
                    row.update(
                        {
                            "anchor_id": anchor_id,
                            "triple_id": triple["triple_id"],
                            "triple_role": role,
                            "A_steps": triple["A_steps"],
                            "B_steps": triple["B_steps"],
                            "x_step": int(triple["x_step"]),
                            "x_phase": triple["x_phase"],
                            "removed_step": int(triple["removed_step"]),
                            "DR_gap": float(dr_gap),
                        }
                    )
                    triple_rows.append(row)

    return {
        "neighborhood_rows": neighborhood_rows,
        "triple_rows": triple_rows,
        "n_true_g_calls": int(cache.n_true_g_calls),
    }


def _safe_out_dir(out_dir: Path, *, resume: bool) -> None:
    name = out_dir.name
    if not name.startswith("phase4_schedule_neighborhood_"):
        raise SystemExit(
            "--out_dir must be a SHA-tagged phase4_schedule_neighborhood_<gitsha> directory"
        )
    if out_dir.exists() and not resume:
        existing_payload = [
            child
            for child in out_dir.iterdir()
            if child.name
            in {
                "neighborhood_raw.json",
                "triples_raw.json",
                "aggregate_stats.json",
                "interpretation.md",
            }
            or child.name.startswith(("neighborhood_raw.shard", "triples_raw.shard"))
        ]
        if existing_payload:
            raise SystemExit(f"Refusing to overwrite existing diagnostic payload in {out_dir}; use --resume")


def write_config(out_dir: Path, args: argparse.Namespace, *, git_sha: str, shard_id: str) -> None:
    config = {
        "phase": "phase4_schedule_neighborhood_diagnostics",
        "git_sha": git_sha,
        "git_dirty": _git_dirty(),
        "T": args.T,
        "K": args.K,
        "B_values": [int(x) for x in args.B_values.split(",") if x.strip()],
        "seed_start": args.seed_start,
        "neighborhood_max_neighbors": args.neighborhood_max_neighbors,
        "all_neighbors": bool(args.all_neighbors),
        "dr_triples_per_context": args.dr_triples_per_context,
        "dr_anchor_types": args.dr_anchor_types,
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        "shard_id": shard_id,
        "surrogate": bool(args.surrogate),
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256_truncated": _checkpoint_sha256(args.checkpoint),
        "protocol_a_dir": args.protocol_a_dir,
        "phase2b_dir": args.phase2b_dir,
        "phase3a_dir": args.phase3a_dir,
        "rng_seed": args.rng_seed,
        "invocation_command": " ".join(sys.argv),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _json_dump(out_dir / f"config.{shard_id}.json", config)
    if args.shard_count == 1 or not (out_dir / "config.json").exists():
        _json_dump(out_dir / "config.json", config)


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    *,
    git_sha: str,
    shard_id: str,
    extra: Mapping[str, Any],
) -> None:
    try:
        import torch  # type: ignore

        torch_version = getattr(torch, "__version__", "unknown")
        cuda_version = getattr(torch.version, "cuda", None)
    except ImportError:
        torch_version = "not_imported"
        cuda_version = None
    manifest = {
        "phase": "phase4_schedule_neighborhood_diagnostics",
        "git_sha": git_sha,
        "git_dirty": _git_dirty(),
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "hpc_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "shard_id": shard_id,
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        **dict(extra),
    }
    _json_dump(out_dir / f"manifest.{shard_id}.json", manifest)
    if args.shard_count == 1:
        _json_dump(out_dir / "manifest.json", manifest)


def append_command_log(out_dir: Path, *, shard_id: str, lines: Sequence[str], single_shard: bool) -> None:
    text = "\n".join(lines) + "\n"
    (out_dir / f"command_log.{shard_id}.txt").write_text(text)
    if single_shard:
        (out_dir / "command_log.txt").write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 schedule-neighborhood diagnostics")
    p.add_argument("--surrogate", action="store_true", help="Use CPU surrogate generator")
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--K", type=int, default=10, help="Number of paired seeds; K=10 is the pilot default")
    p.add_argument("--B_values", type=str, default="2,3,4")
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--protocol_a_dir", type=str, required=True)
    p.add_argument("--phase2b_dir", type=str, default="results/phase2b_proseco_owt")
    p.add_argument("--phase3a_dir", type=str, default="results/phase3a_proseco_owt")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--backend", type=str, default="auto", choices=("auto", "proseco_owt", "proseco_llada_sft"))
    p.add_argument("--corrector_steps", type=int, default=1)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--all_neighbors", action="store_true")
    p.add_argument("--neighborhood_max_neighbors", type=int, default=32)
    p.add_argument("--dr_triples_per_context", type=int, default=3)
    p.add_argument("--dr_anchor_types", type=str, default="cd_g,bs_ag,top_mc,random_mc,middle")
    p.add_argument("--rng_seed", type=int, default=20260510)
    p.add_argument("--sigma_gain", type=float, default=0.005)
    p.add_argument("--gamma_surrogate", type=float, default=0.008)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.shard_count <= 0:
        raise SystemExit("--shard_count must be positive")
    if args.shard_idx < 0 or args.shard_idx >= args.shard_count:
        raise SystemExit("--shard_idx must satisfy 0 <= shard_idx < shard_count")

    git_sha = _git_hash(short=True)
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "results" / f"phase4_schedule_neighborhood_{git_sha}"
    _safe_out_dir(out_dir, resume=bool(args.resume))
    (out_dir / "per_seed").mkdir(parents=True, exist_ok=True)
    shard_id = f"shard{args.shard_idx}-of-{args.shard_count}"
    write_config(out_dir, args, git_sha=git_sha, shard_id=shard_id)
    append_command_log(
        out_dir,
        shard_id=shard_id,
        single_shard=args.shard_count == 1,
        lines=[
            f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}",
            f"cwd={Path.cwd()}",
            f"command={' '.join(sys.argv)}",
            f"git_sha={git_sha}",
            f"git_dirty={_git_dirty()}",
            f"hostname={socket.gethostname()}",
            f"slurm_job_id={os.environ.get('SLURM_JOB_ID')}",
        ],
    )

    B_values = [int(x) for x in args.B_values.split(",") if x.strip()]
    dr_anchor_types = {x.strip() for x in args.dr_anchor_types.split(",") if x.strip()}

    print("=" * 72)
    print("Phase 4 — schedule-neighborhood diagnostics")
    print("=" * 72)
    print(f"  out_dir:       {out_dir}")
    print(f"  git_sha:       {git_sha}")
    print(f"  K:             {args.K}")
    print(f"  B_values:      {B_values}")
    print(f"  neighbors:     {'all' if args.all_neighbors else args.neighborhood_max_neighbors}")
    print(f"  DR triples:    {args.dr_triples_per_context} per eligible anchor")
    print(f"  shard:         {args.shard_idx}/{args.shard_count}")

    by_seed = load_protocol_a(Path(args.protocol_a_dir), K=args.K, seed_start=args.seed_start)
    seeds_all = sorted(by_seed)
    my_seeds = [seed for idx, seed in enumerate(seeds_all) if idx % args.shard_count == args.shard_idx]
    print(f"  my_seeds:      {my_seeds}")

    phase2b_dir = Path(args.phase2b_dir)
    phase3a_dir = Path(args.phase3a_dir)
    cd_by_key = _index_single_rows(
        _load_rows(phase3a_dir / "cd_raw.json"),
        schedule_key="schedule_final",
        gain_key="G_final",
    )
    bs_by_key = _index_single_rows(
        _load_rows(phase3a_dir / "bs_raw.json"),
        schedule_key="schedule_final",
        gain_key="G_final",
    )
    mc_by_key = _index_mc_rows(_load_rows(phase2b_dir / "mc_raw.json"))
    policy_by_key = _index_policy_rows(_load_rows(phase2b_dir / "policy_raw.json"))
    gen = build_generator(args)

    all_neighborhood: list[dict[str, Any]] = []
    all_triples: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    started = time.time()
    for seed_idx, seed in enumerate(my_seeds):
        seed_path = out_dir / "per_seed" / f"diagnostics_rows_seed{seed}.json"
        if args.resume and seed_path.exists():
            print(f"[seed {seed}] SKIP resume")
            obj = json.loads(seed_path.read_text())
            all_neighborhood.extend(obj.get("neighborhood_rows", []))
            all_triples.extend(obj.get("triple_rows", []))
            seed_summaries.append(obj.get("summary", {"seed": seed, "resumed": True}))
            continue

        anchors_by_budget = {
            B: build_anchor_specs(
                seed=seed,
                B=B,
                T=args.T,
                cd_by_key=cd_by_key,
                bs_by_key=bs_by_key,
                mc_by_key=mc_by_key,
                policy_by_key=policy_by_key,
                random_anchor_seed=args.rng_seed,
            )
            for B in B_values
        }
        print(
            f"[seed {seed}] {seed_idx + 1}/{len(my_seeds)} anchors="
            f"{sum(len(v) for v in anchors_by_budget.values())}",
            flush=True,
        )
        seed_started = time.time()
        diagnostics = evaluate_seed_diagnostics(
            gen=gen,
            seed=seed,
            seed_data=by_seed[seed],
            B_values=B_values,
            T=args.T,
            anchors_by_budget=anchors_by_budget,
            neighborhood_max_neighbors=args.neighborhood_max_neighbors,
            all_neighbors=bool(args.all_neighbors),
            dr_triples_per_context=args.dr_triples_per_context,
            dr_anchor_types=dr_anchor_types,
            rng_seed=args.rng_seed,
            git_sha=git_sha,
            shard_id=shard_id,
        )
        summary = {
            "seed": int(seed),
            "wall_time_seconds": float(time.time() - seed_started),
            "n_neighborhood_rows": len(diagnostics["neighborhood_rows"]),
            "n_triple_rows": len(diagnostics["triple_rows"]),
            "n_true_g_calls": int(diagnostics["n_true_g_calls"]),
        }
        seed_obj = {
            "summary": summary,
            "anchors_by_budget": anchors_by_budget,
            "neighborhood_rows": diagnostics["neighborhood_rows"],
            "triple_rows": diagnostics["triple_rows"],
        }
        _json_dump(seed_path, seed_obj)
        all_neighborhood.extend(diagnostics["neighborhood_rows"])
        all_triples.extend(diagnostics["triple_rows"])
        seed_summaries.append(summary)
        print(
            f"[seed {seed}] done rows={summary['n_neighborhood_rows']}+"
            f"{summary['n_triple_rows']} true_G={summary['n_true_g_calls']} "
            f"wall={summary['wall_time_seconds']:.1f}s",
            flush=True,
        )

    _json_dump(out_dir / f"neighborhood_raw.{shard_id}.json", all_neighborhood)
    _json_dump(out_dir / f"triples_raw.{shard_id}.json", all_triples)
    if args.shard_count == 1:
        _json_dump(out_dir / "neighborhood_raw.json", all_neighborhood)
        _json_dump(out_dir / "triples_raw.json", all_triples)

    write_manifest(
        out_dir,
        args,
        git_sha=git_sha,
        shard_id=shard_id,
        extra={
            "walltime_seconds": float(time.time() - started),
            "n_seeds_evaluated": len(my_seeds),
            "n_neighborhood_rows": len(all_neighborhood),
            "n_triple_rows": len(all_triples),
            "seed_summaries": seed_summaries,
        },
    )
    print("-" * 72)
    print(f"Shard {shard_id} complete in {time.time() - started:.1f}s")
    print(f"  neighborhood rows: {len(all_neighborhood)}")
    print(f"  triple rows:       {len(all_triples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
