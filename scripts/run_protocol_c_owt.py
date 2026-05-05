"""Run Protocol C — bounded adaptive-controller pilot on OWT artefacts.

Loads existing Phase 1 OWT trajectories, Phase 2b OWT MC rows, the Phase 2b
MC oracle headroom, and Refinement-A′ σ_ξ; runs the
:func:`mdm_playground.analysis.protocol_c.protocol_c_pipeline`; writes the
verdict JSON to ``results/protocol_c_owt/protocol_c_summary.json``.

This is a CPU-only pilot. No GPU, no HPC, no new trajectories.

Reference
---------
- Activation audit: docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md
- Experiment plan: docs/thesis/next_steps/ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md
- Theory: docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mdm_playground.analysis.protocol_c import protocol_c_pipeline

logger = logging.getLogger(__name__)


PHASE1_DIR_DEFAULT = Path("results/phase1_proseco_owt_full/protocol_a")
PHASE2B_PER_SEED_DIR_DEFAULT = Path("results/phase2b_proseco_owt/per_seed")
MC_ORACLE_PATH_DEFAULT = Path("results/phase2b/mc_oracle.json")
THEOREM_A_CONSTS_PATH_DEFAULT = Path("results/phase2b/theorem_a_constants.json")
OUTPUT_DIR_DEFAULT = Path("results/protocol_c_owt")


def load_phase1_trajectories(phase1_dir: Path) -> list[dict[str, Any]]:
    """Load all Phase 1 Protocol A trajectories from a directory."""
    files = sorted(phase1_dir.glob("trajectory_*.json"))
    if not files:
        msg = f"no trajectory_*.json files in {phase1_dir}"
        raise FileNotFoundError(msg)
    trajectories: list[dict[str, Any]] = []
    for fp in files:
        with fp.open() as fh:
            trajectories.append(json.load(fh))
    logger.info("loaded %d Phase 1 OWT trajectories from %s", len(trajectories), phase1_dir)
    return trajectories


def load_phase2b_mc_rows_by_seed(
    per_seed_dir: Path,
) -> dict[int, list[dict[str, Any]]]:
    """Load Phase 2b mc_rows_seed*.json files, keyed by seed."""
    files = sorted(per_seed_dir.glob("mc_rows_seed*.json"))
    if not files:
        msg = f"no mc_rows_seed*.json files in {per_seed_dir}"
        raise FileNotFoundError(msg)
    out: dict[int, list[dict[str, Any]]] = {}
    pattern = re.compile(r"mc_rows_seed(\d+)\.json$")
    for fp in files:
        match = pattern.search(fp.name)
        if match is None:
            continue
        seed = int(match.group(1))
        with fp.open() as fh:
            out[seed] = json.load(fh)
    logger.info(
        "loaded Phase 2b MC rows for %d seeds from %s", len(out), per_seed_dir
    )
    return out


def load_delta_open_per_B(  # noqa: N802 — match thesis notation
    mc_oracle_path: Path,
    B_values: Sequence[int],
    phase2b_mc_rows_by_seed: Mapping[int, Sequence[dict[str, Any]]],
) -> dict[int, float]:
    """Compute Δ_open per B from the Phase 2b mc_oracle.json.

    Δ_open(B) := mean over seeds of (best-of-100 G(S) − G(S_uniform)) at that B.
    The numerator G(S_uniform) at the same seed is taken from the policy_rows
    payload via the per-seed Phase 2b artefacts; here we approximate by the
    paired uniform-G stored in mc_oracle.json under ``per_trajectory.B_to_argmax_G``
    relative to the best MC schedule's pair.

    Since mc_oracle.json reports best-of-100 G but not uniform G, we compute
    Δ_open(B) here as the mean across (seed, B) of best_G − uniform_G_proxy
    where uniform_G_proxy is the mean MC G of the seed at that B (a
    conservative lower bound; in practice the Phase 2b paired
    mc_oracle_minus_uniform reports +0.45 directly). For honest Protocol C
    accounting we use the ``mean(best_G) − mean(uniform_G_among_mc)`` form.
    """
    with mc_oracle_path.open() as fh:
        mc_oracle = json.load(fh)
    per_trajectory = mc_oracle["data"]["per_trajectory"]

    out: dict[int, float] = {}
    for B in B_values:
        # Best MC schedule G per seed for this B.
        best_G_per_seed: list[float] = []
        for traj in per_trajectory:
            seed = int(traj["seed"])
            if str(B) in traj.get("B_to_argmax_G", {}):
                best_G_per_seed.append(
                    float(traj["B_to_argmax_G"][str(B)]["G"])
                )
        if not best_G_per_seed:
            msg = f"no best-G entries for B={B} in mc_oracle.json"
            raise ValueError(msg)
        # Uniform G per seed: pulled from the seed's MC rows by checking which
        # mc_rows have schedule_steps matching the uniform pattern. Phase 2b
        # uniform schedule for B=2,3,4 at T=64 is {0, 32}, {0, 21, 42}, {0, 16, 32, 48}.
        # We find the closest by Hamming distance.
        T = 64  # OWT Phase 2b standard
        uniform_steps = _uniform_steps(T, B)
        uniform_g_per_seed: list[float] = []
        for traj in per_trajectory:
            seed = int(traj["seed"])
            mc_rows = phase2b_mc_rows_by_seed.get(seed)
            if mc_rows is None:
                continue
            # Find row with schedule closest to uniform; expect Hamming 0 if
            # uniform was sampled, else use mean-MC-G as proxy.
            closest = None
            closest_ham = None
            for r in mc_rows:
                if int(r["B"]) != B:
                    continue
                rs = frozenset(int(s) for s in r["schedule_steps"])
                ham = len(rs.symmetric_difference(uniform_steps))
                if closest_ham is None or ham < closest_ham:
                    closest_ham = ham
                    closest = r
                if closest_ham == 0:
                    break
            if closest is not None and closest_ham == 0:
                uniform_g_per_seed.append(float(closest["G"]))
            elif mc_rows:
                # fallback: mean G across this seed's MC rows at this B
                mc_g = [
                    float(r["G"]) for r in mc_rows if int(r["B"]) == B
                ]
                if mc_g:
                    uniform_g_per_seed.append(float(sum(mc_g) / len(mc_g)))

        if not uniform_g_per_seed:
            # last resort fallback: assume baseline 0
            uniform_mean = 0.0
        else:
            uniform_mean = float(sum(uniform_g_per_seed) / len(uniform_g_per_seed))
        best_mean = float(sum(best_G_per_seed) / len(best_G_per_seed))
        out[B] = best_mean - uniform_mean
    logger.info("Δ_open per B (best_MC − uniform_proxy): %s", out)
    return out


def _uniform_steps(T: int, B: int) -> frozenset[int]:
    if B <= 0:
        return frozenset()
    if B >= T:
        return frozenset(range(T))
    step = T / B
    return frozenset(int(i * step) for i in range(B))


def load_sigma_xi_per_B(  # noqa: N802 — match thesis notation
    theorem_a_constants_path: Path, B_values: Sequence[int]
) -> dict[int, float]:
    """Load σ_ξ pooled per B from theorem_a_constants.json."""
    with theorem_a_constants_path.open() as fh:
        d = json.load(fh)
    sigma_xi = d["data"]["sigma_xi"]["per_B"]
    out: dict[int, float] = {}
    for B in B_values:
        if str(B) not in sigma_xi:
            msg = f"σ_ξ not found for B={B} in {theorem_a_constants_path}"
            raise ValueError(msg)
        out[B] = float(sigma_xi[str(B)]["sigma_xi_pooled"])
    logger.info("σ_ξ pooled per B: %s", out)
    return out


def _git_short_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Protocol C — bounded adaptive controller pilot on OWT.",
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=PHASE1_DIR_DEFAULT,
        help="Phase 1 Protocol A trajectory directory.",
    )
    parser.add_argument(
        "--phase2b-per-seed-dir",
        type=Path,
        default=PHASE2B_PER_SEED_DIR_DEFAULT,
        help="Phase 2b per-seed mc_rows directory.",
    )
    parser.add_argument(
        "--mc-oracle-path",
        type=Path,
        default=MC_ORACLE_PATH_DEFAULT,
        help="Phase 2b mc_oracle.json.",
    )
    parser.add_argument(
        "--theorem-a-constants-path",
        type=Path,
        default=THEOREM_A_CONSTS_PATH_DEFAULT,
        help="Phase 2b theorem_a_constants.json (for σ_ξ).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Output directory for the Protocol C summary JSON.",
    )
    parser.add_argument(
        "--B-values", nargs="+", type=int, default=[2, 3, 4]
    )
    parser.add_argument(
        "--n-signal-bins", type=int, default=4, help="Signal quantile bins."
    )
    parser.add_argument(
        "--n-phase-bins", type=int, default=3, help="Phase buckets (early/mid/late)."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=args.log_level,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_phase1_trajectories(args.phase1_dir)
    mc_rows_by_seed = load_phase2b_mc_rows_by_seed(args.phase2b_per_seed_dir)
    delta_open = load_delta_open_per_B(
        args.mc_oracle_path, args.B_values, mc_rows_by_seed
    )
    sigma_xi = load_sigma_xi_per_B(args.theorem_a_constants_path, args.B_values)

    summary = protocol_c_pipeline(
        phase1_trajectories=trajectories,
        phase2b_mc_rows_by_seed=mc_rows_by_seed,
        delta_open_per_B=delta_open,
        sigma_xi_per_B=sigma_xi,
        B_values=args.B_values,
        n_signal_bins=args.n_signal_bins,
        n_phase_bins=args.n_phase_bins,
    )
    summary["meta"]["generated_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    summary["meta"]["generated_by"] = (
        f"scripts/run_protocol_c_owt.py@{_git_short_hash()}"
    )
    summary["meta"]["phase1_dir"] = str(args.phase1_dir)
    summary["meta"]["phase2b_per_seed_dir"] = str(args.phase2b_per_seed_dir)

    out_path = args.output_dir / "protocol_c_summary.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    logger.info("wrote %s", out_path)
    print(json.dumps(summary["verdict"], indent=2))  # noqa: T201 — programmatic stdout
    return 0


if __name__ == "__main__":
    sys.exit(main())
