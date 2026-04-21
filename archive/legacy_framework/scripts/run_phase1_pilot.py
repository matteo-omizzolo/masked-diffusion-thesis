#!/usr/bin/env python3
"""Phase 1 Pilot: entropy-proxy experiment (Protocol A + B).

Runs both surrogate mode (local, no GPU) and real MDLM mode (HPC).

Usage:
    # Surrogate (pipeline validation, local)
    python scripts/run_phase1_pilot.py --surrogate \
        --T 64 --N 30 --M 20 --P 100 --B_values 4,8,16 --seed 42

    # Real MDLM (HPC — called from phase1_pilot.sbatch)
    python scripts/run_phase1_pilot.py \
        --T 128 --N 50 --M 30 --P 300 --B_values 8,16,32 --seed 42 \
        --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
        --out_dir results/phase1_real

Outputs (under --out_dir):
    protocol_a/trajectory_{i}.json     per-trajectory gain tables
    protocol_b/schedule_{j}.json       sampled schedule G/A/residual
    protocol_b/pairs.json              pairwise xi estimates
    summary.json                       aggregate epsilon, eta_B, gamma, T_low
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow running from repo root
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from mdm_playground.scheduling import (
    allocate_budget,
    compute_signals,
    estimate_single_step_gain,
    evaluate_schedule,
)
from mdm_playground.scheduling.surrogate import SurrogateGenerator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 entropy-proxy pilot experiment")
    p.add_argument("--surrogate", action="store_true",
                   help="Use surrogate generator (no GPU needed)")
    p.add_argument("--T", type=int, default=64, help="Predictor steps")
    p.add_argument("--N", type=int, default=30, help="Number of trajectories (Protocol A)")
    p.add_argument("--M", type=int, default=20, help="Sampled schedules per B (Protocol B)")
    p.add_argument("--P", type=int, default=100, help="Pairs for pairwise gamma (Protocol B)")
    p.add_argument("--B_values", type=str, default="4,8,16",
                   help="Comma-separated budget values for Protocol B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="MDLM checkpoint path (real mode only)")
    p.add_argument("--out_dir", type=str, default="results/phase1_pilot")
    # Surrogate-specific
    p.add_argument("--gamma", type=float, default=0.008,
                   help="Pairwise interaction scale (surrogate)")
    p.add_argument("--sigma_gain", type=float, default=0.005,
                   help="Gain noise (surrogate)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Protocol A: per-step one-loop gain
# ---------------------------------------------------------------------------

def run_protocol_a(gen, N: int, T: int, seed_base: int, out_dir: Path):
    """Run Protocol A for N trajectories. Returns per-step mean Δ_t array."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []

    for i in range(N):
        seed = seed_base + i
        print(f"  Trajectory {i+1}/{N} (seed={seed}) ...", flush=True)
        y_base = gen.run_base(seed=seed)

        per_t = []
        for t in range(T):
            y_branch = gen.run_branch(t_corrected=t, seed=seed)
            gain = estimate_single_step_gain(y_base, y_branch, F="neg_nll")
            sigs = y_base["per_step_signals"][t]
            per_t.append({
                "t": t,
                "delta": gain["delta"],
                "tcr": gain["tcr"],
                "f_base": gain["f_base"],
                "f_branch": gain["f_branch"],
                "n_changed": gain["n_changed"],
                "entropy": sigs["entropy"],
                "inverse_margin": sigs["inverse_margin"],
                "quality_mass_proxy": sigs["quality_mass_proxy"],
                "unmasked_fraction": sigs["unmasked_fraction"],
            })
        record = {"seed": seed, "T": T, "per_t": per_t}
        (out_dir / f"trajectory_{i}.json").write_text(json.dumps(record, indent=2))
        all_records.append(per_t)

    print(f"  Protocol A done — {N} trajectories saved.")
    return all_records


# ---------------------------------------------------------------------------
# Protocol B: joint gain vs additive surrogate
# ---------------------------------------------------------------------------

def run_protocol_b(
    gen,
    all_records: List,
    B_values: List[int],
    M: int,
    P: int,
    T: int,
    seed_base: int,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build mean delta_trace from Protocol A
    delta_mean = np.zeros(T)
    for traj in all_records:
        for rec in traj:
            delta_mean[rec["t"]] += rec["delta"]
    delta_mean /= len(all_records)
    delta_trace = {t: float(delta_mean[t]) for t in range(T)}

    # --- η_B estimation ---
    schedule_records = []
    j = 0
    for B in B_values:
        for m in range(M):
            seed = seed_base + j * 7 + 1000
            rng = np.random.default_rng(seed)
            steps = sorted(int(s) for s in rng.choice(T, size=min(B, T), replace=False))
            allocation = {s: 1 for s in steps}

            result = evaluate_schedule(
                allocation, delta_trace, gen, F="neg_nll", seed=seed
            )
            rec = {"j": j, "B": B, "seed": seed, **result}
            (out_dir / f"schedule_{j}.json").write_text(json.dumps(rec, indent=2))
            schedule_records.append(rec)
            j += 1

    print(f"  Protocol B (eta_B): {j} schedules done.")

    # --- Pairwise γ estimation ---
    pairs_records = []
    rng_pairs = np.random.default_rng(seed_base + 9999)
    for p_idx in range(P):
        t, tp = int(rng_pairs.integers(0, T)), int(rng_pairs.integers(0, T))
        while tp == t:
            tp = int(rng_pairs.integers(0, T))
        seed = seed_base + p_idx * 13 + 5000

        y_base = gen.run_base(seed=seed)
        y_t = gen.run_branch(t_corrected=t, seed=seed)
        y_tp = gen.run_branch(t_corrected=tp, seed=seed)
        y_pair = gen.run_with_schedule(allocation={t: 1, tp: 1}, seed=seed)

        delta_t = estimate_single_step_gain(y_base, y_t, F="neg_nll")["delta"]
        delta_tp = estimate_single_step_gain(y_base, y_tp, F="neg_nll")["delta"]
        g_pair = y_pair["neg_nll"] - y_base["neg_nll"]
        xi = g_pair - (delta_t + delta_tp)

        pairs_records.append({"p_idx": p_idx, "t": t, "tp": tp,
                               "delta_t": delta_t, "delta_tp": delta_tp,
                               "G_pair": g_pair, "xi": xi})

    (out_dir / "pairs.json").write_text(json.dumps(pairs_records, indent=2))
    print(f"  Protocol B (pairs): {P} pairs done.")

    return schedule_records, pairs_records, delta_trace


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def compute_summary(
    all_records: List,
    schedule_records: List,
    pairs_records: List,
    B_values: List[int],
    T: int,
    out_dir: Path,
) -> dict:
    """Compute aggregate ε, η_B, γ, T_low and save summary.json."""

    # ε: per-signal, estimated as RMS of (delta - signal) after z-score alignment
    deltas = np.array([[r["delta"] for r in traj] for traj in all_records])  # (N, T)
    entropy = np.array([[r["entropy"] for r in traj] for traj in all_records])
    margin = np.array([[r["inverse_margin"] for r in traj] for traj in all_records])
    quality = np.array([[r["quality_mass_proxy"] for r in traj] for traj in all_records])

    mean_delta = deltas.mean(axis=0)   # (T,)
    mean_delta_all = mean_delta.mean()
    std_delta = deltas.std() + 1e-12

    def calibrate_and_epsilon(signal_arr):
        """Linear calibration of signal → delta. Return residual and epsilon."""
        sig_flat = signal_arr.flatten()
        d_flat = deltas.flatten()
        # Least squares: delta ≈ a * signal + b
        A_mat = np.column_stack([sig_flat, np.ones_like(sig_flat)])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, d_flat, rcond=None)
        a, b = coeffs
        psi = a * signal_arr + b
        residual = deltas - psi
        eps_rms = float(np.sqrt((residual ** 2).mean()))
        eps_max = float(np.abs(residual).max())
        # Spearman per trajectory, then mean
        from scipy.stats import spearmanr
        spearman_per = [spearmanr(signal_arr[i], deltas[i]).statistic
                        for i in range(len(deltas))]
        return {
            "eps_rms": eps_rms,
            "eps_max": eps_max,
            "spearman_mean": float(np.mean(spearman_per)),
            "spearman_std": float(np.std(spearman_per)),
            "calib_a": float(a),
            "calib_b": float(b),
        }

    eps_entropy = calibrate_and_epsilon(entropy)
    eps_margin = calibrate_and_epsilon(margin)
    eps_quality = calibrate_and_epsilon(quality)

    # η_B per B value
    eta_by_B = {}
    for B in B_values:
        recs = [r for r in schedule_records if r["B"] == B]
        if recs:
            resids = [abs(r["residual"]) for r in recs]
            eta_by_B[str(B)] = {
                "eta_95": float(np.percentile(resids, 95)),
                "eta_mean": float(np.mean(resids)),
                "n_schedules": len(recs),
            }

    # γ (pairwise interaction bound)
    xis = [abs(r["xi"]) for r in pairs_records]
    gamma_est = {
        "gamma_95": float(np.percentile(xis, 95)),
        "gamma_mean": float(np.mean(xis)),
        "gamma_max": float(max(xis) if xis else 0.0),
        "n_pairs": len(xis),
    }

    # T_low: steps where mean Δ_t ≤ some fraction of the peak
    peak_delta = mean_delta.max()
    threshold_05 = 0.5 * peak_delta
    threshold_03 = 0.3 * peak_delta
    T_low_05 = [int(t) for t in np.where(mean_delta <= threshold_05)[0]]
    T_low_03 = [int(t) for t in np.where(mean_delta <= threshold_03)[0]]

    # Theorem A bound check for each B
    theorem_A_check = {}
    for B in B_values:
        eta_rec = eta_by_B.get(str(B), {})
        eta_95 = eta_rec.get("eta_95", 0.0)
        eps = eps_entropy["eps_rms"]
        bound = 2 * B * eps + 2 * eta_95
        # Estimate G(Ŝ_B) using top-B by entropy from Protocol A mean signals
        mean_entropy = entropy.mean(axis=0)
        top_B_idx = np.argsort(mean_entropy)[::-1][:B]
        G_top_B_estimate = float(mean_delta[top_B_idx].sum())
        theorem_A_check[str(B)] = {
            "bound_2Be_2eta": round(bound, 5),
            "G_top_B_estimate": round(G_top_B_estimate, 5),
            "bound_useful": bool(bound < G_top_B_estimate),
        }

    summary = {
        "T": T,
        "N_trajectories": len(all_records),
        "calibration": {
            "entropy": eps_entropy,
            "inverse_margin": eps_margin,
            "quality_mass_proxy": eps_quality,
        },
        "eta_by_B": eta_by_B,
        "gamma": gamma_est,
        "T_low": {
            "threshold_50pct_peak": T_low_05[:10],  # first 10 for readability
            "threshold_30pct_peak": T_low_03[:10],
            "peak_delta_mean": float(peak_delta),
        },
        "theorem_A_bound_check": theorem_A_check,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary written to {out_dir}/summary.json")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    B_values = [int(x) for x in args.B_values.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1 Entropy-Proxy Pilot Experiment")
    print("=" * 60)
    print(f"  Mode:     {'SURROGATE' if args.surrogate else 'REAL MDLM'}")
    print(f"  T={args.T}, N={args.N}, M={args.M}, P={args.P}")
    print(f"  B values: {B_values}")
    print(f"  Out dir:  {out_dir}")
    print()

    t_start = time.time()

    if args.surrogate:
        gen = SurrogateGenerator(
            T=args.T,
            sigma_gain=args.sigma_gain,
            gamma=args.gamma,
            seed_base=args.seed,
        )
    else:
        # Import real backend only on HPC
        try:
            from mdm_playground.scheduling.backends.mdlm import MDLMGenerator
            gen = MDLMGenerator(checkpoint=args.checkpoint, T=args.T)
        except ImportError as e:
            print(f"ERROR: Real MDLM backend not available: {e}")
            print("Use --surrogate for local testing.")
            sys.exit(1)

    # --- Protocol A ---
    print("Running Protocol A ...")
    all_records = run_protocol_a(
        gen, N=args.N, T=args.T,
        seed_base=args.seed,
        out_dir=out_dir / "protocol_a",
    )

    # --- Protocol B ---
    print("\nRunning Protocol B ...")
    schedule_records, pairs_records, delta_trace = run_protocol_b(
        gen,
        all_records=all_records,
        B_values=B_values,
        M=args.M,
        P=args.P,
        T=args.T,
        seed_base=args.seed,
        out_dir=out_dir / "protocol_b",
    )

    # --- Summary ---
    print("\nComputing summary ...")
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("WARNING: scipy not found, installing...")
        os.system(f"{sys.executable} -m pip install scipy --break-system-packages -q")
        from scipy.stats import spearmanr

    summary = compute_summary(
        all_records=all_records,
        schedule_records=schedule_records,
        pairs_records=pairs_records,
        B_values=B_values,
        T=args.T,
        out_dir=out_dir,
    )

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"\n{'=' * 60}")
    print("KEY RESULTS")
    print("=" * 60)
    cal = summary["calibration"]
    print(f"  ε (entropy, RMS):  {cal['entropy']['eps_rms']:.5f}")
    print(f"  ε (margin, RMS):   {cal['inverse_margin']['eps_rms']:.5f}")
    print(f"  Spearman(H, Δ):    {cal['entropy']['spearman_mean']:.3f} "
          f"± {cal['entropy']['spearman_std']:.3f}")
    print(f"  γ (95th pct):      {summary['gamma']['gamma_95']:.5f}")
    print()
    for B in B_values:
        chk = summary["theorem_A_bound_check"].get(str(B), {})
        print(f"  B={B}: 2Bε+2η = {chk.get('bound_2Be_2eta', '?'):.4f}  "
              f"G(Ŝ_B) ≈ {chk.get('G_top_B_estimate', '?'):.4f}  "
              f"{'✓ non-vacuous' if chk.get('bound_useful') else '✗ vacuous'}")
    print("=" * 60)
    print(f"\nAll results saved in {out_dir}/")
    print("Next: python scripts/analyze_phase1.py --results_dir", out_dir)


if __name__ == "__main__":
    main()
