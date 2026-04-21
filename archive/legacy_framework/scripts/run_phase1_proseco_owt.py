#!/usr/bin/env python3
"""Phase 1 Experiment — ProSeCo-OWT Backend: Protocol A + B.

Uses kuleshov-group/proseco-owt (co-trained backbone) with the
annealed-refinement corrector on UNMASKED positions.

Usage:
    # Surrogate (pipeline validation, CPU-only)
    python scripts/run_phase1_proseco_owt.py --surrogate \
        --T 64 --N 10 --M 8 --P 50 --B_values 4,8,16 --seed 42

    # ProSeCo-OWT (HPC, requires GPU + staged checkpoint)
    python scripts/run_phase1_proseco_owt.py \
        --T 64 --N 20 --M 15 --P 120 --B_values 4,8,16 --seed 42 \
        --checkpoint ~/mdm/checkpoints/proseco_owt \
        --corrector_steps 1 \
        --out_dir results/phase1_proseco_owt

    # Full run (HPC, --time 14:00:00)
    python scripts/run_phase1_proseco_owt.py \
        --T 64 --N 50 --M 30 --P 300 --B_values 4,8,16 --seed 42 \
        --checkpoint ~/mdm/checkpoints/proseco_owt \
        --corrector_steps 1 \
        --out_dir results/phase1_proseco_owt_full

Outputs (under --out_dir):
    protocol_a/trajectory_{i}.json     per-trajectory gain tables
    protocol_b/schedule_{j}.json       schedule G/A/residual records
    protocol_b/pairs.json              pairwise xi estimates
    summary.json                       aggregate epsilon, eta_B, gamma, T_low
    run_config.json                    full run configuration

Key differences from run_phase1_pilot.py:
    - Uses ProSeCoGenerator (not MDLMGenerator or SurrogateGenerator)
    - Corrector applies to UNMASKED positions (not masked)
    - Signals computed over UNMASKED positions (ProSeCo action set)
    - Corrector = annealed iterative refinement (not one-shot Gibbs resample)
    - run_branch does NOT record signals (cost saving; only run_base does)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from mdm_playground.scheduling import (
    allocate_budget,
    estimate_single_step_gain,
    evaluate_schedule,
)
from mdm_playground.scheduling.surrogate import SurrogateGenerator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1 ProSeCo experiment — Protocol A + B"
    )
    p.add_argument("--surrogate", action="store_true",
                   help="Use surrogate generator (CPU, no GPU needed)")
    p.add_argument("--T", type=int, default=64,
                   help="Predictor steps")
    p.add_argument("--N", type=int, default=20,
                   help="Number of trajectories (Protocol A)")
    p.add_argument("--M", type=int, default=15,
                   help="Sampled schedules per budget B (Protocol B)")
    p.add_argument("--P", type=int, default=120,
                   help="Pairwise pairs for gamma estimation (Protocol B)")
    p.add_argument("--B_values", type=str, default="4,8,16",
                   help="Comma-separated corrector budget values for Protocol B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="MDLM checkpoint path (required for real ProSeCo run)")
    p.add_argument("--corrector_steps", type=int, default=2,
                   help="Inner corrector iterations per loop (default 2)")
    p.add_argument("--out_dir", type=str, default="results/phase1_proseco")
    # Surrogate-specific
    p.add_argument("--gamma_surrogate", type=float, default=0.008)
    p.add_argument("--sigma_gain", type=float, default=0.005)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Protocol A: per-step one-loop marginal gain
# ---------------------------------------------------------------------------

def run_protocol_a(gen, N: int, T: int, seed_base: int, out_dir: Path) -> List:
    """Run Protocol A: for each of N trajectories, measure Δ_t at all T steps.

    Returns list of per-trajectory per-step records (each entry = list of T dicts).
    Signals come from run_base (per_step_signals). Δ_t from run_branch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []

    for i in range(N):
        seed = seed_base + i
        print(f"  Trajectory {i+1}/{N} (seed={seed}) ...", flush=True)

        # Base trajectory: records per-step signals
        y_base = gen.run_base(seed=seed)
        base_signals = y_base["per_step_signals"]  # length T

        per_t = []
        for t in range(T):
            # Branch: no signal recording needed (only final neg_nll)
            y_branch = gen.run_branch(t_corrected=t, seed=seed)
            gain = estimate_single_step_gain(y_base, y_branch, F="neg_nll")
            sigs = base_signals[t]

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
                "n_revisable": sigs.get("n_revisable", 0),
                "n_masked": sigs.get("n_masked", 0),
            })

        record = {"seed": seed, "T": T, "per_t": per_t}
        (out_dir / f"trajectory_{i}.json").write_text(json.dumps(record, indent=2))
        all_records.append(per_t)
        print(f"    f_base={y_base['neg_nll']:.4f}  "
              f"delta_range=[{min(r['delta'] for r in per_t):.4f}, "
              f"{max(r['delta'] for r in per_t):.4f}]", flush=True)

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
) -> tuple:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build mean delta_trace from Protocol A
    delta_mean = np.zeros(T)
    for traj in all_records:
        for rec in traj:
            delta_mean[rec["t"]] += rec["delta"]
    delta_mean /= max(len(all_records), 1)
    delta_trace = {t: float(delta_mean[t]) for t in range(T)}

    # --- η_B estimation: random schedules ---
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
        t = int(rng_pairs.integers(0, T))
        tp = int(rng_pairs.integers(0, T))
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

        pairs_records.append({
            "p_idx": p_idx, "t": t, "tp": tp,
            "delta_t": delta_t, "delta_tp": delta_tp,
            "G_pair": g_pair, "xi": xi,
        })

    (out_dir / "pairs.json").write_text(json.dumps(pairs_records, indent=2))
    print(f"  Protocol B (pairs): {P} pairs done.")

    return schedule_records, pairs_records, delta_trace


# ---------------------------------------------------------------------------
# Signal-based schedule policies comparison (beyond uniform baseline)
# ---------------------------------------------------------------------------

def run_policy_comparison(
    gen,
    all_records: List,
    delta_trace: Dict[int, float],
    B_values: List[int],
    T: int,
    seed_base: int,
    out_dir: Path,
) -> List[Dict]:
    """Compare scheduling policies for each budget B.

    Policies evaluated:
      uniform, front, back, middle,
      top_B (entropy), top_B (margin), top_B (quality_mass),
      burn_in_gated (entropy),
      oracle (top-B by measured mean Δ_t),
      bottom_B (sanity check / worst case)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-signal mean traces from Protocol A
    mean_entropy = np.array([
        np.mean([traj[t]["entropy"] for traj in all_records])
        for t in range(T)
    ])
    mean_margin = np.array([
        np.mean([traj[t]["inverse_margin"] for traj in all_records])
        for t in range(T)
    ])
    mean_quality = np.array([
        np.mean([traj[t]["quality_mass_proxy"] for traj in all_records])
        for t in range(T)
    ])
    mean_delta = np.array([delta_trace[t] for t in range(T)])

    policies = {
        "uniform": ("uniform", {}),
        "front": ("front", {}),
        "back": ("back", {}),
        "middle": ("middle", {}),
        "entropy_top_B": ("top_B", {}),
        "margin_top_B": ("margin_top_B", {}),
        "quality_top_B": ("quality_top_B", {}),
        "entropy_burn_in_gated": ("burn_in_gated", {"low_gain_threshold": 0.0}),
        "oracle": None,  # handled separately using mean_delta directly
        "bottom_B": ("bottom_B", {}),  # worst case sanity check
    }

    signal_traces = {
        "uniform": mean_delta,
        "front": mean_delta,
        "back": mean_delta,
        "middle": mean_delta,
        "entropy_top_B": mean_entropy,
        "margin_top_B": mean_margin,
        "quality_top_B": mean_quality,
        "entropy_burn_in_gated": mean_entropy,
        "oracle": mean_delta,
        "bottom_B": mean_entropy,
    }

    comparison_records = []
    seed = seed_base + 88888

    for B in B_values:
        print(f"  Policy comparison, B={B} ...", flush=True)
        y_base = gen.run_base(seed=seed)

        for policy_name, policy_spec in policies.items():
            trace = signal_traces[policy_name]

            if policy_name == "oracle":
                allocation = allocate_budget(trace, B, "top_B")
            elif policy_spec is not None:
                policy_fn, policy_kwargs = policy_spec
                allocation = allocate_budget(trace, B, policy_fn, policy_kwargs)
            else:
                continue  # shouldn't reach here

            result = evaluate_schedule(
                allocation, delta_trace, gen, F="neg_nll", seed=seed
            )
            rec = {
                "B": B, "policy": policy_name,
                "G": result["G"], "A": result["A"],
                "residual": result["residual"],
                "f_base": result["f_base"],
                "f_schedule": result["f_schedule"],
                "allocation": sorted(allocation.keys()),
                "seed": seed,
            }
            comparison_records.append(rec)
            print(f"    {policy_name:30s} G={result['G']:+.4f}  A={result['A']:+.4f}",
                  flush=True)

    (out_dir / "policy_comparison.json").write_text(
        json.dumps(comparison_records, indent=2)
    )
    print(f"  Policy comparison done — {len(comparison_records)} records.")
    return comparison_records


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(
    all_records: List,
    schedule_records: List,
    pairs_records: List,
    comparison_records: List,
    B_values: List[int],
    T: int,
    out_dir: Path,
) -> dict:
    """Compute aggregate ε, η_B, γ, T_low and policy comparison summary."""
    from scipy.stats import spearmanr

    deltas = np.array([[r["delta"] for r in traj] for traj in all_records])  # (N, T)
    entropy = np.array([[r["entropy"] for r in traj] for traj in all_records])
    margin = np.array([[r["inverse_margin"] for r in traj] for traj in all_records])
    quality = np.array([[r["quality_mass_proxy"] for r in traj] for traj in all_records])
    unmasked = np.array([[r["unmasked_fraction"] for r in traj] for traj in all_records])

    mean_delta = deltas.mean(axis=0)

    def calibrate_and_epsilon(signal_arr, label):
        sig_flat = signal_arr.flatten()
        d_flat = deltas.flatten()
        A_mat = np.column_stack([sig_flat, np.ones_like(sig_flat)])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, d_flat, rcond=None)
        a, b = coeffs
        psi = a * signal_arr + b
        residual = deltas - psi
        eps_rms = float(np.sqrt((residual ** 2).mean()))
        eps_max = float(np.abs(residual).max())
        spearman_per = []
        for i in range(len(deltas)):
            r, _ = spearmanr(signal_arr[i], deltas[i])
            spearman_per.append(float(r) if not np.isnan(r) else 0.0)
        return {
            "label": label,
            "eps_rms": eps_rms,
            "eps_max": eps_max,
            "spearman_mean": float(np.nanmean(spearman_per)),
            "spearman_std": float(np.nanstd(spearman_per)),
            "calib_a": float(a),
            "calib_b": float(b),
        }

    eps_entropy = calibrate_and_epsilon(entropy, "entropy")
    eps_margin = calibrate_and_epsilon(margin, "inverse_margin")
    eps_quality = calibrate_and_epsilon(quality, "quality_mass_proxy")
    eps_unmasked = calibrate_and_epsilon(unmasked, "unmasked_fraction")

    # η_B per B
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

    # γ (pairwise)
    xis = [abs(r["xi"]) for r in pairs_records] if pairs_records else [0.0]
    gamma_est = {
        "gamma_95": float(np.percentile(xis, 95)),
        "gamma_mean": float(np.mean(xis)),
        "gamma_max": float(max(xis)),
        "n_pairs": len(pairs_records),
    }

    # T_low from burn-in diagnostics
    peak_delta = mean_delta.max() if mean_delta.max() > 0 else 1.0
    T_low_50 = [int(t) for t in np.where(mean_delta <= 0.5 * peak_delta)[0]]
    T_low_30 = [int(t) for t in np.where(mean_delta <= 0.3 * peak_delta)[0]]
    n_positive = int((mean_delta > 0).sum())
    t_first_positive = int(np.argmax(mean_delta > 0)) if n_positive > 0 else T

    # Theorem A checks
    theorem_A_check = {}
    for B in B_values:
        eta_rec = eta_by_B.get(str(B), {})
        eta_95 = eta_rec.get("eta_95", 0.0)
        eps = eps_entropy["eps_rms"]
        bound = 2 * B * eps + 2 * eta_95
        # G(Ŝ_B) estimate from oracle schedule
        oracle_idx = np.argsort(mean_delta)[::-1][:B]
        G_oracle_est = float(mean_delta[oracle_idx].sum())
        theorem_A_check[str(B)] = {
            "bound_2Be_2eta": round(bound, 5),
            "G_oracle_estimate": round(G_oracle_est, 5),
            "bound_useful": bool(bound < G_oracle_est),
            "eps_entropy_rms": round(eps, 5),
            "eta_95": round(eta_95, 5),
        }

    # Policy comparison summary
    policy_summary: Dict = {}
    for rec in comparison_records:
        B = rec["B"]
        policy = rec["policy"]
        key = f"B{B}"
        if key not in policy_summary:
            policy_summary[key] = {}
        policy_summary[key][policy] = {
            "G": round(rec["G"], 5),
            "beats_uniform": None,  # filled below
        }
    for B_key, policies in policy_summary.items():
        uniform_G = policies.get("uniform", {}).get("G", 0.0)
        for policy, vals in policies.items():
            vals["beats_uniform"] = bool(vals["G"] > uniform_G)

    summary = {
        "T": T,
        "N_trajectories": len(all_records),
        "backend": "ProSeCo-style corrector (MDLM backbone)",
        "n_positive_delta_steps": n_positive,
        "t_first_positive_delta": t_first_positive,
        "peak_mean_delta": float(peak_delta),
        "calibration": {
            "entropy": eps_entropy,
            "inverse_margin": eps_margin,
            "quality_mass_proxy": eps_quality,
            "unmasked_fraction": eps_unmasked,
        },
        "eta_by_B": eta_by_B,
        "gamma": gamma_est,
        "T_low": {
            "threshold_50pct_peak": T_low_50[:10],
            "threshold_30pct_peak": T_low_30[:10],
            "n_positive_delta": n_positive,
            "t_first_positive": t_first_positive,
        },
        "theorem_A_bound_check": theorem_A_check,
        "policy_comparison": policy_summary,
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

    run_config = {
        "backend": "surrogate" if args.surrogate else "proseco_owt",
        "T": args.T, "N": args.N, "M": args.M, "P": args.P,
        "B_values": B_values, "seed": args.seed,
        "checkpoint": args.checkpoint,
        "corrector_steps": args.corrector_steps,
        "out_dir": str(out_dir),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print("=" * 60)
    print("Phase 1 ProSeCo-OWT Experiment")
    print("=" * 60)
    print(f"  Backend:  {'SURROGATE' if args.surrogate else 'PROSECO-OWT (co-trained backbone)'}")
    print(f"  T={args.T}, N={args.N}, M={args.M}, P={args.P}")
    print(f"  B values: {B_values}")
    print(f"  Corrector steps: {args.corrector_steps}")
    print(f"  Out dir:  {out_dir}")
    print()

    t_start = time.time()

    if args.surrogate:
        gen = SurrogateGenerator(
            T=args.T, sigma_gain=args.sigma_gain,
            gamma=args.gamma_surrogate, seed_base=args.seed,
        )
    else:
        if args.checkpoint is None:
            print("ERROR: --checkpoint required for ProSeCo run.")
            sys.exit(1)
        try:
            from mdm_playground.scheduling.backends.proseco_owt import ProSeCoOWTGenerator
            gen = ProSeCoOWTGenerator(
                checkpoint=args.checkpoint,
                T=args.T,
                corrector_steps=args.corrector_steps,
            )
            print(f"  {gen.corrector_description()}")
        except Exception as e:
            print(f"ERROR: Could not load ProSeCo-OWT backend: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    # --- Protocol A ---
    print("\nRunning Protocol A ...")
    all_records = run_protocol_a(
        gen, N=args.N, T=args.T, seed_base=args.seed,
        out_dir=out_dir / "protocol_a",
    )

    # --- Protocol B ---
    print("\nRunning Protocol B ...")
    schedule_records, pairs_records, delta_trace = run_protocol_b(
        gen, all_records=all_records,
        B_values=B_values, M=args.M, P=args.P,
        T=args.T, seed_base=args.seed,
        out_dir=out_dir / "protocol_b",
    )

    # --- Policy comparison ---
    print("\nRunning policy comparison ...")
    comparison_records = run_policy_comparison(
        gen, all_records=all_records, delta_trace=delta_trace,
        B_values=B_values, T=args.T, seed_base=args.seed,
        out_dir=out_dir / "policy_comparison",
    )

    # --- Summary ---
    print("\nComputing summary ...")
    try:
        from scipy.stats import spearmanr
    except ImportError:
        os.system(f"{sys.executable} -m pip install scipy -q")
    summary = compute_summary(
        all_records=all_records,
        schedule_records=schedule_records,
        pairs_records=pairs_records,
        comparison_records=comparison_records,
        B_values=B_values,
        T=args.T,
        out_dir=out_dir,
    )

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"\n{'=' * 60}")
    print("KEY RESULTS")
    print("=" * 60)
    print(f"  Steps with Δ_t > 0:  {summary['n_positive_delta_steps']} / {args.T}")
    print(f"  First positive step: {summary['t_first_positive_delta']}")
    print(f"  Peak mean Δ_t:       {summary['peak_mean_delta']:.5f}")
    cal = summary["calibration"]
    print(f"  ε (entropy, RMS):    {cal['entropy']['eps_rms']:.5f}")
    print(f"  ε (margin, RMS):     {cal['inverse_margin']['eps_rms']:.5f}")
    print(f"  Spearman(H, Δ):      {cal['entropy']['spearman_mean']:.3f} "
          f"± {cal['entropy']['spearman_std']:.3f}")
    print(f"  γ (95th pct):        {summary['gamma']['gamma_95']:.5f}")
    print()
    for B in B_values:
        chk = summary["theorem_A_bound_check"].get(str(B), {})
        print(f"  B={B}: 2Bε+2η = {chk.get('bound_2Be_2eta', '?'):.4f}  "
              f"G_oracle ≈ {chk.get('G_oracle_estimate', '?'):.4f}  "
              f"{'✓ non-vacuous' if chk.get('bound_useful') else '✗ vacuous'}")
    print()
    pol = summary.get("policy_comparison", {})
    print("  Policy comparison (G values):")
    for B_key, policies in pol.items():
        print(f"    {B_key}:")
        for policy, vals in sorted(policies.items(), key=lambda kv: -kv[1].get("G", 0)):
            beat = "✓" if vals.get("beats_uniform") else " "
            print(f"      {beat} {policy:30s} G={vals.get('G', 0):+.4f}")
    print("=" * 60)
    print(f"\nAll results saved in {out_dir}/")
    print("Next: python scripts/analyze_phase1_proseco.py --results_dir", out_dir)


if __name__ == "__main__":
    main()
