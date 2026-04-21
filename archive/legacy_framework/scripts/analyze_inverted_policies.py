#!/usr/bin/env python3
"""Offline analysis: test inverted signal policies on ProSeCo-OWT Protocol A data.

The Phase 1 pilot showed that uncertainty signals (entropy/margin/quality) are
NEGATIVELY correlated with Δ_t.  Signal top_B therefore selects WRONG steps.

This script tests:
  1. Inverted policies: entropy_BOT_B, margin_BOT_B, quality_BOT_B
     (select B steps with LOWEST signal = latest steps = high Δ_t region)
  2. New positional signals: unmasked_fraction_TOP_B
     (select B steps with HIGHEST fraction of committed tokens)
  3. Spearman correlation of ALL signals (including inverted) with Δ_t
  4. G = mean gain over N trajectories for each policy at B∈{4,8,16}

No new HPC run needed — uses existing results/phase1_proseco_owt/protocol_a/.
"""

import json
import pathlib
import numpy as np
from scipy import stats

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "phase1_proseco_owt"
TRAJ_DIR = RESULTS_DIR / "protocol_a"
B_VALUES = [4, 8, 16]


def load_trajectories():
    trajs = []
    for p in sorted(TRAJ_DIR.glob("trajectory_*.json")):
        trajs.append(json.loads(p.read_text()))
    return trajs


def policy_gain(per_t, B, step_selector):
    """Mean gain when corrector is applied at exactly B steps chosen by selector."""
    steps = step_selector(per_t, B)
    return sum(per_t[t]["delta"] for t in steps)


def top_B_by(per_t, B, key, descending=True):
    vals = [(per_t[t][key], t) for t in range(len(per_t))]
    vals.sort(reverse=descending)
    return [t for _, t in vals[:B]]


def make_policies(B):
    return {
        "uniform":              lambda pt, b: list(range(0, len(pt), max(1, len(pt)//b)))[:b],
        "front":                lambda pt, b: list(range(b)),
        "back":                 lambda pt, b: list(range(len(pt)-b, len(pt))),
        "middle":               lambda pt, b: list(range(len(pt)//2 - b//2, len(pt)//2 - b//2 + b)),
        # Original (wrong direction)
        "entropy_top_B":        lambda pt, b: top_B_by(pt, b, "entropy", descending=True),
        "margin_top_B":         lambda pt, b: top_B_by(pt, b, "inverse_margin", descending=True),
        "quality_top_B":        lambda pt, b: top_B_by(pt, b, "quality_mass_proxy", descending=True),
        # Inverted (correct direction)
        "entropy_bot_B":        lambda pt, b: top_B_by(pt, b, "entropy", descending=False),
        "margin_bot_B":         lambda pt, b: top_B_by(pt, b, "inverse_margin", descending=False),
        "quality_bot_B":        lambda pt, b: top_B_by(pt, b, "quality_mass_proxy", descending=False),
        # Unmasked-fraction signal (positively correlated with Δ_t)
        "unmasked_top_B":       lambda pt, b: top_B_by(pt, b, "unmasked_fraction", descending=True),
        # oracle
        "oracle":               lambda pt, b: top_B_by(pt, b, "delta", descending=True),
        "bottom_B":             lambda pt, b: top_B_by(pt, b, "delta", descending=False),
    }


def main():
    trajs = load_trajectories()
    N = len(trajs)
    T = len(trajs[0]["per_t"])
    print(f"Loaded {N} trajectories, T={T}\n")

    # ---- 1. Spearman per signal ----
    print("=" * 60)
    print("Spearman(signal, Δ_t) per trajectory — mean ± std")
    print("=" * 60)
    signals = ["entropy", "inverse_margin", "quality_mass_proxy", "unmasked_fraction"]
    spearman_results = {}
    for sig in signals:
        corrs = []
        for traj in trajs:
            pt = traj["per_t"]
            x = np.array([pt[t][sig] for t in range(T)])
            y = np.array([pt[t]["delta"] for t in range(T)])
            r, _ = stats.spearmanr(x, y)
            corrs.append(r)
        mean_r = np.mean(corrs)
        std_r = np.std(corrs)
        spearman_results[sig] = (mean_r, std_r)
        direction = "NEG (anti-corr)" if mean_r < 0 else "POS"
        useful = "✓ RIGHT direction" if mean_r > 0.10 else ("✓ RIGHT (inverted)" if mean_r < -0.10 else "~ weak")
        print(f"  {sig:<25} r={mean_r:+.4f} ± {std_r:.4f}  {direction}  {useful}")

    # ---- 2. Policy gains per B ----
    print()
    print("=" * 60)
    print("Policy G = mean total gain (sum of Δ_t at chosen B steps)")
    print("=" * 60)

    all_policy_G = {}
    for B in B_VALUES:
        policies = make_policies(B)
        print(f"\n  B={B}:")
        gains = {}
        for name, sel in policies.items():
            G_list = []
            for traj in trajs:
                pt = traj["per_t"]
                steps = sel(pt, B)
                G_list.append(sum(pt[t]["delta"] for t in steps))
            G_mean = np.mean(G_list)
            G_std  = np.std(G_list) / np.sqrt(N)
            gains[name] = (G_mean, G_std)
        uniform_G = gains["uniform"][0]
        oracle_G  = gains["oracle"][0]
        for name, (G, se) in sorted(gains.items(), key=lambda x: -x[1][0]):
            beats = "✓" if G > uniform_G and name != "uniform" else " "
            is_oracle = " [oracle]" if name == "oracle" else ""
            print(f"    {beats} {name:<25} G={G:+.4f} ± {se:.4f}{is_oracle}")
        all_policy_G[B] = gains

    # ---- 3. Summary table ----
    print()
    print("=" * 60)
    print("Summary: does the policy beat uniform?")
    print("=" * 60)
    header = f"{'Policy':<25}" + "".join(f"  B={B}" for B in B_VALUES)
    print(header)
    policy_names = list(make_policies(4).keys())
    for name in policy_names:
        row = f"{name:<25}"
        for B in B_VALUES:
            G = all_policy_G[B][name][0]
            uniform_G = all_policy_G[B]["uniform"][0]
            mark = "✓" if G > uniform_G and name != "uniform" else ("=" if name == "uniform" else "✗")
            row += f"  {mark}({G:+.3f})"
        print(row)

    # ---- 4. Mean Δ_t profile ----
    mean_delta = np.zeros(T)
    for traj in trajs:
        for t in range(T):
            mean_delta[t] += traj["per_t"][t]["delta"]
    mean_delta /= N

    print()
    print("Mean Δ_t profile (first 10 and last 10 steps):")
    for t in list(range(10)) + list(range(T-10, T)):
        bar = "█" * int(max(0, mean_delta[t]) * 20)
        print(f"  t={t:2d}  Δ={mean_delta[t]:+.4f}  {bar}")

    # ---- 5. Save compact results ----
    out = {
        "spearman": {s: {"mean": float(v[0]), "std": float(v[1])} for s, v in spearman_results.items()},
        "policy_G": {
            str(B): {name: {"G": float(v[0]), "se": float(v[1])}
                     for name, v in all_policy_G[B].items()}
            for B in B_VALUES
        },
        "mean_delta_profile": mean_delta.tolist(),
    }
    out_path = RESULTS_DIR / "inverted_policy_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
