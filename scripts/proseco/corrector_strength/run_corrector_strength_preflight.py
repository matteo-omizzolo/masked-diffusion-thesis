"""
run_corrector_strength_preflight.py
=====================================
Phase B preflight: test whether corrector strength variants produce
meaningfully different correction behaviors on ProSeCo-OWT.

Strength levels tested
----------------------
  no_correction  : corrector call is a no-op; Delta_t = G_pair = xi = 0 (sanity)
  strength_0     : corrector_steps=0  (single argmax pass, no annealed refinement)
  strength_1     : corrector_steps=1  (standard; must reproduce canonical values)
  strength_2     : corrector_steps=2  (two annealed refinement passes)

For each strength level this script measures, per seed × timestep:
  - f_base, f_branch_single (Delta_t)
  - corrector_n_changed : tokens flipped at the corrector call itself
  - final_n_changed     : Hamming(tokens_base, tokens_branch) at end of trajectory

And per seed × pair:
  - f_branch_pair, G_pair, A_pair, xi

Outputs: results/corrector_strength_preflight_<sha>/
  raw_deltas.json       — per (seed, t, strength) Delta_t + n_changed
  raw_pairs.json        — per (seed, pair_id, strength) G_pair + xi
  manifest.json         — run metadata + gate checks
  summary.json          — aggregate per-strength stats

Usage
-----
  # Debug (CPU, T=16, 3 pairs per seed, ~2 min locally)
  python3.11 scripts/proseco/corrector_strength/run_corrector_strength_preflight.py \\
      --checkpoint ~/mdm/checkpoints/proseco_owt --debug

  # Full preflight (GPU, T=64, 8 pairs per seed)
  python3.11 scripts/proseco/corrector_strength/run_corrector_strength_preflight.py \\
      --checkpoint ~/mdm/checkpoints/proseco_owt \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

XI_RAW_PATH = REPO_ROOT / "results" / "phase1_interaction_diag_nogit" / "xi_raw.json"
PROTOCOL_A_DIR = REPO_ROOT / "results" / "phase1_proseco_owt_full" / "protocol_a"

PREFLIGHT_SEEDS = [42, 43]
STRENGTH_LEVELS = ["no_correction", "strength_0", "strength_1", "strength_2"]
STRENGTH_CORRECTOR_STEPS = {
    "no_correction": None,
    "strength_0": 0,
    "strength_1": 1,
    "strength_2": 2,
}

# Canonical reproduction tolerance (Delta_t from strength_1 vs Protocol A)
CANONICAL_TOL = 1e-3


# ---------------------------------------------------------------------------
# TrackedGenerator wrapper
# ---------------------------------------------------------------------------

def _make_tracked_generator(checkpoint: str, T: int, corrector_steps: int, device: str):
    """Build a ProSeCoOWTGenerator subclass that tracks corrector change counts."""
    from mdm_playground.scheduling.backends.proseco_owt import ProSeCoOWTGenerator
    import torch

    class CorrectorStrengthGenerator(ProSeCoOWTGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._no_correction = False
            self._corrector_changes_list: list[int] = []

        def _reset_tracking(self) -> None:
            self._corrector_changes_list = []

        @torch.no_grad()
        def _apply_corrector(self, x, t, p_x0=None):
            if self._no_correction:
                return x
            x_before = x.clone()
            x_after = super()._apply_corrector(x, t, p_x0=p_x0)
            n = int((x_after != x_before).sum().item())
            self._corrector_changes_list.append(n)
            return x_after

        def run_branch(self, t_corrected: int, seed: int = 0, return_trace: bool = False):
            self._reset_tracking()
            result = super().run_branch(t_corrected, seed=seed, return_trace=return_trace)
            result["corrector_n_changed"] = (
                self._corrector_changes_list[0] if self._corrector_changes_list else 0
            )
            return result

        def run_with_schedule(self, allocation, seed: int = 0, return_trace: bool = False):
            self._reset_tracking()
            result = super().run_with_schedule(allocation, seed=seed, return_trace=return_trace)
            result["corrector_n_changed_list"] = list(self._corrector_changes_list)
            return result

    return CorrectorStrengthGenerator(
        checkpoint=checkpoint,
        T=T,
        corrector_steps=corrector_steps,
        device=device,
    )


# ---------------------------------------------------------------------------
# Preflight pair selection
# ---------------------------------------------------------------------------

def select_preflight_pairs(
    seed: int,
    xi_raw: list[dict[str, Any]],
    n_per_stratum: int = 3,
) -> list[dict[str, Any]]:
    """Pick pairs for the preflight: n_per_stratum from low/mid/high A_pair quartiles.

    Selection is fully deterministic: sort each stratum by distance, pick
    evenly spaced indices. This guarantees the same pairs are selected every run.
    """
    seed_rows = [r for r in xi_raw if r["seed"] == seed]
    a = np.array([r["A_pair"] for r in seed_rows])
    q33, q67 = float(np.percentile(a, 33)), float(np.percentile(a, 67))

    def pick(stratum: list[dict]) -> list[dict]:
        stratum = sorted(stratum, key=lambda r: r["distance"])
        n = len(stratum)
        if n <= n_per_stratum:
            return stratum
        idxs = np.round(np.linspace(0, n - 1, n_per_stratum)).astype(int)
        return [stratum[i] for i in idxs]

    low = [r for r in seed_rows if r["A_pair"] < q33]
    mid = [r for r in seed_rows if q33 <= r["A_pair"] < q67]
    high = [r for r in seed_rows if r["A_pair"] >= q67]
    selected = pick(low) + pick(mid) + pick(high)

    for i, r in enumerate(selected):
        r = dict(r)
        r["pair_id"] = i
        selected[i] = r
    return selected


def get_needed_timesteps(pairs: list[dict]) -> list[int]:
    """All unique timesteps needed for single-step branches."""
    ts = set()
    for p in pairs:
        ts.add(p["t"])
        ts.add(p["t_prime"])
    return sorted(ts)


# ---------------------------------------------------------------------------
# Canonical Delta_t loading (for reproduction check)
# ---------------------------------------------------------------------------

def load_canonical_deltas(seed: int, proto_a_dir: Path) -> dict[int, float] | None:
    """Load Protocol A Delta_t values for a given seed. Returns None if unavailable."""
    seed_map = {}
    for f in proto_a_dir.glob("trajectory_*.json"):
        try:
            d = json.loads(f.read_text())
            seed_map[d["seed"]] = d
        except Exception:
            continue
    if seed not in seed_map:
        return None
    rows = seed_map[seed]["per_t"]
    return {int(r["t"]): float(r["delta"]) for r in rows}


# ---------------------------------------------------------------------------
# Core preflight run
# ---------------------------------------------------------------------------

def run_preflight(
    checkpoint: str,
    out_dir: Path,
    *,
    debug: bool = False,
    device: str = "cuda",
    seeds: list[int] | None = None,
    strength_levels: list[str] | None = None,
) -> dict[str, Any]:
    seeds = seeds or PREFLIGHT_SEEDS
    strength_levels = strength_levels or STRENGTH_LEVELS
    T = 16 if debug else 64
    n_per_stratum = 1 if debug else 3

    out_dir.mkdir(parents=True, exist_ok=True)
    sha = _get_short_sha()

    print(f"[preflight] T={T}, seeds={seeds}, strengths={strength_levels}, debug={debug}")
    print(f"[preflight] device={device}, out={out_dir}")

    # Load xi_raw for pair selection + canonical comparison
    xi_raw = json.loads(XI_RAW_PATH.read_text()) if XI_RAW_PATH.exists() else []
    if not xi_raw:
        raise FileNotFoundError(f"xi_raw.json not found at {XI_RAW_PATH}")

    # Select preflight pairs per seed
    seed_pairs: dict[int, list[dict]] = {}
    seed_timesteps: dict[int, list[int]] = {}
    for seed in seeds:
        pairs = select_preflight_pairs(seed, xi_raw, n_per_stratum=n_per_stratum)
        seed_pairs[seed] = pairs
        seed_timesteps[seed] = get_needed_timesteps(pairs)
        print(f"  seed {seed}: {len(pairs)} pairs, {len(seed_timesteps[seed])} unique timesteps")

    # Load canonical deltas for reproduction check
    canonical: dict[int, dict[int, float]] = {}
    for seed in seeds:
        cd = load_canonical_deltas(seed, PROTOCOL_A_DIR)
        if cd:
            canonical[seed] = cd
            print(f"  seed {seed}: canonical deltas loaded ({len(cd)} steps)")
        else:
            print(f"  seed {seed}: canonical deltas not available (reproduction check skipped)")

    # Load model ONCE with corrector_steps=1 (standard); we'll swap corrector_steps
    print(f"\n[preflight] Loading model...", flush=True)
    t0 = time.time()
    gen = _make_tracked_generator(checkpoint, T=T, corrector_steps=1, device=device)
    print(f"[preflight] Model loaded in {time.time()-t0:.1f}s", flush=True)

    raw_deltas: list[dict] = []
    raw_pairs: list[dict] = []
    t_start_total = time.time()

    for strength_name in strength_levels:
        k = STRENGTH_CORRECTOR_STEPS[strength_name]
        print(f"\n[preflight] Strength: {strength_name} (corrector_steps={k})", flush=True)

        # Configure strength
        if k is None:
            gen._no_correction = True
            gen.corrector_steps = 1  # doesn't matter, corrector is disabled
        else:
            gen._no_correction = False
            gen.corrector_steps = k

        for seed in seeds:
            pairs = seed_pairs[seed]
            timesteps = seed_timesteps[seed]

            # --- Base run (same for all strengths, recomputed for CRN audit) ---
            t0 = time.time()
            y_base = gen.run_base(seed=seed)
            f_base = float(y_base["neg_nll"])
            t_base = time.time() - t0
            print(f"  seed {seed}: f_base={f_base:.4f} ({t_base:.1f}s)", flush=True)

            # --- Single-step branches ---
            delta_by_t: dict[int, float] = {}
            corrector_nch_by_t: dict[int, int] = {}
            final_nch_by_t: dict[int, int] = {}
            tokens_base = np.asarray(y_base["tokens"])

            for t_step in timesteps:
                t0 = time.time()
                y_branch = gen.run_branch(t_corrected=t_step, seed=seed)
                f_branch = float(y_branch["neg_nll"])
                delta = f_branch - f_base
                corr_n = int(y_branch.get("corrector_n_changed", 0))
                final_n = int((tokens_base != np.asarray(y_branch["tokens"])).sum())

                delta_by_t[t_step] = delta
                corrector_nch_by_t[t_step] = corr_n
                final_nch_by_t[t_step] = final_n

                raw_deltas.append({
                    "seed": seed,
                    "t": t_step,
                    "strength": strength_name,
                    "corrector_steps": k,
                    "f_base": f_base,
                    "f_branch": f_branch,
                    "delta_t": delta,
                    "corrector_n_changed": corr_n,
                    "final_n_changed": final_n,
                    "wall_s": time.time() - t0,
                })

            # --- Pair branches ---
            for pair in pairs:
                t_a, t_b = pair["t"], pair["t_prime"]
                t0 = time.time()
                alloc = {t_a: 1, t_b: 1}
                y_pair = gen.run_with_schedule(allocation=alloc, seed=seed)
                f_pair = float(y_pair["neg_nll"])
                G_pair = f_pair - f_base
                delta_a = delta_by_t.get(t_a, float("nan"))
                delta_b = delta_by_t.get(t_b, float("nan"))
                A_pair = delta_a + delta_b
                xi = G_pair - A_pair
                nch_list = y_pair.get("corrector_n_changed_list", [])
                corr_nch_total = sum(nch_list)
                final_nch_pair = int((tokens_base != np.asarray(y_pair["tokens"])).sum())

                raw_pairs.append({
                    "seed": seed,
                    "pair_id": pair["pair_id"],
                    "t": t_a,
                    "t_prime": t_b,
                    "phase_t": pair["phase_t"],
                    "phase_tp": pair["phase_tp"],
                    "distance": pair["distance"],
                    "strength": strength_name,
                    "corrector_steps": k,
                    "f_base": f_base,
                    "f_pair": f_pair,
                    "G_pair": G_pair,
                    "delta_t": delta_a,
                    "delta_tp": delta_b,
                    "A_pair": A_pair,
                    "xi": xi,
                    "corrector_n_changed_total": corr_nch_total,
                    "corrector_n_changed_list": nch_list,
                    "final_n_changed": final_nch_pair,
                    "canonical_G_pair": pair.get("G_pair"),
                    "canonical_A_pair": pair.get("A_pair"),
                    "canonical_xi": pair.get("xi"),
                    "wall_s": time.time() - t0,
                })

    total_wall = time.time() - t_start_total
    print(f"\n[preflight] Total wall time: {total_wall:.1f}s", flush=True)

    # --- Save raw data ---
    _write_json(out_dir / "raw_deltas.json", raw_deltas)
    _write_json(out_dir / "raw_pairs.json", raw_pairs)

    # --- Compute summary + gate checks ---
    summary = _compute_summary(raw_deltas, raw_pairs, canonical, strength_levels)
    summary["sha"] = sha
    summary["T"] = T
    summary["debug"] = debug
    summary["seeds"] = seeds
    summary["total_wall_s"] = total_wall
    summary["n_deltas"] = len(raw_deltas)
    summary["n_pairs"] = len(raw_pairs)
    _write_json(out_dir / "summary.json", summary)

    # --- Manifest ---
    manifest = {
        "sha": sha,
        "T": T,
        "debug": debug,
        "seeds": seeds,
        "strength_levels": strength_levels,
        "checkpoint": str(checkpoint),
        "device": device,
        "total_wall_s": total_wall,
        "gate_pass": summary["gate_pass"],
    }
    _write_json(out_dir / "manifest.json", manifest)

    # --- Print gate summary ---
    _print_gate_summary(summary)

    return {
        "out_dir": str(out_dir),
        "summary": summary,
        "gate_pass": summary["gate_pass"],
    }


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_summary(
    raw_deltas: list[dict],
    raw_pairs: list[dict],
    canonical: dict[int, dict[int, float]],
    strength_levels: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    gates: dict[str, dict] = {}

    # Per-strength aggregate stats
    per_strength: dict[str, dict] = {}
    for sl in strength_levels:
        d_rows = [r for r in raw_deltas if r["strength"] == sl]
        p_rows = [r for r in raw_pairs if r["strength"] == sl]

        if not d_rows:
            per_strength[sl] = {"n": 0}
            continue

        deltas = np.array([r["delta_t"] for r in d_rows])
        corr_nch = np.array([r["corrector_n_changed"] for r in d_rows])
        final_nch = np.array([r["final_n_changed"] for r in d_rows])

        p_stats: dict[str, float] = {}
        if p_rows:
            G = np.array([r["G_pair"] for r in p_rows])
            xi = np.array([r["xi"] for r in p_rows if not (
                isinstance(r["A_pair"], float) and (
                    r["A_pair"] != r["A_pair"]))])  # skip NaN A_pair
            A = np.array([r["A_pair"] for r in p_rows])
            p_stats = {
                "mean_G_pair": float(G.mean()),
                "mean_A_pair": float(A.mean()),
                "mean_xi": float(xi.mean()) if len(xi) else float("nan"),
                "p_xi_pos": float((xi > 0).mean()) if len(xi) else float("nan"),
            }

        per_strength[sl] = {
            "n_delta_rows": len(d_rows),
            "n_pair_rows": len(p_rows),
            "mean_delta_t": float(deltas.mean()),
            "mean_abs_delta_t": float(np.abs(deltas).mean()),
            "mean_corrector_n_changed": float(corr_nch.mean()),
            "mean_final_n_changed": float(final_nch.mean()),
            **p_stats,
        }

    summary["per_strength"] = per_strength

    # Gate 1: canonical reproduction (strength_1 vs Protocol A)
    std1_rows = [r for r in raw_deltas if r["strength"] == "strength_1"]
    canon_errors: list[float] = []
    canon_checked = False
    for r in std1_rows:
        s, t = r["seed"], r["t"]
        if s in canonical and t in canonical[s]:
            err = abs(r["delta_t"] - canonical[s][t])
            canon_errors.append(err)
            canon_checked = True

    if canon_checked and canon_errors:
        max_err = float(max(canon_errors))
        gates["G1_canonical_reproduction"] = {
            "pass": max_err < CANONICAL_TOL,
            "max_abs_error": max_err,
            "n_checked": len(canon_errors),
            "threshold": CANONICAL_TOL,
        }
    else:
        gates["G1_canonical_reproduction"] = {
            "pass": None,  # cannot check
            "note": "canonical deltas not available",
        }

    # Gate 2: strength variants are non-trivial (strength_0/1/2 have corr_n_changed > 0)
    # Check that standard (strength_1) has non-zero corrector changes at some steps
    std1_nch = np.array([r["corrector_n_changed"] for r in raw_deltas
                         if r["strength"] == "strength_1"])
    gates["G2_standard_nontrivial"] = {
        "pass": bool(len(std1_nch) > 0 and std1_nch.max() > 0),
        "max_corrector_n_changed_strength_1": int(std1_nch.max()) if len(std1_nch) else 0,
    }

    # Gate 3: strength_0 and strength_2 differ from strength_1 in corrector changes
    s0_nch = np.array([r["corrector_n_changed"] for r in raw_deltas
                       if r["strength"] == "strength_0"])
    s2_nch = np.array([r["corrector_n_changed"] for r in raw_deltas
                       if r["strength"] == "strength_2"])
    s1_nch = std1_nch

    # Check ordering: mean_corrector_n_changed: s0 <= s1 <= s2 (expected)
    # OR that they simply differ
    s0_mean = float(s0_nch.mean()) if len(s0_nch) else float("nan")
    s1_mean = float(s1_nch.mean()) if len(s1_nch) else float("nan")
    s2_mean = float(s2_nch.mean()) if len(s2_nch) else float("nan")
    variants_differ = not (
        abs(s0_mean - s1_mean) < 0.5 and abs(s2_mean - s1_mean) < 0.5
    )
    gates["G3_strength_variants_differ"] = {
        "pass": variants_differ,
        "mean_corrector_n_changed_s0": s0_mean,
        "mean_corrector_n_changed_s1": s1_mean,
        "mean_corrector_n_changed_s2": s2_mean,
        "note": "variants are non-idempotent if they differ",
    }

    # Gate 4: no_correction is a true no-op (all Delta_t = 0 for no_correction)
    no_corr_deltas = np.array([abs(r["delta_t"]) for r in raw_deltas
                               if r["strength"] == "no_correction"])
    gates["G4_no_correction_noop"] = {
        "pass": bool(len(no_corr_deltas) == 0 or no_corr_deltas.max() < 1e-6),
        "max_abs_delta_no_correction": float(no_corr_deltas.max()) if len(no_corr_deltas) else 0.0,
    }

    # Gate 5: CRN confirmed — f_base is same across all strengths for same seed
    crn_ok = True
    for seed in set(r["seed"] for r in raw_deltas):
        bases = [r["f_base"] for r in raw_deltas if r["seed"] == seed]
        if bases and (max(bases) - min(bases)) > 1e-6:
            crn_ok = False
            break
    gates["G5_crn_consistent"] = {
        "pass": crn_ok,
    }

    # Overall gate
    gate_pass_flags = [g["pass"] for g in gates.values() if g.get("pass") is not None]
    summary["gates"] = gates
    summary["gate_pass"] = all(gate_pass_flags) if gate_pass_flags else False

    return summary


def _print_gate_summary(summary: dict[str, Any]) -> None:
    print("\n=== PREFLIGHT GATE RESULTS ===")
    for gname, gval in summary["gates"].items():
        p = gval.get("pass")
        status = "PASS" if p is True else ("SKIP" if p is None else "FAIL")
        print(f"  [{status}] {gname}")
        for k, v in gval.items():
            if k != "pass":
                print(f"           {k}: {v}")
    overall = "PREFLIGHT PASS" if summary["gate_pass"] else "PREFLIGHT FAIL"
    print(f"\n  === {overall} ===")

    print("\n=== PER-STRENGTH SUMMARY ===")
    for sl, stats in summary["per_strength"].items():
        if stats.get("n_delta_rows", 0) == 0:
            continue
        print(f"  {sl}:")
        for k in ["mean_delta_t", "mean_corrector_n_changed", "mean_final_n_changed",
                  "mean_G_pair", "mean_xi", "p_xi_pos"]:
            v = stats.get(k)
            if v is not None:
                print(f"    {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_short_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, cwd=str(REPO_ROOT))
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"Not JSON-serializable: {type(o)}")


# ---------------------------------------------------------------------------
# Surrogate mode (for local unit testing without model)
# ---------------------------------------------------------------------------

def _run_surrogate_preflight(out_dir: Path, seeds: list[int], debug: bool) -> dict[str, Any]:
    """Use the surrogate generator for pipeline tests (no GPU/model required)."""
    from mdm_playground.scheduling.surrogate import SurrogateGenerator
    T = 8 if debug else 16
    sha = _get_short_sha()
    out_dir.mkdir(parents=True, exist_ok=True)

    xi_raw = json.loads(XI_RAW_PATH.read_text()) if XI_RAW_PATH.exists() else []
    n_per_stratum = 1

    raw_deltas: list[dict] = []
    raw_pairs: list[dict] = []

    for sl in STRENGTH_LEVELS:
        k = STRENGTH_CORRECTOR_STEPS[sl]
        gen = SurrogateGenerator(T=T, seed_base=k or 0)

        for seed in seeds:
            if xi_raw:
                # Filter xi_raw to only pairs with timesteps valid for this T
                xi_valid = [r for r in xi_raw
                            if r["t"] < T and r["t_prime"] < T]
                pairs = select_preflight_pairs(seed, xi_valid, n_per_stratum=n_per_stratum)
                pairs = pairs[:3]
            else:
                pairs = [{"t": 2, "t_prime": 5, "phase_t": "early", "phase_tp": "middle",
                          "distance": 3, "pair_id": 0, "A_pair": 0.2, "xi": -0.05,
                          "G_pair": 0.15}]
            timesteps = get_needed_timesteps(pairs)

            y_base = gen.run_base(seed=seed)
            f_base = float(y_base["neg_nll"])

            delta_by_t: dict[int, float] = {}
            for t_step in timesteps:
                y_br = gen.run_branch(t_corrected=t_step, seed=seed)
                d = float(y_br["neg_nll"]) - f_base
                delta_by_t[t_step] = d
                raw_deltas.append({
                    "seed": seed, "t": t_step, "strength": sl, "corrector_steps": k,
                    "f_base": f_base, "f_branch": float(y_br["neg_nll"]),
                    "delta_t": d,
                    "corrector_n_changed": 0 if k is None else (k + 1) * 10,
                    "final_n_changed": abs(int(d * 100)),
                    "wall_s": 0.0,
                })

            for pair in pairs:
                t_a, t_b = pair["t"], pair["t_prime"]
                alloc = {t_a: 1, t_b: 1}
                y_pr = gen.run_with_schedule(allocation=alloc, seed=seed)
                f_pair = float(y_pr["neg_nll"])
                G = f_pair - f_base
                A = delta_by_t.get(t_a, 0.0) + delta_by_t.get(t_b, 0.0)
                xi = G - A
                raw_pairs.append({
                    "seed": seed, "pair_id": pair["pair_id"],
                    "t": t_a, "t_prime": t_b,
                    "phase_t": pair["phase_t"], "phase_tp": pair["phase_tp"],
                    "distance": pair["distance"],
                    "strength": sl, "corrector_steps": k,
                    "f_base": f_base, "f_pair": f_pair,
                    "G_pair": G, "delta_t": delta_by_t.get(t_a, 0.0),
                    "delta_tp": delta_by_t.get(t_b, 0.0),
                    "A_pair": A, "xi": xi,
                    "corrector_n_changed_total": 0,
                    "corrector_n_changed_list": [],
                    "final_n_changed": abs(int(G * 100)),
                    "canonical_G_pair": pair.get("G_pair"),
                    "canonical_A_pair": pair.get("A_pair"),
                    "canonical_xi": pair.get("xi"),
                    "wall_s": 0.0,
                })

    _write_json(out_dir / "raw_deltas.json", raw_deltas)
    _write_json(out_dir / "raw_pairs.json", raw_pairs)

    summary = _compute_summary(raw_deltas, raw_pairs, {}, STRENGTH_LEVELS)
    summary.update({"sha": sha, "T": T, "debug": debug, "seeds": seeds,
                    "surrogate": True, "n_deltas": len(raw_deltas),
                    "n_pairs": len(raw_pairs)})
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "manifest.json", {"sha": sha, "surrogate": True,
                                             "gate_pass": summary["gate_pass"]})
    _print_gate_summary(summary)
    return {"out_dir": str(out_dir), "summary": summary, "gate_pass": summary["gate_pass"]}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B corrector-strength preflight")
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path.home() / "mdm/checkpoints/proseco_owt"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true",
                        help="T=16, 1 pair per stratum (~2 min on GPU or surrogate)")
    parser.add_argument("--surrogate", action="store_true",
                        help="Use surrogate generator (no model/GPU needed)")
    parser.add_argument("--seeds", type=int, nargs="+", default=PREFLIGHT_SEEDS)
    parser.add_argument("--strengths", type=str, nargs="+", default=STRENGTH_LEVELS)
    args = parser.parse_args()

    sha = _get_short_sha()
    if args.out_dir is None:
        args.out_dir = REPO_ROOT / "results" / f"corrector_strength_preflight_{sha}"

    if args.surrogate:
        result = _run_surrogate_preflight(args.out_dir, args.seeds, debug=args.debug)
    else:
        result = run_preflight(
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            debug=args.debug,
            device=args.device,
            seeds=args.seeds,
            strength_levels=args.strengths,
        )

    sys.exit(0 if result["gate_pass"] else 1)


if __name__ == "__main__":
    main()
