"""Surrogate MDLM generator for pipeline testing and local surrogate runs.

The surrogate mimics key qualitative properties of real MDLM one-loop gain
dynamics without requiring the HPC or a GPU:

1. **Mid-trajectory peak.** Real masked diffusion correctors are most
   effective in the middle of the trajectory (moderate unmasked fraction),
   not at the very beginning (most tokens masked, context poor) or at the
   very end (most tokens fixed, little room to correct). The surrogate
   models Δ_t as a bell curve in t, peaking near 40-60% unmasked.

2. **Signal correlation.** Entropy H_t peaks near the start (maximum
   uncertainty), while the one-loop gain Δ_t peaks in the middle. This
   creates realistic *imperfect correlation* between entropy and gain,
   exactly the regime Theorem A's ε term captures.

3. **Approximate additivity.** G(S) ≈ ∑ Δ_t + η, where interaction
   residuals ξ_{t,t'} are drawn from a Gaussian with scale γ.

4. **Common random numbers.** Branch trajectories share the base seed;
   only the corrector step differs. This reduces Δ_t variance.

Parameters (set on construction):

- T          : int    — predictor steps (default 64 for fast local runs)
- D          : int    — sequence length (default 128 tokens)
- peak_frac  : float  — trajectory fraction where Δ_t peaks (default 0.55)
- sigma_gain : float  — noise on each Δ_t (default 0.005)
- gamma      : float  — pairwise interaction scale (default 0.008)
- signal_noise : float — noise added to each s_t signal (default 0.02)
- seed_base  : int    — master seed (default 0)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class SurrogateGenerator:
    """Surrogate MDLM-like generator for Phase 1 pipeline testing.

    All dynamics are analytic and CPU-cheap. The generator is
    deterministic given a seed.

    Usage
    -----
    gen = SurrogateGenerator(T=64, D=128)
    y_base = gen.run_base(seed=0)
    y_branch = gen.run_with_schedule(allocation={32: 1}, seed=0)
    """

    def __init__(
        self,
        T: int = 64,
        D: int = 128,
        peak_frac: float = 0.55,
        sigma_gain: float = 0.005,
        gamma: float = 0.008,
        signal_noise: float = 0.02,
        seed_base: int = 0,
    ):
        self.T = T
        self.D = D
        self.peak_frac = peak_frac
        self.sigma_gain = sigma_gain
        self.gamma = gamma
        self.signal_noise = signal_noise
        self.seed_base = seed_base

        # Pre-compute the deterministic per-step Δ_t profile (mean).
        self._delta_mean = self._compute_delta_mean()
        # Pre-compute per-step signal profiles.
        self._entropy_mean = self._compute_entropy_profile()
        self._margin_mean = self._compute_margin_profile()
        self._quality_mean = self._compute_quality_profile()

    # ------------------------------------------------------------------
    # Analytic profiles
    # ------------------------------------------------------------------

    def _compute_delta_mean(self) -> np.ndarray:
        """Mean one-loop marginal gain profile as function of step index."""
        t = np.linspace(0, 1, self.T)
        # Bell curve centred at peak_frac, width ~0.25
        mu = self.peak_frac
        sigma = 0.20
        gain = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
        # Scale so max gain ≈ 0.05 (plausible NLL improvement)
        gain = 0.050 * gain / gain.max()
        # Small non-negative floor
        return np.clip(gain, 0.002, None)

    def _compute_entropy_profile(self) -> np.ndarray:
        """Mean entropy signal: peaks near t=0 (all masked), decays toward end."""
        t = np.linspace(0, 1, self.T)
        # Monotone decrease + slight bump in middle
        base = 2.5 * (1.0 - t) ** 1.3 + 0.3
        return np.clip(base, 0.0, None)

    def _compute_margin_profile(self) -> np.ndarray:
        """Mean inverse-margin signal: low when entropy high, increases late."""
        t = np.linspace(0, 1, self.T)
        return np.clip(0.2 + 0.6 * t ** 1.2, 0.0, 1.0)

    def _compute_quality_profile(self) -> np.ndarray:
        """Mean quality-mass proxy: 1 − p_argmax, mirrors margin."""
        return np.clip(1.0 - self._compute_margin_profile() + 0.05, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Signal trace generation
    # ------------------------------------------------------------------

    def signal_trace(self, seed: int = 0) -> list[dict]:
        """Return a per-step list of signal dicts for one trajectory."""
        rng = np.random.default_rng(self.seed_base + seed * 100)
        trace = []
        for t in range(self.T):
            noise = rng.normal(0, self.signal_noise, 3)
            trace.append({
                "t": t,
                "entropy": float(max(0.0, self._entropy_mean[t] + noise[0])),
                "inverse_margin": float(
                    np.clip(self._margin_mean[t] + noise[1], 0.0, 1.0)
                ),
                "quality_mass_proxy": float(
                    np.clip(self._quality_mean[t] + noise[2], 0.0, 1.0)
                ),
                "unmasked_fraction": float((t + 1) / self.T),
            })
        return trace

    # ------------------------------------------------------------------
    # Trajectory runners (Generator protocol)
    # ------------------------------------------------------------------

    def run_base(self, seed: int = 0) -> dict:
        """Run base trajectory (no correction)."""
        rng = np.random.default_rng(self.seed_base + seed * 100 + 1)
        tokens = rng.integers(0, 100, size=self.D)
        # Synthetic NLL: roughly decreases across trajectory
        base_nll = 3.5 + rng.normal(0, 0.05)
        return {
            "tokens": tokens,
            "neg_nll": float(-base_nll),  # negative NLL; higher = better quality
            "per_step_signals": self.signal_trace(seed=seed),
            "seed": seed,
        }

    def run_branch(self, t_corrected: int, seed: int = 0) -> dict:
        """Run branch trajectory: one corrector loop at step t_corrected."""
        base = self.run_base(seed=seed)
        rng = np.random.default_rng(self.seed_base + seed * 100 + 7 + t_corrected)

        # The corrector changes some tokens at step t
        delta_mean = self._delta_mean[t_corrected]
        delta = float(delta_mean + rng.normal(0, self.sigma_gain))

        # Update neg_nll: +delta means NLL decreases (quality improves)
        new_neg_nll = base["neg_nll"] + delta

        # Token changes: TCR ∝ entropy at t (more entropy → more changes)
        entropy_t = base["per_step_signals"][t_corrected]["entropy"]
        tcr_mean = 0.05 + 0.15 * (entropy_t / 3.0)  # rough scale
        tcr = float(np.clip(tcr_mean + rng.normal(0, 0.02), 0.0, 1.0))
        n_changed = int(tcr * self.D)

        # Perturb n_changed tokens (use CRN: same tokens, just some positions differ)
        changed_idx = rng.choice(self.D, size=n_changed, replace=False)
        tokens_branch = base["tokens"].copy()
        tokens_branch[changed_idx] = rng.integers(0, 100, size=n_changed)

        return {
            "tokens": tokens_branch,
            "neg_nll": float(new_neg_nll),
            "delta_true": delta,
            "t_corrected": t_corrected,
            "per_step_signals": base["per_step_signals"],
            "seed": seed,
        }

    def run_with_schedule(
        self, allocation: Dict[int, int], seed: int = 0
    ) -> dict:
        """Run trajectory with a multi-step allocation.

        Models G(S) with pairwise interactions:
            G(S) = ∑_{t ∈ S} Δ_t + ∑_{{t,t'} ⊂ S} ξ_{t,t'}
        """
        base = self.run_base(seed=seed)
        steps = sorted(allocation.keys())

        # Sum individual gains
        total_delta = 0.0
        for t in steps:
            rng_t = np.random.default_rng(self.seed_base + seed * 100 + 7 + t)
            d_mean = self._delta_mean[t]
            total_delta += d_mean + rng_t.normal(0, self.sigma_gain)

        # Add pairwise interactions
        total_interaction = 0.0
        for i, t in enumerate(steps):
            for t2 in steps[i + 1:]:
                rng_pair = np.random.default_rng(
                    self.seed_base + seed * 100 + 999 + t * 1000 + t2
                )
                total_interaction += rng_pair.normal(0, self.gamma)

        G = total_delta + total_interaction
        new_neg_nll = base["neg_nll"] + G

        return {
            "tokens": base["tokens"].copy(),  # simplified: token content not critical
            "neg_nll": float(new_neg_nll),
            "G_true": G,
            "schedule_steps": steps,
            "seed": seed,
        }

    # ------------------------------------------------------------------
    # Ground-truth Δ_t (for analysis validation)
    # ------------------------------------------------------------------

    def true_delta_profile(self) -> np.ndarray:
        """Return the noiseless Δ_t profile (for comparison with estimates)."""
        return self._delta_mean.copy()
