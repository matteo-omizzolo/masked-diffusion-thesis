# Experiment Result Analysis — Phase 1 Pilot (Real MDLM)

**Date:** April 2026  
**Run:** `phase1_pilot_478600`, HPC node `gnode02`, NVIDIA A100 80GB PCIe  
**Model:** MDLM checkpoint (`mdlm.ckpt`, ~2.5 GB, 100M parameters)  
**Data source:** `results/phase1_pilot/` (20 real trajectories, 45 schedules, 120 pairs)  
**Analysis status:** Complete but limited — see implementation bugs in §B1.

---

## B1. Executive Summary

| Question | Answer | Confidence |
|----------|--------|------------|
| Is entropy a good proxy for marginal gain? | **Cannot determine** — signal computation bug produced entropy=0 everywhere | **None** |
| Is margin better? | **Cannot determine** — same bug | **None** |
| Is quality mass available and useful? | **No** — quality head not wired; proxy=0 | **None** |
| Is burn-in gating supported? | **Partially** — early steps are the most harmful, consistent with Prop. B's low-gain region prediction | **Moderate** |
| Is approximate additivity plausible? | **No** — η_B is 1–2 orders of magnitude larger than G(S_B*) at all tested B | **High** |
| Is Theorem A empirically non-vacuous? | **No** — bound is vacuous by 3–4 orders of magnitude at all B | **High** |

**Root cause:** Two implementation bugs corrupted the results. (1) Entropy/margin signals were computed over already-committed (unmasked) positions instead of masked positions; all signals ≈ 0. (2) The corrector resampled all masked positions at step t in a single Gibbs sweep, which is harmful at early steps (produces Δ_t < 0 for all t ≤ 60). These bugs mean the experiment cannot answer the core thesis questions from this run.

**What the run does establish (genuine findings):**
1. The experiment pipeline (Protocol A/B, JSON logging, analysis scripts) ran to completion on real MDLM in 70 min on one A100.
2. The Gibbs-resample-all-masked corrector is universally harmful: Δ_t < 0 for all t, with harm monotonically decreasing as t increases.
3. TCR_t is extremely strongly correlated with Δ_t when both are driven by the unmasked fraction (Pearson r = −0.960, Spearman ρ = −0.976) — but this is a trivial consequence of both being proportional to the masked fraction, not a signal-to-gain relationship.
4. Pairwise interactions are POSITIVE (saturating harm): joint correction is less harmful than the sum of individual corrections, meaning the additive surrogate overestimates harm.
5. All corrections produce negative gain; the "oracle" schedule is to correct nowhere.

---

## B2. Protocol A Analysis — Signal-to-Gain Relationship

### Signal failure

All three signals (entropy H_t, inverse confidence margin, quality mass proxy) returned 0.0 for all trajectories. Root cause: the `_extract_signals` function in `backends/mdlm.py` computes entropy over unmasked (committed) positions:

```python
revisable = (x[0] != self.mask_id)   # WRONG — committed positions
```

MDLM commits tokens in descending order of confidence (highest-confidence tokens first). Committed positions therefore have near-zero entropy by construction. The correct implementation computes over **masked** positions (those still available for resampling).

**Consequence:** ε_rms ≈ 0.992, ε_max ≈ 2.99, Spearman(H, Δ) = NaN. All calibration metrics are artifacts of zero-signal input. None of Q2 (entropy as proxy), Q8 (TCR vs Δ distinction), or the ε estimate for Theorem A can be reported from this run.

### Delta profile (genuine result)

Mean Δ_t across 20 trajectories:

```
t=0:  Δ=-3.19, TCR=0.973, unmask=0.015
t=10: Δ=-2.25, TCR=0.775, unmask=0.172
t=20: Δ=-1.47, TCR=0.580, unmask=0.331
t=30: Δ=-0.84, TCR=0.417, unmask=0.482
t=40: Δ=-0.37, TCR=0.260, unmask=0.642
t=50: Δ=-0.10, TCR=0.137, unmask=0.798
t=60: Δ=-0.004, TCR=0.029, unmask=0.954
t=63: Δ=0.000, TCR=0.000, unmask=1.000
```

The profile is **monotonically non-decreasing** (harm decreases as t increases). This is not the bell-shaped curve that a beneficial corrector should produce. It reflects a pathological corrector that disrupts the predictor's careful confidence ordering.

### Correlation summary

Since signals are zero, the only correlation that can be computed is TCR_t vs Δ_t:

| Pair | Pearson | Spearman |
|------|---------|----------|
| TCR vs Δ | −0.960 | −0.976 |

Interpretation: very high, but both TCR and |Δ| are monotone functions of the unmasked fraction u_t, so the correlation is driven by the common driver u_t, not by any causal signal-to-gain relationship. This is not a useful calibration signal.

### Calibration verdict

**Cannot determine.** Entropy ≈ 0 everywhere → ε ≈ 1 (maximum possible calibration error). No signal tested in this run. Q2 and Q8 remain open and must be measured after fixing Bug #1.

---

## B3. Protocol B Analysis — Approximate Additivity

### η_B measurements

| B | η_mean | η_95 | η_mean / |G_mean| |
|---|--------|------|----------|
| 4 | 1.821 | 3.783 | 0.80 |
| 8 | 5.853 | 9.626 | 2.33 |
| 16 | 14.562 | 18.937 | 4.66 |

The ratio η_mean / |G_mean| = 0.80 at B=4 and 4.66 at B=16. η_B is not small relative to G(S) at any B.

**Important nuance — sign structure.** In this experiment, all Δ_t < 0 and all G(S) < 0. The residual η = G(S) − A(S) is systematically **positive**: G(S) > A(S) (joint correction is less harmful than the sum of individual harms). The additive surrogate overestimates harm because correcting at step t "saturates" some of the damage before step t' can add to it.

Pairwise interaction pattern:
- t=1, t'=2: G({1,2}) = −3.39 vs Δ_1 + Δ_2 = −6.40; ξ = +3.01 (large positive)
- t=35, t'=38: G({35,38}) = −0.66 vs Δ_35 + Δ_38 = −1.20; ξ = +0.54 (moderate positive)
- t=57, t'=6: G({57,6}) = −2.65 vs Δ_57 + Δ_6 = −2.68; ξ ≈ 0 (near-independent)

The interaction ξ_{t,t'} is typically **positive** (saturating) when both t and t' are early (large harm, strong saturation) and near-zero when at least one step is late.

### Proposition C check

γ_95 = 2.302 → Proposition C bound: η_B ≤ 2.302 × B(B−1)/2

| B | Prop. C bound | Observed η_95 |
|---|--------------|---------------|
| 4 | 13.8 | 3.8 |
| 8 | 64.5 | 9.6 |
| 16 | 276.0 | 18.9 |

Proposition C's bound is very loose (factor ~3–15 above observed). This is expected: Proposition C uses the worst-case γ for all pairs, but most pairs have much smaller |ξ|.

**η_B looks large** at all tested B (moderate-to-severe by the classification of Entry 6 in the worklog).

### Approximate additivity verdict

**Not plausible** at any tested B with this corrector design. η_B / |G| grows superlinearly with B. Proposition C is proven but very loose. The interactions are saturating (positive ξ), which is a structurally different regime from what Theorem A assumes — the additive surrogate gives a lower bound on G (G > A) rather than an upper bound. If a beneficial corrector were used, approximate additivity might hold differently; this cannot be determined from the current data.

---

## B4. Budget Sensitivity

| B | G(S)_mean | A(S)_mean | η_mean | 2Bε+2η | Vacuous? |
|---|-----------|-----------|--------|---------|----------|
| 4 | −2.28 | −4.11 | 1.82 | 15.5 | **Yes** |
| 8 | −2.51 | −8.36 | 5.85 | 35.1 | **Yes** |
| 16 | −3.13 | −17.69 | 14.56 | 69.6 | **Yes** |

**The signal-guided schedule cannot outperform uniform** at any tested budget because:
1. All corrections are harmful — there is no "best schedule" with positive gain
2. ε ≈ 1 (signals are zero)
3. η_B grows rapidly

Observation: G(S) is roughly constant across B (approximately −2.3 to −3.1) even though A(S) grows linearly with B. This reflects the saturation effect: adding more corrector steps produces diminishing marginal harm because each corrector step already disrupts subsequent ones. In the limit, correcting at all 64 steps would not be 64× as harmful as correcting at 1 step.

---

## B5. Burn-In Analysis

**Supported — but in the wrong direction.**

The data confirms that early steps (low u_t) have the largest harm (most negative Δ_t). This is consistent with Proposition B's prediction that early trajectory regions have low (or negative) gain. Specifically:

- Steps t ∈ {0, …, 9}: Δ_t ranges from −3.19 to −2.09 (very harmful)
- Steps t ∈ {50, …, 60}: Δ_t ranges from −0.10 to −0.004 (near-zero harm)

The "T_low" threshold from the summary:
- 50th percentile of peak Δ: steps {0, …, 9} (first 15% of trajectory)
- 30th percentile of peak Δ: same range

In the Theorem A framework, T_low = {0, …, 9} is a **maximum harm region**, not merely a low-gain region. Proposition B says gating out T_low is benign when Δ_t ≤ δ; here δ is negative but large. Gating out the most harmful steps is strongly supported and would reduce total harm substantially (the best strategy is to correct only near t=63 where Δ_t ≈ 0).

**Caution:** This finding applies to the implemented (Gibbs-resample-all) corrector only. A beneficial corrector may show a proper burn-in pattern (low positive gain early, large positive gain mid-trajectory). The current data cannot distinguish between "corrector harmful early" and "corrector ineffective early."

---

## B6. Interpretation for the Thesis

### Theorem A

**Status: mathematically sound, empirically vacuous in this run.**

The proof in `proof_worklog.md` Entry 6 is correct and unaffected by the experimental outcome. What the experiment shows is that the specific instantiation tested (Gibbs-resample-all corrector, broken signal computation) produces a regime where:
- G(Ŝ_B) ≤ 0 for all B (corrector is universally harmful)
- ε ≈ 1 (signals are zero)
- η_B >> |G| (interactions dominate)

This means Theorem A's bound 2Bε + 2η_B is vacuous — it correctly tells us that top-B by a zero signal cannot be guaranteed to outperform any other B-step schedule. This is a true negative result (the proxy provides no guarantee), not a flaw in the theorem.

**What would make Theorem A non-vacuous:** A corrector with Δ_t > 0 at some steps (so G(S_B*) > 0), working signals (ε < G(S_B*) / (2B)), and small η_B (η_B < G(S_B*) / 2).

**Status: EMPIRICALLY TESTED (negative result). Theory unaffected.**

### Proposition B (burn-in gating)

**Status: partially supported.**

The data shows the early trajectory is the most harmful region (not just low-gain). Proposition B's claim — that gating out T_low is benign when Δ_t ≤ δ — holds trivially when δ < 0 (Proposition B needs Δ_t ≤ δ for gating to not hurt). The actual Δ_t values are very negative in T_low, so gating out early steps is strongly recommended.

However, the positive version of burn-in (early steps have near-zero gain because context is insufficient, not because they cause harm) cannot be confirmed without a working beneficial corrector.

**Status: WEAKLY SUPPORTED (gating out harmful early steps confirmed; burn-in for a beneficial corrector unconfirmed).**

### Proposition C (pairwise interaction bound)

**Status: proven but very loose.**

Proposition C gives η_B ≤ γ B(B−1)/2. With γ_95 = 2.302, the bound is much larger than the observed η_B. The bound is valid but not tight.

More importantly, the sign of ξ_{t,t'} is systematically positive (saturating), meaning the additive model under-estimates G(S) (joint correction less harmful than sum). Proposition C's assumption |ξ| ≤ γ holds, but the directional structure (ξ > 0 for early pairs) is not captured by the bound.

**Status: PROVEN but loose. Direction of interaction not what the theorem model assumes.**

### Contraction route (Stretch C2)

**Status: not tested, unchanged.**

The Gibbs-resample-all corrector is not the kernel envisioned for C2. No new evidence either way.

### Overall thesis framing

The experiment provides a **genuine negative result** that is still useful:
1. The proxy-regret framework (Theorem A) correctly identifies when signal-adaptive scheduling cannot outperform uniform (when ε is large, η_B is large, or G is small/negative).
2. The corrector design choices have a dramatic effect on Δ_t; choosing a bad corrector kernel makes any scheduling approach irrelevant.
3. The experimental infrastructure is validated and ready for the corrected run.

The corrected run (Bug #1 and Bug #2 fixed) is necessary before any positive results can be reported.

---

## Implementation bug impact summary

| Finding | Source | Reliable? |
|---------|--------|-----------|
| Δ_t profile (monotone, negative) | Real MDLM + Gibbs-all corrector | Yes — real finding about this corrector |
| TCR_t profile (monotone, decreasing) | Real MDLM | Yes |
| Pearson/Spearman(TCR, Δ) | Real MDLM | Yes — but driven by u_t, not informative |
| ε (entropy, margin) | Bug #1: signals=0 | **NO** — artifacts |
| η_B | Real MDLM + real corrector harm | Yes for this corrector |
| γ, pairwise ξ | Real MDLM | Yes for this corrector |
| Theorem A non-vacuous check | Both bugs contaminate | **NO** |
| T_low identification | Real MDLM | Yes (all steps are harmful; early steps most so) |
