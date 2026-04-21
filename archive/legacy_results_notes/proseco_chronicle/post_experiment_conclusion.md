# Post-Experiment Conclusion

**Date:** April 2026  
**Based on:** `phase1_pilot_478600` — Real MDLM run, 20 trajectories, T=64, B ∈ {4,8,16}  
**Status:** Negative result from a broken implementation — not a theory failure.

---

## 1. Main empirical finding

The Phase 1 pilot ran the full Protocol A + B pipeline on real MDLM (A100, 70 min). Two critical implementation bugs corrupted the results:

1. **Signal computation bug** (`_extract_signals`): entropy and margin computed over unmasked (committed) positions → all signals = 0. No calibration data.

2. **Corrector design bug** (`_apply_corrector`): full Gibbs resample of all masked positions at step t → universally negative Δ_t (corrector always harmful, worst at early steps). The corrector is not a beneficial corrector kernel.

**Because of these bugs, the main thesis questions (Q2, Q8, Theorem A vacuousness) cannot be answered from this run.**

What is genuinely established:
- The experiment infrastructure is complete and correct (Protocol A/B pipeline, JSON logging, analysis scripts).
- The Gibbs-resample-all corrector produces monotonically decreasing harm from t=0 (Δ≈−3.19) to t=63 (Δ≈0).
- Pairwise interactions are large and saturating (ξ > 0, joint harm < summed harm).
- The "oracle" schedule for this corrector is to correct at no steps or only the last few.

---

## 2. Best signal right now

**None.** Signal computation produced zeros for all three signals (entropy, inverse margin, quality mass). No ranking ability can be assessed. After the signal bug is fixed, the expectation is:

- Entropy over masked positions will peak early (many masks = high uncertainty) and decline
- Inverse margin will rise later (model becomes more decisive)
- Whether either correlates with Δ_t depends on the corrector design

For a corrected beneficial corrector, confidence-margin at masked positions is the most likely strong signal (KLASS, Zhao et al., and ProSeCo all suggest margin-based signals).

---

## 3. Is uniform scheduling beaten?

**Not applicable.** All corrections are harmful with the current corrector. Any schedule (including uniform) produces G(S) < 0. The "best" strategy is zero corrections (B=0). Signal-adaptive scheduling cannot outperform uniform when all individual corrections are detrimental.

After re-running with a beneficial corrector, the comparison becomes meaningful.

---

## 4. Is burn-in gating justified?

**Yes, for the wrong reason.** Early steps are the most harmful (not merely low-gain), so gating them out reduces harm. This validates Proposition B's structural claim that gating T_low is benign (here it would be actively beneficial). The positive version — early steps are low-gain for context reasons — cannot be confirmed yet.

**Practical recommendation:** Any re-run should automatically exclude steps t where u_t < 0.20 (fewer than 20% tokens unmasked) from the corrector budget. The data strongly supports this.

---

## 5. Is the additive surrogate usable?

**No, at the tested budgets.** η_B / |G| = 0.80 at B=4 and 4.66 at B=16. The additive surrogate is not a useful approximation for B ≥ 4 with this corrector.

The saturating interaction structure (ξ > 0) means the surrogate over-estimates harm. In the positive-gain regime (after the corrector design is fixed), saturating interactions would mean the surrogate over-estimates gain, which is the dangerous direction for Theorem A's bound. This is why measuring η_B empirically (not just bounding via Proposition C) matters.

---

## 6. Strongest defensible thesis version now

**The Theorem A framework is mathematically sound and the proof is complete. The experimental contribution is currently a negative result, with the positive result pending a corrected re-run.**

The defensible thesis version is:

> **Core theorem** (Theorem A): Under binary placement, approximate additivity, and calibrated proxy, the proxy-regret bound G(S_B*) − G(Ŝ_B) ≤ 2Bε + 2η_B holds. The bound is non-vacuous iff 2Bε + 2η_B < G(S_B*).

> **Phase 1 negative result**: The Gibbs-resample-all corrector does not satisfy Δ_t > 0. Using this corrector, the scheduler has no beneficial steps to allocate, and Theorem A's bound is vacuous. This is a correct diagnosis, not a theorem failure.

> **Phase 1 positive structural result**: Burn-in gating (excluding early steps) is strongly supported. The low-gain region T_low = {t : u_t < 0.20} accounts for the most harmful steps.

> **Phase 2 target**: With a confidence-guided partial corrector and corrected signal computation, test whether ε, η_B can be made small enough to produce a non-vacuous bound at some B.

The weaker, more immediately defensible version (if the re-run cannot happen in time):

> We establish the Theorem A framework, identify the key quantities ε, η_B, γ, and provide the experimental methodology. We show the framework correctly diagnoses why a naive corrector fails (large ε, large η_B, no positive gain). We demonstrate the infrastructure and report what would be needed for a non-vacuous positive result.

---

## 7. Next steps in theory

1. **No changes needed to Theorem A** — the proof is solid.
2. **Proposition B refinement**: Strengthen the δ threshold characterization. The current run shows Δ_t < 0 for all t with this corrector; for a beneficial corrector, identifying the δ threshold from the bell-curve shape of Δ_t is the empirical task.
3. **Proposition C tightening**: Investigate whether a temporal decay model for ξ (nearby steps interact more than distant steps) gives a tighter bound. The data shows ξ is larger for early pairs (both early = strong saturation) than for pairs with one late step.
4. **Expectation-version of ε and η_B**: Write out the formal expected-regret bound as insurance against heavy-tailed Δ_t in the corrected run.

---

## 8. Next steps in experiments

**Priority 1 (before anything else):**

Fix Bug #1 — signal computation (1 line change):
```python
# In backends/mdlm.py, _extract_signals()
# Line 338: change
revisable = (x[0] != self.mask_id)   # WRONG
# to
revisable = (x[0] == self.mask_id)   # CORRECT
```

Fix Bug #2 — corrector design:
Replace `_apply_corrector` with a partial confidence-guided resample:
- Identify the K lowest-confidence masked positions (K = fraction of masked positions, e.g., K = masked_count × α where α ∈ (0, 1))
- Resample only those K positions from p_x0(·|x_t)
- Leave all other positions (masked and unmasked) unchanged

This mimics the ReMDM conf-refinement approach and is more likely to produce positive Δ_t.

**Priority 2:** Resubmit Phase 1 pilot with the corrected implementation:
```bash
T=64, N=30, M=20, P=200, B_values=4,8,16,32
corrector: conf_guided (bottom 20% confidence masked positions)
```

**Priority 3:** If re-run shows positive Δ_t peak, run Protocol A/B with full N=50, T=128.

**Priority 4:** Read ProSeCo carefully to determine whether it uses signal-adaptive scheduling and whether any overlap with Theorem A's formulation exists (Q4 in open_questions).
