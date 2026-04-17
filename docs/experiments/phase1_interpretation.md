# Phase 1 Pilot — Results Interpretation

**Date:** April 2026
**Run mode:** Surrogate (local, no GPU)
**Configuration:** T=64, N=40 trajectories, M=25 schedules/B, P=200 pairs
**Budget values:** B ∈ {4, 8, 16}
**Data:** `results/phase1_pilot/`, **Figures:** `figures/phase1_pilot/`

> **⚠ Surrogate caveat.** All numerical results below come from the
> deterministic surrogate generator (`src/mdm_playground/scheduling/surrogate.py`),
> which mimics qualitative properties of real MDLM dynamics but is not a real
> language model. The surrogate is designed as a pipeline validator and as a
> clean testbed for the Theorem A analysis. Numbers labelled **[SURROGATE]**
> must be re-measured on the HPC with the real MDLM checkpoint before being
> cited in the thesis. The **analysis logic, figures, and interpretation
> framing** are production-ready.

---

## 1. Protocol A — Signal Calibration (ε)

### Results

| Signal | Spearman(ψ, Δ) | ε_rms | ε_max |
|--------|---------------|-------|-------|
| Entropy H_t | −0.22 ± 0.05 | 0.0167 | 0.042 |
| Inv. Margin M̃_t | +0.22 ± 0.05 | 0.0172 | 0.040 |
| Quality mass proxy Q_t | −0.22 ± 0.05 | 0.0172 | 0.040 |

**[SURROGATE]**

### Interpretation

The surrogate is constructed so that **entropy H_t peaks early** in the
trajectory (steps t ≈ 0–15, when most tokens are masked) while the **one-loop
marginal gain Δ_t peaks in the middle** (steps t ≈ 30–40, roughly 55% into
the trajectory). This generates the negative Spearman correlation: high-entropy
steps have low gain; mid-trajectory steps have high gain.

This is qualitatively the most important and thesis-relevant finding in the
surrogate: **raw entropy is a misaligned proxy for one-loop gain** precisely
because of the burn-in effect captured by Proposition B. The surrogate
illustrates the pathology cleanly; the real MDLM experiment must test whether
the same qualitative shape holds.

**What the thesis predicts.** The theory (Theorem A + Proposition B) predicts:
1. If entropy ranks gain correctly, ε is small → bound is tight → top-B by
   entropy is near-optimal.
2. If entropy inversely ranks gain (as in this surrogate), ε is large and
   the proxy schedule actively picks the *wrong* steps.
3. Excluding the low-gain region T_low (steps 0–9 here) should improve
   alignment — this is Proposition B's claim.

The inverse-margin signal shows symmetric (positive) correlation of similar
magnitude, meaning it points in the right direction but at similar quality.
In the surrogate both signals have similar absolute error ε ≈ 0.017. On the
real MDLM, the question is whether confidence sharpens faster in the mid-
trajectory and thus correlates better with Δ_t.

**Key insight from `calibration_scatter.png`.** The scatter is diffuse — all
three signals produce broad clouds with no clear monotone relationship. The
linear calibration slopes (a ≈ ±0.007–0.015) are small, confirming the signals
are weakly informative in the surrogate. For real MDLM, we want either a
tighter cloud (small ε) or, failing that, to identify which signal has the
best ranking accuracy even if the linear fit is weak.

**Key insight from `delta_vs_t.png`.** Three panels tell the story:
- Panel 1 (Δ_t vs t): bell curve peaking at t ≈ 35/64, near-zero for t < 10.
- Panel 2 (normalised signals vs t): entropy starts high and decays while Δ_t
  rises and then falls — the mismatch in shape is the source of large ε.
- Panel 3 (TCR_t vs Δ_t): TCR increases monotonically with entropy (as
  expected — more entropy → more tokens changed), while Δ_t is non-monotone.
  These two are manifestly distinct quantities (Q8 confirmed).

---

## 2. Protocol A — Low-Gain Region T_low (Proposition B)

### Results

| Threshold | T_low steps |
|-----------|-------------|
| 50% of peak Δ | steps 0–9 (first 15% of trajectory) |
| 30% of peak Δ | steps 0–9 (same; threshold change minor) |
| Peak Δ (mean) | 0.050 per step |

**[SURROGATE]**

### Interpretation

A clean low-gain region exists in the surrogate: the first ~15% of steps
(t < 10 at T=64) consistently show Δ_t < 30–50% of the peak gain. This
matches the burn-in intuition: when few tokens are unmasked, a corrector loop
cannot reduce factorization error effectively.

**Proposition B application.** Excluding T_low = {0, …, 9} from the top-B
selection and then picking the remaining top-B by signal should outperform
naive top-B-by-entropy (which assigns budget to the early, low-gain steps).
This is a directly testable prediction: in Phase 2, a `burn_in_gated` policy
with threshold calibrated from Protocol A data should beat `top_B` by entropy
on real MDLM.

**Caution.** In the surrogate, T_low is read off the mean Δ_t curve trivially
because the generator is analytic. On the real MDLM, Protocol A data will show
a noisier Δ_t curve; T_low must be estimated with appropriate CIs. The
50th-percentile threshold is a reasonable operating point but is a hyperparameter
to be cross-validated.

---

## 3. Protocol A — TCR_t ≠ Δ_t (Q8 Confirmed)

### Results

- **Pearson(TCR_t, Δ_t) ≈ 0.18**, **Spearman ≈ 0.18** — weak positive
  correlation but far from 1. **[SURROGATE]**

### Interpretation

The `tcr_vs_delta.png` figure shows a broad scatter with a mild positive
trend. TCR peaks early (high-entropy, many-masked steps) while Δ_t peaks
in the middle. If TCR were used as the calibration signal instead of Δ_t,
it would produce a proxy with even larger ε: the top-B-by-TCR schedule
would also pick early steps. This confirms that Protocol A must measure Δ_t
directly (via branched trajectories with a quality functional F), not TCR.

This distinction (Q8) is non-trivial for real experiments because measuring
Δ_t requires running generation to completion T−t steps after the branch,
while TCR is cheaply computed from a single corrector call. The surrogate
confirms that the extra compute of full-trajectory Δ_t measurement is necessary.

---

## 4. Protocol B — Approximate Additivity (η_B)

### Results

| B | η_B (95th pct) | η_B (mean) |
|---|----------------|------------|
| 4 | 0.041 | 0.017 |
| 8 | 0.079 | 0.033 |
| 16 | 0.134 | 0.059 |

**[SURROGATE]** — surrogate γ = 0.016 (95th pct), η_B ≈ γ B(B−1)/2.

### Interpretation

η_B grows roughly linearly with B at mean level, but super-linearly at the
95th percentile (consistent with Proposition C: γ B(B−1)/2 is quadratic).
The `eta_vs_B.png` figure shows the empirical η_B alongside the Proposition C
bound: they are compatible but the bound is loose because not all pairs have
maximal |ξ| = γ.

**Key number.** Mean η_B at B=8 is 0.033, compared to individual Δ_t ≈ 0.01–0.05.
So the additivity slack is of similar order to the individual step gains, meaning
interactions are *not* negligible even at B=8. This is important: in the surrogate,
approximate additivity holds at the 95th-percentile level (η_B = 0.079 at B=8)
but this is already ~2× a typical single-step gain.

**Implication for Theorem A.** The 2η_B term in the bound is significant.
Even if ε = 0, the regret bound would be 2×0.079 = 0.158 at B=8, which
is larger than G(Ŝ_B) ≈ 0.023. This means Theorem A is vacuous in the
surrogate not primarily because of signal miscalibration (ε) but because
the interactions are non-negligible at B≥8. The theorem becomes useful only
at small B (say B ≤ 4) or if the interactions on real MDLM are smaller.

**What changes on real MDLM.** The surrogate uses γ = 0.008 (pairwise noise
scale). On the real MDLM, the interaction structure depends on the model's
conditional distribution and how corrector steps at different t affect
subsequent predictor steps. It is possible that interactions are smaller on
the real model if corrector effects are local (don't propagate far through
subsequent predictor steps). This is the key empirical question Protocol B
must answer on the HPC.

---

## 5. Protocol B — Pairwise Interaction γ (Proposition C)

### Results

- γ (95th pct) = 0.016, γ (mean) = 0.007, γ (max) = 0.028 **[SURROGATE]**
- Proposition C bound: η_B ≤ γ B(B−1)/2 → at B=8: 0.016 × 28 = 0.45

### Interpretation

The `pairwise_xi_hist.png` histogram shows that most pairwise interactions
|ξ_{t,t'}| are small (clustered near zero), with a light tail. The 95th
percentile (γ̂ = 0.016) is a conservative estimate; using the mean (0.007)
would give a tighter but less robust bound. Proposition C's bound with 95th-
percentile γ is conservative but tracks the growth of η_B correctly.

**Whether pairs show temporal structure.** In the surrogate, ξ is a random
draw independent of |t − t'|; on real MDLM, nearby steps may interact more
strongly (correcting at t changes Z_{t+1} more than Z_{t+10}). A more refined
temporal interaction model is a natural thesis appendix topic.

---

## 6. Theorem A Non-Vacuousness Check

### Results

| B | 2Bε + 2η_B | G(Ŝ_B) estimate | Useful? |
|---|-----------|-----------------|---------|
| 4 | 0.215 | 0.008 | ✗ vacuous |
| 8 | 0.425 | 0.023 | ✗ vacuous |
| 16 | 0.802 | 0.098 | ✗ vacuous |

**[SURROGATE]**

### Interpretation

The `theorem_A_budget.png` figure shows the regret bound dominating G(Ŝ_B)
at all budget levels in the surrogate. The bound is vacuous primarily because:

1. **Large ε.** The signal is misaligned with Δ_t, making calibration error
   the leading term at small B (2Bε is the dominant contribution for B ≤ 8).
2. **Non-trivial η_B.** Interactions grow with B, compounding the issue.
3. **Small G(Ŝ_B).** The top-B-by-entropy schedule picks high-entropy
   early steps, which have LOW gain. G(Ŝ_B) is thus not the best achievable
   gain — it is the gain of a *bad* schedule. The oracle G(S*_B) would be
   larger (approximately B × peak Δ_t ≈ B × 0.05), but the proxy's
   misalignment means Ŝ_B is far from S*_B.

**What this means for the thesis.** The surrogate illustrates exactly the
negative result Theorem A can encode: *if the signal is badly calibrated
(large ε), the proxy-regret bound is vacuous, and the proxy schedule provides
no improvement guarantee.* This is a meaningful statement, not a failure of
the theory — it correctly identifies when signal-adaptive scheduling will not
help.

For a **positive result** (bound is non-vacuous, proxy schedule provably beats
uniform), we need one of:
- A signal with smaller ε on real MDLM (the primary empirical question)
- A corrector with larger per-step gains Δ_t (e.g., a stronger kernel)
- A small-budget regime where 2Bε + 2η_B < G(S*_B)

The surrogate thus motivates the experimental design for Phase 2: test whether
the real MDLM's gain profile has a larger peak Δ_t relative to the noise level,
and whether excluding T_low reduces ε enough to make the bound non-vacuous.

---

## 7. Surrogate vs Real MDLM: What Will Change

| Quantity | Surrogate (analytic) | Real MDLM (HPC) |
|----------|---------------------|-----------------|
| Δ_t profile shape | Bell curve, peak at 55% of T | Unknown; hypothesised bell curve |
| Max Δ_t | ~0.050 per step | Unknown; depends on corrector + F |
| Spearman(H, Δ) | −0.22 (misaligned) | Unknown; key experimental question |
| ε_rms | 0.017 | Unknown; what Phase 1 HPC measures |
| η_B (B=8) | 0.079 (95th pct) | Unknown; depends on inter-step propagation |
| γ | 0.016 | Unknown; depends on corrector locality |
| T_low | steps 0–9 (first 15%) | Unknown; approximately where G is small |
| Theorem A status | Vacuous at all B | Depends on above |

The most important unknowns are: (a) the actual Δ_t magnitude and shape on
MDLM-OWT at T=128, and (b) the correlation between signals and Δ_t on real
trajectories. These are answered by running `scripts/run_phase1_pilot.py`
without `--surrogate` on the HPC.

---

## 8. Immediate Next Steps

### HPC experiment (from your laptop)

```bash
# Step 1: Push updated code to HPC
bash hpc/push.sh

# Step 2: Submit Phase 1 pilot (real MDLM, T=64, N=20)
bash hpc/submit.sh phase1-pilot

# Step 3: Monitor
ssh 3316152@slogin.hpc.unibocconi.it 'squeue -u 3316152'
ssh 3316152@slogin.hpc.unibocconi.it 'tail -f ~/mdm/masked-diffusion-thesis/out/phase1_pilot_*.out'

# Step 4: Pull results
bash hpc/pull.sh

# Step 5: Re-run analysis (same script, no surrogate flag)
python scripts/analyze_phase1.py \
    --results_dir results/phase1_pilot \
    --out_dir figures/phase1_real
```

### Interpretation update after HPC run

Replace this document's **[SURROGATE]** values with real numbers. The key
interpretive questions to answer with real data:

1. Is Spearman(H, Δ) positive? (Entropy is a useful signal)
   - If yes → ε is small → Theorem A could be non-vacuous at small B
   - If no → confirm surrogate's burn-in prediction; test burn_in_gated
2. Is η_B small at B=8? (Interactions are negligible)
   - If yes → Theorem A is non-vacuous for the right signal
   - If no → restrict to B ≤ 4 regime
3. Does T_low exist? (Proposition B applicable)
   - If yes → measure δ threshold and re-run Protocol A excluding T_low
4. Does any signal give a non-vacuous bound?
   - If yes → the thesis has a positive result
   - If no → negative result (report that naive entropy scheduling cannot
     be theoretically justified at typical budgets; requires either a better
     signal or stronger corrector)

Both outcomes make a thesis contribution: the Theorem A framework and the
ε/η_B decomposition clarify *why* signal scheduling succeeds or fails,
which is the novelty.

---

## Figures Summary

| File | Content | Key message |
|------|---------|-------------|
| `calibration_scatter.png` | ψ vs Δ_t, 3 signals | All signals weakly correlated with Δ_t in surrogate |
| `delta_vs_t.png` | Δ_t, signals, TCR across t | Bell-shaped gain; entropy inversely tracks gain |
| `eta_vs_B.png` | η_B vs B + Proposition C bound | η_B grows with B; interactions non-negligible |
| `pairwise_xi_hist.png` | \|ξ\| histogram | Mostly small but tailed; γ ≈ 0.016 |
| `theorem_A_budget.png` | 2Bε+2η_B vs G(Ŝ_B) | Bound vacuous in surrogate; calibration needed |
| `tcr_vs_delta.png` | TCR vs Δ scatter | TCR ≠ Δ_t; Q8 confirmed |
