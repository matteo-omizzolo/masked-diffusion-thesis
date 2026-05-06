# Research Direction

> **Current source of truth.** Updated 2026-05-06.

---

## Thesis question

> For a fixed predictor schedule and fixed **corrector-placement budget** B in
> masked diffusion language models, **when is informed correction timing
> reducible to marginal signal ranking, and when does it require
> interaction-aware or search-based scheduling?**

The original marginal-signal question — "can aggregate signals (entropy,
inverse margin, quality mass) predict Δ_t well enough to outperform uniform?"
— remains the **baseline hypothesis** tested by Theorem A. The current
thesis asks the broader **regime question**: marginal/rankable vs
interaction-driven vs higher-order/chaotic vs online-decision-like timing
(Diagnostic Framework C).

---

## Problem formalization

Modern masked/discrete diffusion work separates base models (MDLM, SEDD,
LLaDA, Dream), remasking samplers (ReMDM), informed-refinement mechanisms and
quality signals (PRISM), and informed corrector kernels (ProSeCo). This thesis
fixes a predictor/corrector/backend and asks when to spend a corrector placement.
With a fixed predictor schedule and a fixed
**corrector-placement budget** B (corresponding extra-NFE budget
B_NFE = c_corr · B; c_corr = 2 for ProSeCo annealed refinement), the open
question is **where along the trajectory** those B corrector placements should
go and **what regime** the (model, corrector, F, B) triple lies in.

The thesis formalizes this through a layered framework:

The framework is **model- and corrector-agnostic**: for any fixed predictor,
informed corrector, quality functional F, and corrector-placement budget B, it
defines G(S), Δ_t, A(S), ξ_{t,t′}, Q(S), and regime diagnostics. ProSeCo-OWT is
the primary empirical case study, not an assumption of the theory.

- **Theorem A (baseline).** Proxy-regret bound for separable-ranker scheduling:
  G(S_B^*) − G(Ŝ_B) ≤ 2Bε + 2η_B under uniform calibration / additivity.
- **Theorem B / B′.** Pairwise surrogate regret framework: when |G − Q| is
  small on a fixed candidate pool C_B, optimising the pairwise surrogate is
  near-optimal for G on the pool.
- **Diagnostic Framework C.** Five-regime taxonomy
  (no-op / marginal / interaction-driven / chaotic / online-decision)
  with measurable diagnostics U_B^{MC,N}, R_B, I_B, P_B, C_B^{MC,N}.
- **Empirical Ranker-Class Limitation.** Time-only / seed-averaged separable
  rankers are bounded by the mean-Δ̄ envelope on the additive surrogate;
  empirically on ProSeCo-OWT, the tested separable rankers do not recover
  the MC-oracle headroom and the mean-Δ̄ envelope itself enters the
  no-detectable-gain band by B = 8.

Detailed statements, proofs, and theory-to-experiment map are in
`research/candidate_theorems.md` §0–§7.

**Baseline empirical verdict (April 2026, pending Phase 0 re-confirmation):**
Tested separable rankers do not recover MC-oracle headroom on ProSeCo-OWT.
Search procedures (CD-G, BS-AG) recover 49–84 % of MC-oracle headroom at
B ∈ {2, 3, 4} using true-G feedback / true-G rollouts. Whether a
*deployable* (Level-3 feature-conditioned) pairwise scheduler exists is
the open empirical question for Phase 1 / Phase 2.

---

## Scope boundaries (what this thesis IS about)

- Trajectory-level allocation of a fixed corrector-placement budget B
  under a fixed predictor schedule.
- Signal-to-gain calibration (Theorem A, A′/A″ diagnostics).
- Additivity / interaction structure of corrector placements (Theorem B / B′).
- Regime classification (Diagnostic Framework C).
- Surrogate regret bounds with explicit no-leakage candidate-pool construction.

---

## Non-goals (what this thesis is NOT about)

- **Corrector kernel design** — how to correct (Barker/MPF/ancestral). Out of scope.
- **Token-selection policies** — which tokens to correct within a step. Out of scope.
- **Predictor schedule optimization** — when/how many tokens to unmask. Out of scope.
- **Remasking methods** — revisiting already-committed tokens by remasking. Out of scope.
- **Full policy optimization** — reinforcement learning over corrector policies. Out of scope.
- **Claim of generality** — results are on ProSeCo-OWT; one LLaDA-SFT probe was
  inconclusive.

---

## Contribution as currently understood

The thesis is reframed as a **theory-first regime-diagnostic study of
fixed-budget corrector timing**. Detailed theorem statements live in
`research/candidate_theorems.md`; this is a one-paragraph summary.

**Theorem stack (analytical).**
- **Theorem A** (uniform marginal proxy regret 2Bε + 2η_B) — proved baseline.
- **Diagnostics A′, A″** — additivity scale and rankability; **not**
  unconditional regret refinements (demoted from prior status).
- **Theorem A as B′(Q := A)** — safe finite-pool selected-schedule corollary.
- **Theorem B / B′** — pairwise surrogate regret framework (central new
  framework). B′ is the finite-candidate-pool, high-probability,
  estimator-aware form usable in experiments with a no-leakage candidate
  pool.
- **Diagnostic Framework C** — regime classification protocol over five
  regimes (no-op / marginal / interaction-driven / chaotic / online-decision)
  using disciplined MC-pool / pool-oracle notation.
- **Empirical Ranker-Class Limitation** (replaces "Negative-Result
  Corollary") — formal part for time-only / seed-averaged separable ψ;
  empirical part on tested separable rankers on ProSeCo-OWT.
- Theorem D (online controller) and Lemma E (clipped-F_C burn-in) are
  appendix-grade only.

**Empirical (on ProSeCo-OWT, prior April 2026 baseline; pending Phase 0
re-confirmation):**
- Tested separable rankers do not recover MC-oracle headroom; the
  mean-Δ̄_t envelope enters the no-detectable-gain band by B = 8.
- CD-G recovers 74–84 % and BS-AG 49–64 % of MC-oracle headroom at
  B ∈ {2, 3, 4} (with true-G feedback / true-G rollouts respectively).
- PRISM-as-separable-score is in the ranker class limited by the
  Empirical Ranker-Class Limitation; non-separable PRISM uses are
  optional / future and not pursued in this thesis.

**Negative (honest):**
- Theorem A's uniform L∞ form is empirically vacuous at B ≥ 4; the
  operative form is the finite-pool corollary (Theorem A as B′(Q := A)).
- State-conditional ranking (Protocol C) does not recover headroom on OWT.
- Results do not transfer to LLaDA-SFT at tested resolution (Tier 3).

---

## Caveats

1. Primary results are on a single backbone (ProSeCo-OWT). External
   validity is limited by the inconclusive LLaDA-SFT probe.
2. CD-G uses the true pipeline-evaluated G for every accept/reject — it
   is a structural existence result, not a deployable inference-time
   scheduler.
3. BS-AG is practical (O(B) G-calls per round) but still uses true G for
   rollouts.
4. Theorem A's uniform L∞ form is empirically vacuous; the safe
   selected-schedule statement is the finite-pool corollary.
5. The "MC-oracle" used as a practical upper bound is **best-of-N random
   schedules**, not the exhaustive (T choose B) maximizer.
6. A deployable Theorem-B claim requires Level-3 (feature-conditioned)
   evaluation on held-out seeds; population-only (Level 2) success is not
   sufficient for a deployability claim.

---

## What might need reconsideration

- Whether to add one more backbone (not currently authorized; would require HPC and Zanella
  approval as a Phase 4).
- Whether to strengthen the thesis contribution to a deployable inference-time scheduler
  (would require function-approximator work; currently out of scope).
- Whether the negative result on PRISM/rankers alone is a sufficient thesis contribution
  without the Phase 3a positive (it is not; Phase 3a is load-bearing).
