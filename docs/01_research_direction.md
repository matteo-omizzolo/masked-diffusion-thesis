# Research Direction

> **Current source of truth.** Updated 2026-05-05.

---

## Thesis question

> For a fixed predictor schedule and fixed corrector NFE budget in masked diffusion
> language models, can aggregate trajectory signals — entropy, confidence margin, or
> quality mass — predict the marginal value of a corrective refinement loop well
> enough to outperform uniform corrector placement?

---

## Problem formalization

Modern masked diffusion LMs (MDLM, ReMDM, ProSeCo, PRISM, LLaDA) run a predictor that
unmasks tokens over T steps and optionally interleave a corrector that refines
already-committed tokens. With a fixed predictor schedule and a fixed global budget B
of extra corrector NFEs, the open question is **where along the trajectory** those B
corrector loops should be placed.

The thesis formalizes this as a **proxy-regret problem**: given a trajectory-level
quality functional F, each step t has an unknown one-loop marginal gain
Δ_t = F(y_t^{+1}) − F(y_base). A signal-driven proxy ψ(s_t) derived from aggregate
trajectory signals (entropy H_t, inverse margin M_t^{-1}, quality mass QM_t — historical files use `Q_t`) is used
to pick a budget-B schedule Ŝ_B.

**Thesis story (empirical verdict, April 2026):** fixed-budget corrector allocation is
a combinatorial trajectory-control problem. Cheap greedy rankers are the wrong solution
class. Search procedures over schedules (CD-G, BS-AG) recover most oracle headroom.

---

## Scope boundaries (what this thesis IS about)

- Trajectory-level allocation of a fixed corrector NFE budget under a fixed predictor
  schedule.
- Signal-to-gain calibration (measuring how well ψ(s_t) predicts Δ_t).
- Additivity/interaction of corrector placements.
- Proxy-regret bound for top-B scheduling (Theorem A and refinements).

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
5. The "MC oracle" used as a practical upper bound is **best-of-N random
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
