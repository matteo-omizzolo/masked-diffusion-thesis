# Research Direction

> **Current source of truth.** Updated 2026-05-05.
> Synthesized from `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` and
> `docs/thesis_direction.md` (both archived after this cleanup).

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
trajectory signals (entropy H_t, inverse margin M_t^{-1}, quality mass Q_t) is used
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

**Theorem (analytical):** Theorem A (proxy-regret bound for top-B scheduling) + Refinements
A′ (variance-form η_B) and A″ (rank-based ε_R) + Negative-Result Corollary (ranker class
bounded by NULL band by B = 8 on OWT). All formally proved under explicit assumptions.
Appendix F: Theorem A-ad (adaptive/state-conditional generalization, honest negative).

**Empirical (on ProSeCo-OWT):**
- Greedy ranker negative: confirmed across all 10 policies, K = 30 seeds.
- Search positive: CD-G and BS-AG recover 49–84 % of oracle headroom.
- PRISM rejection: not because no structure exists, but because PRISM is in the ranker class.

**Negative (honest):**
- Theorem A L∞ form is empirically vacuous at all tested B ≥ 4 on OWT.
- State-conditional ranking (Protocol C) does not recover headroom on OWT.
- Results do not transfer to LLaDA-SFT at tested resolution.

---

## Caveats

1. Primary results are on a single backbone (ProSeCo-OWT). External validity is limited
   by the inconclusive LLaDA-SFT probe.
2. CD-G uses the true pipeline-evaluated G for every accept/reject — it is a structural
   existence result, not a deployable inference-time scheduler.
3. BS-AG is practical (O(B) G-calls per round) but still uses true G for rollouts.
4. The proxy-regret bound uses L∞ calibration error ε, which is empirically vacuous;
   the rank-based ε_R (Refinement A″) is the operative quantity.

---

## What might need reconsideration

- Whether to add one more backbone (not currently authorized; would require HPC and Zanella
  approval as a Phase 4).
- Whether to strengthen the thesis contribution to a deployable inference-time scheduler
  (would require function-approximator work; currently out of scope).
- Whether the negative result on PRISM/rankers alone is a sufficient thesis contribution
  without the Phase 3a positive (it is not; Phase 3a is load-bearing).
