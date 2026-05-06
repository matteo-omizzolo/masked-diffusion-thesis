> TECHNICAL WORKLOG.
> Current thesis status starts at `START_HERE.md`. Summary in `docs/03_theory.md`.
> Current theory-first plan: `docs/06_theory_first_research_plan.md`.

# Current Theory Direction — May 2026

> This file defines the **active theorem stack** for the theory-first corrector
> timing programme. The April 2026 (pre-revision) theorem stack — including
> proofs of "Refinement A′", "Refinement A″", the "Negative-Result Corollary",
> γ B(B−1)/2 "Proposition C", and "Theorem A-ad" — was moved to
> `docs/archive/old_theory_stack/candidate_theorems_pre_2026_05.md` and is
> **superseded**. Do not cite that file as current. Git history preserves the
> full editorial trail.
>
> **Backbone.** Theorem A (marginal baseline) → Theorem B / B′ (pairwise
> surrogate regret framework, central) → Diagnostic Framework C (regime
> classification). Theorem D and Lemma E are appendix-grade only. Refinements
> A′ and A″ are diagnostics, not regret theorems. The Empirical Ranker-Class
> Limitation (§1.5) replaces the previous "Negative-Result Corollary".

## 0. Formal problem setup

### 0.1 Trajectory and schedules

Let a masked diffusion language model run a fixed predictor over T denoising
steps with time index t ∈ {1, …, T}. A **corrector schedule** is a subset

  S ⊆ {1, …, T},   |S| = B,

with at most one corrector loop per step (binary placement). The integer B is
the **fixed corrector-placement budget**; the corresponding extra-NFE budget is

  B_NFE := c_corr · B,

where c_corr is the number of extra forward passes per corrector placement
(c_corr = 2 for ProSeCo annealed refinement). All schedules with |S| = B have
equal compute cost. We use "placement budget" by default and note "NFE budget"
only when emphasising compute.

Let y^S denote the final sample produced when the corrector is applied at
exactly the times in S, and y^∅ the corrector-free baseline. Let
F : Y → ℝ be a trajectory-level quality functional. In experiments we use
F = − GPT-2 NLL on a fixed 512-token window; F is treated as a **relative
within-run** metric, not an absolute-quality claim.

Define the **joint schedule gain** and **oracle**

  G(S) := F(y^S) − F(y^∅),       S_B^* ∈ argmax_{|S|=B} G(S).

S_B^* is exact in theorem statements; in experiments it is approximated by the
**MC-oracle** (best-of-N random schedules, paired CRN), which is a practical
upper bound — *not* an exhaustive (T choose B) maximizer.

### 0.2 Marginal gain, additive surrogate, and pairwise surrogate

Define the **single-step marginal gain** Δ_t := G({t}), the **additive
surrogate**

  A(S) := ∑_{t ∈ S} Δ_t,

and the **operational second difference**

  ξ_{t,t'} := G({t, t'}) − Δ_t − Δ_{t'}    (t ≠ t').

> **Operational caveat.** ξ_{t,t'} is the discrete second difference of the
> schedule-value function G; it is **not** a claim that the neural dynamics
> decompose mechanistically into local pairwise effects. ξ aggregates direct
> effects, downstream state changes, altered context at later steps, stochastic
> sensitivity, and metric effects. We use the term "pairwise" operationally.

Define the **pairwise (second-order) surrogate**

  Q(S) := ∑_{t ∈ S} Δ_t  +  ∑_{t < t', t,t' ∈ S} ξ_{t,t'}.

Q(·) captures all pair effects but ignores triples and higher orders.

### 0.3 Signals and separable rankers

Let s_t denote observable trajectory signals at step t. Active signals:

  H_t        — aggregate entropy over the revisable set;
  M_t^{-1}   — inverse confidence margin over the revisable set;
  QM_t       — PRISM-style "quality mass" over the revisable set;
  u_t        — revisable / unmasked fraction;
  φ_t = t/T  — normalized phase.

> **Notation.** We write **QM_t** for the quality-mass signal to avoid collision
> with the pairwise surrogate Q(S). Historical result files use `Q_t` for the
> same signal; treat the two as identical.

A **separable ranker** is any policy of the form
Ŝ_B(ψ) := top-B_t  ψ(s_t), scoring each step independently.

The corrector acts on the **revisable set** R_t (ProSeCo: already-unmasked
positions). Signals must be computed over the same R_t the corrector acts on;
historical bug-#1 (signals computed over wrong positions) must not reappear.

### 0.4 Online state

For the online direction (Theorem D), define a compact state

  z_t = (φ_t, H_t, M_t^{-1}, QM_t, u_t, b_t, h_t),

where b_t ∈ {0, …, B} is the remaining placement budget and h_t an optional
compact history summary. We distinguish:

- **Offline schedule selection.** Choose S after observing/estimating
  trajectory-level quantities for the whole trajectory.
- **Online control.** At each step t observe z_t and choose
  a_t ∈ {0, 1} under ∑_t a_t = B.

### 0.5 Levels of analysis (seed-wise / population / feature-conditioned)

The objects G, Δ, ξ, Q can live at three different levels. The level matters
for what experiments can establish.

**Level 1 — Seed-wise schedule value.** For seed i, fix the trajectory and
its predictor randomness (CRN); evaluate

  G_i(S),   Δ_{i,t} = G_i({t}),   ξ_{i,t,t'},   Q_i(S).

These are well-defined objects on a single trajectory. Computing them requires
counterfactual G-evaluations on the *same* trajectory; they are diagnostic
quantities, not deployable signals.

**Level 2 — Population / mean schedule value.** Average over a paired-seed
distribution:

  Ḡ(S) := 𝔼_i[G_i(S)],   Δ̄_t := 𝔼_i[Δ_{i,t}],
  ξ̄_{t,t'} := 𝔼_i[ξ_{i,t,t'}],   Q̄(S) := A(S; Δ̄) + Σ ξ̄_{t,t'}.

Empirical estimators average over K paired seeds. Optimizing Q̄ produces a
**single global schedule** that is good on average; it does not adapt to
per-instance trajectory features.

**Level 3 — Feature-conditioned / held-out scheduler.** Build

  Q̂_i(S) — estimated from observable features of trajectory i,

trained/calibrated on **other** seeds and evaluated on held-out seed i. This
is the only level that supports a **trajectory-conditioned** scheduler at
inference without true G_i queries. It is the level required to claim a
*deployable* scheduler.

**Theorem-level convention.** Theorem A applies at any level provided its
assumptions are stated at that level. Theorem B is stated for an arbitrary G
and arbitrary surrogates Q, Q̂; instantiating it at Levels 1, 2, 3 produces
diagnostic, population, and feature-conditioned forms. The thesis must not
claim a deployable scheduler unless the Level-3 form has empirically supported
assumptions on held-out seeds.

### 0.6 Randomness and pairing

All paired comparisons (G(S) − G(uniform), G(S) − A(S), etc.) use **common
random numbers** (CRN): same seed, same predictor randomness, same tokenizer
draws — the only thing that differs is the corrector schedule. Estimators
report point estimates with BCa bootstrap 95 % CIs over seeds. Quantities
without subscript i are seed-averaged; seed-wise quantities carry the i index.

---

## 1. Theorem A — Marginal/ranker baseline (active main theorem, baseline)

**Role.** Theorem A is the *baseline* theory-first theorem: it states the
conditions under which a separable ranker is provably near the oracle. It is
**not** the central new contribution; the experiments test whether its
assumptions hold, and Theorem B takes over when they do not.

### 1.1 Statement (proved)

Assume:

  (A1) Binary placement, ∑_t k_t = B, k_t ∈ {0, 1}.
  (A2) Approximate additivity. ∃ η_B ≥ 0 with |G(S) − A(S)| ≤ η_B for all |S| ≤ B.
  (A3) Proxy calibration. ∃ ε ≥ 0 with |Δ_t − ψ(s_t)| ≤ ε for all t.

Let Ŝ_B = top-B_t ψ(s_t). Then

  G(S_B^*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

### 1.2 Proof

By (A2) at S = S_B^* and S = Ŝ_B,

  G(S_B^*) − G(Ŝ_B) ≤ A(S_B^*) − A(Ŝ_B) + 2 η_B.    (★)

Let S_A^* = argmax_{|S|=B} A(S). By Lemma A1 (below) S_A^* is the B indices of
largest Δ_t, so A(S_B^*) ≤ A(S_A^*). By Lemma A2 with proxy ψ on Δ-values,
A(S_A^*) − A(Ŝ_B) ≤ 2 B ε. Substituting into (★) gives the bound. ∎

Lemma A1 (oracle top-B under exact additivity), Lemma A2 (calibration regret
under exact additivity, swap-by-swap exchange) are stated and proved in the
Historical Provenance section below.

### 1.3 Diagnostics A′, A″ (additivity scale, rankability — *not* unconditional regret refinements)

> **Status change (2026-05).** A′ and A″ were previously presented as
> "proved under modeling assumptions" regret refinements of Theorem A. On
> mathematical re-examination they do **not** in general control the
> selected-schedule regret that Theorem A's combining step uses, and the
> previously circulated formulas were not internally consistent. They are
> **demoted to empirical diagnostics**. A safe finite-pool refinement of
> Theorem A is recovered as a corollary of Theorem B′ applied with Q := A
> (see §2.7 below).

**Diagnostic A′ — additivity residual scale.**
Define, for a fixed schedule distribution D_B (e.g. uniform over a sampled
size-B schedule pool C_B):

  η^typ_{B, D} := 𝔼_{S ∼ D_B} |G(S) − A(S)|,
  σ^pool_{ξ, B} := sd_{S ∈ C_B}(G(S) − A(S)).

These quantify the *typical* additivity residual on the chosen schedule
distribution. They are **not** a regret bound for selected schedules
(S_B^*, top-B-by-ψ) unless converted into a uniform or
high-probability bound over a candidate pool. The legacy
"σ_ξ √(B/2)" mixing-form is treated as a heuristic scaling claim and is
not used as an active theorem.

**Diagnostic A″ — rankability / calibration.**
Define:

  R_B := ρ_Spearman(A(S), G(S))   over the schedule sample;
  R_ψ := ρ_Spearman(ψ(S), A(S))   over the schedule sample.

These summarise whether the additive surrogate or a signal-derived score is
informative about G ranking. They do **not** by themselves bound the regret
of the optimizer-selected top-B schedule: a proxy may have high global
correlation and still misrank top schedules. Any "ε_R" notation is an
empirical scale on rank-correlation, **not** a theorem constant. We avoid
using "ε_R" in active theorem statements; if it appears in plots or
historical files it is a diagnostic.

### 1.4 Safe finite-pool ranking corollary

The **only** rigorous selected-schedule consequence of A is its finite-pool
version, obtained by applying Theorem B′ (§2.3) with Q := A and an estimated
proxy Â, or by applying the original Theorem A with η_B and ε defined as
finite-pool suprema:

  η_{B, C} := sup_{S ∈ C_B} |G(S) − A(S)|,
  ε_{B, C} := sup_{S ∈ C_B} |A(S) − Â(S)|.

Then top-Â over C_B has additive regret at most 2 ε_{B, C} + 2 η_{B, C}
within the pool. This is exactly Theorem B′ with Q = A.

### 1.5 Empirical Ranker-Class Limitation (scoped; replaces "Negative-Result Corollary")

> **Status change (2026-05).** The previous "Negative-Result Corollary"
> made an overbroad claim about separable per-step rankers being bounded by the
> `mean_delta_oracle` envelope. This overclaims: a *seed-conditioned*
> separable ranker ψ(s_{i,t}) can in principle out-rank a time-only
> seed-averaged envelope. We replace it with a paired (formal + empirical)
> statement.

**Formal part (time/mean-profile separable rankers).** Consider the
seed-averaged additive problem with mean profile Δ̄_t := 𝔼_i Δ_{i,t}.
For any ranker that depends only on t or on a feature φ(t) — equivalently,
any policy whose score is a function of t alone — the top-B set is a
function of the order statistics of Δ̄_t. Hence among **time-only
separable rankers**, none exceeds the `mean_delta_oracle` envelope
(top-B by Δ̄_t) on the additive surrogate Ā.

**Empirical part (ProSeCo-OWT).** All separable rankers tested in Phase 2b
— including the cheating ψ = paired Δ̂_t — do not recover the MC-oracle
headroom; the `mean_delta_oracle` envelope itself enters the
no-detectable-gain band by B = 8.

**Scope warning.** The empirical part does **not** rule out:
(i) feature-conditioned separable rankers ψ(s_{i,t}) outside the tested
class; (ii) non-separable PRISM uses; (iii) pairwise schedulers (Theorem B);
(iv) online controllers; (v) search with true-G feedback (CD-G/BS-AG). The
formal part applies only to time-only / seed-averaged-feature scores and
only on the additive mean-value surrogate Ā.

### 1.6 Empirical observables and falsifiers (Theorem A)

| Quantity | Definition | Tests |
|---|---|---|
| η_B (uniform) | sup_{|S|≤B} |G(S) − A(S)|; in practice take 95th percentile or finite-pool sup | (A2) |
| η_{B,C} | sup over candidate pool C_B | (A2) within pool |
| η^typ_{B,D} | 𝔼_{S∼D} |G(S) − A(S)| | A′ diagnostic |
| ε (uniform) | sup_t |Δ_t − ψ(s_t)| | (A3) |
| R_B = ρ(A, G) | Spearman over schedule sample | A″ diagnostic |
| ranker headroom | G(Ŝ_B) − G(uniform) | usefulness of A |

Theorem A (uniform form) is *empirically useful in regime R* only if
2Bε + 2η_B < ranker headroom in R. Otherwise it is **vacuous**. The
**regime is wrong for marginal scheduling** if any of:

  (F-A1) R_B is small;
  (F-A2) Empirical Ranker-Class Limitation regime: tested separable rankers
         do not recover the MC-oracle headroom;
  (F-A3) MC-oracle headroom large but no top-B separable ranker
         (including the cheating ψ = paired Δ̂_t) closes more than a small
         fraction of it.

Status: Theorem A **proved** under (A1)–(A3) (uniform form). Diagnostics
A′ and A″ are not regret refinements; the safe selected-schedule statement
is the finite-pool Theorem A (§1.4) or Theorem B′ with Q := A.

---

## 2. Theorem B — Pairwise surrogate regret (proposed central theorem)

**Role.** Theorem B is the proposed central new theorem. The inequality
itself is a generic surrogate-regret statement; what makes it scientifically
valuable is the empirical hypothesis that Q is a substantially better
approximation to G than A.

> **Honesty paragraph.** Theorem B is a generic surrogate-regret theorem — a
> short standard argument. Its thesis value comes from whether the pairwise
> approximation hypothesis is empirically true for masked diffusion corrector
> timing on the tested (model, corrector, F, B) triple, not from the
> inequality alone. Theorem B is "central" because it formalises a
> **falsifiable interaction hypothesis** and hands the experiments a clean
> set of quantities (ζ_B, α_B, P_B vs η_B, R_B, ranker headroom) to estimate.

### 2.1 Theorem B (exact-Q form)

Assume (A1) and:

  (B2) Pairwise additivity. ∃ ζ_B ≥ 0 with |G(S) − Q(S)| ≤ ζ_B for all |S| ≤ B.

Let S_Q^* = argmax_{|S|=B} Q(S) and let Ŝ satisfy
Q(S_Q^*) − Q(Ŝ) ≤ ω_B (surrogate optimization gap). Then

  G(S_B^*) − G(Ŝ) ≤ 2 ζ_B + ω_B.

**Proof.** By (B2),  G(S_B^*) ≤ Q(S_B^*) + ζ_B ≤ Q(S_Q^*) + ζ_B  and
G(Ŝ) ≥ Q(Ŝ) − ζ_B ≥ Q(S_Q^*) − ω_B − ζ_B. Subtract. ∎

### 2.2 Theorem B (estimated-Q̂ form)

Assume (A1), (B2), and

  (B3) Surrogate estimation error. ∃ α_B ≥ 0 with |Q(S) − Q̂(S)| ≤ α_B
       for all |S| ≤ B.

Let Ŝ_Q̂ satisfy Q̂(S_Q̂^*) − Q̂(Ŝ_Q̂) ≤ ω_B where S_Q̂^* maximises Q̂. Then

  G(S_B^*) − G(Ŝ_Q̂) ≤ 2 ζ_B + 2 α_B + ω_B.

**Proof.** Convert the Q̂-optimization gap to a Q-optimization gap. For any
S with Q̂(Ŝ_Q̂) ≥ Q̂(S_Q̂^*) − ω_B,

  Q(Ŝ_Q̂) ≥ Q̂(Ŝ_Q̂) − α_B
          ≥ Q̂(S_Q̂^*) − ω_B − α_B
          ≥ Q̂(S_Q^*) − ω_B − α_B            (since S_Q̂^* maximises Q̂)
          ≥ Q(S_Q^*) − 2 α_B − ω_B.

So Ŝ_Q̂ has Q-optimization gap at most ω_B + 2 α_B. Apply §2.1 with this
gap to obtain  G(S_B^*) − G(Ŝ_Q̂) ≤ 2 ζ_B + (ω_B + 2 α_B). ∎

> **Constant 2 α_B (not 4 α_B).** Under the symmetric two-sided definition
> |Q − Q̂| ≤ α_B used here, the correct constant is 2. A 4 α_B form arises
> only under one-sided definitions or if Q̂ is also (incorrectly) used in
> place of G when reasoning about S_B^*. We adopt 2 α_B.

### 2.3 Theorem B′ — finite candidate pool / high-probability form

The uniform bound (B3) over all (T choose B) schedules is unrealistic. In
practice estimation error is controlled only over a finite **candidate pool**
C_B ⊆ {S : |S| = B} (sampled schedules, beam candidates, MC-oracle pool).

Assume (A1) and:

  (B2′) Pairwise additivity over C_B.
        ζ_{B,C} := sup_{S ∈ C_B} |G(S) − Q(S)|.
  (B3′) High-probability estimation over C_B.
        With probability ≥ 1 − δ over the training pool,
        sup_{S ∈ C_B} |Q(S) − Q̂(S)| ≤ α_{B,δ}.

Let S_C^* = argmax_{S ∈ C_B} G(S) be the **pool oracle** and let Ŝ
approximately maximise Q̂ over C_B with optimization gap ω_B. Then with
probability ≥ 1 − δ,

  G(S_C^*) − G(Ŝ) ≤ 2 ζ_{B,C} + 2 α_{B,δ} + ω_B.

To compare against the **full oracle**, introduce the candidate-pool
approximation term

  κ_B := G(S_B^*) − G(S_C^*) ≥ 0.

Then with probability ≥ 1 − δ,

  G(S_B^*) − G(Ŝ) ≤ κ_B + 2 ζ_{B,C} + 2 α_{B,δ} + ω_B.

**Proof.** Apply §2.2's argument restricted to C_B; the κ_B term is the
definition. ∎

> **Honesty.** κ_B is **not directly estimable** against the true
> S_B^*. In experiments we estimate it relative to the MC-oracle pool, which
> itself is a practical upper bound. The thesis must not claim full-space
> regret unless C_B is argued to contain (or arbitrarily approximate) the
> full oracle.

> **Data-dependence caveat.** The candidate pool C_B must be **fixed
> independently of held-out evaluation G**. Acceptable constructions: random
> sampling (MC pool); beam candidates produced from training-seed Δ̂, ξ̂;
> Q̂-greedy candidates from training-seed Q̂. **Not acceptable**: pools
> selected using held-out G or by an optimizer with G-feedback on test
> seeds. If the pool depends on training data, evaluate on held-out seeds;
> if it depends on test G, the bound does not apply. (See no-leakage
> protocol in `research/open_questions.md` OQ-T2.)

### 2.4 Diagnostic / population / feature-conditioned hierarchy

Theorem B can be instantiated at three levels (cf. §0.5). They differ in
what experimental claim they support.

| Level | Form | Allowed inputs | Supports claim of |
|---|---|---|---|
| 1. Diagnostic | uses true ξ_i from counterfactual G-calls | true G_i({t,t'}) | structural pairwise hypothesis |
| 2. Population | optimises Q̄ to choose one global schedule | mean Δ̄_t, ξ̄_{t,t'} estimated over training seeds | global schedule good on average |
| 3. Feature-conditioned | builds Q̂_i from observable features of trajectory i | only s_t-style features of held-out seed | **deployable inference-time scheduler** |

Phase 1 tests Levels 1–2. Phase 2 tests Level 3. The thesis claims
"interaction-aware scheduling beats rankers" only at the level demonstrated;
**deployability** requires Level 3.

### 2.5 Empirical observables — bound constants

| Quantity | Definition | Tests |
|---|---|---|
| ζ_B / ζ_{B,C} | sup over candidate pool C_B of |G(S) − Q(S)| | (B2) / (B2′) |
| α_B / α_{B,δ} | sup over C_B of |Q(S) − Q̂(S)| (uniform / w.p. 1−δ), paired held-out seeds | (B3) / (B3′) |
| ω_B | optimizer gap Q̂(argmax_{S ∈ C_B} Q̂) − Q̂(Ŝ_Q̂) | optimizer |
| ζ_B / η_B | improvement ratio of pairwise vs additive approximation on C_B | regime gate |
| κ_B vs MC pool | G(S_B^{MC,N}) − G(S_C^*) | candidate-pool tightness against MC pool |

Pairs ξ_{i,t,t'} must be estimated using paired CRN — they are differences of
differences and are very noisy without pairing.

### 2.6 Empirical observables — Level-specific metrics

For each level, predictability and closure are measured at that level:

| Metric | Level | Definition | Supports |
|---|---|---|---|
| P_B^seed | Level 1 | within-seed ρ_Spearman(Q_i(S), G_i(S)) over a schedule sample, then averaged over seeds | seed-wise pairwise diagnostic |
| P_B^pop  | Level 2 | ρ_Spearman(Q̄(S), Ḡ(S)) over the schedule sample | global / population schedule structure |
| P_B^feat | Level 3 | held-out ρ_Spearman(Q̂_i(S), G_i(S)) over schedules within seed i, averaged over held-out seeds | feature-conditioned (deployable) scheduler |
| C_B^pop  | Level 2 | (G(Ŝ^pop) − G(uniform)) / (G(S_C^{MC,N}) − G(uniform)) on test seeds, with Ŝ^pop a single global schedule | population scheduler closure (vs MC pool) |
| C_B^feat | Level 3 | (𝔼_i[G_i(Ŝ_{Q̂_i})] − G(uniform)) / (G(S_C^{MC,N}) − G(uniform)) on held-out seeds | feature-conditioned scheduler closure |
| held-out gain | Level 3 | G(Ŝ_{Q̂_i}) − G(uniform) per held-out seed | Level-3 usefulness |

**Phase 1 settles Levels 1 and 2** (population and seed-wise structure of ξ).
**Phase 2b settles Level 3** (deployable feature-conditioned scheduler).
The thesis can claim a **deployable scheduler** only if C_B^feat or P_B^feat
is statistically significant on held-out seeds.

### 2.7 Theorem A as a special case of Theorem B′

Setting Q := A in Theorem B′ recovers a finite-pool, high-probability,
estimator-aware version of Theorem A:

  G(S_B^*) − G(Ŝ) ≤ κ_B + 2 η_{B,C} + 2 ε_{B,C,δ} + ω_B

where η_{B,C} := sup_{S ∈ C_B} |G(S) − A(S)| and ε_{B,C,δ} :=
sup_{S ∈ C_B} |A(S) − Â(S)| (held-out, w.p. ≥ 1 − δ). This is the safe
rigorous regret statement that A′ and A″ were trying to provide.

### 2.8 Falsifiers (Theorem B)

Theorem B is *useful* on regime R if ζ_{B,C} < η_{B,C} **and/or**
P_B^{level} > R_B at the relevant level, with uncertainty accounted for,
and the held-out scheduler at the matching level beats top-B separable rankers.

Theorem B is *falsified as a useful theory* on R if any of:
  (F-B1) ζ_{B,C} ≈ η_{B,C} (pairwise approx no better than additive on C_B);
  (F-B2) α_{B,δ} so large that 2 α_{B,δ} swamps 2 (η_{B,C} − ζ_{B,C});
  (F-B3) Held-out Level-3 Q̂-scheduler does not beat separable rankers.

(F-B1) → regime IV (higher-order / chaotic). (F-B2) → undersampled surrogate;
shrinkage or larger training pool. (F-B3) → optimizer or feature-conditioning
is the bottleneck; CD-G remains as comparison.

Status: §2.1, §2.2, §2.3 statements **proved** under stated assumptions
(uniform-bound or finite-pool / high-probability). The non-trivial empirical
work is estimating ζ_{B,C}, α_{B,δ}, P_B^{level}, κ_B with valid seed splits
and a fixed, no-leakage candidate-pool construction.

---

## 3. Diagnostic Framework C — Regime classification

**Role.** A diagnostic taxonomy. Not a theorem; rather a statistical
framework for classifying a (model, corrector, F, B) triple into one of five
regimes, each with a recommended policy class. Renamed from "Proposition C"
because it is a definition + protocol, not a proven proposition.

### 3.1 Diagnostics

For a fixed (model, corrector, F, B) triple over a paired-seed sample, fix a
pre-specified candidate pool C_B and sampling distribution (e.g. MC pool of
N random size-B schedules). Define:

  U_B^{MC,N} := G(S_B^{MC,N}) − G(S_B^{uniform})    (MC-oracle headroom; N is part of the diagnostic)
  U_B^{pool}  := G(S_C^*) − G(S_B^{uniform})        (pool-oracle headroom on C_B)
  R_B := ρ_Spearman(A(S), G(S))                      (marginal rankability)
  I_B := σ(G(S) − A(S)) / (σ(A(S)) + δ_0)            (interaction strength; δ_0 small)
  P_B := ρ_Spearman(Q(S), G(S))                      (pairwise predictability; cf. §2.6 Level-specific variants)
  C_B^{MC,N}(method) := [G(method) − G(uniform)] / [G(S_B^{MC,N}) − G(uniform)]
                                                     (closure ratio against MC pool; defined when denominator
                                                      CI excludes 0)

> **Notation discipline.**
> - **U_B^{MC,N}**: best-of-N MC-oracle headroom (N and the schedule
>   sampling distribution are part of the diagnostic).
> - **U_B^{pool}**: best-on-pool oracle headroom on a fixed C_B.
> - **U_B^***: reserved for the unobservable exhaustive (T choose B)
>   oracle headroom; **never reported** from experiments.
> - "MC-oracle" / "pool oracle" / "exhaustive oracle" are distinct objects;
>   active docs must use the precise term.

All quantities are point estimates with BCa bootstrap 95 % CIs over seeds.
For ξ-based or Q-based diagnostics where pair / schedule sampling is sparse,
see Phase 1 uncertainty protocol in `docs/06_theory_first_research_plan.md`.

### 3.2 Operational "high"/"low"

"High" / "low" are **statistical comparisons**, not universal thresholds.
Specifically:

- "U_B > 0" means the BCa CI for U_B excludes 0.
- "R_B high" / "P_B > R_B" means the bootstrap CI of P_B − R_B excludes 0.
- "I_B high" means I_B's BCa CI lies above the analogous quantity at a
  reference (for instance, ξ-shuffled control).
- "C_B(method) statistically positive" means CI excludes 0.

We do not impose universal numerical thresholds; classification decisions are
taken at supervisor meetings against the measured CIs.

### 3.3 Regime taxonomy

| Regime | Diagnostic signature | Appropriate policy class |
|---|---|---|
| **I. No-op** | U_B ≈ 0 | corrector timing not meaningful at this B |
| **II. Marginal/rankable** | U_B > 0; R_B high; I_B low; ranker C_B ≈ 1 | top-B-by-ψ rankers (Theorem A) |
| **III. Interaction-driven** | U_B > 0; R_B low/moderate; I_B high; P_B > R_B | pairwise surrogate / CD-G / BS-AG (Theorem B) |
| **IV. Higher-order / chaotic** | U_B > 0; R_B low; P_B low; only true-G search closes any C_B | search with G feedback (CD-G); no compact predictive surrogate |
| **V. Online-decision** | offline structure exists *and* compact z_t has predictive value *and* budget-aware policy beats non-budget-aware ranker | online controller (Theorem D) |

### 3.4 Provisional classification of ProSeCo-OWT

Under the prior baseline (April 2026 results, to be confirmed by Phase 0):
U_B > 0 at B ∈ {2,3,4} (MC-oracle headroom +0.45); R_B moderate;
A″-style rank diagnostic R_B 0.60 → 0.46 across B ∈ {2,3,4}; tested separable
rankers do not recover U_B^{MC,N} (Empirical Ranker-Class Limitation, §1.5);
CD-G/BS-AG close 49–84 % of U_B^{MC,N}.

This **provisionally** places ProSeCo-OWT into Regime III or IV at
B ∈ {2,3,4}. Phase 1 (interaction diagnostics) is the experiment that
distinguishes them.

### 3.5 Use in the thesis

Diagnostic Framework C is **not** a universality claim. It is a
diagnostic *protocol*: given a new (model, corrector, F, B), measure
(U_B, R_B, I_B, P_B, C_B), classify the regime, choose the policy class.
The thesis claim is that this protocol — together with the per-regime
policy recommendation — is well-defined and operationally useful. Status:
**definitional**; empirical content is contributed by Phase 0 + Phase 1.

---

## 4. Theorem D — Online budgeted controller (optional / appendix)

**Role.** Optional, appendix-grade. Theorem D abstracts "when to correct" as
a finite-horizon budgeted online decision problem. Conceptually useful but
**generic**; does not by itself establish that current trajectory signals
are a sufficient state. Empirical value depends on a separate
state-sufficiency test that has so far been negative (Protocol C).

### 4.1 Setup and statement (proof sketch)

State z_t ∈ 𝒵, action a_t ∈ {0,1}, budget recursion b_{t+1} = b_t − a_t,
b_1 = B, terminal reward F(y_T). Value function

  V_t(z, b) = sup_π 𝔼[F(y_T) | z_t = z, b_t = b, π],

Bellman recursion V_t(z, b) = max_{a ≤ b} 𝔼[V_{t+1}(z_{t+1}, b − a) | z, a].
Let V̂_t be an approximation and π̂ its one-step-greedy policy.

**Performance-loss bound (standard ADP).** If |V_t(z, b) − V̂_t(z, b)| ≤ δ
for all reachable (t, z, b), then

  V_1(z_1, B) − 𝔼[F(y_T) | π̂] ≤ 2 T δ.

**Proof sketch.** Standard ADP suboptimality: at each step the greedy gap
relative to true V is at most 2δ; errors aggregate over T decision points
because all T greedy choices depend on V̂_{t+1} accuracy, including those
choosing a_t = 0. A sharper c B δ form is **not** available in general;
adopting it requires structural assumptions we cannot currently justify. ∎

### 4.2 Connection to Protocol C and falsifiers

Protocol C tested z_t = (signal_quartile, phase) on 12 buckets and found
ε̃ / ε ∈ [0.983, 0.986] — essentially no bucket-level value-function
approximation. Under Theorem D this means either z_t is too coarse or the
conditional gain is unstructured at that resolution. Phase 4 (only if
reached) should test richer continuous z_t with a learned approximator.

Theorem D is *useful* if (a) some compact z_t admits small ‖V − V̂‖_∞ and
(b) the resulting π̂ beats the best Theorem-A or Theorem-B policy at the
same B. Otherwise it stays in the appendix as the formal companion to
Protocol C's honest negative.

Status: standard ADP; thesis value depends on Phase 4. **Optional /
appendix unless promoted by empirical result.** First to cut if time tightens.

---

## 5. Lemma E — Burn-in exclusion under bounded-Lipschitz F (optional)

**Role.** Optional side lemma. The naive Lipschitz statement against F = −GPT-2
NLL is **fragile**: a single token change can dramatically alter later-token
likelihoods, so F is not Lipschitz in normalized Hamming distance in general.

### 5.1 Conditional statement

Let F_C be a **clipped or bounded-Lipschitz surrogate** of F (e.g. clipped
per-token NLL contributions) satisfying

  |F_C(y) − F_C(y')| ≤ L_F · d_H(y, y').

Then |Δ_t^{F_C}| ≤ L_F · |R_t| / D, and excluding step t from the candidate
schedule costs at most L_F · |R_t| / D in the additive surrogate (and at most
B L_F · |R_t| / D + 2 η_B in joint gain by Theorem A's combining argument).

> **Warning.** The actual experimental F = −GPT-2 NLL is **not** guaranteed
> to satisfy a normalized-Hamming Lipschitz condition without clipping. Use
> Lemma E as intuition or apply to a clipped surrogate F_C; do not rely on it
> in the main theorem stack.

### 5.2 Empirical alternative

The operational way to justify burn-in exclusion is to **measure** R_t, TCR_t,
Δ_t directly (Phase 0). If R_t ≈ ∅ for t ≤ T_burn implies measured
Δ_t ≈ 0, that is the empirical statement; Lemma E is then optional commentary.

Status: **conditional sketch**; appendix or side lemma only.

---

## 6. Backbone and contribution classification

**Main body of thesis:**
- §0 Formal setup, levels of analysis, notation.
- §1 Theorem A (marginal baseline, uniform form, proved). Diagnostics A′, A″
  (additivity scale, rankability — *not* unconditional regret refinements).
  Safe finite-pool ranking corollary (Theorem A as a special case of B′).
  Empirical Ranker-Class Limitation (formal time-only part; empirical part
  on tested separable rankers).
- §2 Theorem B exact, Theorem B estimated, Theorem B′ (finite candidate pool /
  high probability), §2.6 Level-specific metrics, §2.7 Theorem A as B′(Q := A).
- §3 Diagnostic Framework C (regime classification, MC-oracle / pool-oracle
  notation discipline).
- Experiments testing (A2)/(A3)/(B2)/(B2′)/(B3)/(B3′) with no-leakage pool
  construction and Phase 1 uncertainty protocol.

**Appendix / optional:**
- §4 Theorem D online controller; first to cut.
- §5 Lemma E burn-in (conditional/clipped form).
- Theorem A-ad (Protocol C honest negative).
- Stretch C2 (Gibbs contraction; not on critical path).

Backbone narrative:

> Theorem A says when separable rankers should work. Diagnostics test (A2)/(A3).
> If tested separable rankers do not recover MC-oracle headroom (Empirical
> Ranker-Class Limitation regime,
> §1.5), Theorem B says when pairwise scheduling should work; Theorem B′ is
> the finite-pool, high-probability, experimentally usable form. Diagnostic
> Framework C classifies which regime the (model, corrector, F, B) triple is
> in. Theorem D and Lemma E are optional / appendix.

---

## 7. Theory-to-experiment map

| Theorem / object | Assumption / prediction | Empirical test (Phase) | If supported | If falsified |
|---|---|---|---|---|
| Theorem A (A2) | |G − A| ≤ η_B (uniform or finite-pool η_{B,C}) | η_{B,C} on candidate pool, Phase 0/2b | Theorem A non-vacuous on C_B | → Theorem B |
| Theorem A (A3) | |Δ − ψ| ≤ ε on C_B | ε on candidate pool, Phase 0 | Marginal regime II | Empirical Ranker-Class Limitation; → B |
| Theorem A util | 2Bε + 2η_{B,C} < ranker headroom | plug-in vs measured on C_B | Theorem A operative on C_B | Structural bound only |
| Diagnostic A′ | typical η^typ_{B,D} small on schedule distribution D | Phase 0/2b residuals | A is informative on D | A is uninformative on D |
| Diagnostic A″ | R_B = ρ(A, G) high | Phase 0 ranking | A is rank-informative | Move to richer surrogate |
| Empirical Ranker-Class Limitation (formal) | time-only separable ψ ≤ mean-Δ̄ envelope on Ā | Phase 2b separable-ranker comparison | Tested rankers bounded; → B | n/a (formal part is exact) |
| Empirical Ranker-Class Limitation (empirical) | tested separable rankers do not recover MC-oracle headroom on ProSeCo-OWT | Phase 2b paired CIs | Negative result on tested class | Tested ranker recovers headroom |
| Theorem B (B2) | ζ_{B,C} < η_{B,C}; P_B^{level} > R_B at level | Phase 1 schedule-level validation after sparse pair diagnostics | Interaction regime III | → Regime IV |
| Theorem B (B3)/(B3′) | |Q − Q̂| ≤ α_{B,δ} on C_B (held-out) | Phase 1/2 leave-seed-out split, fixed pool | Pairwise scheduler buildable on C_B | Surrogate undersampled / pool data-dependent |
| Theorem B′ κ_B | C_B near MC pool oracle | Phase 2 pool comparison | Within-pool regret meaningful for MC pool | Restrict claim to within-pool |
| Theorem B util | C_B^feat statistically positive on held-out seeds | Phase 2b feature-conditioned eval | **Theorem B central; Level-3 deployable** | Population-only or non-deployable |
| Diagnostic C | regime CIs stable at K=30 | Phase 0 + Phase 1 with seed+pair-pool resampling | Diagnostic framework valid | Single-backbone case study |
| Theorem D | compact z_t admits small ‖V−V̂‖_∞ | Phase 4 (if reached) | Online controller in main | Appendix only |
| Lemma E | clipped F_C is L_F-Lipschitz; R_t small ⇒ Δ_t^{F_C} small | Phase 0 audit of R_t, L_F on F_C | Side lemma in main | Side remark only |

## Provenance — superseded theorem stack

> The April 2026 theorem stack (Phase 3b vintage), including the
> previously-circulated proofs of "Refinement A′", "Refinement A″",
> the "Negative-Result Corollary", "Proposition C" (γ B(B−1)/2 pairwise
> interaction bound), and "Theorem A-ad", was moved to
> `docs/archive/old_theory_stack/candidate_theorems_pre_2026_05.md`.
> It is **superseded**. Do not cite it as current. The active
> replacements are stated in §1.3 (A′, A″ as diagnostics), §1.4
> (safe finite-pool corollary), §1.5 (Empirical Ranker-Class Limitation),
> §2 (Theorem B / B′), §3 (Diagnostic Framework C). Git history
> preserves the full editorial trail.
