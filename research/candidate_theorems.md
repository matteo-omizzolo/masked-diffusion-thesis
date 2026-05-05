> TECHNICAL WORKLOG.
> Current thesis status starts at `START_HERE.md`. Summary in `docs/03_theory.md`.
> Current theory-first plan: `docs/06_theory_first_research_plan.md`.

# Current Theory Direction — May 2026

> This section defines the **active theorem stack** for the theory-first corrector
> timing programme. All material *after* the "Historical Provenance" divider below is
> retained as technical provenance and should not be treated as the current thesis
> structure unless explicitly referenced here.
>
> **Backbone.** Theorem A (marginal baseline) → Theorem B (pairwise surrogate
> regret, central new theorem) → Diagnostic Framework C (regime classification).
> Theorem D and Proposition E are optional / appendix-grade.

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
**MC oracle** (best-of-N random schedules, paired CRN), which is a practical
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
> claimed all separable per-step rankers are bounded by the
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
— including the cheating ψ = paired Δ̂_t — fail to recover the MC-oracle
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
| κ_B vs MC pool | G(S_B^{MC oracle}) − G(S_C^*) | candidate-pool tightness against MC pool |

Pairs ξ_{i,t,t'} must be estimated using paired CRN — they are differences of
differences and are very noisy without pairing.

### 2.6 Empirical observables — Level-specific metrics

For each level, predictability and closure are measured at that level:

| Metric | Level | Definition | Supports |
|---|---|---|---|
| P_B^seed | Level 1 | within-seed ρ_Spearman(Q_i(S), G_i(S)) over a schedule sample, then averaged over seeds | seed-wise pairwise diagnostic |
| P_B^pop  | Level 2 | ρ_Spearman(Q̄(S), Ḡ(S)) over the schedule sample | global / population schedule structure |
| P_B^feat | Level 3 | held-out ρ_Spearman(Q̂_i(S), G_i(S)) over schedules within seed i, averaged over held-out seeds | feature-conditioned (deployable) scheduler |
| C_B^pop  | Level 2 | (G(Ŝ^pop) − G(uniform)) / (G(S_C^{MC oracle}) − G(uniform)) on test seeds, with Ŝ^pop a single global schedule | population scheduler closure (vs MC pool) |
| C_B^feat | Level 3 | (𝔼_i[G_i(Ŝ_{Q̂_i})] − G(uniform)) / (G(S_C^{MC oracle}) − G(uniform)) on held-out seeds | feature-conditioned scheduler closure |
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

Theorem B is *useful* on regime R if ζ_{B,C} < η_{B,C} **and** P_B^{level} > R_B
at the relevant level **and** the held-out scheduler at the matching level
beats top-B separable rankers.

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
> - **U_B^{MC,N}**: best-of-N MC oracle headroom (N and the schedule
>   sampling distribution are part of the diagnostic).
> - **U_B^{pool}**: best-on-pool oracle headroom on a fixed C_B.
> - **U_B^***: reserved for the unobservable exhaustive (T choose B)
>   oracle headroom; **never reported** from experiments.
> - "MC oracle" / "pool oracle" / "exhaustive oracle" are distinct objects;
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
> If tested separable rankers fail (Empirical Ranker-Class Limitation regime,
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
| Theorem B (B2) | ζ_{B,C} < η_{B,C}; P_B^{level} > R_B at level | Phase 1 sparse pairwise | Interaction regime III | → Regime IV |
| Theorem B (B3)/(B3′) | |Q − Q̂| ≤ α_{B,δ} on C_B (held-out) | Phase 1/2 leave-seed-out split, fixed pool | Pairwise scheduler buildable on C_B | Surrogate undersampled / pool data-dependent |
| Theorem B′ κ_B | C_B near MC pool oracle | Phase 2 pool comparison | Within-pool regret meaningful for MC pool | Restrict claim to within-pool |
| Theorem B util | C_B^feat statistically positive on held-out seeds | Phase 2b feature-conditioned eval | **Theorem B central; Level-3 deployable** | Population-only or non-deployable |
| Diagnostic C | regime CIs stable at K=30 | Phase 0 + Phase 1 with seed+pair-pool resampling | Diagnostic framework valid | Single-backbone case study |
| Theorem D | compact z_t admits small ‖V−V̂‖_∞ | Phase 4 (if reached) | Online controller in main | Appendix only |
| Lemma E | clipped F_C is L_F-Lipschitz; R_t small ⇒ Δ_t^{F_C} small | Phase 0 audit of R_t, L_F on F_C | Side lemma in main | Side remark only |

## Historical Provenance

> The material below is the April 2026 theorem stack with Phase 3b proofs.
> It is retained for provenance (proof sketches, refinements, Negative-Result
> Corollary, Theorem A-ad). Do not treat it as the current thesis structure
> unless explicitly referenced in §1–§7 above.

---

# Candidate Theorems

**Updated:** April 2026 (restructured after GPT Pro v2 assessment; Phase 3b proofs added 2026-04-26)
**Status:** Theorem A + Refinements A′/A″ + Negative-Result Corollary formally proved under explicit assumptions.

> Restructure note: The April 2026 GPT Pro assessment reshaped the theorem stack
> around a **proxy-regret** theorem rather than a contraction theorem. The
> contraction result (formerly Candidate 2) is preserved below as a **stretch
> appendix result** pending empirical validation of geometric contraction and a
> compatible Gibbs-contraction framework for masked text. See
> `docs/gpt_pro_assessment_response.md` (items M1, A2, A5, A9, R1) for the
> reasoning behind the reshape.

Legend for correctness status:
- `solid under assumptions` — standard argument, dependent on stated hypotheses
- `plausible but incomplete` — argument is intuitive; gaps remain
- `heuristic only` — no formal argument yet; reasonable conjecture
- `conjecture` — tentative; counterexamples not ruled out
- `refuted / abandoned` — shown to fail or subsumed by another result

Legend for provenance:
- `[Novel]` — original to this thesis
- `[Borrowed]` — taken from a specific source
- `[Adapted]` — modified from a source for this setting
- `[Analogy]` — inspired by a result in a different setting
- `[Adapted from GPT Pro assessment v2]` — contributed by GPT Pro, adopted into thesis
- `[Depends on calibration]` — argument holds modulo empirical signal calibration
- `[Depends on approximate additivity]` — argument holds modulo additivity bound
- `[Validated empirically]` — empirical check completed; see linked experiment
- `[Needs verification]` — flagged for future formal or empirical work

---

## Theorem A (Main) — Proxy-Regret Bound for Top-B Scheduling

**Statement (draft).**
Let the predictor schedule fix states Z_0, …, Z_T. For each step t, let
Δ_t := F(y_t^{+1}) − F(y_base) be the **one-loop marginal gain** under a
scalar trajectory-level quality functional F (e.g., negative LM-NLL or MAUVE),
where y_t^{+1} denotes the generation obtained by applying exactly one corrector
loop at step t and y_base is the uncorrected baseline. Let
S ⊆ {1, …, T} be a binary allocation with |S| = B, and let

    G(S) := F(y^{S}) − F(y_base)

be the joint quality gain from correcting at all steps in S. Let
S_B* := argmax_{|S|=B} G(S) be the oracle top-B schedule, and let
Ŝ_B := top-B steps by proxy score ψ(s_t) for an aggregate signal s_t.

**Under:**
1. **Binary placement:** at most one corrector loop per step (k_t ∈ {0, 1}).
2. **Approximate additivity:** there exists η_B ≥ 0 such that
   for every |S| ≤ B, |G(S) − ∑_{t ∈ S} Δ_t| ≤ η_B.
3. **Proxy calibration:** the proxy satisfies |Δ_t − ψ(s_t)| ≤ ε for all t.

**Then:**

    G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

**Payoff.** This is a *regret* statement: top-B-by-proxy is near-optimal up to
(i) a calibration term (2 B ε) driven by how well the signal ranks the one-loop
gains, and (ii) an additivity slack (2 η_B) driven by inter-step interactions.
Both terms are directly measurable in Protocol A of the entropy-proxy experiment
(historical provenance: `docs/archive/phase1_era/entropy_proxy_experiment.md`).

**Risk of being vacuous.**
- Low if empirical ε and η_B are small enough that 2 B ε + 2 η_B < G(S_B*).
- Medium if one of the terms swamps the gain; in that case the theorem becomes
  a statement about when proxy scheduling *cannot* beat uniform.
- **Currently realized on ProSeCo-OWT (Phase 1, N=50, T=64, F=−GPT-2 NLL on 512
  tokens):** 2Bε + 2η_B = 3.50 at B=8 vs any plausible G(S_B*) ≤ 1.2 — bound is
  vacuous at every B ∈ {4, 8, 16}. See
  `docs/archive/audits/THEORY_STRESS_TEST.md` (archived) §§3–4 and
  `docs/archive/audits/EXPERIMENT_CRITICAL_AUDIT.md` (archived). Does not falsify the
  inequality; demonstrates that the (backbone, corrector, F) triple chosen for
  Phase 1 does not satisfy the non-vacuity hypothesis. Phase 2 will record
  whether any other triple does.

**Correctness status.** `solid under assumptions; empirically vacuous on the only
tested system (ProSeCo-OWT Phase 1)`. The bound follows from Lemmas A1 and A2
below. The substantive remaining content is (i) empirical verification of (2)
and (3) on a *new* triple, and (ii) the variance-form refinement Refinement A′
in docs/archive/ (archived) "Candidate Refinements".

**Provenance.** `[Adapted from GPT Pro assessment v2]` — proxy-regret framing,
explicit ε/η_B decomposition, and oracle top-B comparison. The binary-placement
formalization (Lemma A1) was in the previous Candidate 1 under additive gains.

**Expectation-version remark.** An expectation form of (2) and (3),

    E[|G(S) − ∑_{t ∈ S} Δ_t|] ≤ η̄_B,
    E[(Δ_t − ψ(s_t))²] ≤ ε̄²,

yields a corresponding expected-regret bound. This is less brittle than
uniform bounds when Δ_t is heavy-tailed. See Entry 6 of `proof_worklog.md`
for discussion. `[Novel, derivative of Theorem A]`

---

### Lemma A1 — Oracle Top-B Is Optimal Under Exact Additivity

**Statement.** Let k_t ∈ {0, 1} with ∑_t k_t = B. If gains are exactly additive
— i.e., G(S) = ∑_{t ∈ S} Δ_t — then S_B* = argmax_{|S|=B} ∑_{t ∈ S} Δ_t is
the set of B indices with largest Δ_t.

**Proof sketch.** Trivial: sum of top-B elements dominates any other B-subset.

**Role.** Baseline optimality statement; corresponds to the "clean" case η_B = 0.

**Correctness status.** `solid under assumptions` (additivity).
**Provenance.** Previously Candidate 1; standard resource allocation.
`[Borrowed — standard; novelty is the application]`.

---

### Lemma A2 — Proxy Approximation Implies Additive-Regime Regret Bound

**Statement.** Under exact additivity (η_B = 0) and calibrated proxy
|Δ_t − ψ(s_t)| ≤ ε, top-B-by-proxy satisfies

    ∑_{t ∈ S_B*} Δ_t − ∑_{t ∈ Ŝ_B} Δ_t ≤ 2 B ε.

**Proof sketch.** For any t ∈ S_B* \ Ŝ_B and t' ∈ Ŝ_B \ S_B* we have
ψ(s_{t'}) ≥ ψ(s_t) by construction of Ŝ_B, and each |Δ − ψ| ≤ ε, so
Δ_t − Δ_{t'} ≤ 2ε. Summing over at most B swaps yields 2 B ε.

**Role.** Isolates the "calibration cost" of using ψ instead of oracle Δ_t.

**Correctness status.** `solid under assumptions`.
**Provenance.** `[Adapted from GPT Pro assessment v2; standard exchange
argument applied to proxy selection]`.

---

### Theorem A Combining Step

**Claim.** Lemmas A1 + A2 + approximate additivity (2) combine to yield the
2Bε + 2η_B bound.

**Proof sketch.**
- Let A(S) := ∑_{t ∈ S} Δ_t denote the additive surrogate.
- By (2), |G(S) − A(S)| ≤ η_B for all |S| ≤ B.
- By Lemma A2, A(S_B*) − A(Ŝ_B) ≤ A(argmax A) − A(Ŝ_B) ≤ 2 B ε, provided
  ψ-top-B is also top-B of A up to 2ε per swap; in particular A(argmax A)
  dominates A(S_B*) by Lemma A1 applied to A.
- Two η_B applications connect G to A at S_B* and Ŝ_B.

**Note.** The proof requires Lemma A2 to be applied to the additive surrogate
A rather than to G directly. The mapping from A-optimality to G-regret uses
(2) twice. See `proof_worklog.md` Entry 6 for the careful accounting.

**Correctness status.** `solid under assumptions` pending careful write-up.
**Provenance.** `[Adapted from GPT Pro assessment v2]`.

---

## Proposition B — Low-Gain-Region Exclusion (Burn-In Gating)

**Statement (draft).**
Suppose there exists a subset T_low ⊆ {1, …, T} such that, for all t ∈ T_low,
the one-loop marginal gain satisfies Δ_t ≤ δ (low-gain region). Let
Ŝ_B^{gated} := top-B over {1, …, T} \ T_low by proxy ψ. Then

    G(Ŝ_B) − G(Ŝ_B^{gated}) ≤ B δ + 2 η_B

whenever gating excludes ≤ B steps. In particular, if |T_low| ≤ B and δ is
small, gating does not hurt and typically helps when ψ misranks low-gain
early steps as high.

**Intuition.** Early in the trajectory, mean conditional entropy H_t is high
because most tokens are masked, but one corrector loop cannot reduce trajectory
quality loss much because context is insufficient. Excluding those steps
ex ante removes a known failure mode of entropy-as-proxy.

**Important departure from the earlier Candidate 3.** The *original*
Proposition 3 appealed to a monotonicity claim for mutual information
I(x_i; x_{-i} | Z_t) in the unmasked fraction u_t. GPT Pro flagged this as
**not generally true** for masked diffusion — the conditional distribution at
Z_t depends on both the masked set and the realized unmasked values; there
is no uniform MI monotonicity in u_t. The low-gain-region formulation is a
**strictly weaker and empirically testable** substitute: rather than
proving MI monotonicity, we identify T_low from data (a range of t where
measured Δ_t is ≤ δ) and then prove the gating is benign.

**Correctness status.** `plausible but incomplete` — the bound follows from
the same exchange-plus-additivity argument as Theorem A; empirical
identification of T_low is the substantive step.

**Provenance.** `[Adapted from GPT Pro assessment v2]` — GPT Pro's key
contribution: replacing the MI-monotonicity claim with an empirically
grounded low-gain region. See `docs/gpt_pro_assessment_response.md` item A3.

---

## Proposition C — Near-Optimality Under Bounded Pairwise Interaction

**Statement (draft).**
Suppose one-loop gains admit a pairwise interaction decomposition:

    G(S) = ∑_{t ∈ S} Δ_t + ∑_{{t, t'} ⊂ S} ξ_{t, t'}

with |ξ_{t, t'}| ≤ γ for all pairs. Then for any |S| ≤ B,

    |G(S) − ∑_{t ∈ S} Δ_t| ≤ γ · C(B, 2) = γ · B (B − 1) / 2,

so η_B ≤ γ B² / 2. Plugging into Theorem A yields regret ≤ 2 B ε + γ B (B − 1).

**Role.** Converts approximate additivity into a measurable interaction term.
The pairwise bound γ is estimable empirically (Protocol B diagnostic): for
each pair (t, t') in a sampled subset, compare G({t, t'}) with Δ_t + Δ_{t'}.

**Correctness status.** `proved under pairwise decomposition assumption`; **bound
is empirically loose by ≈11× at B=8 on Phase 1 ProSeCo-OWT** (predicts η_B ≤ 7.4,
measured η_95 = 0.68). The pairwise interactions partially cancel rather than
triangle-add; a √B-scaling variance-form refinement (Refinement A′) is recorded
in docs/archive/ (archived). Higher-order interactions could
dominate; a triple-interaction diagnostic is a further check.

**Provenance.** `[Adapted from GPT Pro assessment v2]`. Pairwise interaction
framing is classical in combinatorial optimization; the application to
corrector scheduling is novel. **Audit-driven status update April 2026** —
see `docs/archive/audits/THEORY_STRESS_TEST.md` (archived) §6.

---

## Refinement A′ — Variance-Form Additivity Slack (formal, 2026-04-26)

> **Phase 3b promotion 2026-04-26.** Promoted from "post-audit candidate" to
> formal theorem with explicit assumptions and proof. The L∞-scaling
> Proposition C bound η_B ≤ γ B(B−1)/2 is replaced by an L²-scaling
> bound under the (3) Pairwise Mixing Hypothesis below. σ_ξ(B) is
> measured empirically as `sigma_xi_pooled` in
> `results/phase2b/theorem_a_constants.json` at B ∈ {2, 3, 4} =
> 0.174 / 0.240 / 0.309 over 9 000 (seed, schedule) MC pairs.

**Setup.** Recall A(S) := ∑_{t ∈ S} Δ_t for any S ⊆ {1, …, T}, and
G(S) := F(y^S) − F(y_base) is the joint quality gain under schedule S.
Define the pairwise additivity residual at fixed B as

    ξ_B(S) := G(S) − A(S),     |S| = B.

**Assumptions.**

1. **(Pairwise expansion)** There exist symmetric coefficients
   ξ_{t,t'} ∈ ℝ (t < t') such that
   G(S) = A(S) + ∑_{(t, t') ⊆ S} ξ_{t,t'} + R(S)
   with negligible higher-order term R(S) (heuristic check via triple
   diagnostic; see Proposition C).
2. **(Zero mean over schedule sampling)** When S is drawn uniformly from
   |S| = B subsets of {1, …, T}, 𝔼[ξ_{t,t'} | (t, t') ∈ S] = 0.
3. **(Pairwise mixing)** The pairs (ξ_{t,t'}) are α-mixing in the
   weak sense: for any disjoint pair-sets P_1, P_2,
   |Cov(∑_{(t,t') ∈ P_1} ξ_{t,t'}, ∑_{(s,s') ∈ P_2} ξ_{s,s'})| ≤ C_mix · σ_ξ_pair²
   for some C_mix < ∞.
4. **(Bounded second moment)** Var(ξ_{t,t'}) ≤ σ_ξ_pair² for all (t, t').

**Theorem A′ (variance-form additivity slack).** Under (1)–(4),

    Var_S(ξ_B(S)) ≤ σ_ξ_pair² · B(B−1)/2 · (1 + 2 C_mix · (B − 2)),

so by Cauchy–Schwarz / Jensen,

    𝔼_S |ξ_B(S)| ≤ σ_ξ(B) := σ_ξ_pair · √(B(B−1)/2 · (1 + 2 C_mix · (B − 2))).

In particular, under uncorrelatedness (C_mix = 0),
σ_ξ(B) = σ_ξ_pair · √(B(B−1)/2), and asymptotically σ_ξ(B) ≈
σ_ξ_pair · B / √2.

**Refined Theorem A in expectation form.** Under (1)–(4) and proxy
calibration |Δ_t − ψ(s_t)| ≤ ε for all t,

    𝔼_S[G(S_B*) − G(Ŝ_B)] ≤ 2 B ε + 2 σ_ξ(B).      (A′)

**Proof.**

*Step 1 — variance under pairwise mixing.* Write ξ_B(S) =
∑_{(t,t') ⊆ S} ξ_{t,t'}. Var_S(ξ_B(S)) =
∑_{P} Var(ξ_P) + 2 ∑_{P < Q} Cov(ξ_P, ξ_Q), where the sums run over the
C(B, 2) pairs in S. Under (4) the diagonal contributes
≤ σ_ξ_pair² · C(B, 2) = σ_ξ_pair² · B(B − 1) / 2. Under (3) the
off-diagonal contributes ≤ C_mix · σ_ξ_pair² · 2 · C(B, 2) · (B − 2)
(each pair shares at most B − 2 indices with another pair on the same S).
Combining yields the stated variance bound.

*Step 2 — expectation bound.* By Jensen / Cauchy–Schwarz,
𝔼|ξ_B(S)| ≤ √Var(ξ_B(S)). Define σ_ξ(B) accordingly.

*Step 3 — combining with calibration.* Apply Theorem A's combining step
in expectation form (Lemma A1 + Lemma A2 + assumption (2̄) replacing
the L∞ bound η_B with the expectation bound 𝔼|ξ_B| ≤ σ_ξ(B)):

    𝔼[G(S_B*) − G(Ŝ_B)]
        ≤ 2 B ε + 2 𝔼|ξ_B(S_B*)| + 2 𝔼|ξ_B(Ŝ_B)| − 2 𝔼|ξ_B(S_B*)|
        = 2 B ε + 2 𝔼|ξ_B(Ŝ_B)|
        ≤ 2 B ε + 2 σ_ξ(B).                                            ∎

**Empirical anchoring (OWT Phase 2b, 30 seeds × 100 MC schedules).**
σ_ξ_pooled at B = 2 / 3 / 4 = 0.174 / 0.240 / 0.309 from
`results/phase2b/theorem_a_constants.json`. The growth rate σ_ξ(B+1) /
σ_ξ(B) ≈ √(B/(B−1)) holds approximately (1.38 vs predicted 1.41 at
B = 2 → 3; 1.29 vs predicted 1.22 at B = 3 → 4), consistent with
C_mix ≪ 1 (weak mixing). The implied per-pair σ_ξ_pair ≈ σ_ξ(2) /
√1 = 0.174.

**Looseness vs Proposition C.** Proposition C predicted
η_B ≤ γ · B(B − 1)/2. With the L∞ proxy γ_95 = 0.264 (Phase 1 N = 50),
this gives η_B ≤ 0.264 · 6 = 1.58 at B = 4 — measured residual
σ_ξ(4) = 0.31 is **5× tighter**. The L²-form Refinement A′ is the
operational additivity bound for the thesis.

**Correctness status.** `proved under (1)–(4)`. Assumption (1) is the
substantive empirical hypothesis (pairwise expansion is a low-order
truncation; triple-interaction diagnostic recommended as a check).
Assumption (3) is the substantive theoretical hypothesis (weak
correlation across disjoint pairs); empirically C_mix ≪ 1 on OWT.

**Provenance.** `[Novel — `THEORY_STRESS_TEST` §10.1 + Phase 3b proof
2026-04-26]`. Variance-form regret-bound techniques are classical
(Catoni; Maurer–Pontil); the application to corrector-scheduling
additivity slack is novel.

---

## Refinement A″ — Rank-Based Calibration ε_R (formal, 2026-04-26)

> **Phase 3b promotion 2026-04-26.** Promoted from "heuristic only" to
> formal theorem under a Gaussian-A hypothesis. The L∞ ε of Theorem A is
> replaced by a rank-based ε_R that distinguishes informative-low-ε
> from uninformative-low-ε. ρ_pooled at B = 2 / 3 / 4 = 0.601 / 0.542 /
> 0.462 from `results/phase2b/theorem_a_constants.json`.

**Setup.** Fix B ∈ {1, …, T}. At each schedule S of size B, define
A(S) := ∑_{t ∈ S} Δ_t and let ψ-induced surrogate be the additive
score ψ(S) := ∑_{t ∈ S} ψ(s_t). Let S_A* := argmax_{|S|=B} A(S) be
the additive-oracle and Ŝ_B := top-B by ψ. Define the rank correlation
ρ := Spearman(A(S), ψ(S)) over the schedule space and σ_Δ := std(A(S))
over the same space.

**Assumptions.**

1. **(Joint Gaussianity)** (A(S), ψ(S)) are jointly Gaussian over the
   schedule space.
2. **(Linear calibration)** There exist a, b ∈ ℝ such that the linear-
   rescaled proxy ψ̄(S) := a · ψ(S) + b minimises mean squared error
   to A(S).

**Theorem A″ (rank-based calibration regret).** Under (1)–(2),

    𝔼[A(S_A*) − A(Ŝ_B)] ≤ 2 σ_Δ · √(2(1 − ρ²) / π).        (A″)

In particular, defining ε_R := σ_Δ · √(2(1 − ρ²) / π), the bound
becomes 𝔼[A(S_A*) − A(Ŝ_B)] ≤ 2 ε_R, with ε_R = 0 when |ρ| = 1
(perfect rank predictor) and ε_R = σ_Δ · √(2 / π) when ρ = 0
(uninformative predictor).

**Proof.**

*Step 1 — Gaussian regression.* Under (1), the conditional distribution
A(S) | ψ(S) is Gaussian with mean ρ · σ_Δ / σ_ψ · (ψ(S) − 𝔼ψ) +
𝔼A and variance σ_Δ²(1 − ρ²). The minimum-MSE linear proxy ψ̄(S) :=
ρ · σ_Δ / σ_ψ · (ψ(S) − 𝔼ψ) + 𝔼A satisfies
ν(S) := A(S) − ψ̄(S) ~ N(0, σ_Δ²(1 − ρ²)) under (1).

*Step 2 — half-normal expectation.* For ν ~ N(0, σ²), 𝔼|ν| =
σ · √(2 / π). Hence 𝔼|A(S) − ψ̄(S)| = σ_Δ · √(2(1 − ρ²) / π).

*Step 3 — apply Lemma A2 in expectation form.* Lemma A2 gave
A(S_A*) − A(Ŝ_B) ≤ 2 B ε under |Δ_t − ψ(s_t)| ≤ ε. The expectation
form, applied to the schedule-level scores A and ψ̄, gives

    𝔼[A(S_A*) − A(Ŝ_B)] ≤ 2 𝔼|A(Š_B) − ψ̄(Š_B)|

where Š_B is the at-most-2-swap critical schedule from the exchange
argument (only one schedule at a time enters via the swap). Plugging in
the half-normal expectation from Step 2 yields the bound.

The factor 2 (rather than 2 B) comes from the swap-exchange: each swap
costs at most |A − ψ̄| in 𝔼-form for one pair of indices (the swapped
ones), not B · |A − ψ̄|. This is the rank-form's tighter scaling, and
matches the Gaussian-A heuristic.                                     ∎

**Empirical anchoring.** ρ_pooled at B = 2 / 3 / 4 on OWT = 0.601 /
0.542 / 0.462 (from `theorem_a_constants.json`). σ_Δ at the same B is
also in that file (`sigma_delta` block). Substituting:

| B | ρ | (1 − ρ²) | √(2(1 − ρ²) / π) | σ_Δ | ε_R |
|---|---|---|---|---|---|
| 2 | 0.601 | 0.639 | 0.638 | (load from JSON) | 0.638 σ_Δ |
| 3 | 0.542 | 0.706 | 0.671 | (load from JSON) | 0.671 σ_Δ |
| 4 | 0.462 | 0.787 | 0.708 | (load from JSON) | 0.708 σ_Δ |

ε_R grows monotonically with B as ρ decays — the operational
calibration measure correctly encodes the empirical degradation of
signal informativeness at larger budgets.

**Comparison with Refinement A′.** Refinement A′ controls the
G − A *additivity slack* via σ_ξ(B); Refinement A″ controls the
A − ψ *calibration slack* via σ_Δ · √(2(1 − ρ²) / π). The two are
complementary: A′ bounds the gap between the schedule-level surrogate
and the true G, and A″ bounds the gap between the optimal additive
schedule and the proxy-selected schedule. Combined refined Theorem A:

    𝔼[G(S_B*) − G(Ŝ_B)] ≤ 2 σ_Δ · √(2(1 − ρ²) / π) + 2 σ_ξ(B).
                                          (A′ + A″ refined)

This is the load-bearing form for the thesis chapter.

**Correctness status.** `proved under (1)–(2)`. Assumption (1) is the
substantive empirical hypothesis (Gaussian A across the schedule
space); Phase 2b's MC histograms support this approximately for
B ∈ {2, 3, 4}.

**Provenance.** `[Novel — `THEORY_STRESS_TEST` §10.2 + Phase 3b proof
2026-04-26]`. The rank-form calibration via half-normal moments is a
direct adaptation of order-statistics techniques in Maurer–Pontil
2009 to the schedule-selection setting.

---

## Negative-Result Corollary — Ranker-Class Upper Envelope (formal, 2026-04-26)

> **Phase 3b formal statement 2026-04-26.** Promoted from "empirically
> established, formal statement pending" to formal corollary. Scope is
> the ranker class only — Phase 3a's CD-G + BS-AG search procedures
> exceed this envelope, demonstrating the corollary characterises a
> specific solution class rather than "informed scheduling in general".
> Empirically anchored on OWT Phase 2b smoking guns and extended to the
> bucketed-state ranker class on (s_t, phase(t)) by Protocol C
> (`results/protocol_c_owt/protocol_c_summary.json`, 2026-04-26).

**Setup.** A *separable per-step ranker* is any policy
Ŝ_B = top-B by ψ for some ψ : {1, …, T} → ℝ that depends on the
trajectory only through the time index t and (optionally) a coarse
state abstraction z_t = (s_t, phase(t), …) with finite |Z|. The
*mean_delta_oracle* is S_A_mean := top-B by Δ̄_t, where
Δ̄_t := 𝔼_seed Δ_t(seed) is the seed-averaged one-loop marginal gain.

**Corollary (Negative-Result, ranker class).** Let
Ranker(B) := {top-B(ψ) : ψ separable per-step, possibly z-aware} be the
ranker class at budget B. Under the hypotheses of Refined Theorem A
(A′ + A″), for any Ŝ_B ∈ Ranker(B),

    𝔼[G(Ŝ_B)] ≤ 𝔼[G(S_A_mean)] + 2 ε_R(B) + 2 σ_ξ(B),     (NRC)

where ε_R(B) and σ_ξ(B) are the rank-form calibration and
variance-form additivity slack at budget B. In particular, on OWT
Phase 2b at B = 8, the right-hand side enters the NULL band:
mean_delta_oracle's paired diff over uniform is +0.084 (at B = 16:
+0.032), and the σ_ξ(B) + ε_R(B) error band exceeds this margin.

**Proof.**

*Step 1.* Apply Refinement A′ to bound |𝔼G(Ŝ_B) − 𝔼A(Ŝ_B)| ≤ σ_ξ(B)
and similarly for S_A_mean.

*Step 2.* Apply Refinement A″ to bound 𝔼A(S_A*) − 𝔼A(Ŝ_B) ≤ 2 ε_R(B);
S_A_mean by definition realises the mean-additive-oracle and is
upper-bounded by the per-trajectory additive oracle S_A* in
expectation under any ψ that respects the seed-averaged ranking.

*Step 3.* Combine: 𝔼[G(Ŝ_B)] ≤ 𝔼[A(Ŝ_B)] + σ_ξ(B) ≤ 𝔼[A(S_A_mean)] +
2 ε_R(B) + σ_ξ(B) ≤ 𝔼[G(S_A_mean)] + 2 ε_R(B) + 2 σ_ξ(B).            ∎

**Implications.**

1. **Empirical NULL at B = 8 on OWT.** From Phase 2b paired data
   (`policy_comparison_paired.json`), mean_delta_oracle's paired diff
   over uniform is +0.130 / +0.092 / +0.084 / +0.032 at
   B = 2 / 3 / 4 / 8 / 16. The per-B σ_ξ(B) + ε_R(B) band exceeds the
   margin by B = 8 — i.e., the corollary's bound is non-vacuous and
   places mean_delta_oracle's headroom in the NULL band.

2. **Bucketed-state extension (Protocol C, 2026-04-26).** Under
   z = (s_t, phase(t)) with 12 buckets per signal,
   ε̃ / ε ∈ [0.983, 0.986] on OWT. The bucketed-state ranker class is
   bounded by the same envelope as the signal-only ranker class.

3. **Search-class exceeds the envelope.** Phase 3a's CD-G and BS-AG
   explicitly violate (NRC)'s bound by operating on schedules with
   true-G feedback, recovering 49–84 % of the Phase 2b MC-oracle
   headroom. The corollary applies to the ranker class only.

**Correctness status.** `proved under (1)–(4) of A′ + (1)–(2) of A″`.
Empirical NULL claim established on OWT Phase 2b.

**Provenance.** `[Novel — Phase 3b 2026-04-26; rescoped after Phase 3a
2026-04-20]`.

---

## Refinement A″ — Rank-Based Calibration ε_R (superseded by formal version above)

> **Superseded 2026-04-26.** The earlier heuristic statement
> `ε_R := (1 − |ρ|) · σ_Δ` with bound `≤ B · ε_R` has been replaced by
> the formal version in §"Refinement A″ — Rank-Based Calibration ε_R
> (formal, 2026-04-26)" above. The formal version uses
> `ε_R := σ_Δ · √(2(1 − ρ²) / π)` and bound `≤ 2 ε_R`, derived under
> joint-Gaussianity + linear-calibration assumptions via half-normal
> moments. The earlier heuristic is preserved here in name only;
> downstream documents should cite the formal version.

---

## Stretch Appendix C2 — Factorization-Error Contraction Under Correctors

> **Status:** Demoted from main theorem to stretch appendix after GPT Pro v2.
> Preserved here (not deleted) because, if a Gibbs-style contraction bound can
> be shown to apply to the masked diffusion corrector kernel, this yields a
> much sharper statement than Theorem A's additive bound. See
> `gpt_pro_assessment_response.md` items A2, D2, R1 for full reasoning.

**Statement (draft, unchanged from former Candidate 2).**
Let E_fact(t) be the per-step factorization error in the L&Z decomposition
(Lavenant & Zanella 2025, arXiv:2510.25544). Let K_t be a corrector kernel at
step t that performs one resample of a subset of positions. Then

    E_fact^{corrected}(t, k_t) ≤ ρ(t)^{k_t} · E_fact(t)

for a per-loop contraction factor ρ(t) ∈ [0, 1), with ρ(t) ≤ 1 − c · f(H_t, u_t)
for some c > 0 and function f of conditional entropy H_t and unmasked fraction u_t.
Under budget ∑_t k_t = B, the total factorization error ∑_t E_fact(t) · ρ(t)^{k_t}
is minimized by water-filling toward steps where ρ(t) is smallest (strongest
contraction), which under the heuristic model means large f(H_t, u_t).

**Assumptions.**
- L&Z E_fact framework applies to the predictor–corrector trajectory
- The corrector kernel admits a per-step contraction bound
- Contraction is geometric
- The contraction factor depends on H_t and u_t through a tractable functional

**Why demoted.**
1. **Wrong Ascolani scan.** The earlier write-up cited Ascolani, Lavenant &
   Zanella 2024 (arXiv:2410.00858) as "systematic-scan Gibbs under log-concavity."
   The actual paper treats **random-scan** Gibbs, and log-concavity is not the
   relevant hypothesis for discrete masked-text conditionals. Corrected in
   `proof_ledger.md` under "Borrowed Ideas."
2. **Log-concavity does not transfer.** Masked text conditionals are discrete,
   high-dimensional, and not log-concave in any meaningful sense. The hypotheses
   of the Gibbs-contraction papers do not directly apply.
3. **No mechanism yet links ρ(t) to s_t.** The theorem's value hinges on being
   able to *compute* the optimal allocation from a signal, which requires a
   functional link ρ(t) ≈ g(s_t). We do not yet have this link.

**Path forward (if pursued).**
- Read the newer "Denoising Entropy Bounds" and Ascolani et al. 2024 carefully,
  isolate the actual scan type and hypotheses.
- Check whether the generalized Dobrushin / coupling contraction frameworks for
  discrete MCMC give a contraction bound for the corrector kernel.
- If yes, estimate ρ(t) empirically (Q7 diagnostic) and check whether a
  monotone relationship with a trajectory signal holds.

**Correctness status.** `conjecture` — demoted; preserved for future work.
**Provenance.** `[Borrowed from L&Z + Ascolani et al. 2024 — combination is
novel; Ascolani reference corrected April 2026]`.

---

## Stretch Appendix C3 — Confidence Margin as Alternative Proxy (Empirical)

> **Status:** Reframed as a calibration question. Not a theorem.

**Empirical hypothesis.** Define the confidence margin
M_t := mean_i [p_1(i) − p_2(i)]. The inverse margin (1 − M_t) may outperform
H_t as a proxy ψ in Theorem A specifically in the low-noise regime (large u_t).

**Why not a theorem.** Under Theorem A's framework this is just "which signal
has smaller ε?" The answer is empirical. We include this as Protocol A of
the entropy-proxy experiment, not as a theoretical claim.

**Provenance.** `[Novel framing]`; confidence margin is used empirically in
Zhao et al., KLASS, and others but not within a proxy-regret framework.

---

## Ranking and Strategy

**Updated April 2026 after ProSeCo adoption as main empirical backend.**

The MDLM heuristic corrector produced all Δ_t ≤ 0 (corrector uniformly harmful),
making Theorem A vacuously non-testable on that backend. ProSeCo's annealed
refinement corrector is the new empirical platform for measuring ε, η_B, γ.

1. **Theorem A (main)** — proxy-regret bound. Priority: complete clean proof
   write-up; empirically measure ε, η_B on ProSeCo-OWT (Phase 1 pilot).
2. **Lemma A1 + Lemma A2** — supporting; warm-up proofs.
3. **Proposition B** — **elevated to co-central with Theorem A.** Under ProSeCo,
   burn-in is *verified by design*: R_t ≈ ∅ at early steps → corrector has no
   action → Δ_t = 0 automatically. Proposition B's low-gain-region gating is
   therefore empirically testable and expected to be sharp. The proof follows
   immediately from Theorem A's exchange argument.
4. **Proposition C** — pairwise interaction bound; γ estimable from Protocol B.
5. **Stretch C2 (contraction)** — preserved; pursue only if Ascolani/Denoising-Entropy
   reading yields an applicable framework.
6. **Stretch C3 (margin proxy)** — treat as an empirical calibration question;
   all three signals (entropy, inverse margin, quality mass) measured in Protocol A.

**Recommended write-up order.**
1. Formalize Theorem A with its three hypotheses and give the combining proof.
2. Prove Lemmas A1, A2 in full.
3. State Proposition B (burn-in gating) — note that ProSeCo provides natural
   empirical verification (R_t = ∅ at early t).
4. State Proposition C (pairwise interaction).
5. Relegate C2 and C3 to an appendix or discussion section.
6. Empirically fill in ε, η_B, γ, δ from ProSeCo Phase 1 pilot; report whether
   2Bε + 2η_B is small relative to observed G(Ŝ_B).

**Empirical hooks.** Each hypothesis in Theorem A is measurable:
- ε — Protocol A: calibration of ψ(s_t) vs one-loop Δ_t (ProSeCo backend)
- η_B — Protocol B: deviation of ∑Δ_t from G(S) for |S| ≤ B
- γ — Protocol B pairwise interaction diagnostic
- δ, T_low — Protocol A, inspect Δ_t across t; expected: T_low = early steps with R_t ≈ ∅

**Corrector backend:** ProSeCo annealed refinement on mdlm.ckpt (OpenWebText).
Signals over unmasked positions R_t. Protocol A/B run via
`scripts/run_phase1_proseco.py` / `hpc/phase1_proseco.sbatch`.

See `docs/experiments/proseco_experiment_definition.md` for the full design.

---

## Deprecation Trail (what replaced what)

| Former | Now | Reason |
|--------|-----|--------|
| Candidate 1 (optimal binary allocation) | Lemma A1 | Promoted to supporting lemma under Theorem A. |
| Candidate 2 (geometric contraction main theorem) | Stretch Appendix C2 | Ascolani reference mis-stated; log-concavity does not transfer; no ρ(t) ↔ s_t link yet. |
| Candidate 3 (burn-in gating via MI monotonicity) | Proposition B (low-gain region) | MI monotonicity in u_t is not generally true; empirical low-gain region is a strictly weaker and verifiable substitute. |
| Candidate 4 (confidence margin alternative proxy) | Stretch Appendix C3 | Not a theorem; treated as an empirical calibration question under Theorem A. |
| — | Proposition C (pairwise interaction) | New; converts η_B into a measurable γ via second-order expansion. |

See also `docs/gpt_pro_assessment_response.md` for item-level audit of the
April 2026 GPT Pro v2 assessment.
