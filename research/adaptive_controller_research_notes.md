> **STATUS:** WORKING NOTES (Phase 3 + Phase 4 of adaptive-controller research study)
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Mathematical adaptation of shortlisted frameworks to the thesis object, plus
> the framework shortlist decision. Scratch document. Polished prose lives in
> `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` (to be written in Phase 5).

---

# Adaptive Budgeted Controllers — Research Notes

## 0. Positioning

This document discharges Phases 3 and 4 of the research study launched from
`docs/thesis/next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md`. Its job is to
take the four frameworks surviving Phase 2 and write each of them in the thesis's
own variables, then decide on a small set (one normative + one algorithmic +
optionally one foil) to carry into Phase 5.

Phases completed before this file:
- Phase 0 — repo read-through.
- Phase 1 — skeptical audit (lives in
  `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md`).
- Phase 2 — web literature scan (folded directly into §2 below and §7's
  references; sources cited inline).

The four surviving frameworks are:
- F1. Finite-horizon constrained MDP (FH-CMDP).
- F2. Control-as-inference / Feynman-Kac path measure.
- F3. Particle Gibbs / conditional SMC.
- F4. Adaptive submodularity.

---

## 1. Thesis object — formal restatement

Fix a masked diffusion language model with a pre-trained predictor and a fixed
informed-corrector kernel. Let T be the number of denoising steps. The
thesis's fixed-budget corrector-scheduling problem is:

Choose a binary allocation a = (a_1, …, a_T) ∈ {0,1}^T with Σ_{t=1}^T a_t = B
that maximizes a trajectory quality functional G(S_a), where S_a = { t : a_t = 1 }.

### 1.1 Open-loop (current thesis)

The current thesis works in the **open-loop regime**: a is chosen from information
available before or independently of the trajectory realisation. Equivalently,
the policy is a deterministic subset S ⊆ {1, …, T} with |S| = B, and the scoring
proxy ψ(s_t) is computed from Protocol-A trajectory signals averaged across seeds.

The central guarantee is **Theorem A** (see `research/candidate_theorems.md`):

    G(S_B*) − G(Ŝ_B)  ≤  2 B ε  +  2 η_B,

where ψ is proxy-calibrated with |Δ_t − ψ(s_t)| ≤ ε and the joint gain is
approximately additive, |G(S) − Σ_{t ∈ S} Δ_t| ≤ η_B. All Phase 2b / 3a
experiments sit under this inequality.

### 1.2 Adaptive (candidate extension)

An adaptive controller is a policy

    π : z_t ↦ a_t ∈ {0,1}

where z_t is **the corrector-deciding information state at step t just before
deciding whether to correct**. Concretely, z_t has components that the current
thesis already computes:

- the most recent predictor state x_{t-1} (or its compressed summary),
- the trajectory signal s_t (entropy H_t, inverse margin M_t^{-1}, or quality
  mass Q_t over R_t),
- any signal traces from earlier steps the controller has chosen to carry,
- the **remaining budget** b_t := B − Σ_{τ<t} a_τ.

The dynamics are:

    a_t   = π(z_t),             with the hard constraint a_t ≤ b_t,
    b_{t+1} = b_t − a_t,        (b_1 := B),
    z_{t+1} ∼ P(· | z_t, a_t),  (the corrector kernel induces a stochastic
                                 transition on the compressed state).

The adaptive-scheduling problem is:

    max_π  𝔼_π [ G(S_π) ]        subject to   Σ_t a_t ≤ B  almost surely.

Let π*_B denote any optimiser. The **adaptive oracle** with full knowledge of the
true per-step marginals Δ_t(z_t) and joint structure plays π*_B; the **best
open-loop schedule** is S_B*; and the thesis's current policy plays
Ŝ_B = top-B_t ψ(s̄_t) with s̄_t = mean signal across pilot seeds.

### 1.3 The three regret gaps

Three quantities matter:

(i)   **Oracle gap Δ_open := 𝔼[G(S_π*_B)] − 𝔼[G(S_B*)]  ≥  0.**
      This is the *headroom* adaptive control can recover over the best fixed
      schedule. Phase 2b did not directly bound this; the MC-oracle headroom
      of +0.45 nats at B ∈ {2,3,4} is a noisy surrogate, not Δ_open.

(ii)  **Open-loop gap Δ_proxy := 𝔼[G(S_B*)] − 𝔼[G(Ŝ_B)]  ≤  2 B ε + 2 η_B.**
      This is what Theorem A controls today.

(iii) **Adaptive proxy gap Δ_ad-proxy := 𝔼[G(S_π*_B)] − 𝔼[G(S_π̂)]  = ?**
      No bound on this exists in the thesis yet. Phase 3 must write it down
      per framework; Phase 4 must decide whether it is tractable enough to
      commit to.

A clean way to keep this visible throughout the remainder of the notes:

    𝔼[G(S_π*_B)] − 𝔼[G(S_π̂)]  =  Δ_open  +  Δ_proxy  −  Δ_close(π̂),

where Δ_close(π̂) ≥ 0 measures how much of the adaptive headroom π̂ actually
recovers. An honest adaptive theorem bounds Δ_ad-proxy, not Δ_open alone.

### 1.4 What is not in z_t

The adaptive controller does **not** include:

- **Which tokens to correct at step t** (that's token-selection; scope-guarded by
  the thesis direction, see `docs/thesis_direction.md` §"Scope Boundaries").
- **How to correct** (that's kernel design; fixed to ProSeCo's informed
  corrector in the thesis's Tier A platform).
- **Whether to remask** (that's remasking; studied by RemeDi, PRISM, outside
  scope).

Preserving this boundary is a **non-negotiable**: if the adaptive controller
reaches into a_t's token subset, the thesis collapses into a token-selection
paper.

---

## 2. Framework F1 — Finite-horizon constrained MDP

### 2.1 Problem statement

Formalise the adaptive problem as a finite-horizon CMDP

    M = (Z, A, P, r, c, T, B),

with:
- state space Z (compressed signal + x_{t-1}),
- action space A = {0, 1},
- transition kernel P(· | z, a),
- reward r(z, a) := a · Δ(z)  where Δ(z) = 𝔼[one-loop marginal | z],
- cost c(z, a) := a  (each correction spends one unit of budget),
- horizon T, budget B.

The **Bellman optimality equation** under a Lagrangian relaxation with
multiplier λ ≥ 0 is

    V_λ(z, b, t)  =  max_{a ∈ {0, min(1,b)}}  { r(z,a) − λ · c(z,a)
                      + 𝔼_{z' ∼ P(·|z,a)}[ V_λ(z', b − a, t+1) ] },

with terminal V_λ(·, ·, T+1) = 0. The optimal Lagrangian policy is

    π*_λ(z, b, t)  =  𝟙[ Δ(z) > λ ]  ∧  𝟙[ b > 0 ].

Strong duality for CMDPs (Altman 1999) gives: there exists λ* ≥ 0 such that
π*_{λ*} attains the CMDP optimum, and the 'threshold rule "correct iff
Δ(z) > λ*"' is the structurally correct adaptive controller **at oracle Δ**.

### 2.2 Adaptive Theorem A analogue

Suppose we have a **state-conditional proxy** ψ̃(z) with

    | Δ(z) − ψ̃(z) |  ≤  ε̃     (ε̃ — adaptive calibration error)

and an **approximate-additivity** assumption in the adaptive setting:

    | 𝔼_π [G(S_π)]  −  𝔼_π [ Σ_t a_t · Δ(z_t) ] |  ≤  η̃_B,

i.e. the path-level quality functional is within η̃_B of the realised sum of
marginal gains when we follow policy π. Then the policy π̂_λ that plays
a_t = 𝟙[ ψ̃(z_t) > λ ] ∧ 𝟙[b_t > 0] (with λ tuned so that the expected
expenditure equals B) satisfies

    𝔼[G(S_π*_B)] − 𝔼[G(S_π̂_λ)]  ≤  2 B · ε̃  +  2 η̃_B  +  𝒪(√(B/N))   (A-ad)

where N is the number of calibration roll-outs used to tune λ and ψ̃. The
last term is a standard Lagrangian-root estimation error and is small when B
is modest and N is in the hundreds.

**Why this is the right adaptive shape.** The open-loop Theorem A bound
2Bε + 2η_B drops out as the degenerate case where ψ̃(z) depends on z only
through s_t (no state conditioning) and the additivity slack η̃_B reduces to
η_B. So (A-ad) is a *strict generalisation* — it never worsens the open-loop
bound, and improves it precisely to the extent ε̃ < ε (state-conditional
signals reduce calibration error).

This is important: **an honest adaptive theorem does not claim stronger
guarantees than Theorem A; it claims a better constant on the same shape.**

### 2.3 Sample-complexity worry

Estimating ψ̃(z) requires samples of Δ_t at each z, not just the marginal at
each t. If Z is high-dimensional (e.g. z_t contains the full predictor
hidden state), ψ̃ needs a function-approximator and calibration budget grows
polynomially in dim(Z). A tractable design rule:

    z_t  :=  (s_t, b_t, phase(t))       — a 3-dim sufficient statistic

where phase(t) ∈ {early, mid, late} is a coarse bucketing of t. This keeps
the calibration budget within the same order as Protocol B.

### 2.4 Cost

- **Formalisable:** Yes, cleanly, with published guarantees (Altman 1999;
  Paternain et al. 2019; Ding et al. 2021 for primal-dual).
- **Tractable to learn a policy:** Depends on dim(Z). With the coarse z_t
  above, yes.
- **Regret bound available:** Yes — Theorem (A-ad) above gives a clean
  shape and is a direct analogue of Theorem A.
- **Matches Phase 2b evidence:** It's consistent with the 62 % within-seed
  variance (state z_t naturally captures the seed-conditional part of the
  signal) and with the top-k-vs-oracle Jaccard ≈ 1.2× (an adaptive threshold
  rule need not coincide with top-B of a fixed per-t oracle).
- **Fits existing infrastructure:** Medium — Protocol B can be re-used but
  needs state-conditional Δ(z) estimates, not just Δ_t.

---

## 3. Framework F2 — Control-as-inference / Feynman-Kac

### 3.1 Problem statement

Define a reference path measure ℙ_0 over trajectories (z_0, z_1, …, z_T) by
running the predictor + uniform-baseline-corrector schedule. Define a target
path measure by tilting with an **exponential twist**

    ℙ*_β(dz_{0:T})  ∝  ℙ_0(dz_{0:T}) · exp( β · G(S_z) )  · 𝟙[|S_z| = B].

The **optimal control measure** ℙ* that maximises 𝔼[G | Σ a_t = B] is the
β → ∞ limit; for finite β it implements soft-constraint KL-regularised
control (Todorov 2009; Kappen 2012). The Feynman-Kac interpretation gives an
explicit forward decomposition

    ℙ*_β(dz_{0:t})  ∝  ℙ_0(dz_{0:t}) · ψ*_t(z_t),

where ψ*_t(z_t) = 𝔼_{ℙ_0}[ exp(β G_{>t}) · 𝟙[Σ a_{>t} = B − Σ a_{≤ t}] | z_t ]
is the **twist function** (backward-recursive value function).

### 3.2 Equivalence to F1

The log-twist log ψ*_t(z_t) is the soft-Bellman optimal value function of the
F1 CMDP at state z_t with remaining budget b_t = B − Σ a_{≤ t} and inverse
temperature β. Specifically, with b inserted into the state,

    log ψ*_t(z, b)  =  V*_β(z, b, t)           (soft Bellman).

So F2 is not an independent framework; it's a path-measure re-expression of
F1. Its value is:

- It makes the **twist explicit** and therefore suggests SMC-based sampling
  of near-optimal trajectories (this is exactly what F3 exploits).
- It connects directly to the **entropy-regularised RL** literature
  (Haarnoja et al. 2018; Levine 2018), which gives variational lower bounds
  and consistency results.

### 3.3 Role

**Glue, not a standalone pick.** Cite it to explain why F1 and F3 are two
sides of the same object; don't try to derive an independent regret theorem
in F2. An attempt would reduce to the F1 theorem (A-ad) after undoing the
twist representation.

---

## 4. Framework F3 — Particle Gibbs / conditional SMC

### 4.1 Problem statement

Given the Feynman-Kac representation in §3, approximate ℙ*_β by an SMC
algorithm with N particles:

    {z_{0:T}^{(i)}, w_T^{(i)}}_{i=1}^N ,

where particles are propagated under ℙ_0 and weighted incrementally by

    w_t^{(i)}  ∝  w_{t-1}^{(i)} · exp( β · a_t^{(i)} · Δ(z_t^{(i)}) ).

Resample using ESS-adaptive rescheduling: resample only when ESS/N drops
below a threshold (e.g. 0.5). The approximation error scales as 𝒪(1/N)
uniformly in T under standard ergodicity (Del Moral 2004).

Conditional SMC (Andrieu-Doucet-Holenstein 2010) gives a valid MCMC sweep
over trajectories: pick one particle as the "reference" trajectory, resample
the rest, and construct a Markov chain on path space whose invariant
distribution is exactly ℙ*_β. **Particle Gibbs** iterates this sweep.

### 4.2 Relevance to thesis

- **ProSeCo already uses SMC-like self-consistency reweighting** at each
  corrector loop (this was the reason ProSeCo was adopted as the Phase 2b
  backbone). So F3 is the framework closest to the thesis's existing
  computational primitives.
- **PG-DLM (arXiv:2507.08390, July 2025)** gives exactly the Particle Gibbs
  recipe for diffusion language models with Propositions 1–2 on
  convergence and Theorem 1 on asymptotic unbiasedness. This is the
  strongest published proof-of-concept that SMC-for-masked-diffusion works
  at language-modelling scale.
- **E-SMC (arXiv:2512.21336, Dec 2025)** uses entropy-adaptive SMC for
  masked diffusion decoding but **does not study fixed-budget regret bounds**
  — so there is still an open niche for "SMC + fixed B + regret shape (A-ad)".

### 4.3 Adaptive Theorem A analogue via SMC

The cleanest regret-style statement in the SMC world is

    | 𝔼_{ℙ̂_β^N} [G(S)]  −  𝔼_{ℙ*_β} [G(S)] |  ≤  C(T, β) / √N          (SMC)

(Del Moral 2004, Chapter 7). Combining with (A-ad) applied to the β → ∞
limit (or to a finite β with a standard KL-vs-maximum gap log|Z|/β),

    𝔼[G(S_π*_B)] − 𝔼[G(S_π̂_{SMC})]  ≤  2Bε̃ + 2η̃_B + C'/√N + log|Z|/β.

All four terms are estimable from data. This is the cleanest "honest
theorem" available to the thesis within the adaptive setting, but it
depends on (A-ad) — i.e. F3 does not avoid the state-conditional calibration
assumption; it inherits it.

### 4.4 Cost

- **Formalisable:** Yes, via established SMC machinery (Del Moral 2004;
  Andrieu-Doucet-Holenstein 2010); recent language-model-specific
  applications (PG-DLM, E-SMC) confirm tractability at scale.
- **Tractable computationally:** Medium — N particles per seed; reuses
  ProSeCo's self-consistency reweighting at no extra engineering cost.
- **Regret bound available:** Yes — (SMC) + (A-ad). Has a 1/√N term which
  makes the bound **sample-size-aware** in a way that open-loop doesn't.
- **Matches Phase 2b evidence:** Yes — within-seed variance is *exactly*
  what SMC resampling is designed to exploit.
- **Fits existing infrastructure:** **Highest** — matches ProSeCo's
  existing computational primitives.

---

## 5. Framework F4 — Adaptive submodularity (foil)

### 5.1 Problem statement

A set-function f: 2^{[T]} → ℝ is submodular if marginal gains shrink with
growing sets: f(S ∪ {t}) − f(S) ≥ f(S' ∪ {t}) − f(S') for S ⊆ S'. Under
**adaptive submodularity** (Golovin-Krause 2011), the adaptive greedy
policy π_{greedy}(z) = argmax_t 𝔼[ Δ_t | z ] achieves
(1 − 1/e) · 𝔼[G(S_π*_B)] — a classical approximation-ratio guarantee.

### 5.2 Why F4 is the right foil, not the right main pick

The Phase 3a negative result (BS-AG 49-64 %, CD-G 74-84 %) says:
**ranker classes that pick top-B from a per-step score recover at most
∼80 % of MC-oracle headroom.** The natural adaptive-greedy analogue — "at
each step, take the locally-best marginal" — is still a ranker, just a
state-aware one. Under adaptive submodularity it would achieve 63 % of the
*adaptive* oracle, **but** adaptive submodularity requires the marginal-gain
function to shrink monotonically in context. Our corrector kernel has
**non-monotone interactions** (Proposition C's γ > 0 pairwise term), so
adaptive submodularity fails as an assumption.

This is a useful foil because:

- It shows that even the strongest classical adaptive-greedy bound does not
  automatically apply to this problem.
- It isolates **which assumption fails**: adaptive submodularity's
  monotonicity, not the adaptivity itself.
- It reinforces that the thesis's path to adaptive gains requires going
  beyond ranker-class policies (threshold-in-Δ(z), SMC-reweighted paths), not
  a refined ranking.

### 5.3 Role

Include as a **one-paragraph comparator** in the theory chapter; it is what
a naive reader might guess is the right answer, and we need to say why it
isn't.

---

## 6. Comparison table

| Criterion | F1 (FH-CMDP) | F2 (CaI/FK) | F3 (PG/cSMC) | F4 (Adaptive submod) |
|---|---|---|---|---|
| Formalisable in thesis variables | ✓ clean Bellman | ✓ path-measure twist | ✓ particle filter | ✓ set-function |
| Regret bound for proxy | **(A-ad)** 2Bε̃+2η̃_B | reduces to F1 | (A-ad)+𝒪(1/√N) | (1−1/e) only if mono-supermod |
| Matches Phase 2b evidence | ✓ within-seed | ✓ trivially | ✓✓ exploits within-seed | ✗ monotonicity fails |
| Fits existing ProSeCo infra | △ needs calibr. | △ explanatory only | ✓✓ reuses reweighting | ✗ needs different policy class |
| Sample complexity | polynomial in dim(Z) | same as F1 | polynomial in N | polynomial in T (known) |
| Published MDM precedent | — | — | ✓ PG-DLM '25, E-SMC '25 | — |
| Role in thesis | normative backbone | glue | algorithmic | foil |
| Risk of scope creep | low (threshold policy) | low | medium (engineering) | low |
| Theorem quality at thesis scale | mid-tier CMDP | redundant with F1 | **highest-quality honest bound** | not applicable |

---

## 7. Decisive picks (Phase 4)

### 7.1 One normative: F1 (FH-CMDP)

F1 gives the cleanest formalisation of the thesis object and the cleanest
analogue to Theorem A. Specifically:

- The threshold-in-Δ(z) controller is a **natural generalisation** of the
  top-B-in-ψ(s_t) controller that Theorem A already studies.
- The bound (A-ad): 2Bε̃ + 2η̃_B + 𝒪(√(B/N)) has the **same shape** as
  Theorem A, so the thesis's central theorem survives the adaptive
  extension instead of being replaced by one.
- The scope is preserved: the policy outputs a_t ∈ {0,1}, not a token subset.
- It connects cleanly to published CMDP guarantees (Altman 1999;
  Paternain et al. 2019) and to the recent Lagrangian test-time
  reinforcement literature (Adaptive Test-Time CPO, arXiv:2507.xxxxx).

### 7.2 One algorithmic: F3 (Particle Gibbs / conditional SMC)

F3 is the **only framework that reuses ProSeCo's existing computational
primitives** and has a published language-modelling proof-of-concept. The
SMC bound + (A-ad) gives a quantitatively honest theorem (all four terms
estimable from data).

An explicit variant to commit to: **ESS-adaptive conditional SMC with the
F1 threshold-λ policy as proposal**. This is the cleanest hybrid; its
analysis reduces to F1 in the N→∞ limit, and its finite-N error is bounded
by the standard SMC theorem.

### 7.3 One foil: F4 (adaptive submodularity)

Include F4 as a one-paragraph comparator in the theory chapter to show that
the classical approximation-ratio guarantee does not apply and to explain
which assumption fails. This is instructive, not load-bearing.

### 7.4 F2: glue, not a pick

F2's role is to make the F1↔F3 equivalence visible. Cite it to motivate
the SMC representation and to connect to entropy-regularised RL. Do not
write an independent regret theorem in F2.

---

## 8. What an honest adaptive theorem commits the thesis to

If Phase 5 commits to F1 (normative) + F3 (algorithmic), the theorem is:

> **Theorem A-ad (adaptive proxy-regret bound).** Let π̂_λ,N be the
> ESS-adaptive conditional-SMC policy with N particles using the threshold
> rule a_t = 𝟙[ψ̃(z_t) > λ] ∧ 𝟙[b_t > 0], with ψ̃ proxy-calibrated to Δ(z)
> within ε̃ and Lagrangian λ tuned so 𝔼[Σ a_t] = B. Assume approximate
> additivity in the adaptive regime with slack η̃_B. Then
>
>    𝔼[G(S_π*_B)] − 𝔼[G(S_π̂_λ,N)]  ≤  2 B ε̃  +  2 η̃_B  +  C/√N  +
>                                         𝒪(√(B/N_cal))  +  log|Z|/β.

**What this requires empirically** (Phase 2c / Protocol C, to be defined):
- Estimate ε̃ (state-conditional calibration error) on the Phase 2b
  trajectories. Requires sampling Δ(z) at each (s_t, b_t, phase(t))
  bucket.
- Estimate η̃_B (adaptive additivity slack). Requires paired measurements
  of G(S_π) for a handful of candidate policies on ≥ 8 seeds.
- Report (1−1/e) comparator to show F4's submodularity assumption fails
  (already available from Protocol B's pairwise interaction γ).

**What this does NOT commit the thesis to:**
- It does not require training a learned controller. The λ-threshold rule
  is non-parametric.
- It does not pivot the thesis; the open-loop Theorem A remains the main
  theorem and A-ad is an extension theorem in the Future Work chapter
  (consistent with Phase 1 recommendation §5 and §7).

### 8.1 Minimal, honest Phase 2c specification

If the thesis were to collect any adaptive evidence at all beyond Phase 2b,
the smallest scientifically honest addition is:

- **Protocol C (pilot, bounded).** On the same 8-seed LLaDA-SFT Phase 2b
  platform, after Phase 2b writes per-seed policy rows, do the following
  *ex post* (no new trajectories):
  1. Compute bucketed state z_t = (s_t, b_t, phase(t)).
  2. Estimate ε̃ by the residuals Δ_t(z) − ψ̃(z) where ψ̃(z) is the
     bucket-mean of Δ_t over the 8 seeds.
  3. Estimate η̃_B by replaying the λ-threshold policy over the 8 seeds
     and measuring G(S_π̂) − Σ_t a_t Δ̂_t.
  4. Report a single scalar: Δ_close(π̂_λ,1) / Δ_open, the fraction of
     oracle headroom recovered by the N=1 (deterministic) threshold
     policy.
- **No new expensive GPU work.** Phase 2b's existing policy_raw and
  mc_raw are re-used.

If Δ_close(π̂_λ,1) / Δ_open > 0.5 on the pilot: present as preliminary
Future Work evidence. If < 0.5: state the negative result and position the
theorem as existence-only.

**This is the smallest justified next step** for Phase 6's recommendation.

---

## 9. Links back to the repo

- Open-loop Theorem A and its lemmas/propositions: `research/candidate_theorems.md`.
- Proof worklog (Entry 6 = Theorem A restructure): `research/proof_worklog.md`.
- Provenance / tag ledger: `research/proof_ledger.md`.
- Skeptical audit (Phase 1): `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md`.
- Phase 2b / 3a findings (within-seed variance, ranker-class negative):
  `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`,
  `docs/thesis/next_steps/INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md`.
- Combinatorial diagnostics headline numbers:
  `results/phase2b/combinatorial_diagnostics.json`.

---

## 10. Open decisions remaining for Phase 5–6

1. Does the thesis commit to writing Theorem A-ad as a formal statement, or
   only as a sketched extension in Future Work? Phase 1's recommendation
   was Future Work. **Tentative Phase 4 answer:** state the theorem
   formally in an appendix or Future Work section *if and only if* Phase 2c
   is run and returns a non-trivial Δ_close. Otherwise sketch only.

2. If Phase 2c runs, does it use Δ_t estimates from MC oracle (9,000 rows
   at B ∈ {2,3,4}) or requires new Protocol C trajectories? **Tentative
   answer:** MC oracle rows are sufficient for ε̃; no new trajectories
   needed.

3. Is F3 used purely as an analytical device, or does the thesis actually
   run a Particle-Gibbs sweep? **Tentative answer:** analytical device;
   the thesis does not ship a PG implementation. PG-DLM already covers
   that empirically.

4. What's the precise state abstraction for z_t? **Tentative:**
   (s_t, b_t, phase(t)) with s_t ∈ {H_t, M_t^{-1}, Q_t} and
   phase(t) ∈ {early, mid, late}, giving |Z| ≲ 3 × 9 × 3 = 81 buckets per
   signal at B=4. This is the minimum non-trivial state space.

---

## 11. References (Phase 2 findings, cited inline above)

- Altman, 1999. *Constrained Markov Decision Processes.* CRC Press.
- Paternain, Chamon, Calvo-Fullana, Ribeiro, 2019. "Constrained RL has zero
  duality gap." NeurIPS.
- Ding, Wei, Yang, Wang, Jovanović, 2021. "Provably efficient safe
  exploration via primal-dual policy optimization." AISTATS.
- Del Moral, 2004. *Feynman-Kac Formulae.* Springer.
- Andrieu, Doucet, Holenstein, 2010. "Particle Markov chain Monte Carlo
  methods." JRSS-B.
- Todorov, 2009. "Efficient computation of optimal actions." PNAS.
- Kappen, Gomez, Opper, 2012. "Optimal control as a graphical model
  inference problem." Machine Learning.
- Haarnoja, Zhou, Abbeel, Levine, 2018. "Soft Actor-Critic." ICML.
- Levine, 2018. "Reinforcement learning and control as probabilistic
  inference: tutorial and review." arXiv:1805.00909.
- Golovin, Krause, 2011. "Adaptive submodularity: theory and applications
  in active learning and stochastic optimization." JAIR.
- Ascolani, Lavenant, Zanella, 2024. "Entropy contraction of the
  Gibbs sampler under log-concavity." [with Provenance-ledger correction:
  random-scan only.]
- **PG-DLM**: arXiv:2507.08390, 2025. "Particle-Gibbs for diffusion
  language models."
- **E-SMC**: arXiv:2512.21336, 2025. "Entropy-adaptive SMC for masked
  diffusion."
- Adaptive Test-Time CPO: arXiv preprint 2025 (Lagrangian for test-time
  policy optimization, cited in Phase 2 notes).

---

*End of research notes. Promote to
`docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` in Phase 5.*
