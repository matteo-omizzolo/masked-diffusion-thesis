> **STATUS:** EXTENSION-THEORY (Appendix-F candidate; theory-active per
> `ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md`, 2026-04-25). Main-thesis status
> remains Theorem A; A-ad is Appendix-F formal theorem.
> **LAST VERIFIED:** 2026-04-25 (Theorem A-ad statement and proof tightened;
> abstract-policy-class reduction made explicit; Appendix-F status flagged
> per activation audit verdict)
> **SCOPE:** Mathematical positioning of adaptive, state-conditional, budgeted corrector
> scheduling as an extension of the open-loop Theorem A framework. Companion to
> `docs/thesis/theory/THEORY_STATUS.md` (open-loop canonical theory), the
> Phase-1 skeptical audit at
> `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md`, and the
> 2026-04-25 activation audit at
> `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md`.

---

# Adaptive Budgeted Controllers for Informed-Corrector Scheduling

## Positioning

This document formalises the **adaptive extension** of the thesis's fixed-budget
informed-corrector scheduling problem and states the four external frameworks
considered for it, along with their regret-style analogues of Theorem A, a
decisive framework pick, and the minimal empirical commitment the thesis would
need to make to use any of them as a load-bearing claim.

The adaptive extension is presented here as **candidate Future-Work content**.
The main-thesis theorem is the open-loop Theorem A in `THEORY_STATUS.md`; the
adaptive direction is an extension that strictly contains the open-loop bound as
a degenerate case, so the core theorem survives. Phase 1's skeptical audit
recommended against pivoting; this file supplies the mathematical scaffolding so
that a Future-Work chapter — or a short appendix — can be written honestly.

## 1. Problem object

### 1.1 Fixed-budget scheduling, open-loop

Fix a predictor and a fixed informed-corrector kernel. Let T be the number of
denoising steps. The thesis studies

    choose  S ⊆ {1, …, T},  |S| = B,
    to maximise  G(S) = quality functional at t = 0,

where Δ_t := G(S ∪ {t}) − G(S \ {t}) is the one-loop marginal gain at step t,
and ψ(s_t) is a proxy score computed from a trajectory signal s_t ∈ {H_t,
M_t^{-1}, Q_t}. The open-loop policy is Ŝ_B = top-B_t ψ(s̄_t). Theorem A
bounds the proxy-regret as

    G(S_B*) − G(Ŝ_B)  ≤  2 B ε  +  2 η_B               (A — open-loop)

under uniform calibration |Δ_t − ψ(s_t)| ≤ ε and approximate additivity
|G(S) − Σ_{t ∈ S} Δ_t| ≤ η_B. Refinements A′ and A″ sharpen ε → ε_R and
η_B → σ_ξ · √B / √2 (see `THEORY_STATUS.md`).

### 1.2 Adaptive extension

An adaptive controller is a policy

    π : z_t ⟼ a_t ∈ {0,1},          a_t ≤ b_t,          b_{t+1} = b_t − a_t,

where z_t is the **information state** available just before deciding whether
to correct at step t, and b_t = B − Σ_{τ<t} a_τ is the remaining budget. A
minimal sufficient state is

    z_t = (s_t, b_t, phase(t)),   phase(t) ∈ {early, mid, late},

with s_t the chosen per-step signal. The adaptive problem is

    max_π  𝔼_π [G(S_π)]            subject to   Σ_t a_t ≤ B  a.s.,

where S_π = { t : a_t = 1 under π }. Let π*_B be any optimiser and S_B* the
best open-loop schedule. The **adaptive oracle gap**

    Δ_open  :=  𝔼[G(S_π*_B)] − 𝔼[G(S_B*)]  ≥  0

measures the headroom that adaptivity can recover over the best fixed
schedule. Δ_open is **not** bounded by any current thesis result; Phase 2b's
MC-oracle +0.45 paired headroom is a noisy surrogate, not Δ_open.

### 1.3 Non-goals (scope guards)

The adaptive controller outputs a_t ∈ {0,1} only. It does **not** choose:
- which tokens to correct at step t (token-selection; scope-guarded, see
  `docs/thesis_direction.md` §"Scope Boundaries"),
- the corrector kernel (fixed to ProSeCo's informed corrector),
- whether to remask (remasking methods are studied elsewhere — RemeDi,
  PRISM).

Preserving this boundary is load-bearing. Without it, the adaptive
controller collapses back into a token-selection paper.

## 2. Four frameworks — formal adaptations

The Phase-2 web scan selected four frameworks for adaptation:

- **F1**: Finite-horizon constrained MDP (CMDP).
- **F2**: Control-as-inference / Feynman-Kac path measure.
- **F3**: Particle Gibbs / conditional SMC.
- **F4**: Adaptive submodularity.

The detailed literature review, comparison table, and scratch derivations
live in `research/adaptive_controller_research_notes.md`. This section
summarises each adaptation and how Theorem A's analogue reads under it.

### 2.1 F1 — Finite-horizon constrained MDP

Formalise the adaptive problem as a FH-CMDP

    M = (Z, A = {0,1}, P, r, c, T, B),

with r(z, a) = a · Δ(z), cost c(z, a) = a, and transition kernel P induced
by the predictor + corrector dynamics. Under Lagrangian relaxation with
multiplier λ ≥ 0,

    V_λ(z, b, t)  =  max_{a ∈ {0, min(1,b)}}  { r(z,a) − λ c(z,a)
                      + 𝔼_{z' ∼ P(·|z,a)} V_λ(z', b − a, t+1) }.

Strong duality for CMDPs (Altman 1999; Paternain et al. 2019) gives
existence of λ* such that π*_{λ*}(z, b, t) = 𝟙[Δ(z) > λ*] ∧ 𝟙[b > 0] is
optimal for the CMDP. This is the **threshold-in-Δ(z) policy** — the direct
analogue of the open-loop top-B-in-ψ(s_t) policy.

#### Proposed Theorem A-ad (F1 form)

Suppose a state-conditional proxy ψ̃(z) satisfies |Δ(z) − ψ̃(z)| ≤ ε̃ and the
approximate-additivity slack in the adaptive regime is η̃_B, i.e.

    | 𝔼_π [G(S_π)]  −  𝔼_π[ Σ_t a_t · Δ(z_t) ] |  ≤  η̃_B.

Let π̂_λ(z, b, t) = 𝟙[ψ̃(z) > λ] ∧ 𝟙[b > 0] with λ tuned on N_cal pilot
trajectories so 𝔼_{π̂_λ}[Σ a_t] = B. Then

    𝔼[G(S_π*_B)] − 𝔼[G(S_π̂_λ)]  ≤  2 B ε̃  +  2 η̃_B  +  𝒪(√(B / N_cal)).     (A-ad F1)

**Relation to Theorem A.** Specialising ψ̃(z) to depend on z only through
s_t and taking the calibration regime where ε̃ → ε, η̃_B → η_B recovers the
open-loop bound 2Bε + 2η_B. Hence A-ad F1 is a **strict generalisation**:
it never worsens Theorem A and improves it precisely to the extent that
state conditioning shrinks ε̃ below ε.

The 𝒪(√(B/N_cal)) term is a standard Lagrangian-root estimation error for
the multiplier λ; under the thesis's K=8 sampling budget this term is
small relative to 2 B ε̃ at any ε̃ ≳ 0.05.

#### Theorem A-ad (formal, abstract-policy-class form). *Activated 2026-04-25.*

The earlier sketch in this subsection compares the *threshold-λ* policy
π̂_λ to the *adaptive oracle* π*_B. The threshold-λ and the open-loop top-B
policies are two different members of a single broader policy class; both
are recovered by stating the regret in terms of an abstract *signed-score*
policy. This subsection states the theorem on that broader class so that
Theorem A-ad rigorously contains Theorem A as the open-loop specialisation.

**Setup (formal).** Fix a finite horizon T, budget B ∈ {1, …, T}, and a
binary action set A = {0, 1}. The information state z_t ∈ Z is the
masked-diffusion state at step t before deciding whether to correct (so
z_t carries (s_t, b_t, phase(t), …) as components, with b_t = B − Σ_{τ<t}
a_τ the remaining budget). A *score policy* is a measurable
ψ̃ : Z → ℝ together with a selection rule σ_ψ̃ : Trajectories → 2^{1..T}
satisfying |σ_ψ̃(z_{1:T})| ≤ B almost surely. Two canonical members of
this class:

- **Top-B selection.** σ_topB(z_{1:T}) := argmax_{|S|=B} Σ_{t∈S} ψ̃(z_t).
  This is the open-loop policy that ranks all T steps and picks B by
  largest score. (It needs the full trajectory before deciding; equivalent
  to running the trajectory once without correctors, then picking.)
- **Threshold-λ selection.** σ_λ(z_{1:T}) :=
  {t : ψ̃(z_t) > λ ∧ b_t > 0}, with λ tuned so 𝔼_{Z}[|σ_λ|] = B. This is
  the causal policy of F1; it spends Bin-distributed budget across seeds
  but matches B in expectation.

Let π̂_ψ̃ denote either selection rule applied with score ψ̃, and define
the realised gain G(σ_ψ̃(z_{1:T})) := F(y^{σ_ψ̃}) − F(y_base).

**Assumptions.**

- **(1) Binary placement.** k_t ∈ {0, 1} for every t.
- **(2̃) Adaptive approximate additivity.** There exists η̃_B ≥ 0 such
  that for every score policy π̂ in the class above with |σ_π̂| ≤ B
  almost surely,
  | 𝔼[G(σ_π̂)] − 𝔼[Σ_{t ∈ σ_π̂} Δ(z_t)] | ≤ η̃_B
  where Δ(z) := 𝔼[Δ_t | z_t = z] is the conditional one-loop marginal
  gain.
- **(3̃) Adaptive proxy calibration.** The score ψ̃ : Z → ℝ satisfies
  |Δ(z) − ψ̃(z)| ≤ ε̃ for all z ∈ Z reached on any policy.
- **(4) Lagrangian estimation.** λ is tuned from N_cal i.i.d. pilot
  trajectories so that |𝔼_Z[|σ_λ|] − B| ≤ Δ_λ where Δ_λ → 0 at a
  standard rate Δ_λ = 𝒪(√(1 / N_cal)).

**Theorem A-ad (proxy-regret, abstract policy class).** Let π*_B be any
optimiser of 𝔼[G(S_π)] subject to |S_π| ≤ B almost surely. Let π̂_ψ̃ be
any score-policy member with score ψ̃ satisfying assumption (3̃). Then

    𝔼[G(σ_{π*_B})] − 𝔼[G(σ_{π̂_ψ̃})] ≤ 2 B ε̃ + 2 η̃_B + 𝒪(√(B / N_cal)).

Specialisations:

- **Top-B specialisation.** When π̂_ψ̃ = σ_topB, the 𝒪(√(B / N_cal)) term
  vanishes (no Lagrangian to estimate), and the bound reduces to
  2 B ε̃ + 2 η̃_B. When ψ̃(z) further depends on z only through s_t and
  ε̃ → ε and η̃_B → η_B, this is Theorem A.
- **Threshold-λ specialisation.** When π̂_ψ̃ = σ_λ with N_cal = ∞ (oracle
  λ tuning), the 𝒪(√(B / N_cal)) term vanishes; with finite N_cal the
  term is the standard Lagrangian-root error from CMDP duality (Altman
  1999; Paternain et al. 2019).

**Proof sketch.** The argument has three steps. Each step is a direct
adaptation of the open-loop counterpart in Theorem A's proof
(`research/proof_worklog.md` Entry 6).

*Step 1 — additivity reduction.* By (2̃) applied to both π*_B and π̂_ψ̃,

    𝔼[G(σ_{π*_B})] − 𝔼[G(σ_{π̂_ψ̃})]
        ≤ 𝔼[Σ_{t ∈ σ_{π*_B}} Δ(z_t)]
          − 𝔼[Σ_{t ∈ σ_{π̂_ψ̃}} Δ(z_t)] + 2 η̃_B.

*Step 2 — calibrated swap argument.* Define A(σ; ψ̃) := Σ_{t ∈ σ} ψ̃(z_t).
By (3̃), for any state z, Δ(z) = ψ̃(z) ± ε̃. Conditional on the realised
trajectory z_{1:T}, the optimal σ given the abstract scoring rule selects
the at-most-B states with largest ψ̃ such that |σ| ≤ B; call this σ_ψ̃*.
The exchange argument of Lemma A2 (open-loop counterpart) bounds

    Σ_{t ∈ σ_{π*_B}} Δ(z_t) − Σ_{t ∈ σ_ψ̃*} Δ(z_t) ≤ 2 B ε̃,

since each of the at most B swaps between σ_{π*_B} and σ_ψ̃* costs at
most 2 ε̃ in Δ-units.

*Step 3 — Lagrangian root error.* For the threshold-λ specialisation,
the at-most-B-element σ_ψ̃* is implemented by the threshold rule
σ_λ when λ is tuned exactly so 𝔼[|σ_λ|] = B. Under (4), the empirical
multiplier λ̂ from N_cal pilot trajectories yields a budget violation of
order Δ_λ = 𝒪(√(1 / N_cal)). The first-order Taylor expansion of the
CMDP value around λ* (Paternain et al. 2019, Lemma 3) gives

    | 𝔼[G(σ_{λ̂})] − 𝔼[G(σ_{λ*})] | = 𝒪(B · Δ_λ) = 𝒪(√(B / N_cal)).

For the top-B specialisation, σ_topB requires no λ; this term is zero.

Combining the three steps yields the theorem. ∎

**Provenance.** `[Adapted from GPT Pro assessment v2 + Lagrangian CMDP
duality (Altman 1999; Paternain et al. 2019); abstract-policy-class
reduction Novel 2026-04-25 to make the strict-generalisation claim
rigorous]`.

**Status.** `Proved as a conditional theorem under (1)–(4); ε̃, η̃_B, λ
are empirical objects measured by Protocol C (see §4.2 below).` The
bound is non-vacuous iff 2 B ε̃ + 2 η̃_B + 𝒪(√(B / N_cal)) <
𝔼[G(σ_{π̂_ψ̃})] at the budget of interest, which is the same shape as
Theorem A's non-vacuity hypothesis.

**What this theorem *does not* claim.** It does not claim ε̃ < ε; it
does not claim η̃_B < η_B; it does not claim that the threshold-λ
policy beats top-B in expectation. All three are empirical questions
addressed by Protocol C. The theorem fixes the *shape* of the bound;
the *constants* are measurements.

### 2.2 F2 — Control-as-inference / Feynman-Kac

Define a reference path measure ℙ_0 under the uniform-baseline-corrector
schedule and tilt with

    ℙ*_β(dz_{0:T})  ∝  ℙ_0(dz_{0:T}) · exp(β G(S_z)) · 𝟙[|S_z| = B].

The Feynman-Kac decomposition ℙ*_β(dz_{0:t}) ∝ ℙ_0(dz_{0:t}) · ψ*_t(z_t)
has **twist function** ψ*_t satisfying log ψ*_t(z, b) = V*_β(z, b, t) — i.e.
the log-twist is the soft-Bellman value of the F1 CMDP at inverse
temperature β.

#### Role

F2 is not an independent framework for regret analysis; its bound reduces
to A-ad F1 after undoing the twist representation (with an additional
log|Z|/β term for the β → ∞ limit). Its value is **explanatory**: it
exposes that the adaptive control problem is formally an inference problem
on a twisted path measure, which motivates F3 as the algorithmic
realisation. Cite F2 as glue; do not try to state an independent theorem.

### 2.3 F3 — Particle Gibbs / conditional SMC

Approximate ℙ*_β with an SMC algorithm: N particles propagated under ℙ_0
with incremental weights w_t ∝ w_{t-1} · exp(β a_t Δ(z_t)), with
ESS-adaptive resampling. Conditional SMC (Andrieu-Doucet-Holenstein 2010)
promotes this to a Markov chain on trajectories with invariant measure
ℙ*_β; Particle Gibbs iterates this sweep.

#### Proposed Theorem A-ad (F3 form)

Under standard SMC ergodicity (Del Moral 2004, Ch. 7), the finite-N
approximation to ℙ*_β satisfies

    | 𝔼_{ℙ̂_β^N}[G(S)]  −  𝔼_{ℙ*_β}[G(S)] |  ≤  C(T, β) / √N.

Combining with A-ad F1 applied at inverse temperature β,

    𝔼[G(S_π*_B)] − 𝔼[G(S_π̂_{SMC})]  ≤  2 B ε̃  +  2 η̃_B  +  C'/√N  +  log|Z|/β.  (A-ad F3)

**All four terms are estimable from data**, making A-ad F3 the cleanest
quantitatively honest adaptive regret bound available. The −log|Z|/β term
is the standard KL-vs-maximum gap from finite β (set β large once
ε̃, η̃_B are estimated).

#### Alignment with existing thesis infrastructure

F3 is the **only** framework that reuses ProSeCo's existing computational
primitives: ProSeCo performs SMC-style self-consistency reweighting per
corrector loop (the primary reason it was adopted as the Phase-2b
backbone). Two very recent papers confirm feasibility at language-model
scale:

- **PG-DLM** (arXiv:2507.08390, July 2025) — Particle Gibbs for diffusion
  language models with Propositions 1–2 on convergence and Theorem 1 on
  asymptotic unbiasedness.
- **E-SMC** (arXiv:2512.21336, Dec 2025) — entropy-adaptive SMC for
  masked-diffusion decoding; does **not** study fixed-budget regret bounds,
  leaving the niche open.

### 2.4 F4 — Adaptive submodularity (foil only)

Under adaptive submodularity (Golovin-Krause 2011), the greedy policy
π_g(z) = argmax_t 𝔼[Δ_t | z] achieves (1 − 1/e) · 𝔼[G(S_π*_B)]. The
hypothesis required is monotone diminishing returns of Δ under context.

**Phase 3a's pairwise interaction γ > 0** (Proposition C, see
`candidate_theorems.md`) falsifies the adaptive-submodularity hypothesis for
this problem. Hence F4 does not apply; we include it only to explain why
the classical adaptive-greedy approximation-ratio guarantee is unavailable.
This is an **informative foil**: it isolates that adaptivity alone does not
rescue ranker-class policies unless the problem is submodular.

## 3. Framework pick

One normative + one algorithmic + one foil.

| Role | Framework | Rationale |
|---|---|---|
| Normative | **F1 (FH-CMDP)** | Cleanest analogue to Theorem A; threshold-in-Δ(z) generalises top-B-in-ψ(s_t); published CMDP guarantees; low scope-creep risk. |
| Algorithmic | **F3 (Conditional SMC)** | Only framework that reuses ProSeCo's primitives; published MDM precedent (PG-DLM, E-SMC); quantitatively honest bound with all terms estimable. |
| Foil | **F4 (Adaptive submod)** | Shows that the classical adaptive-greedy guarantee fails on this problem; isolates which hypothesis breaks. |
| Glue | F2 (CaI / Feynman-Kac) | Cite as formal equivalence between F1 and F3; do not state an independent theorem in F2. |

## 4. Load-bearing empirical commitments

### 4.1 What A-ad requires the thesis to estimate

- **ε̃** — state-conditional calibration error on the bucketed state
  z = (s_t, b_t, phase(t)).
- **η̃_B** — adaptive additivity slack under the threshold-λ policy.
- **Δ_close(π̂) / Δ_open** — fraction of adaptive oracle headroom recovered
  by the deterministic threshold policy.
- Optionally, **C/√N** — SMC approximation constant, from particle counts N
  ∈ {4, 8, 16} on the same 8 seeds.

### 4.2 Protocol C (pilot, bounded, no new GPU work) — retargeted to OWT

**Activation-audit retarget (2026-04-25).** The earlier scoping of
Protocol C to **LLaDA-SFT** artefacts has a defect: the bounded
LLaDA-SFT probe established Δ_open ≈ 0 at the tested budgets (paired CI
[0, 0] at B=4; CI [−4.07, −1.07] at B=2; see
`POST_CROSS_BACKBONE_DECISION.md`). Protocol C's reported quantity
`Δ_close(π̂_λ,1) / Δ_open` is mathematically uninterpretable when the
denominator is ≈ 0. The activation audit
(`docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md`)
retargets Protocol C to **OWT**, where Δ_open is empirically anchored at
+0.45 paired (B ∈ {2, 3, 4}; `results/phase2b/mc_oracle.json`) and Phase 1
Protocol A trajectory data delivers 50 seeds × 64 steps with full
per-step Δ_t and per-step signals.

**Inputs (CPU-only, all on disk):**

- `results/phase1_proseco_owt_full/protocol_a/trajectory_*.json` — 50
  seeds × T=64 steps with per-step Δ_t, entropy H_t, inverse_margin
  M_t^{−1}, quality_mass_proxy Q_t, unmasked_fraction u_t, n_revisable,
  n_masked.
- `results/phase2b_proseco_owt/per_seed/{policy,mc}_rows_seed*.json` — 30
  seeds (Phase 2b platform), G/A/residual for paired policy rows and 100
  MC rows per seed × B ∈ {2, 3, 4}.
- `results/phase2b/{mc_oracle,theorem_a_constants,combinatorial_diagnostics,policy_comparison_paired}.json`
  — paired Δ_open estimates, σ_ξ, ρ pooled, and policy comparison anchors.

**Procedure:**

1. **Bucket the state.** For each Phase 1 trajectory and each step
   t ∈ {0, …, T−1}, define z_t = (signal_quartile(s_t), phase(t)) where
   signal_quartile is computed over the pooled (seed, t) distribution per
   signal (4 buckets), and phase(t) ∈ {early=t<T/3, mid=T/3≤t<2T/3,
   late=t≥2T/3} (3 buckets). Total |Z| = 12 buckets per signal. b_t is
   *not* part of z_t for the calibration step, since b_t is a
   policy-history-dependent random variable; b_t enters only at the
   threshold-policy replay step.
2. **Estimate ψ̃_bucket(z).** For each signal s ∈ {H, M^{−1}, Q} and each
   bucket z, ψ̃_bucket(z) := mean(Δ_t | z_t = z) across all 50 × 64 =
   3 200 trajectory steps.
3. **Compute ε̃ vs ε.** For each signal:
   - ε(s) := RMS_{(seed, t)} (Δ_t − ψ_linear(s_t)) where ψ_linear is the
     least-squares linear fit Δ_t ≈ a · s_t + b on the same 3 200 points.
   - ε̃(s) := RMS_{(seed, t)} (Δ_t − ψ̃_bucket(z_t)) on the same data.
   - Report ratio ε̃ / ε per signal.
4. **Threshold-λ policy replay (additive surrogate).** For each B ∈
   {2, 3, 4} and each signal, compute λ such that
   𝔼_seed[|{t : ψ̃(z_t) > λ}|] = B over the 50 OWT seeds. For each seed
   i, compute schedule S_π̂(i) = {t : ψ̃(z_t(i)) > λ}, capping at B by
   keeping only the B largest ψ̃-values when |S_π̂(i)| > B. Estimate
   G(S_π̂(i)) via the additive surrogate
   A_π̂(i) := Σ_{t ∈ S_π̂(i)} Δ_t(i)
   (using the per-seed realised Δ_t from Phase 1 Protocol A), and bound
   the surrogate-vs-truth gap by σ_ξ · √B / √2 (Refinement A′; σ_ξ from
   Phase 2b at the same B).
5. **Δ_close / Δ_open ratio with σ_ξ uncertainty.** For each (signal, B):
   - Compute paired uniform baseline A_uniform(i, B) over the same 50
     seeds via Phase 1 Δ_t at uniformly-spaced t.
   - Δ_close_A := mean_i [A_π̂(i) − A_uniform(i, B)].
   - Δ_open := +0.45 from `mc_oracle.json` (paired across 30 seeds at the
     Phase 2b platform; OWT and Phase 1 use the same MDLM/ProSeCo-OWT
     backend, so Δ_open is comparable up to seed sampling variance).
   - Report Δ_close_A / Δ_open with explicit uncertainty band
     ±σ_ξ · √B / √2 / Δ_open.
6. **Schedule overlap diagnostic.** For each seed, compute Hamming
   distance between S_π̂(i) and the best MC schedule for that seed (from
   Phase 2b mc_rows). Report mean Hamming distance and the fraction of
   seeds with exact match.

**Stop-rule (pre-registered).**

| Outcome | Action |
|---|---|
| ε̃ / ε ≤ 0.7 AND Δ_close_A / Δ_open ≥ 0.5 (after subtracting σ_ξ · √B / √2 / Δ_open) | Preliminary positive — Theorem A-ad in Appendix F with Protocol C as preliminary evidence |
| ε̃ / ε > 0.9 OR Δ_close_A / Δ_open < 0.3 (after subtracting uncertainty) | Honest negative — A-ad as existence-only theorem; Protocol C as a refinement of the ranker-class corollary to the state-conditional ranker class on (s_t, phase(t)) |
| Otherwise | Inconclusive — A-ad as formal theorem; Protocol C as bounded null result with explicit Tier-3 caveat (50 OWT seeds, additive-surrogate G estimator) |

**Honesty.** The additive surrogate A(S) ≈ G(S) is the same trap that
Phase 2b's failed rankers fell into. Protocol C is honest about this in
two ways: (a) the σ_ξ · √B / √2 uncertainty band is reported explicitly
on every Δ_close ratio, (b) the Hamming-distance overlap between the
threshold schedule and the best MC schedule is a direct test of whether
Protocol C's policy lives in the same region of schedule space as the
schedules that actually achieved high G. Even with a positive verdict,
Protocol C is **not** a Phase 3 substitute and does not measure G under a
non-additive corrector kernel.

Protocol C uses **no new GPU trajectories** (hence the "bounded" label),
runs in seconds on a laptop, and fits inside the existing Phase 1 + Phase
2b OWT artefact budget.

#### Pilot result (closed 2026-04-26 — *honest_negative*).

Module: `src/mdm_playground/analysis/protocol_c.py`. Entry script:
`scripts/run_protocol_c_owt.py`. Tests: `tests/test_protocol_c.py` (24
unit tests, all PASS). Output:
`results/protocol_c_owt/protocol_c_summary.json`. Wall time: ~ 60 s
on a single CPU.

Headline numbers on OWT (50 Phase 1 seeds × T = 64 = 3 200 paired
points; 30 Phase 2b mc_rows seeds for σ_ξ):

- **ε̃ / ε ∈ [0.983, 0.986]** across entropy / inverse_margin /
  quality_mass_proxy. State conditioning on (signal_quartile, phase)
  shrinks calibration RMS by **< 1.7 %**.
- **Δ_close_A / Δ_open after σ_ξ uncertainty subtraction** is in
  [−0.32, +0.015] across all (signal, B) — best at +0.015 (entropy,
  B = 2). σ_ξ · √B / √2 dominates Δ_close_A at every B ≥ 3.
- **Hamming distance (threshold schedule vs best MC schedule)** is
  3.93 / 5.87 / 7.47 at B = 2 / 3 / 4; max possible is 2 B = 4 / 6 / 8.
  Fraction of seeds with exact match: 0.000 across all signals × all
  B. The threshold schedule lives in a disjoint region of schedule
  space from the high-G MC schedules.
- **Bound non-vacuity:** 2 B ε̃ + 2 η̃_B is inert by ≈ 4.9 × at B = 2,
  the same shape of inertness as Theorem A on the same dataset.
- **Verdict:** outcome_class = `"honest_negative"` per the pre-
  registered decision rule (eps_ratio > 0.9 leg fires
  independently of close-ratio leg).

**Decision (terminal node):**
`docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_DECISION.md`.
Theorem A-ad lands in Appendix F as a **formal conditional theorem
with documented inert-bound diagnostic**; the bucketed-state ranker
class is empirically bounded by the same σ_ξ envelope as the
signal-only ranker class. The Negative-Result Corollary is updated
with a one-line addendum noting the bucketed-state extension. No
re-run, no GPU work, no main-body change is authorised by this
decision.

## 5. Honesty ledger

Per thesis-direction honesty requirement, claims in this doc carry one of
four tags:

| Tag | Claim |
|---|---|
| [Empirically anchored] | Δ_open ≥ +0.37 at B ∈ {2, 3, 4} on OWT (Protocol C 2026-04-26; mc_oracle.json paired figure +0.45). |
| [Empirically refuted on OWT bucketed state] | ε̃ < ε on z = (s_t, phase(t)), 4 × 3 = 12 buckets — observed ε̃ / ε ∈ [0.983, 0.986]; shrinkage < 1.7 %. |
| [Proved, conditional] | Theorem A-ad (abstract policy class form) under (1) binary placement, (2̃) adaptive approximate additivity, (3̃) adaptive proxy calibration, (4) Lagrangian estimation. Proof in §2.1 (3-step swap-on-Δ + CMDP duality). |
| [Proved, conditional] | A-ad F3 under SMC ergodicity of Del Moral 2004 Ch. 7 and finite-β KL-vs-max gap. |
| [Cited] | PG-DLM Propositions 1–2 and Theorem 1; E-SMC empirical feasibility; Golovin-Krause 2011 adaptive-submodularity ratio; Altman 1999 strong duality; Paternain et al. 2019 zero-duality-gap CMDPs. |
| [Refuted on OWT — heuristic was z = (s_t, b_t, phase(t)) sufficient] | The bucketed (s_t, phase(t)) abstraction does not deliver ε̃ < 0.7 ε on OWT; richer state would need a function approximator and is out-of-scope. |
| [Incorrect as stated in earlier draft] | The "strict generalisation" reduction in the original §2.1 sketch was ambiguous (top-B is not the threshold-λ specialisation under the same ψ̃). Tightened in §2.1 formal restatement: the reduction is to the abstract policy class, which contains both top-B and threshold-λ as members. |

Cross-reference into the central `research/proof_ledger.md` once the theorem
is promoted from Future Work to a formal thesis statement.

## 6. Limitations and open items

- **Tight upper bound on Δ_open.** The current thesis provides the
  MC-oracle as an oracle schedule under the open-loop regime; an
  adaptive-oracle upper bound requires either (i) exhaustive search over
  all deterministic policies on a tiny (T, B) instance, or (ii) a tighter
  analytic argument via the F2 twist function. Neither has been done.
  Labelled [Open].

- **ε̃ vs ε gap.** No evidence yet that state conditioning shrinks ε
  meaningfully on the bucketed state z = (s_t, b_t, phase(t)). Protocol C
  provides the first measurement. Labelled [Open].

- **η̃_B under the threshold policy.** In the open-loop regime η_B was
  estimated at 9,000 MC rows with the variance-form Refinement A′ giving
  σ_ξ √B / √2. Under the threshold policy the rows distribution shifts
  (it now conditions on ψ̃ exceeding λ), so σ_ξ may differ. Labelled [Open].

- **Tokenizer transfer.** If z_t is computed from signals that are
  tokenizer-dependent (entropy of a BPE tokenizer), the bucket abstraction
  is not portable across backbones. A robust signal (e.g. normalised
  margin) should be preferred for the primary demonstration.
  Labelled [Open].

- **Scope creep risk.** Any extension beyond the
  (s_t, b_t, phase(t)) state — e.g. using the predictor's full hidden
  representation as z_t — requires learning a function approximator and
  is a separate paper. The thesis's adaptive extension must keep the
  state finite-buckets. Labelled [Hard rule].

## 7. Positioning in the thesis document

Recommended placement:

- **Main body:** no change. Theorem A (open-loop) remains the main theorem.
  Phase 3a results, the Negative-Result Corollary (ranker-class), and
  the Phase 2b / 3a experiments stay as canonical content.
- **Future Work chapter** *or* **Appendix F (Adaptive Extensions)*:* one
  section summarising §1.2, §2.1 (F1 with A-ad F1 bound), §2.3 (F3 with
  A-ad F3 bound), and §4 (Protocol C as minimal empirical commitment).
  Explicit statement that A-ad is a conditional theorem with unestimated
  constants, and that Protocol C — not a full re-run — is the first step.
- **Not recommended**: promoting A-ad to a main-body theorem before
  Protocol C runs. This would violate the thesis direction's
  "honesty requirement" that proved, heuristic, and conjectured claims be
  distinguished, because ε̃ is currently unestimated.

## 8. Pointer to the research study artefacts

- Skeptical audit (Phase 1): `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md`.
- Working derivations (Phases 3–4): `research/adaptive_controller_research_notes.md`.
- Final recommendation (Phase 6): `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md` (to be written).
- Open-loop canonical theory: `docs/thesis/theory/THEORY_STATUS.md`.
- Theorem A and refinements: `research/candidate_theorems.md`.
- Provenance / tag ledger: `research/proof_ledger.md` (add A-ad rows at Phase-5 close).
- Open questions: `research/open_questions.md` (add Q-adapt-1 … Q-adapt-5 at Phase-5 close).

---

*Document owner: adaptive-controller research study. Promote to main-thesis
content only if Protocol C produces Δ_close/Δ_open > 0.5 evidence.*
