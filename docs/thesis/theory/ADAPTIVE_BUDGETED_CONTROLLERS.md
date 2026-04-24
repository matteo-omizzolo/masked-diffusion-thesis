> **STATUS:** EXTENSION-THEORY (Future-Work appendix candidate, not main-thesis canonical)
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Mathematical positioning of adaptive, state-conditional, budgeted corrector
> scheduling as a strict extension of the open-loop Theorem A framework. Companion to
> `docs/thesis/theory/THEORY_STATUS.md` (open-loop canonical theory) and to the
> Phase-1 skeptical audit at
> `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md`.

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

### 4.2 Protocol C (pilot, bounded, no new GPU work)

The minimal empirically honest evidence add-on is:

1. Re-use Phase 2b's `per_seed/policy_rows_seed{seed}.json` and
   `per_seed/mc_rows_seed{seed}.json` (LLaDA-SFT, 8 seeds).
2. Bucket by z = (s_t, b_t, phase(t)). Estimate ψ̃_bucket(z) as the
   bucket-mean of Δ_t over the 8 seeds.
3. Residuals Δ_t(z) − ψ̃_bucket(z) give ε̃ (per signal, per B).
4. Replay the λ-threshold policy over the 8 seeds; measure G(S_π̂) −
   Σ_t a_t Δ̂_t to estimate η̃_B.
5. Report a single scalar per (signal, B): Δ_close(π̂_λ,N=1) / Δ_open.

**Stop-rule.** If Δ_close / Δ_open > 0.5 on pilot: present as preliminary
Future-Work evidence. If < 0.5: present the negative result and position
the theorem as existence-only.

Protocol C uses **no new GPU trajectories** (hence the "bounded" label)
and fits inside the Phase 2b artefact budget.

## 5. Honesty ledger

Per thesis-direction honesty requirement, claims in this doc carry one of
four tags:

| Tag | Claim |
|---|---|
| [Conjecture] | Δ_open > 0 at thesis-relevant B. (Phase 2b's MC-oracle +0.45 is a surrogate, not a bound on Δ_open.) |
| [Proved, conditional] | A-ad F1 under the stated assumptions |Δ(z) − ψ̃(z)| ≤ ε̃ and η̃_B-slack. Derivation mirrors open-loop Theorem A; see `research/adaptive_controller_research_notes.md` §2.2. |
| [Proved, conditional] | A-ad F3 under SMC ergodicity of Del Moral 2004 Ch. 7 and finite-β KL-vs-max gap. |
| [Cited] | PG-DLM Propositions 1–2 and Theorem 1; E-SMC empirical feasibility; Golovin-Krause 2011 adaptive-submodularity ratio; Altman 1999 strong duality. |
| [Heuristic] | z_t = (s_t, b_t, phase(t)) is a sufficient state; smaller abstractions (e.g. s_t alone) may work for entropy but not for margin or quality; justified by within-seed variance 62 % at B = 4 but not proved. |
| [Incorrect as stated] | None yet on this file. Any claim later shown false is appended here, not silently removed. |

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
