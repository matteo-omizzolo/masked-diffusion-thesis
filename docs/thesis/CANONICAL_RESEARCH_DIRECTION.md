# Canonical Research Direction

**Author:** Matteo Omizzolo — MSc Thesis, Bocconi University
**Supervisor:** Prof. Giacomo Zanella
**Last updated:** 2026-04-23 (bounded LLaDA-SFT external-validity probe closed — Phase 2b partial-transfer verdict, Phase 3a NOT authorized on that backbone; OWT Phase 2b / Phase 3a remains the main discovery backbone and is not displaced)

> **Phase-3a-aware framing (2026-04-20).** Phase 3a (job 479941, K=30 paired,
> combinatorial baselines on ProSeCo-OWT) closes the Phase 3 empirical
> contract. **CD-G** (coordinate descent, true-G feedback) and **BS-AG**
> (beam search, cheap-A ranking + true-G rollouts) both PASS at every
> B ∈ {2, 3, 4, 8}; CD-G recovers 74–84 %, BS-AG 49–64 % of the Phase 2b
> MC-oracle +0.45 paired headroom; both still PASS at B = 8 — the budget at
> which `mean_delta_oracle` itself enters the NULL band (Phase 2b smoking
> gun #1). The thesis story tightens to:
>
> > **Fixed-budget corrector allocation is a combinatorial trajectory-control
> > problem; cheap greedy rankers are the wrong solution class for it. Search
> > procedures over schedules — even ones that use the cheap-A surrogate to
> > prune candidates — recover most of the oracle headroom.**
>
> Phase 2b's three smoking guns (mean_delta_oracle ceiling at B=8; top-10 MC
> ∩ oracle Jaccard ≈ 1.2–1.3× random baseline across B ∈ {2,3,4} per
> `combinatorial_diagnostics.json`; top-10 MC internal Jaccard ≈ bottom-10)
> still hold and still falsify the PRISM premise (PRISM is a learned
> per-token quality signal, i.e. a member of the ranker class bounded by the
> rescoped Negative-Result Corollary). The PRISM rejection therefore stands
> with a refined reason: *not* "no recoverable structure exists" (Phase 3a
> refutes that), but "the recoverable structure does not factor through any
> separable per-step score". See `PHASE3A_COMBINATORIAL_RESULTS.md`,
> `PHASE3_DIRECTION_AUDIT.md` (with post-Phase-3a addendum),
> `PHASE3_ALTERNATIVE_PLAN.md`, `RESULTS_STATUS.md` §§13–14, and
> `THEORY_STATUS.md` (Honesty Ledger, ranker-class Negative-Result Corollary
> entry).

This file is the single authoritative summary of the thesis research direction.
It supersedes any prior framing where the two disagree. Sibling canonical file:
`CANONICAL_EXPERIMENT_OVERVIEW.md`.

---

## Thesis question

> For a fixed predictor schedule and fixed corrector NFE budget in masked diffusion
> language models, can aggregate trajectory signals — entropy, confidence margin, or
> quality mass — predict the marginal value of a corrective refinement loop well
> enough to outperform uniform corrector placement?

(Verbatim from `docs/thesis_direction.md`.)

---

## Problem being solved

Modern masked diffusion language models (MDLM, ReMDM, ProSeCo, PRISM, LLaDA) run a
predictor that unmasks tokens over T steps and optionally interleave a *corrector*
that refines already-committed tokens. With a fixed predictor schedule and a fixed
global budget B of extra corrector NFEs, the open question is where along the
trajectory those B corrector loops should be placed. Adjacent work covers corrector
kernel design, token-selection policies, predictor-side schedule theory, and
remasking, but does not provide a principled trajectory-level allocator of a fixed
corrector budget.

The thesis formalizes this as a proxy-regret problem: given a trajectory-level
quality functional F, each step t has an unknown one-loop marginal gain
Δ_t = F(y_t^{+1}) − F(y_base). A signal-driven proxy ψ(s_t) derived from aggregate
trajectory signals (entropy H_t, inverse margin M_t^{-1}, quality mass Q_t) is used
to pick a budget-B schedule Ŝ_B. The question is whether Ŝ_B can compete with the
oracle top-B schedule S_B* under explicit assumptions on proxy calibration and
additivity of gains across steps.

---

## Scope and non-goals

In-scope: trajectory-level allocation of a fixed corrector NFE budget under a fixed
predictor schedule; signal-to-gain calibration; additivity/interaction of corrector
placements.

Out of scope (per `docs/thesis_direction.md`):

- Informed-corrector kernel design (how to correct).
- Remasking methods for token revisitation.
- Predictor / unmasking schedule optimization (when to unmask).
- Token-selection policies (which tokens to correct).
- Redesigning the training objective or training a learned end-to-end policy.
- Proving universal optimality of entropy as a scheduling signal.

Signals, logit-quality measures, and corrector kernels from adjacent work are
used as inputs and baselines, not as contributions.

---

## Main theorem target

### Theorem A (main) — proxy-regret bound

Let the predictor fix states Z_0,…,Z_T. For each step t, let
Δ_t := F(y_t^{+1}) − F(y_base). For S ⊆ {1,…,T}, let
G(S) := F(y^S) − F(y_base). Let S_B* := argmax_{|S|=B} G(S). Let Ŝ_B := top-B
steps by proxy score ψ(s_t).

Under (1) binary placement k_t ∈ {0,1}, (2) approximate additivity
|G(S) − ∑_{t∈S} Δ_t| ≤ η_B for all |S| ≤ B, (3) proxy calibration
|Δ_t − ψ(s_t)| ≤ ε,

    G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

Status: `solid under assumptions`. Provenance:
`[Adapted from GPT Pro assessment v2]`. Full statement:
`research/candidate_theorems.md` (Theorem A).

### Supporting results

- **Lemma A1** — oracle top-B optimality under exact additivity (trivial
  exchange argument). `[Borrowed — standard]`.
- **Lemma A2** — proxy calibration yields 2Bε regret in the exactly-additive
  regime. `[Adapted from GPT Pro assessment v2]`.
- **Proposition B** — low-gain-region (burn-in) gating is benign: if
  T_low ⊆ {t : Δ_t ≤ δ} and |T_low| ≤ B, gating it out from the allocator
  costs at most B δ + 2 η_B in regret. Replaces the earlier MI-monotonicity
  claim, which was shown not to be uniformly true. `[Adapted from GPT Pro v2]`.
- **Proposition C** — under a pairwise-interaction expansion
  G(S) = ∑ Δ_t + ∑_{t<t'∈S} ξ_{t,t'} with |ξ_{t,t'}| ≤ γ,
  we get η_B ≤ γ B(B−1)/2, making the additivity slack empirically
  estimable from pairwise measurements. `[Adapted from GPT Pro v2]`.

### Contraction route — Stretch Appendix C2

The earlier framing centered on a factorization-error contraction theorem
(Lavenant & Zanella E_fact, Ascolani-style Gibbs contraction) as the main
result. This is now demoted to a stretch appendix, for three reasons
(`research/candidate_theorems.md` Stretch C2):

1. The Ascolani et al. 2024 paper actually treats random-scan Gibbs under
   strong log-concavity, not systematic-scan (reference corrected).
2. Masked-text conditionals are not log-concave, so the hypotheses do not
   transfer.
3. No mechanism yet links a per-step contraction factor ρ(t) to an observable
   trajectory signal s_t, which is required for the theorem to imply a
   signal-adaptive schedule.

Preserved for possible future use via generalized Dobrushin / coupling
frameworks, but not load-bearing for the thesis.

---

## Role of each empirical component

- **MDLM heuristic corrector (`backends/mdlm.py`)** — diagnostic baseline.
  Negative result documented (see Status section). Retained as a comparison
  baseline and as a worked example of why corrector kernel choice matters.
  Not the main platform.
- **ProSeCo annealed-refinement corrector (`backends/proseco.py`)** — intended
  main Phase 1 platform. Layered on the MDLM backbone via
  `external/remdm/diffusion.py` loader. Corrector acts on unmasked positions
  R_t; signals defined over R_t.
- **PRISM (`external/PRISM/`)** — future source of a quality-mass signal
  (Phase 4). Currently submoduled, not cloned into the experimental pipeline.

---

## Empirical calibration quantities

All four appear directly in the theorems and are measured on real trajectories.

| Quantity | Meaning | Estimator |
|---|---|---|
| ε | Proxy calibration error |Δ_t − ψ(s_t)| | RMS residual of least-squares fit of ψ(s_t) to Δ_t across all (trajectory, t) |
| η_B | Additivity slack over |S|=B schedules | 95th percentile of |G(S) − ∑Δ_t| across M sampled schedules |
| γ | Pairwise interaction bound sup |ξ_{t,t'}| | 95th percentile of |G({t,t'}) − Δ_t − Δ_{t'}| across P sampled pairs |
| δ / T_low | Low-gain-region threshold and set | Inspect Δ̄_t profile; sweep δ |

---

## Status — what is established (post-audit, 2026-04-19)

**Theory.**

- Theorem A is `solid under assumptions (1)–(3)`; the proof through Lemmas A1
  and A2 is sketched in `research/proof_worklog.md` Entry 6/7. The combining
  step requires either 2η_B or 3η_B depending on how η_B is defined; the
  bookkeeping is recorded in `THEORY_STATUS.md` "Combining-step bookkeeping"
  and is a write-up choice, not a substantive issue.
- Two post-audit candidate refinements are registered:
  Refinement A′ (variance-form additivity slack, √B-tighter than Proposition C)
  and Refinement A″ (rank-based calibration ε_R, separating informative-low-ε
  from uninformative-low-ε). Both are derived from `THEORY_STRESS_TEST.md`
  §10 and need full proofs for thesis use.
- Lemmas A1, A2, and Proposition B remain `solid under assumptions`.
  Proposition C is `proved under pairwise decomposition` but **empirically
  loose by ≈11×** on Phase 1 ProSeCo-OWT — Refinement A′ replaces it as the
  thesis's working additivity bound pending proof.

**Infrastructure.**

- Protocol A and Protocol B implementations validated end-to-end; surrogate
  sanity PASSED. ProSeCo-OWT backend, MDLM-conf backend, and the
  `mdm_playground.scheduling` package (signals / gain / allocation / evaluate)
  are operational.
- ANALYSIS_SPEC v1 (`docs/thesis/experiments/ANALYSIS_SPEC.md`) defines the
  paired K-seed estimator, A(S) vs G(S) distinction, and Tier T1–T4 evidence
  tiers used by all Phase 2 outputs.
- Phase 2 analysis package (`src/mdm_playground/analysis/`) implements the
  paired-difference statistics, BCa bootstrap, and figures used by Phase 2a/2b.

**Empirics — Phase 1 (under audit, deprecated by Phase 2b for policy claims).**

- ProSeCo-OWT FULL RUN (job 479537, N=50, T=64) produced **valid**
  trajectory-level measurements: 61/64 positive Δ_t steps, peak Δ̄ = 0.157,
  ε_rms = 0.133, η_95(B=8) = 0.680, γ_95 = 0.264. These are the numerical
  inputs to the THEORY_STRESS_TEST.
- The same run's **policy-comparison rows are inadmissible** (single-seed,
  A vs G conflation). The "entropy_bot_B beats uniform by +29%" headline did
  not survive Phase 2b: paired CI = [−0.130, +0.030], a borderline NULL.
- MDLM diagnostic (job 478600) and ProSeCo no-op (job 478929) are retained
  as documented negative results.

**Empirics — Phase 2b (job 479581, K=30 paired, T=64; complete 2026-04-19).**

- 5 (policy, B) cells **PASS** (paired CI > 0): `middle@B=2` (+0.089),
  `entropy_bot_B_pt@B=2` (+0.105), `mean_delta_oracle@B={2,3,4}` (+0.130 to
  +0.092). 30 cells FAIL (CI < 0); 9 borderline (|Δ̂| ≤ 2σ_F).
- All "top" signal policies are **systematically harmful** (Cohen's d down to
  −2.04 for `back@B=16`).
- MC oracle (best-of-100 random schedules per seed) sits at +0.45 above
  uniform across B ∈ {2, 3, 4}, with CI tight to ±0.07. Mean-profile oracle
  captures only +0.09 of that; the remaining +0.36 is **per-instance**.
- Pooled Spearman ρ(A, G) decays monotonically with B: 0.66 (B=2), 0.62 (B=3),
  0.50 (B=4), 0.44 (B=8), 0.45 (B=16). This validates Refinement A″'s
  prediction that ε_R = (1−|ρ|)·σ_Δ grows with B.
- Smoking guns for the "scheduling is combinatorial, not signal-driven"
  hypothesis: (i) `mean_delta_oracle` uses ground-truth Δ_t and saturates by
  B=8; (ii) top-10 MC schedules overlap `mean_delta_oracle`'s picks at only
  1.20 / 1.18 / 1.30× random baseline at B ∈ {2,3,4} (observed in
  `results/phase2b/combinatorial_diagnostics.json`; prior "~1.5×" figure
  used an analytic E[|∩|]/E[|∪|] baseline, which over-states the ratio
  versus the exact MC baseline used by `random_jaccard_baseline`);
  (iii) top-10 MC schedules show internal Jaccard
  indistinguishable from bottom-10. Together: the +0.45 headroom is not
  recoverable by any greedy single-step ranker, learned or otherwise.
- Variance decomposition at B=4: 62 % of total G-variance across MC schedules
  is *within-seed* (i.e. depends on schedule choice, not instance), so the
  gain is real and combinatorial — there is no escape via "sampling lottery"
  explanations.

**Empirics — Phase 3a (job 479941, K=30 paired, T=64; complete 2026-04-20).**

- Two non-greedy search procedures over schedules — **CD-G** (coordinate
  descent with true-G feedback, ≤ 65 G-calls per cell) and **BS-AG** (beam
  search width 8, cheap-A ranking on extension candidates with true-G
  rollouts on top 8) — both **PASS** at every tested budget B ∈ {2, 3, 4, 8}
  with paired BCa CIs strictly above 0.
- Oracle-gap closure at B ∈ {2, 3, 4}: **CD-G recovers 78.9 / 74.1 / 84.3 %**
  of the Phase 2b MC-oracle headroom; **BS-AG recovers 64.1 / 57.1 / 48.8 %**.
- At B = 8 (no Phase 2b MC oracle anchor) both methods still PASS: CD-G
  Δ = +0.322, BS-AG Δ = +0.153. This is the budget at which
  `mean_delta_oracle` (the upper envelope of any single-step ranker) sits
  in the NULL band — a search procedure exceeds the *ranker class* envelope.
- Caveats are load-bearing for the write-up: CD-G uses true-G feedback and is
  a structural / existence result, not a deployable inference-time scheduler;
  BS-AG is closer to practical (cheap-A ranking + O(B) true-G rollouts) but
  still requires a true-G evaluator; Phase 3a was run on ProSeCo-OWT only;
  no theoretical guarantee bounds the recoverable closure ratio (49–84 %).
- **Refines** the Phase 2b verdict from "scheduling is hard" to
  "scheduling is **search**, not signals; the wrong solution class is
  separable greedy ranking, not informed scheduling in general."

Full report: `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`.

## Status — what remains open

**Empirical (Phase 2 closed; Phase 3a closed; Phase 3b active).**

- Q12 (true G(S_B*)): Phase 2b's MC oracle (best-of-100 per seed) gives
  +0.45 over uniform at B ∈ {2, 3, 4}; Phase 3a's CD-G recovers 74–84 % of
  this headroom (search over schedules, true-G feedback) and BS-AG recovers
  49–64 % (search over schedules, cheap-A ranking + true-G rollouts). No
  tighter lower bound on S_B* than Phase 2b MC + Phase 3a CD-G is currently
  planned.
- Q5 / Q11: pooled Spearman ρ(A, G) decays 0.66 → 0.39 across B ∈ {2..16}.
  Refinement A″ is empirically anchored; remaining work is the formal
  order-statistics derivation in Phase 3b.
- Q10: σ_ξ (residual G − A per (seed, schedule)) measured from Phase 2b MC
  rows. Replaces Proposition C's `B(B−1)/2` looseness with the √B-tighter
  variance-form bound (Refinement A′), pending formal proof in Phase 3b.
- Phase 2c MAUVE F-swap **closed without execution.** Cohen's d up to ±2
  in many cells leaves cell ordering robust to metric choice; F-swap was
  not reopened (Phase 3a positive eliminates the need to defend the small-B
  ranker win — the search-class result is the load-bearing positive).
- Phase 3b (theory finalisation, **active**): formal proofs of A′ and A″ +
  Negative-Result Corollary **rescoped to the greedy/separable-ranker class**
  (Phase 3a search procedures explicitly exceed the ranker envelope, so the
  corollary cannot generalise to "informed scheduling in general").

**Theoretical.**

- Formal proofs of Refinement A′ (variance-form, requires mixing hypothesis
  on ξ; σ_ξ already measured) and Refinement A″ (rank-based ε_R, requires
  order-statistics derivation; ρ-decay already measured).
- Negative-Result Corollary (rescoped post-Phase-3a): any policy of the form
  `Ŝ_B = top-B(ψ)` for separable per-step `ψ` is upper-bounded by the
  `mean_delta_oracle` envelope, which on ProSeCo-OWT enters the NULL band by
  B = 8. The corollary characterises the **ranker class only**; Phase 3a's
  CD-G + BS-AG explicitly exceed this envelope.
- Clean LaTeX write-up of Theorem A's combining step (with explicit η-definition
  choice).
- Stretch C2 contraction: only if Ascolani et al. 2024 / Denoising Entropy
  Bounds yield an applicable framework for discrete masked-text conditionals.

**Out of scope as of Phase 3 audit (verdict unchanged after Phase 3a).**

- PRISM fine-tuning. Rejected with refined reason: PRISM is a learned
  per-token quality signal, i.e. a member of the ranker class bounded by the
  rescoped Negative-Result Corollary; even a perfect Δ-predictor cannot
  exceed `mean_delta_oracle`'s envelope, which enters the NULL band by B = 8.
  Phase 3a does **not** rehabilitate PRISM — the recoverable structure does
  not factor through any separable per-step score. Full rationale in
  `PHASE3_DIRECTION_AUDIT.md` (with post-Phase-3a addendum).
- Cross-backbone (MDLM full Phase 2b / Phase 3a replication). Parked. The
  search-vs-ranker dichotomy on a different (backbone, corrector, F) triple
  is out of thesis scope.
- A formal lower bound on what CD-G or BS-AG can recover. The 49–84 %
  closure ratios are empirical descriptors, not theorems; characterising the
  topology of useful schedules formally is sequel-paper material.
- A deployable inference-time scheduler. CD-G uses true-G feedback (≈ 65 ×
  generation cost per cell, structural result only); BS-AG would need
  either a tractable G surrogate or a draft-and-rescore wrapper.

---

## Current empirical plan (Phase 3a closed; Phase 3b active)

- **Phase 3a (HPC) — COMPLETE 2026-04-20** (job 479941). Both CD-G and BS-AG
  PASS at every B ∈ {2, 3, 4, 8}; CD-G recovers 74–84 %, BS-AG 49–64 % of
  the Phase 2b MC-oracle headroom at B ∈ {2, 3, 4}; both still PASS at
  B = 8 where the ranker envelope is NULL. Decision-gate outcome:
  **"scheduling is search, not signals" (positive)** — the thesis chapter
  on combinatorial corrector scheduling lands. Full report:
  `PHASE3A_COMBINATORIAL_RESULTS.md`. Provenance:
  `results/phase3a_proseco_owt/{cd,bs}_{paired,raw}.json`,
  `oracle_gap_closure.json`.
- **Phase 3b (notebook, ≈ 2–3 days) — ACTIVE.** Theory finalisation:
  promote Refinements A′ and A″ from "candidate" to "stated and proven
  (informal)" in `research/candidate_theorems.md`; add the Negative-Result
  Corollary **scoped to the greedy/separable-ranker class** (Phase 3a
  search procedures explicitly exceed the ranker envelope, so the corollary
  must be ranker-class only); update `THEORY_STATUS.md` Honesty Ledger
  entries (already partially anchored). Out of scope for Phase 3b: any
  formal lower bound on CD-G / BS-AG closure ratios.
- **Tier-2 contingencies — closed.** Alt-C external-validity micro-experiment
  on MDLM is **not** triggered (cross-backbone replication of the search-vs-
  ranker dichotomy is out of thesis scope; the Phase 3a positive on
  ProSeCo-OWT is sufficient for the chapter contract). Alt-D Phase 2c
  MAUVE F-swap is **not** triggered (Phase 3a's search-class positive is
  the load-bearing result; no need to defend the small-B ranker win).
- **Bounded cross-backbone appendix probe (ProSeCo-LLaDA-SFT) — CLOSED 2026-04-23.**
  Infrastructure under `hpc/cross_backbone_proseco_llada_sft_{bounded,resume_phase2b}.sbatch`
  and `src/mdm_playground/scheduling/backends/proseco_llada_sft.py`. Protocol A
  and Phase 2b at K=8, T=64, B ∈ {2, 4}, GPT-2 reference, ProSeCo-style
  corrector completed on 2026-04-22; analysis 2026-04-23. The run is an
  **external-validity probe** for the OWT K=30 mainline and is treated as
  bounded / Tier-3 evidence only (`n = 8 < 30`; CIs ~1.94× wider than
  OWT's K=30; corrector semantic shift: LLaDA is time-agnostic, corrector
  iteration is argmax rather than annealed-σ). The probe yields a
  **partial-transfer verdict scoped to the tested bounded setup**:
  (a) the uniform-not-beaten observation from OWT Phase 2b is corroborated
  at T3 tier; (b) positive MC-oracle headroom over uniform did **not**
  transfer at the tested budgets (paired bootstrap CI at B=4 is [0, 0] and
  at B=2 is [−4.07, −1.07]); (c) per-trajectory signal dominance over the
  mean-profile schedule transfers with large effect (cohens_d ≈ 7.8 at B=4).
  The verdict is **not** "uniform is optimal across masked diffusion corrector
  scheduling" — it is bounded external-validity evidence only. Three
  non-discriminable hypotheses (corrector dominance H1, protocol sparseness
  H2, reference mismatch H3) explain the non-transfer at K=8; discrimination
  is outside thesis scope. **Phase 3a on LLaDA-SFT is NOT authorized** by
  the current decision (terminal node:
  `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md`); re-opening
  precondition is documented in §6 of that file. The OWT K=30 Phase 2b /
  Phase 3a mainline is **not displaced** by this probe. Full result report:
  `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` §10–§12.

## Immediate next milestone (Phase 3b — theory finalisation)

With Phase 3a closed, the binding milestone is the Phase 3b theory contract:

1. Formal proof of **Refinement A′** (variance-form additivity slack,
   `𝔼|G − A| ≤ σ_ξ · √B / √2`), using the σ_ξ measured directly from
   Phase 2b MC residuals (9 000 G − A pairs across B ∈ {2, 3, 4}).
2. Formal proof of **Refinement A″** (rank-based ε_R,
   `𝔼[A(S_A*) − A(Ŝ_B)] ≤ B · ε_R`), using the order-statistics derivation
   on ρ(A, G) decay (0.66 → 0.39 across B ∈ {2..16}).
3. Statement and proof of the **Negative-Result Corollary scoped to the
   ranker class**: any policy `Ŝ_B = top-B(ψ)` for separable per-step ψ is
   upper-bounded by the `mean_delta_oracle` envelope; on ProSeCo-OWT this
   envelope enters the NULL band by B = 8. Phase 3a CD-G + BS-AG operate on
   schedules and exceed this envelope, demonstrating the corollary is
   tight to the ranker class and does *not* generalise.
4. Update `research/candidate_theorems.md`, `research/proof_worklog.md`,
   `THEORY_STATUS.md` Honesty Ledger; draft `thesis/chapters/ch6_theory.tex`
   skeleton with refined theorems.

See `CANONICAL_EXPERIMENT_OVERVIEW.md` for the full Protocol A/B
specification, `RESULTS_STATUS.md` §§12–14 for the Phase 2b → Phase 3a
rollout, `PHASE3A_COMBINATORIAL_RESULTS.md` for the Phase 3a report, and
`PHASE3_DIRECTION_AUDIT.md` for the standing PRISM rejection (with refined
post-Phase-3a reason).
