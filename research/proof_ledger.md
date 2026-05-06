> TECHNICAL WORKLOG.
> Current thesis status starts at `START_HERE.md`. Summary in `docs/03_theory.md`.
> Current theory-first plan: `docs/06_theory_first_research_plan.md`.

# Proof Ledger: Provenance Tracking

**Updated:** April 2026 (expanded after GPT Pro v2 assessment).
**Correction:** May 2026 — see correction block below.

## Current correction — May 2026

This ledger records provenance, including **superseded** theory.

Current active theory is: **A → B / B′ → Diagnostic Framework C**.
- A′ and A″ are **diagnostics** only (not regret refinements).
- The old "Negative-Result Corollary" is **superseded** by the
  Empirical Ranker-Class Limitation (`candidate_theorems.md` §1.5).
- The old "Proposition C" (γ B(B−1)/2 pairwise interaction bound) is
  **superseded** by Theorem B / B′ + Diagnostic Framework C.

Older entries below are **historical** unless explicitly referenced by
`research/candidate_theorems.md` §0–§7. Pre-revision proofs are archived
at `docs/archive/old_theory_stack/candidate_theorems_pre_2026_05.md`.

---

> Tracks the origin of every proof ingredient, inequality, decomposition, or
> modeling choice used in the thesis. Every new claim, definition, and
> conjecture added during the April 2026 restructure is tagged here.

### Tag system

| Tag | Meaning |
|-----|---------|
| `[Borrowed]` | Lifted from a cited source with minimal adaptation |
| `[Adapted]` | Modified from a source for the masked-diffusion setting |
| `[Adapted from GPT Pro assessment v2]` | Contributed by the April 2026 GPT Pro critique |
| `[Analogy]` | Inspired by a result in a different setting |
| `[Novel]` | Original to this thesis |
| `[Definition]` | A definition (no truth claim, but still needs review) |
| `[Conjecture]` | Tentative; counterexamples not ruled out |
| `[Incorrect as stated]` | A previous claim now known to be wrong; preserved for history |
| `[Refuted / abandoned]` | Shown to fail and removed from active use |
| `[Validated empirically]` | Empirical check completed; see linked experiment |
| `[Depends on calibration]` | Argument holds modulo empirical signal calibration (ε) |
| `[Depends on approximate additivity]` | Argument holds modulo additivity bound (η_B) |
| `[Depends on bounded pairwise interaction]` | Historical tag: held modulo γ in old Proposition C; superseded by Theorem B/B′ diagnostics |
| `[Needs verification]` | Flagged for future formal or empirical work |
| `[Empirically motivated]` | Supported by data rather than theory |

---

## Borrowed Ideas

### L&Z Error Decomposition
- **Source:** Lavenant & Zanella 2025 (arXiv:2510.25544).
- **What:** KL(p_data ‖ p_alg) ≤ E_learn + E_fact decomposition; information
  profile view; per-step factorization error.
- **Used in:** Problem formalization (Entry 1); information-profile route
  (Entry 5); Stretch Appendix C2.
- **Tag:** `[Borrowed from L&Z]`

### Gibbs Sampling Entropy Contraction
- **Source:** Ascolani, Lavenant & Zanella 2024 (arXiv:2410.00858).
- **What (corrected):** Entropy contraction bounds for **random-scan** Gibbs
  under appropriate conditional-independence and positive-curvature-type
  hypotheses (not log-concavity in the classical sense).
- **Previous (incorrect) citation in this ledger:** "systematic-scan Gibbs
  under log-concavity." Corrected April 2026; see `proof_worklog.md` Entry 6.
- **Used in:** Stretch Appendix C2 only (contraction route is now a stretch
  result, not the main theorem).
- **Status:** `[Incorrect as stated in earlier draft; corrected April 2026]`
  `[Adapted — applicability to masked diffusion correctors not established]`
  `[Needs verification]`

### Informed Proposals for Discrete MCMC
- **Source:** Zanella 2020.
- **What:** Locally-balanced proposal distribution for discrete state spaces.
- **Used in:** Background vocabulary for informed correctors; discussion of
  Zhao et al.'s Barker/MPF kernels.
- **Tag:** `[Borrowed from Zanella 2020 — vocabulary only]`

### Spectral Gap and Non-Uniform Scan
- **Sources:** Roberts & Sahu 1997; *Adapting the Gibbs Sampler* (2018).
- **What:** Coordinate update order affects mixing; non-uniform scan can
  improve over uniform scan.
- **Used in:** Conceptual motivation for non-uniform corrector scheduling.
- **Tag:** `[Analogy to non-uniform scan Gibbs]`

### L&Z Information Profile
- **Source:** Lavenant & Zanella 2025.
- **What:** Decomposition of KL budget into per-step informational content
  captured by the predictor.
- **Used in:** Motivation of Gap E (corrector contribution to E_fact per step).
- **Tag:** `[Borrowed from L&Z]`

---

## Adapted Ideas

### Top-B Proxy-Regret Framework (Theorem A)
- **Source:** Classical resource-allocation + sort-and-swap arguments; the
  specific formulation due to the April 2026 GPT Pro v2 assessment.
- **What:** The statement G(S_B*) − G(Ŝ_B) ≤ 2Bε + 2η_B with explicit
  calibration (ε) and additivity (η_B) terms.
- **Adapted for:** Corrector scheduling in masked diffusion; signal-as-proxy
  framing.
- **Tag:** `[Adapted from GPT Pro assessment v2; standard exchange argument]`

### Approximate Additivity via Pairwise Interaction (Historical Proposition C)
- **Source:** Combinatorial optimization; second-order interaction models.
- **Adapted for:** Linking η_B to an empirically estimable pairwise bound γ.
  Current replacement is Theorem B/B′ plus schedule-level validation of Q(S) ≈ G(S).
- **Tag:** `[Adapted from GPT Pro assessment v2]`

### Resource Allocation Under Budget Constraint
- **Source:** Standard optimization theory (Lagrangian methods, water-filling).
- **What:** Allocate limited budget to maximize total payoff under diminishing
  returns.
- **Adapted for:** Corrector budget allocation across trajectory steps.
- **What is new:** The application to corrector scheduling in masked diffusion;
  the specific identification of what "marginal gain" means.
- **Tag:** `[Adapted from resource allocation theory]`

### Factorization Error Contraction Model (Stretch C2)
- **Source:** L&Z (for E_fact) + Ascolani et al. 2024 (for contraction, with
  corrected scan type).
- **What is new:** Combining L&Z's per-step E_fact with a geometric contraction
  model for the corrector kernel, yielding E_fact(t) · ρ(t)^{k_t}.
- **Status:** Demoted from main theorem to stretch appendix after GPT Pro v2
  critique (see `gpt_pro_assessment_response.md` items A2, D2, R1).
- **Tag:** `[Adapted from L&Z + Ascolani et al. 2024]` `[Conjecture]`
  `[Needs verification]`

---

## Novel Ideas

### Signal-Adaptive Trajectory-Level Scheduler
- **What:** The overall framing of corrector scheduling as a trajectory-level
  resource allocation problem guided by aggregate signals (entropy, confidence
  margin, quality mass).
- **ProSeCo audit complete (April 2026).** ProSeCo (arXiv:2602.11590) uses
  a fixed periodic schedule (every N steps, starting from step K). It does not
  use trajectory signals to decide when to apply the corrector, does not define
  or measure Δ_t, and does not provide a proxy-regret theorem. The thesis uses
  ProSeCo as its corrector backend and provides the theoretical framework ProSeCo
  lacks. Novelty claim is confirmed.
- **Tag:** `[Novel framing]` `[Novelty confirmed vs ProSeCo April 2026]`

### ProSeCo Corrector Kernel (Phase 1 Backend)
- **Source:** ProSeCo paper (arXiv:2602.11590); code in `backends/proseco.py`.
- **What:** Annealed iterative refinement: x̂_0 = argmax(p_θ(x_0|x_t)); run
  `corrector_steps` backbone calls at decreasing τ ∈ {1.0, …, ε}; apply result
  to unmasked positions R_t only. Chosen as Phase 1 backend because the MDLM
  heuristic corrector produced all Δ_t ≤ 0.
- **What is new:** The use of this kernel within the Theorem A proxy-regret
  framework (measuring ε, η_B, γ); signal computation over R_t (unmasked);
  burn-in verification (R_t ≈ ∅ at early steps → Proposition B empirically verified).
- **Tag:** `[Borrowed from ProSeCo (kernel mechanism)]`
  `[Novel in Theorem A application + ε/η_B/γ measurement]`

### MDLM Heuristic — Diagnostic Baseline (Refuted for Scheduling)
- **What:** Gibbs-style one-shot resample of ALL masked positions from p(x_0|x_t).
  Run as Phase 1 diagnostic (job 478600, April 2026).
- **Finding:** All Δ_t ≤ 0 at every step. Corrector is harmful because at early
  steps ~97% of tokens are resampled with <5% context. This is a known-bad
  corrector design, documented as a negative result.
- **Signals:** Bug #1 (signals over wrong positions) also invalidated calibration.
  Both bugs documented in `result_inventory.md`.
- **Status:** Preserved as historical baseline / negative result. Not used in
  Phase 1 ProSeCo analysis.
- **Tag:** `[Refuted / abandoned as scheduling platform]`
  `[Preserved as negative diagnostic result]`

### One-Loop Marginal Gain Δ_t
- **Definition.** Δ_t := F(y_t^{+1}) − F(y_base) for a chosen trajectory-level
  quality functional F.
- **Tag:** `[Definition]` `[Novel in this thesis's formalization;
  closely analogous to leave-one-out marginal contribution in machine learning]`

### Separation of TCR_t and Δ_t
- **What:** GPT Pro v2 flagged that the previous draft conflated the
  token-change rate TCR_t (fraction of positions changed by one corrector
  loop) with the quality gain Δ_t. These are empirically distinct and must
  not be identified.
- **Used in:** Signal-vs-gain calibration protocol (ε estimation).
- **Tag:** `[Adapted from GPT Pro assessment v2]`

### Low-Gain-Region Exclusion (Proposition B)
- **What:** Empirically identify T_low ⊆ {1, …, T} where measured Δ_t ≤ δ,
  then exclude from top-B-by-proxy selection. Replaces the earlier MI
  monotonicity claim (see Refuted section below).
- **Tag:** `[Adapted from GPT Pro assessment v2]` `[Novel substitute for MI-
  monotonicity claim]` `[Empirically motivated]`

### Expectation-Version Remark
- **What:** An expected-value formulation of (ε, η_B) yielding an
  expected-regret bound, as a hedge against heavy-tailed Δ_t.
- **Tag:** `[Novel, derivative of Theorem A]` `[Needs verification]`

---

## Incorrect as Stated — Preserved for History

### MI Monotonicity in Unmasked Fraction
- **Original claim (former Candidate 3):** I(x_i; x_{-i} | Z_t) is monotone
  increasing in u_t (fraction unmasked at step t).
- **Status:** `[Incorrect as stated]`. The conditional distribution at Z_t
  depends on both the masked set and the realized unmasked values; there is
  no uniform monotonicity in u_t.
- **Replaced by:** Proposition B (low-gain-region exclusion).
- **Tag:** `[Incorrect as stated]` `[Refuted / abandoned as a proof step]`.

### Systematic-Scan Citation for Ascolani et al.
- **Original citation:** "Per-step KL contraction for systematic-scan Gibbs
  under log-concavity (Ascolani, Lavenant & Zanella 2024)."
- **Correct:** Random-scan Gibbs; hypotheses are not classical log-concavity.
- **Status:** `[Incorrect as stated]`; citation corrected April 2026.
- **Tag:** `[Incorrect as stated; corrected]`.

---

## Definitions (reviewed April 2026)

- **Corrector at step t.** Markov kernel K_t acting on the current state Z_t,
  resampling a subset of token positions from a distribution informed by the
  model's conditional. Does not change the noise level. `[Definition]`
- **Predictor schedule.** Fixed sequence of T unmasking steps producing
  Z_0, …, Z_T. `[Definition]`
- **One-loop marginal gain Δ_t.** F(y_t^{+1}) − F(y_base). `[Definition]`
- **Token-change rate TCR_t.** Fraction of positions changed by one corrector
  loop at step t. Distinct from Δ_t. `[Definition]`
- **Aggregate trajectory signal s_t.** Scalar function of the state and model
  logits at step t (e.g., mean conditional entropy, inverse margin, quality
  mass). `[Definition]`
- **Proxy score ψ(s_t).** Scoring function used to rank steps for top-B
  selection. May equal s_t or a calibrated transform. `[Definition]`
- **Calibration error ε.** sup_t |Δ_t − ψ(s_t)| (or expected version). `[Definition]`
- **Additivity slack η_B.** sup_{|S| ≤ B} |G(S) − ∑_{t ∈ S} Δ_t|. `[Definition]`
- **Pairwise interaction bound γ.** sup_{t, t'} |ξ_{t, t'}| in the pairwise
  expansion. `[Definition]`
- **Low-gain region T_low.** {t : Δ_t ≤ δ} for a chosen threshold δ. `[Definition]`

---

## Unverified Claims — Flagged Table

| Claim | Used in | Status | Action needed |
|-------|---------|--------|---------------|
| Approximate additivity |G(S) − ∑ Δ_t| ≤ η_B | Theorem A | `[Depends on approximate additivity]` | Protocol B empirical measurement |
| Calibrated proxy \|Δ_t − ψ(s_t)\| ≤ ε | Theorem A | `[Depends on calibration]` | Protocol A empirical measurement |
| Pairwise interaction bound γ | Historical Proposition C | `[Depends on bounded pairwise interaction]` | Superseded by ζ_{B,C}, P_B schedule-level validation in Theorem B/B′ |
| Existence of low-gain region T_low | Proposition B | `[Empirically motivated]` | Protocol A inspection of Δ_t across t |
| Geometric contraction E_fact · ρ^k | Stretch C2 | `[Conjecture]` `[Needs verification]` | Read Ascolani et al. + Denoising Entropy; check whether discrete-MCMC contraction frameworks apply |
| ρ(t) admits tractable functional link to s_t | Stretch C2 | `[Conjecture]` | Only relevant if C2 pursued; empirical fit |
| Signal s_t correctly ranks Δ_t | Everywhere | `[Empirically motivated only]` | Protocol A — central experimental question |
| ProSeCo does not already do signal-adaptive scheduling | Novelty claim | `[Verified — confirmed April 2026]` | See Q4 in open_questions.md; ProSeCo audit complete |
| Expectation-version bound | Theorem A remark | `[Needs verification]` | Formal write-up |
| TCR_t vs Δ_t distinction matters empirically | Proxy design | `[Empirically motivated]` | Measure both in Protocol A |

---

## Cross-Reference Summary

| Object | Provenance tags |
|--------|-----------------|
| Theorem A (proxy-regret) | `[Adapted from GPT Pro assessment v2]` `[Depends on calibration]` `[Depends on approximate additivity]` |
| Lemma A1 (oracle top-B) | `[Borrowed — standard resource allocation]` |
| Lemma A2 (calibration regret) | `[Adapted from GPT Pro assessment v2]` |
| Proposition B (low-gain gating) | `[Adapted from GPT Pro assessment v2]` `[Empirically motivated]` |
| Historical Proposition C (pairwise interaction) | `[Adapted from GPT Pro assessment v2]` — superseded by Theorem B/B′ + Diagnostic Framework C |
| Stretch C2 (contraction) | `[Adapted from L&Z + Ascolani 2024]` `[Conjecture]` `[Needs verification]` |
| Stretch C3 (confidence margin) | `[Novel framing]` `[Empirically motivated]` |
| MI monotonicity (old C3) | `[Incorrect as stated]` `[Refuted]` — preserved in history only |
| Systematic-scan citation | `[Incorrect as stated]` — corrected to random-scan |

### Adaptive Extension (added 2026-04-22; extension only — not main-thesis canonical)

| Object | Provenance tags |
|--------|-----------------|
| Adaptive object (state z_t, action a_t, budget b_t) | `[Definition]` `[Novel framing in thesis context]` |
| Theorem A-ad F1 (CMDP threshold regret 2Bε̃+2η̃_B) | `[Adapted from Altman 1999 + Paternain et al. 2019 + open-loop Theorem A structure]` `[Conjecture]` `[Depends on state-conditional calibration ε̃]` `[Depends on adaptive additivity η̃_B]` `[Needs verification]` |
| Theorem A-ad F3 (conditional SMC + (A-ad F1) with 1/√N + log\|Z\|/β) | `[Adapted from Del Moral 2004 Ch. 7 + Andrieu-Doucet-Holenstein 2010 + A-ad F1]` `[Conjecture]` `[Needs verification]` `[Depends on SMC ergodicity]` |
| Framework F1 as primary normative frame | `[Borrowed — CMDP textbook]` `[Analogy to open-loop Theorem A]` |
| Framework F2 as formal glue only | `[Borrowed — Todorov 2009, Kappen et al. 2012, Levine 2018]` `[Definition]` — no independent regret claim attempted |
| Framework F3 as algorithmic realisation | `[Borrowed — Del Moral 2004, ADH 2010]` `[Empirically motivated via PG-DLM arXiv:2507.08390 and E-SMC arXiv:2512.21336]` |
| Framework F4 (adaptive submodularity) as foil | `[Borrowed — Golovin-Krause 2011]` `[Refuted for this problem]` — Prop C's γ > 0 falsifies monotone diminishing-returns hypothesis |
| z_t = (s_t, b_t, phase(t)) sufficient-state heuristic | `[Heuristic]` `[Empirically motivated by 62% within-seed variance]` `[Needs verification]` |
| Protocol C (pilot, bounded, no new GPU) | `[Definition]` `[Novel — this study]` |

Details: docs/archive/ (archived) and
`research/adaptive_controller_research_notes.md` (scratch derivations).
Phase-1 skeptical audit: docs/archive/ (archived).

---

See `docs/gpt_pro_assessment_response.md` for the item-level audit that drove
these entries.
