> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


> **STATUS:** INDEPENDENT AUDIT (does not defer to prior decision docs)
> **LAST VERIFIED:** 2026-04-24
> **SCOPE:** Skeptical classification of the thesis's current state after (i) OWT
> Phase 2b + 3a, (ii) the bounded LLaDA-SFT K=8 external-validity probe, and
> (iii) the 6-phase adaptive-controller study. Decides where the strongest next
> move lies given the 2025–2026 MDM literature landscape.
>
> This document SUPERSEDES INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md where
> the two conflict, and is intended to be read alongside (not after)
> POST_CROSS_BACKBONE_DECISION.md and POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md.

---

# Next Research Direction — Independent Audit

## 0. Audit stance

This audit deliberately does **not** treat any prior plan, decision doc, or
canonical direction file as authoritative. Every claim is re-derived from one
of three sources:

1. **On-disk experiment artefacts** — `results/phase2b/*.json`,
   `results/phase3a/*.json`, `results/cross_backbone/**/*.json`.
2. **Proven or conjectured theorem text** — `research/candidate_theorems.md`,
   `research/proof_ledger.md`, `research/adaptive_controller_research_notes.md`.
3. **External literature verified during the 2026-04-24 scan** (see §3).

The audit treats disagreements between plan docs and result docs as real
information, not noise. Where plan docs assumed something that did not come
true, this audit reclassifies it rather than filing it under "progress".

---

## Section A — Classification of what the thesis has established

### A.1 SUPPORTED (file-backed, Tier T1–T2)

On ProSeCo-OWT at K = 30, T = 64, B ∈ {2, 3, 4, 8}:

- **Ranker-class null.** No separable per-step ranker policy drawn from the
  tested signal set beats uniform on paired G at B ∈ {2, 3, 4}. Source:
  `results/phase2b/policy_comparison_paired.json` + phase2b memo.
- **Oracle saturation.** `mean_delta_oracle` (the envelope of any separable
  per-step ranker) reaches the uniform baseline by B = 8 on OWT; this is the
  anchor for the Negative-Result Corollary. Source: `results/phase2b/
  mean_delta_oracle_curve.json` + phase2b memo §4.
- **Top-k / oracle set overlap.** Jaccard ratios 1.18–1.30× random across
  B ∈ {2, 3, 4}; within-seed share 0.619–0.691. Source:
  `results/phase2b/combinatorial_diagnostics.json`.
- **MC-oracle headroom on OWT.** MC-oracle at N = 100 random schedules per
  seed exceeds uniform by +0.45 NLL units (paired mean diff, 95 % BCa CI
  excludes 0). Source: phase2b mc_oracle.json.
- **Search-class positive.** CD-G closes 74–84 % of MC-oracle headroom on OWT;
  BS-AG closes 49–64 % with A-rank + G-rollouts. Source: `results/phase3a/
  closure_table.json` + phase3a memo.

On **Theorem A** (the core theoretical contribution):

- **Theorem A (proxy-regret bound).** `G(S_B*) − G(Ŝ_B) ≤ 2Bε + 2η_B`
  under (1) binary placement, (2) approximate additivity with slack η_B,
  (3) proxy calibration with L∞ slack ε. Proof combines Lemma A1 (top-B
  oracle optimality under exact additivity) + Lemma A2 (proxy approximation
  propagates as 2Bε). Source: `research/candidate_theorems.md`, ledger
  entries `[Novel]` + `[Adapted]`.
- **Vacuity regime.** At B = 8 on ProSeCo-OWT Phase 1, plug-in numeric
  2Bε + 2η_B ≈ 3.50 vs plausible G ≤ 1.2 — bound is **vacuous at that
  triple**. Source: candidate_theorems §"Empirical anchor".

On LLaDA-SFT (ProSeCo-LLaDA-SFT backend), bounded probe K = 8, T = 64,
B ∈ {2, 4}, GPT-2 reference, 8 seeds:

- **Uniform-is-un-beaten reproduces strongly.** Paired (uniform − MC-oracle)
  mean diff: **+2.64 at B = 2** and **exactly 0.000 at B = 4** (all 24 MC
  samples per seed equal uniform G to reference precision). Source:
  `results/cross_backbone/proseco_llada_sft_bounded/phase2b/
  policy_comparison_paired.json`, mc_oracle.json.
- **MC-oracle does NOT exhibit positive headroom** over uniform at this
  K = 8 setup — a strictly stronger null than OWT. The OWT +0.45 headroom
  does not transfer at the tested budgets.
- All signal policies tested (mean_delta_oracle, entropy_top_B_mp, front)
  have paired-diff ≤ 0 vs uniform on LLaDA-SFT at this setup.

On the 2025–2026 MDM theory landscape (see §3 for citations):

- **No 2025–2026 paper publishes a proxy-regret bound for fixed-budget
  informed-corrector scheduling.** PG-DLM (2507.08390) does particle
  Gibbs over reward-guided full trajectories — a different algorithmic
  class. E-SMC, EAGS, Learning Unmasking Policies, APD, Soft-Masked
  Diffusion, Entropy-Bounded Unmasking are predictor-side or training-time.
  The thesis's niche is **empirically confirmed open**.

### A.2 PLAUSIBLE BUT NOT PROVEN

- **Approximate additivity at low B breaks at high B.** Partially evidenced
  by mean_delta_oracle saturation by B = 8 on OWT. Not yet measured at
  finer-grained B for a clean break-point. Open question Q5.
- **Refinement A′ (variance η_B = σ_ξ √B / √2) tightens the vacuous
  regime.** Derivation exists; empirical tightness on ProSeCo-OWT
  artefacts not yet computed. Open question Q10.
- **Refinement A″ (rank-based ε_R = (1 − |ρ|)·σ_Δ) replaces the loose
  L∞ calibration slack with Spearman rank correlation.** Derivation
  exists; not yet computed on OWT. Open question Q11.
- **Proposition B (low-gain-region exclusion).** Proof sketch exists in
  candidate_theorems.md; the empirical diagnostic (share of mean gain
  captured by top-k sites) anchors it at 62–69 % within-seed share.
  Not yet a formalized theorem statement in the thesis.
- **Proposition C (pairwise γ bound: η_B ≤ γ·B(B−1)/2).** Derivation exists;
  γ estimate on OWT artefacts not yet computed.
- **The OWT ranker-class negative result generalises to "any z-agnostic
  separable ranker".** Holds by definition for the tested signals; whether
  a not-yet-tested separable signal exists that beats uniform is open —
  but by the mean_delta_oracle envelope, it cannot beat mean_delta_oracle
  under the assumed decomposition.

### A.3 SPECULATIVE (conjectural, not file-backed)

- **Theorem A-ad (adaptive-budgeted controllers) strictly generalises
  Theorem A to achieve non-vacuous gains on some (backbone, corrector,
  task) triple.** Framework derivation exists in
  `ADAPTIVE_BUDGETED_CONTROLLERS.md` §2 (F1 FH-CMDP Bellman form) and §4
  (Theorem A-ad combined statement). Constants ε̃, η̃_B have not been
  estimated; Δ_close/Δ_open ratio unmeasured.
- **Richer state z_t (beyond the (s_t, b_t, phase(t)) 3-tuple) yields
  adaptive-controller gains worth a PhD-level paper.** Out of thesis
  scope regardless of truth value.
- **MC-oracle headroom observed on OWT is representative of the true
  Δ_open structure.** MC-oracle is a *bound* on Δ_open + random-schedule
  variance, not a clean measurement of Δ_open itself. See Q-adapt-2.
- **Ranker-class performance is a fundamental ceiling for separable
  policies under the current thesis object.** Claim is intuitive and
  consistent with the mean_delta_oracle envelope, but not formally
  established for pathological signal constructions.

### A.4 CONTRADICTED BY EVIDENCE

- **"MC-oracle headroom transfers robustly across backbones."** Empirically
  false at the tested LLaDA-SFT K = 8 bounded setup: headroom is zero at
  B = 4 (identical on every MC sample) and negative at B = 2. Status:
  **refuted at that setup**, open at K ≥ 30 LLaDA-SFT (not measured).
- **"Ranker-class policies close most of the headroom on OWT."** False at
  the ranker class on both OWT (Phase 2b) and LLaDA-SFT (bounded) — only
  the search class (CD-G / BS-AG) closes it on OWT.
- **"Adaptive control is naturally the next step from Phase 2b/3a."** The
  LLaDA-SFT null makes this a *conditional* next step, not a natural one:
  it depends on Δ_open being measurable as positive on some triple, which
  is not established on LLaDA-SFT at tested K.
- **"Large-model continuation is strictly informative."** The bounded
  LLaDA-SFT Phase 2b result does not discriminate between H1 (corrector
  over-dominates at K = 8), H2 (protocol sparseness), and H3 (reference
  mismatch). Adding more LLaDA-SFT data at K = 8 cannot separate these;
  only a K ≥ 30 + richer-reference repeat could — which is out of scope.
- **"Adaptive submodularity is the natural theoretical foil."** Falsified
  by Proposition C: γ > 0 means G(·) is not monotone-submodular over
  site subsets in the relevant regime, so adaptive-submodularity theorems
  (Golovin–Krause 2011) do not apply to this problem. See §5 of
  ADAPTIVE_BUDGETED_CONTROLLERS.md.

---

## Section B — The strategic fork: four candidates compared

The user's audit framing proposes four candidates. Each is evaluated against
**cost**, **marginal evidence yield**, **risk of overclaim**, **compatibility
with file-backed state**, and **time-to-thesis**.

### B.1 Candidate (a) — Stronger large-model empirical confirmation

Run LLaDA-SFT at K ≥ 30 with a richer budget grid and refreshed reference
set, to resolve H1/H2/H3 and check whether MC-oracle headroom appears.

| Axis | Assessment |
|---|---|
| Cost | **High.** Roughly 30× the bounded probe (8 seeds × 24 MC samples → 30 seeds × ~100 MC samples) plus reference regeneration. HPC QoS on `stud` partition does not support this without a multi-week plan. |
| Marginal yield | **Conditional.** If K = 30 LLaDA-SFT shows positive MC-oracle headroom, it validates transfer and opens adaptive control as a credible next step. If it shows null, it strengthens the universal-uniform observation — but the thesis already has this claim from two backbones. |
| Overclaim risk | **Medium.** Headroom at K = 30 would not automatically mean "ranker-class gap" — needs paired CD-G / BS-AG on LLaDA-SFT too. |
| File-backed state | **Weak compatibility.** The bounded probe is already informative enough for an external-validity appendix. Further large-model work is diminishing-returns unless a specific hypothesis (not H1/H2/H3 non-discriminability) motivates it. |
| Time-to-thesis | **Negative.** Would delay writing by weeks. |

**Verdict:** NOT the strongest next move at the thesis stage. Defer to
post-thesis paper if external-validity becomes a bottleneck at review.

### B.2 Candidate (b) — Adaptive-budgeted controller extension + Protocol C

Write Theorem A-ad as a main-body theorem and run Protocol C on existing
LLaDA-SFT Phase 2b JSONs to estimate ε̃, η̃_B, Δ_close/Δ_open.

| Axis | Assessment |
|---|---|
| Cost | **Low for Protocol C** (1-day laptop, no new GPU). **Medium for main-body promotion** (writing + committee justification of conditional theorem with unmeasured constants). |
| Marginal yield | **Conditional.** Protocol C as an appendix experiment yields a bounded empirical statement. Main-body promotion yields a theorem with unmeasured constants — that is a thesis liability, not an asset. |
| Overclaim risk | **High if main-body.** A conditional theorem whose constants are bound by a probe that already shows a null on the same backbone would be legitimately attacked at defense. |
| File-backed state | **Compatible as appendix; incompatible as main theorem.** The LLaDA-SFT null means any adaptive bound on that triple is provably inert on the evidence we have. |
| Time-to-thesis | **Neutral for appendix; negative for main-body.** |

**Verdict:** Appendix-F status is appropriate; main-body promotion is not.
Protocol C is a strong candidate for Phase 4 implementation.

### B.3 Candidate (c) — Sharper hybrid around the existing thesis object

Tighten Theorem A with Refinement A′ (variance η_B) + A″ (rank ε_R), formalize
Proposition B (low-gain-region exclusion) and Proposition C (pairwise γ), and
compute each on the on-disk OWT artefacts. The Negative-Result Corollary
(ranker class) + search-class positive (CD-G / BS-AG) becomes the core story.

| Axis | Assessment |
|---|---|
| Cost | **Low.** All inputs are on disk. Requires a ~2-day analysis + a clean theorem write-up in ch6. |
| Marginal yield | **High.** Turns a bound that is vacuous at B = 8 into a bound that is tight by construction at the OWT triple, and establishes which structural ingredient (γ, σ_ξ, ρ) controls each constant. Also turns the "no signal beats uniform" negative into a formally scoped statement. |
| Overclaim risk | **Low.** Every constant would be measured on disk; no cross-backbone generalization is asserted. |
| File-backed state | **Strongest compatibility.** This is the path that turns what the thesis already has into a defensible theorem-plus-experiment pair. |
| Time-to-thesis | **Positive.** Writing and tightening are the same work. |

**Verdict:** This is the strongest single move. See §4.

### B.4 Candidate (d) — Writing-first

Lock the current story, write ch3 (discrete diffusion), ch5 (informed
correctors), ch6 (theoretical contribution), ch7 (experiments) + Zanella
meeting writeup.

| Axis | Assessment |
|---|---|
| Cost | **Baseline — must happen regardless.** |
| Marginal yield | **High on completion probability.** |
| Overclaim risk | **Depends on whether (c) is done first.** Writing without (c) leaves the vacuous-bound concern in ch6; writing after (c) fixes it. |
| File-backed state | **Compatible by construction.** |
| Time-to-thesis | **Strictly positive.** |

**Verdict:** Required, not optional. Should **run concurrently with (c)**:
(c) produces the tighter theorem text that ch6 consumes; writing ch3 / ch5 /
ch7 can proceed in parallel.

### B.5 Non-exclusivity

(c) and (d) are not mutually exclusive — they form the spine. (b)-as-appendix
is additive if time permits. (a) is out of thesis scope.

---

## Section C — What counts as a genuine MDM contribution in 2025–2026

The 2026-04-24 literature scan (see §3 and
`MDM_THEORY_LANDSCAPE_POSITIONING.md`) confirms:

- **Predictor-side scheduling** is saturated. E-SMC, EAGS, Soft-Mask,
  Entropy-Bounded-Unmasking, Learning-Unmasking-Policies all operate on
  when-to-unmask or how-to-choose-tokens-to-unmask.
- **Reward-guided trajectory search** is saturated. PG-DLM (particle Gibbs),
  APD (adaptive parallel decoding), E-SMC all do full-trajectory search.
- **Corrector kernel design** is partially saturated. Zhao et al. 2024,
  PRISM, ProSeCo kernel, DFM all give kernels.
- **Corrector scheduling as a regret problem — under fixed predictor +
  fixed kernel + fixed budget B — has no proxy-regret theorem in the
  published or pre-print record.** The closest adjacent object
  (PG-DLM's particle-Gibbs trajectory measure) covers a strictly
  different algorithmic class.

A genuine MDM contribution from this thesis therefore lands as:

1. **A proxy-regret theorem (Theorem A) for the ranker class** under
   explicit assumptions, with non-vacuous constants via A′ + A″.
2. **A Negative-Result Corollary (ranker-class scope) anchored on
   measured mean_delta_oracle saturation** on OWT, complemented by a
   search-class positive (CD-G / BS-AG) that exceeds the ranker envelope.
3. **A bounded external-validity probe on a second backbone** (LLaDA-SFT
   K = 8) that reproduces the universal-uniform observation and documents
   the MC-oracle non-transfer honestly.
4. **An Appendix-F conditional extension** (Theorem A-ad with bucketed
   state, F1 FH-CMDP framing) that shows the ranker class is a special
   case of a strictly broader adaptive class — with Protocol C either
   supplying preliminary evidence or closing the minimal-bucketed
   adaptive direction with an honest negative.

No single one of items 1–3 is independently novel enough to defend a thesis,
but **together they form a self-contained positive + negative pair with a
clearly scoped empirical anchor** — which is exactly the empirical-formal
pairing that's missing from the landscape.

The adaptive extension is an **asset iff framed as future work**. Promoting
it to a main-body result would weaken the thesis because its constants are
unmeasured.

---

## Section D — Decisive path

### D.1 Core recommendation

**Execute (c) + (d) as the spine; execute (b) as a bounded appendix; reject (a).**

Sequence:

1. **(c) + (d) in parallel** — tighten Theorem A with A′ + A″, formalize
   Prop B + Prop C, anchor each constant on OWT artefacts (~2 days of
   analysis), and write ch6 + ch7 around the resulting tight theorem.
   Concurrently, write ch3 and ch5 (which do not depend on the tightening).
2. **(b) as Protocol C appendix** — run the 1-day laptop analysis on
   existing LLaDA-SFT Phase 2b JSONs per
   `ADAPTIVE_BUDGETED_CONTROLLERS.md` §4.2. Write Appendix F with the
   conditional theorem and either a preliminary positive (Δ_close/Δ_open
   > 0.5) or an honest negative (≤ 0.5).
3. **Zanella meeting writeup** — incorporate A′ / A″ tightening and
   Protocol C result into the meeting doc.
4. **Reject (a)** — do not run additional LLaDA-SFT at K ≥ 30. Document
   the H1/H2/H3 non-discriminability as a clearly scoped open question.

### D.2 Phase 4 smallest justified step

The smallest justified implementation is **the Refinement A′ + A″ + Prop B +
Prop C estimator**, which computes:

- σ_ξ (per-seed MC residual std) on OWT phase2b artefacts → A′ η_B plug-in;
- Spearman ρ(A, G) on OWT phase2b artefacts → A″ ε_R plug-in;
- γ estimate (largest pairwise interaction across co-selected sites) on OWT
  phase2b artefacts → Prop C η_B upper bound;
- low-gain-share (fraction of mean gain captured by top-k sites) on OWT
  phase2b artefacts → Prop B formal anchor.

Output: `results/phase2b/theorem_a_constants.json` and a short memo
integrating into ch6. This unblocks writing ch6 with measured constants and
is a direct candidate for the thesis's "empirical anchor" table.

Protocol C is a secondary Phase 4 candidate — independent, equally low-cost,
and adds Appendix F. Both should happen; neither blocks the other.

### D.3 Explicit rejections

- **Reject K ≥ 30 LLaDA-SFT continuation.** Insufficient marginal yield
  within thesis scope.
- **Reject main-body promotion of Theorem A-ad.** Constants unmeasured;
  conditional theorem without evidence is a liability.
- **Reject Particle Gibbs / cSMC implementation (F3).** Covered empirically
  by PG-DLM; not a contribution.
- **Reject learned-controller study.** Out of scope, separate paper.
- **Reject richer state beyond (s_t, b_t, phase(t)).** Out of scope.
- **Reject cross-backbone Protocol C replication.** Not authorized.
- **Reject any Phase 3a work on LLaDA-SFT.** Per
  `POST_CROSS_BACKBONE_DECISION.md`; no Δ_open evidence at tested K = 8
  setup.
- **Reject adaptive-submodularity as the theoretical foil.** Falsified by
  Prop C; keep F4 only as a cited foil in the Appendix-F literature review.

---

## §3 — External literature scan (2026-04-24, compressed)

| Paper | Class | Collides with thesis object? |
|---|---|---|
| PG-DLM (arXiv:2507.08390) | Particle Gibbs over reward-guided trajectories | No — different algorithmic class (full-trajectory measure, reward-guided) |
| E-SMC / Optimizing Decoding Paths (arXiv:2512.21336) | Entropy-adaptive SMC for MDM decoding | No — predictor-side, no fixed-budget regret |
| Learning Unmasking Policies (arXiv:2512.09106) | MDP over unmasking, learned predictor policy | No — predictor object, not corrector |
| EAGS (arXiv:2411.06438) | Entropy-based noise scheduling + Gibbs | No — predictor + kernel, no regret theorem |
| APD (arXiv:2506.00413) | Adaptive parallel decoding, mix with AR | No — not about correctors |
| Soft-Masked Diffusion (arXiv:2510.17206) | Training-time mask softening | No — different phase |
| Entropy-Bounded Unmasking (arXiv:2505.24857) | Reasoning, unmasking | No — predictor, not corrector |
| Zhao et al. 2024 | Corrector kernel design | No — kernel level, thesis is scheduling one level up |
| PRISM (arXiv:2510.01384) | Learnable quality score | Complementary — signal input to scheduling |
| ProSeCo (arXiv:2602.11590) | Annealed refinement correctors | Complementary — kernel held fixed by thesis |

**Net:** Corrector *scheduling* as a proxy-regret problem is empirically open
in 2025–2026. The thesis niche is genuine.

---

## §4 — Links

- `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` — canonical direction (must
  stay consistent with this audit).
- `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md` — cross-backbone
  verdict.
- `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md` —
  adaptive decision (consistent with §D.3 above).
- `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` — F1 FH-CMDP
  derivation + Protocol C spec.
- `docs/thesis/theory/THEORY_STATUS.md` — open-loop theorem status.
- `research/candidate_theorems.md` — Theorem A + refinements.
- `research/open_questions.md` — Q1–Q12 + Q-adapt-1..5.
- `research/proof_ledger.md` — provenance tags.
- Companion doc (landscape): `MDM_THEORY_LANDSCAPE_POSITIONING.md`.
- Companion doc (decision): `NEXT_RESEARCH_DIRECTION_DECISION.md`.

---

*End of independent audit. The decisive path is (c) + (d) spine, (b)
appendix, (a) rejected. Phase 4 implements the Theorem A-constant estimator
on OWT artefacts.*
