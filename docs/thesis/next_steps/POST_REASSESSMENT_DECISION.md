> **STATUS:** DECISION (closes the 2026-04-26 reassessment cycle)
> **LAST VERIFIED:** 2026-04-26
> **SCOPE:** Final go/no-go on (i) new HPC work, (ii) the adaptive-controller
> direction, and (iii) the remaining critical-path tasks. Reads downstream of
> `NEXT_STEP_REASSESSMENT.md`.

---

# Post-Reassessment Decision

## TL;DR

**No new HPC work is justified.** Every candidate considered (cross-backbone
extension, B = 8 MC oracle on OWT, pairwise γ at B ≥ 8, third-backbone
replication, LLaDA-SFT Phase 3a reopening) fails at least one of: redundant
with existing data, out of thesis scope, or expected gain not load-bearing.

**The adaptive-controller direction is closed at the appendix-active
level.** Theorem A-ad lives in Appendix F as a formally proved conditional
theorem with documented inert-bound diagnostic; Protocol C closed with an
honest negative on bucketed (s_t, phase(t)) state. No re-run is authorised.
The verdict survived a skeptical re-reading: three independent corroborating
lines (eps_ratio, after-uncertainty close ratio, Hamming-2B disjointness)
all support the closure, and the Phase 2b paired data on
`entropy_bot_B_pt` is the empirical TRUE-G version of Protocol C's
threshold policy — its decay from +0.105 (B=2) to −0.051 (B=8) confirms
the additive-surrogate result at TRUE G under the σ_ξ band.

**The thesis critical path is now Phase 3b theory finalisation +
LaTeX writing.** Phase 3b is **complete as of 2026-04-26**: formal proofs
of Refinements A′ and A″ + the Negative-Result Corollary are in
`research/candidate_theorems.md`; an Entry 8 records the proof structure
in `research/proof_worklog.md`; and ch6 LaTeX skeleton is drafted with
the formal theorem statements locked. Remaining: ch3 / ch4 / ch5 / ch7
LaTeX writeup; abstract; introduction; conclusion.

---

## 1. The HPC question, answered

| Candidate | Verdict | Reason |
|---|---|---|
| **No new HPC** | **GO (default)** | Best alternative |
| Bounded adaptive HPC (TRUE-G threshold-λ at K = 30) | NO-GO | Phase 2b's `entropy_bot_B_pt` at K = 30 is empirically the same family; +0.105 / +0.032 / −0.051 at B = 2 / 4 / 8 confirms Protocol C's additive-surrogate result at TRUE G |
| B = 8 MC oracle on OWT | NO-GO | ~30–60 A100-h; qualitative finding "search beats ranker at B = 8" already established without the closure ratio |
| Pairwise γ at B ≥ 8 on OWT | NO-GO | Refinement A′ replaced Proposition C; γ at B = 8 / 16 not on critical path |
| Third-backbone replication | NO-GO | Out of thesis scope (`POST_CROSS_BACKBONE_DECISION.md`); prior MDLM and ReMDM-conf attempts showed structural failure modes |
| LLaDA-SFT Phase 3a reopening | NO-GO | Pre-registered reopening precondition (`mc_oracle_minus_uniform.bootstrap_95_ci_lo > 0.05`) not satisfied |

**Decision: NO-GO on HPC. Default holds.**

## 2. The adaptive-controller question, answered

### 2.1 Is the current adaptive-controller study genuinely closed?

**Yes.** The skeptical re-reading (`NEXT_STEP_REASSESSMENT.md` §B.1)
identifies three independent corroborating lines for the
`honest_negative` verdict:

1. **eps_ratio leg.** ε̃ / ε ∈ [0.983, 0.986] across all three signals
   on OWT bucketed (s_t, phase(t)) state. Bucketing shrinks calibration
   RMS by < 1.7 %.
2. **after-uncertainty close-ratio leg.** Δ_close_A / Δ_open after
   subtracting σ_ξ · √B / √2 / Δ_open is ≤ 0.015 at the best (signal,
   B) and negative at every B ≥ 3.
3. **Hamming-2B disjointness.** Threshold schedule and best MC
   schedule have mean Hamming distance ≈ 2 B (max possible) on OWT;
   0 / 30 exact matches.

The third line is structural and does not depend on the bucket
abstraction. A skeptic cannot explain it away with "richer state
would help."

### 2.2 Phase-2b cross-check

The bucketed-state threshold policy is empirically near-equivalent to
Phase 2b's `entropy_bot_B_pt`, which Phase 2b ran at K = 30 paired
with TRUE G:

| B | uniform G | entropy_bot_B_pt G | paired diff | Verdict |
|---|---|---|---|---|
| 2 | 0.113 | 0.218 | **+0.105** | small positive |
| 4 | 0.282 | 0.314 | +0.032 | borderline |
| 8 | 0.476 | 0.425 | **−0.051** | small negative |

The Protocol C additive-surrogate prediction (Δ_close_A = +0.18 at
B = 2 minus σ_ξ band 0.17 → predicted true diff in [+0.01, +0.35],
matching observed +0.105) confirms the theory. **No new HPC is
needed to verify Protocol C's verdict at TRUE G.**

### 2.3 Verdict on adaptive controllers

**Closed at appendix-active level.** Theorem A-ad is a formally proved
conditional theorem; Protocol C is closed with an honest negative on
bucketed state. Re-opening precondition: a future measurement showing
ε̃ / ε ≤ 0.7 on a richer state abstraction or another backbone.
Absent that, the direction stays out of the critical path.

## 3. The next-step boundary

### 3.1 Main thesis (critical path)

| Task | Status | ETA |
|---|---|---|
| Phase 3b — formal proofs of Refinement A′, Refinement A″, Negative-Result Corollary | **DONE 2026-04-26** | — |
| ch6 — Theoretical Contribution LaTeX skeleton | **DRAFT 2026-04-26** (theorem statements locked; section bodies TODO) | 1–2 days prose |
| ch3 — Discrete Diffusion LaTeX | TODO (placeholder only) | 2–3 days |
| ch4 — Masked Diffusion LaTeX | TODO (placeholder only) | 2–3 days |
| ch5 — Informed Correctors LaTeX | TODO (placeholder only) | 1–2 days |
| ch7 — Experiments LaTeX | TODO (placeholder only) | 2–3 days |
| ch1 — Introduction | TODO | 1 day |
| ch8 — Conclusion | TODO | 1 day |
| Abstract | TODO | 1 day |
| Final pass + bibliography + appendices | TODO | 2–3 days |

Total remaining writing budget: ~ 12–18 working days.

### 3.2 Optional / Appendix / Future Work (NOT critical path)

- **Appendix F (Adaptive Extensions):** formalise Theorem A-ad,
  cite Protocol C's honest-negative pilot. ~ 4 pages. Optional —
  may be condensed to a Future-Work § in ch8 if pages are tight.
- **H1 / H2 / H3 disambiguation on LLaDA-SFT:** future-paper material;
  not in thesis scope.
- **Third-backbone replication:** future-paper material.
- **Richer state abstractions for adaptive control:** future-paper
  material; requires function approximator and is out of scope.
- **Stretch C2 (E_fact contraction):** stretch appendix only if
  Ascolani / Denoising Entropy Bounds yield an applicable framework;
  not in critical path.

### 3.3 What should explicitly NOT be done

- No new HPC runs of any kind.
- No re-run of Protocol C on a different state abstraction.
- No re-run of Protocol C on a different backbone.
- No reopening of LLaDA-SFT Phase 3a.
- No promotion of Theorem A-ad to main body.
- No expansion of the thesis object beyond {fixed predictor, fixed
  informed corrector kernel, fixed extra corrector NFE budget,
  trajectory-level allocation}.
- No third-backbone study within the thesis.
- No function-approximator-based adaptive policies within the thesis.
- No claim that uniform is generically optimal across masked diffusion.
- No claim that MC-oracle headroom is backbone-generic.

## 4. What was done in this reassessment cycle

### 4.1 Documents produced

- `docs/thesis/next_steps/NEXT_STEP_REASSESSMENT.md` — independent
  reassessment with classified evidence, adaptive re-skepticism, and
  HPC-candidate analysis.
- `docs/thesis/next_steps/POST_REASSESSMENT_DECISION.md` — this file.

### 4.2 Theory promotions (Phase 3b)

- `research/candidate_theorems.md` — Refinements A′ and A″ promoted
  from candidate / heuristic to **formally proved under explicit
  assumptions**; Negative-Result Corollary promoted from
  empirically-anchored to **formal corollary** with proof.
- `research/proof_worklog.md` — Entry 8 records the Phase 3b
  promotion structure and empirical anchoring.
- `docs/thesis/theory/THEORY_STATUS.md` — Honesty Ledger rows for
  Refinements A′ / A″ and Negative-Result Corollary updated to
  "formally proved 2026-04-26".

### 4.3 Thesis LaTeX

- `thesis/chapters/ch6_contribution.tex` — replaced placeholder with
  full skeleton: assumptions, Theorem A, Lemmas A1 / A2,
  Theorem A′, Theorem A″, Refined Theorem A, Negative-Result
  Corollary as definitions / theorem environments. Section bodies
  marked TODO; theorem statements are publication-ready.

### 4.4 What was not produced (intentionally)

- No new code (the existing Protocol C / Phase 2b / Phase 3a
  modules are sufficient).
- No new tests beyond the existing Protocol C tests (24 PASS).
- No new HPC sbatch (no new runs authorised).
- No new GPU work.

## 5. Honesty ledger

| Claim | Tag |
|---|---|
| No new HPC experiment is currently justified | `[Decision]` |
| The adaptive-controller direction is closed at appendix-active | `[Decision; supported by three independent corroborating lines]` |
| Phase 3b theory finalisation is complete (formal proofs of A′ and A″ + Negative-Result Corollary in `candidate_theorems.md`) | `[Done 2026-04-26]` |
| ch6 LaTeX skeleton is publication-quality for theorem statements; section bodies are TODO | `[Done 2026-04-26]` |
| Phase 2b's `entropy_bot_B_pt` empirically confirms Protocol C's additive-surrogate prediction at TRUE G | `[Empirically anchored — `policy_comparison_paired.json` paired diffs +0.105 / +0.032 / −0.051 at B = 2 / 4 / 8]` |
| Refinement A″ formal version (with half-normal moment) supersedes the earlier heuristic version (with $(1 - |ρ|)\sigma_Δ$) | `[Resolved 2026-04-26 — see `candidate_theorems.md` §"Refinement A″ — Rank-Based Calibration ε_R (superseded)"]` |
| The Negative-Result Corollary as written places ranker-class headroom in the NULL band by B = 8 on OWT | `[Empirically anchored on `policy_comparison_paired.json` mean_delta_oracle paired diff + Refined Theorem A bound]` |

## 6. Links

- Reassessment: `docs/thesis/next_steps/NEXT_STEP_REASSESSMENT.md`
- Activation audit: `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md`
- Adaptive experiment plan: `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md`
- Adaptive post-decision: `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_DECISION.md`
- Cross-backbone post-decision: `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md`
- Theory: `research/candidate_theorems.md`,
  `docs/thesis/theory/THEORY_STATUS.md`,
  `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`
- Proof worklog: `research/proof_worklog.md` (Entry 8)
- LaTeX: `thesis/chapters/ch6_contribution.tex`
- Phase 2b empirical anchor: `results/phase2b/policy_comparison_paired.json`,
  `results/phase2b/theorem_a_constants.json`,
  `results/phase3a_proseco_owt/oracle_gap_closure.json`

---

*End of post-reassessment decision. NO-GO on HPC. Phase 3b theory
complete. Thesis critical path is LaTeX writeup of ch3 / ch4 / ch5 /
ch7 / ch1 / ch8 + abstract + final pass.*
