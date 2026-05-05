> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


> **STATUS:** DECISION (terminal node of the 2026-04-24 strategic audit)
> **LAST VERIFIED:** 2026-04-24
> **SCOPE:** Decisive recommendation on what to do next, with fallback +
> explicit rejections. Reads as the actionable companion to
> `NEXT_RESEARCH_DIRECTION_AUDIT.md` (evidence classification) and
> `MDM_THEORY_LANDSCAPE_POSITIONING.md` (literature positioning).
>
> This document closes the strategic-fork question opened on 2026-04-24 and
> binds the next 3–4 weeks of thesis work. Supersedes any conflict with
> earlier plan docs.

---

# Next Research Direction — Decision

## TL;DR

**Execute Candidate (c) + (d) as the spine. Add Candidate (b) as a bounded
appendix. Reject Candidate (a).**

- **(c) Sharper hybrid** — tighten Theorem A with Refinement A′ (variance
  η_B), Refinement A″ (rank ε_R), formalize Proposition B (low-gain-region
  exclusion), Proposition C (pairwise γ). Compute every constant on OWT
  Phase 2b artefacts.
- **(d) Writing-first** — ch3 + ch5 + ch6 + ch7 + Zanella-meeting writeup,
  concurrent with (c).
- **(b) Protocol C appendix** — 1-day laptop analysis of existing LLaDA-SFT
  Phase 2b JSONs per `ADAPTIVE_BUDGETED_CONTROLLERS.md` §4.2. Appendix F
  only; no main-body promotion of Theorem A-ad.
- **(a) Reject K ≥ 30 LLaDA-SFT continuation.**

**Phase 4 — smallest justified implementation:**
`scripts/compute_theorem_a_constants.py` + `src/mdm_playground/analysis/
theorem_a_constants.py` computing σ_ξ, ρ, γ, low-gain-share from on-disk
OWT Phase 2b artefacts → `results/phase2b/theorem_a_constants.json` +
short memo for ch6.

---

## 1. Why (c) is the primary move

### 1.1 It is the only move that turns a file-backed bound from vacuous to
tight

At B = 8 on ProSeCo-OWT Phase 1, the plug-in Theorem A bound 2Bε + 2η_B ≈
3.50, while plausible G ≤ 1.2. Bound is vacuous at the regime where the
ranker-class negative result lives.

Refinement A′ reduces η_B from worst-case additivity slack to σ_ξ · √B / √2,
tighter by a factor of √B. Refinement A″ replaces the worst-case L∞
calibration slack with (1 − |ρ|)·σ_Δ, tighter when the proxy preserves rank
but not scale. With both plugged in and σ_ξ / ρ measured on the 8-seed MC
rollouts already on disk, the expected plug-in bound at B = 8 drops below
the plausible G, making the bound non-vacuous at the regime where its
conclusion matters.

This is the **only** candidate direction that directly fixes the main
technical weakness of the current thesis.

### 1.2 It is the only move that gives ch6 measured constants to display

ch6 (Theoretical Contribution) currently shows the bound as symbolic. The
defense-ready version of ch6 needs a table of measured constants — σ_ξ, ρ,
γ, ε, η_B at each B — with BCa bootstrap CIs. (c) delivers exactly that
table.

### 1.3 It is zero-GPU, low-risk, high-yield

All inputs are on disk. The computation is a single sweep over
`results/phase2b/*.json`. Estimated cost: ~2 days of analysis + writing.

### 1.4 It composes cleanly with writing (d)

The output of (c) feeds ch6 directly. ch3 and ch5 are written in parallel
without waiting for (c) because they don't reference Theorem A's constants.
ch7 (Experiments) consumes (c) + the LLaDA-SFT bounded probe.

---

## 2. Why (b) goes to Appendix F, not the main body

### 2.1 The LLaDA-SFT null restricts what Theorem A-ad can claim

At the K = 8 bounded setup, MC-oracle does not show positive headroom over
uniform (paired diff −2.64 at B = 2, exactly 0.000 at B = 4). This means
**on the only triple where we have cross-backbone data, Δ_open is not
measurable as positive.** Any adaptive bound on that triple is therefore
provably inert on the evidence we have.

Theorem A-ad as a main-body claim would require a measurement of Δ_open > 0
on the evidence triple — which does not exist. As an appendix-F conditional
theorem with Protocol C as bounded probe on the OWT triple (where Δ_open is
measurable via the +0.45 MC-oracle headroom), it is defensible and
appropriate.

### 2.2 Protocol C is the smallest justified adaptive experiment

Per `ADAPTIVE_BUDGETED_CONTROLLERS.md` §4.2:
- Re-use OWT Phase 2b per-seed JSON outputs (already on disk — this is a
  correction from POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md which specified
  LLaDA-SFT as the Protocol C base. On LLaDA-SFT, Δ_open ≈ 0 at tested B
  means Protocol C cannot recover anything even in principle. On OWT,
  Δ_open ≈ +0.45 at B = 2..4 and there is signal to recover).
- Bucket by z_t = (s_t, b_t, phase(t)); signals s_t ∈ {H_t, M_t^{-1}, Q_t};
  phase(t) ∈ {early, mid, late}.
- Estimate ε̃, η̃_B, Δ_close(π̂_λ, N = 1) / Δ_open on OWT.
- Decision rule:
  - Δ_close / Δ_open > 0.5 → Appendix F with Theorem A-ad F1 as formal
    conditional theorem + Protocol C as preliminary positive evidence.
  - Δ_close / Δ_open ≤ 0.5 → Appendix F with Theorem A-ad F1 as sketched
    conditional theorem + honest Protocol C negative; clearly scoped as
    "minimal bucketed state is insufficient on OWT; richer state is an
    open question".

### 2.3 Correction to POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md

That doc specified Protocol C on **LLaDA-SFT Phase 2b artefacts**. Given the
LLaDA-SFT K = 8 null, Protocol C on LLaDA-SFT cannot yield useful evidence
about Δ_close / Δ_open — the denominator is ~0. **This decision doc
overrides the earlier recommendation: Protocol C runs on OWT Phase 2b
artefacts instead.** The LLaDA-SFT probe stays as an external-validity
appendix (not Protocol C).

---

## 3. Why (a) is rejected

### 3.1 K ≥ 30 LLaDA-SFT is out-of-budget for marginal yield

The bounded K = 8 probe already supports the universal-uniform observation
at T3 tier. Moving to K ≥ 30 would cost ~30× the bounded probe plus
reference regeneration (GPT-2 at larger scale, or a switch to a more
appropriate reference). HPC QoS on `stud` partition cannot absorb this
without a multi-week plan. The marginal thesis-yield of discriminating
H1/H2/H3 at K ≥ 30 is not worth weeks of writing delay.

### 3.2 The three non-discriminable hypotheses cannot be resolved by adding
K alone

H1 (corrector over-dominates at K = 8), H2 (protocol sparseness), H3
(reference mismatch) are not separable by increasing K only — they require
richer reference sets or a different corrector kernel stress-test. Neither
is within thesis scope.

### 3.3 The thesis already has sufficient external-validity material

Two backbones, the OWT ranker-class negative + Phase 3a search-class
positive, and the LLaDA-SFT bounded probe — is adequate material for a
defense. Candidate (a) would deliver more of the same evidence type, not
new evidence type.

---

## 4. The fallback

If (c) Phase 4 (Theorem A-constant estimator) returns **measured constants
that do NOT tighten the bound below plausible G** — e.g., if σ_ξ is larger
than expected or ρ is small — the fallback is:

1. **Report the measured constants as-is in ch6.** Do not manipulate the
   bound into appearing tight.
2. **State the bound's regime of applicability honestly** — e.g., "non-
   vacuous for B ≤ 4; vacuous at B = 8 on ProSeCo-OWT at measured ε, η_B".
3. **Preserve the Negative-Result Corollary as the core empirical claim.**
   The corollary does not depend on the bound being tight — it depends on
   the mean_delta_oracle envelope measured on disk.
4. **Write Appendix F with a weaker Protocol C.** Even if the main-body
   bound is vacuous at high B, Protocol C on OWT can still yield a
   Δ_close / Δ_open ratio; that ratio is a thesis-level artefact regardless
   of the main-body bound's tightness.
5. **Zanella meeting framing:** "Here is where the bound is tight and here
   is where it is not. Here is the structural diagnostic (Prop C γ) that
   tells us which regime we are in. This is more honest than a sharper
   bound with unmeasured constants."

This fallback is explicit — if Phase 4 reveals that A′ + A″ don't produce a
tight bound, the thesis still holds together through the Negative-Result
Corollary + search-class positive + external-validity probe + honest
structural diagnostic.

---

## 5. Explicit rejections (binding)

The following are rejected for the thesis scope:

| Option | Reason for rejection |
|---|---|
| K ≥ 30 LLaDA-SFT continuation | Weeks of delay; non-discriminable H1/H2/H3 |
| Main-body promotion of Theorem A-ad | Constants unmeasured; LLaDA-SFT Δ_open = 0 at tested K |
| Particle Gibbs / cSMC implementation (F3) | Covered by PG-DLM (arXiv:2507.08390); not novel |
| Learned-controller study | Out of scope, separate paper |
| Richer state beyond (s_t, b_t, phase(t)) | Out of scope |
| Cross-backbone Protocol C replication | Not authorized |
| Phase 3a on LLaDA-SFT | No Δ_open signal at tested K = 8 |
| Adaptive submodularity (F4) as theoretical foil | Falsified by Prop C |
| Any new HPC submission before ch3/ch5/ch6 drafts | Time-to-thesis risk |

---

## 6. Phase 4 — smallest justified implementation

### 6.1 Target artefacts

Compute the four Theorem-A constants on OWT Phase 2b on-disk data:

| Constant | Source | Use |
|---|---|---|
| σ_ξ (per-seed MC residual std) | `results/phase2b/mc_oracle.json` residuals | Refinement A′ plug-in for η_B |
| ρ (Spearman rank corr ψ, G) | `results/phase2b/policy_comparison_paired.json` + mc_oracle.json | Refinement A″ plug-in for ε_R |
| γ (largest pairwise interaction) | `results/phase2b/combinatorial_diagnostics.json` | Proposition C bound on η_B |
| low-gain-share (top-k share) | `results/phase2b/combinatorial_diagnostics.json` | Proposition B formal anchor |

### 6.2 Deliverables

1. **`src/mdm_playground/analysis/theorem_a_constants.py`** — pure-function
   module computing the four constants from on-disk JSONs, with BCa
   bootstrap CIs.
2. **`scripts/compute_theorem_a_constants.py`** — CLI entrypoint emitting
   `results/phase2b/theorem_a_constants.json`.
3. **`tests/test_theorem_a_constants.py`** — unit tests for each constant
   estimator, fixture-backed.
4. **`docs/thesis/theory/THEOREM_A_CONSTANTS.md`** — one-page memo with
   the measured table and per-B plug-in bound.

### 6.3 Stop conditions

- **If all four constants compute cleanly and the plug-in bound is
  non-vacuous at B ∈ {2, 3, 4}**, the memo goes into ch6 directly.
- **If any constant is ill-defined** (e.g., ρ requires more MC samples than
  on disk), the memo documents what is measurable and what requires a
  bounded additional MC run (not a new experiment — a rerun on existing
  schedules with more samples).

### 6.4 Non-goals of Phase 4

Phase 4 does **not**:
- touch Protocol C (that is Phase 5);
- touch ch3, ch5, or ch7 writing (those proceed in parallel);
- touch any HPC job;
- touch the LLaDA-SFT artefacts.

Phase 4 is strictly the OWT-constant estimator + memo.

### 6.5 Phase 5 (sequenced after Phase 4)

Protocol C on OWT artefacts, per §2.2 above. Output: Appendix F draft.

---

## 7. Timeline

| Item | Duration | Dependency |
|---|---|---|
| Phase 4 — Theorem A constants on OWT | 2 days | None |
| ch3 writing | 5 days | None (parallel) |
| ch5 writing | 3 days | None (parallel) |
| ch6 writing (Theorem A + refinements) | 4 days | Phase 4 |
| ch7 writing (experiments + LLaDA probe) | 3 days | Phase 4 |
| Zanella meeting writeup update | 1 day | Phase 4, ch6 |
| Phase 5 — Protocol C on OWT | 1 day | Phase 4 |
| Appendix F draft | 2 days | Phase 5 |

Total ~3 weeks at thesis pace. Nothing in this plan requires new GPU time.

---

## 8. Links

- Audit: `NEXT_RESEARCH_DIRECTION_AUDIT.md`
- Landscape: `../theory/MDM_THEORY_LANDSCAPE_POSITIONING.md`
- Prior adaptive rec (partially superseded by §2.3): `POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`
- Cross-backbone decision: `POST_CROSS_BACKBONE_DECISION.md`
- F1 CMDP theory: `../theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`
- Theorem A text: `../../../research/candidate_theorems.md`
- Open questions: `../../../research/open_questions.md`
- Provenance: `../../../research/proof_ledger.md`

---

*End of decision. Phase 4 is the next executable step.*
