> **STATUS:** SUPPORTING
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Direction audit justifying the Phase 3 pivot (PRISM vs alternatives). Supports CANONICAL_RESEARCH_DIRECTION.md.

---

# Phase 3 Direction Audit — PRISM vs Alternatives

*Created: 2026-04-20. Author: research advisor pass on the post–Phase-2b pivot.*
*Verdict: **REJECT PRISM as the next move.** Pivot to combinatorial-baseline + theory-finalization track. Justification below.*

> **Post-Phase-3a addendum (2026-04-20).** Phase 3a (job 479941, K=30 paired,
> CD-G + BS-AG over schedules) closed the Tier-1 Alt-A workstream with a
> **search-class positive**: both methods PASS at every B ∈ {2, 3, 4, 8};
> CD-G recovers 74–84 %, BS-AG 49–64 % of the +0.45 MC-oracle headroom at
> B ∈ {2, 3, 4}; both still PASS at B = 8 where `mean_delta_oracle` itself is
> NULL. **The PRISM rejection still stands**, with a refined reason: PRISM is
> a learned per-token quality signal, i.e. a member of the ranker class
> bounded by the rescoped Negative-Result Corollary; the recoverable structure
> demonstrated by Phase 3a does **not** factor through any separable per-step
> score, so a learned per-token ranker cannot reach it by construction. What
> changes from this audit's original framing: the broad claim that "the gain
> is irreducible at single-step granularity" was incomplete — it is irreducible
> for *single-step rankers*, but reachable by *search procedures over
> schedules*. Sections A1–A4 below remain valid as the empirical case for
> rejecting PRISM; the only sentence that needs softening is the
> decision-rule line in A3 Tier 1 ("if neither does, the negative result
> strengthens — 'the gain is irreducible at single-step granularity, even with
> search'") — Phase 3a's CD-G and BS-AG are the search procedures referenced,
> and they did recover most of the gap. See
> `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`.

---

## A1. Is PRISM actually the right next step? — **No.**

### A1.1 The headline argument for PRISM (and why it fails)

The post–Phase-2b synthesis ran:
> Hand-crafted signals fail at medium B; the +0.36 mean-profile-to-instance gap is "instance-specific"; therefore a *learned* per-instance signal (PRISM) should close it.

This is a non-sequitur. "Instance-specific" ≠ "single-token-quality predictable". To pivot to PRISM you have to believe the missing structure is recoverable by a per-token quality predictor. The Phase 2b data do **not** support that belief. They actively contradict it.

### A1.2 Three smoking guns from Phase 2b that kill the PRISM hypothesis

**Smoking gun #1 — even the ground-truth signal saturates.**
`mean_delta_oracle` (places correctors at the *true* top-Δ_t steps for each instance, computed by replaying the trajectory) only beats uniform at B ∈ {2, 3, 4}. At B = 8 and B = 16 it falls inside the NULL band. PRISM optimises for token quality; even a perfect per-step Δ-predictor would, by construction, reproduce `mean_delta_oracle`'s ceiling at best — i.e. it would still fail at B ≥ 8.

| B | mean_delta_oracle Δ̂ vs uniform | CI | verdict |
|---|---:|---|---|
| 2 | +0.130 | [+0.043, +0.209] | PASS |
| 3 | +0.103 | [+0.028, +0.182] | PASS |
| 4 | +0.092 | [+0.039, +0.151] | PASS |
| 8 | +0.084 | [−0.020, +0.185] | borderline (NULL) |
| 16 | +0.032 | [−0.075, +0.133] | borderline (NULL) |

**Smoking gun #2 — top MC schedules do not pick the high-Δ steps.**
The MC oracle (best-of-100 random schedules per seed) achieves +0.45 over uniform at B = 4. But the top-10 MC schedules' step sets only overlap `mean_delta_oracle`'s "top-Δ steps" picks at **~1.5× random baseline**:

| B | top-10 MC ∩ oracle Jaccard | random baseline | ratio |
|---|---:|---:|---:|
| 2 | 0.027 | 0.016 | 1.7× |
| 3 | 0.032 | 0.024 | 1.3× |
| 4 | 0.047 | 0.032 | 1.5× |

Conclusion: the +0.45 headroom is **not** sitting on the high-Δ steps. A learned ranker that predicts Δ_t (or any Δ-correlated quality signal — which is exactly what PRISM does at the token level) cannot pick up that headroom.

**Smoking gun #3 — top MC schedules do not share structure with each other.**
If the +0.45 headroom were "learnable" you would expect the top schedules per instance to be *consistent* (e.g. they all pick the same 3–4 critical steps). They are not:

| B | top-10 internal Jaccard | bottom-10 internal Jaccard | random baseline |
|---|---:|---:|---:|
| 2 | 0.054 | 0.066 | 0.016 |
| 3 | 0.053 | 0.049 | 0.024 |
| 4 | 0.049 | 0.057 | 0.032 |

The "best" 10 schedules per instance are no more internally consistent than the "worst" 10. That is the empirical signature of an **unstructured combinatorial gain**, not of a missing signal. A learned ranker — PRISM, a critic head, or anything else that produces a per-step score — cannot resolve this by design.

### A1.3 Variance decomposition closes the case

At B = 4, total variance of G across random schedules decomposes as:

| component | var | share |
|---|---:|---:|
| within-seed (schedule choice) | 0.0299 | 62 % |
| between-seed (instance) | 0.0193 | 38 % |

So scheduling *does* matter (62 % of the variance is in the schedule choice, not in which instance you happened to draw). But that variance is **not** captured by any of the per-step rankings tested, including the cheating ground-truth one. The bottleneck is combinatorial, not informational.

### A1.4 Alternative explanations for the failures (and which are real)

| Candidate explanation | Plausible? | Phase 2b evidence |
|---|---|---|
| Metric noise (F = neg_nll too coarse) | partly | σ_F ≈ 0.19–0.26 across B; Phase 2c could test MAUVE-F; but cell ordering is unlikely to flip given Cohen's d up to ±2 in many cells |
| Schedule-evaluation methodology | weak | paired K=30 BCa is the right protocol; ANALYSIS_SPEC §3 |
| Weak theorem-to-experiment alignment | strong | Refinement A″ (ρ-decay 0.66→0.39) is the empirical fingerprint of additivity breaking down; this is theory-revealing, not theory-failing |
| Wrong action-set abstraction (single-step ranking saturates) | **strongest** | smoking guns #1, #2, #3 |
| Insufficiently strong baseline policies | partial | we did not test combinatorial search baselines (beam, coordinate descent, MCTS); this is the cheap test we should run |
| Missing learned signal (PRISM premise) | **weakest** | mean_delta_oracle uses ground truth and still fails at B=8 |

The dominant explanation is "single-step greedy schemes have a ceiling". The complementary cheap experiments are combinatorial search and variance-form theory. **PRISM does not address the dominant explanation.**

---

## A2. Assumptions of pivoting to PRISM (and which are wrong)

| # | Assumption embedded in a PRISM pivot | Status | Note |
|---|---|---|---|
| 1 | Instance-specific headroom is recoverable by a *learned token-quality* signal | **false** | mean_delta_oracle is the upper bound for any single-step learned ranker, and it saturates by B=8 |
| 2 | PRISM's signal is aligned with *schedule-level* gain | **false** | PRISM is trained for token-level remasking quality, not corrector placement; the additivity gap (Theorem A) is exactly the misalignment |
| 3 | A PRISM checkpoint integrates cleanly with our scheduling protocol | unknown→risky | PRISM operates on MDLM backbone; we are using ProSeCo-OWT; integration is non-trivial, multi-week scope |
| 4 | PRISM is the highest-EV next experiment | **false** | combinatorial-baseline experiments are 1-2 days and directly test the dominant hypothesis; PRISM is 2-4 weeks with high failure risk |
| 5 | The thesis should pivot from "aggregate signals" to "learned signals" | **false** | the original scope (CLAUDE.md) is *aggregate trajectory signals*; pivoting away after a clean negative is scope creep, not science |
| 6 | We should keep adding new model branches until something wins | **false** | "the gain is unstructured" is itself a thesis-grade contribution under the original scope |

---

## A3. Strongest alternatives, ranked

### Tier 1 — execute immediately (this week)

**Alt-A. Combinatorial scheduling baselines** *(estimate: 1-2 days)*
Test whether any *non-greedy* search procedure recovers the MC-oracle headroom. Two candidates:
1. **Coordinate descent over schedules**: start from uniform, swap one corrector position at a time, accept if paired-G improves. K seeds × T positions × B per swap ≈ a few hours.
2. **Beam search with rollouts**: maintain a beam of B-step partial schedules, rollout-evaluate each, expand top-k.
**Decision-relevance**: if either matches MC oracle, the thesis story becomes "scheduling needs search, not signals" — concrete and publishable. If neither does, the negative result strengthens — "the gain is irreducible at single-step granularity, even with search".

**Alt-B. Theory finalisation** *(estimate: 2-3 days)*
Refinements A′ (variance-form additivity slack) and A″ (rank-based ε_R) are *empirically validated* by Phase 2b. Tighten and prove them. Add a Negative-Result Corollary: "when top-K MC schedules show internal Jaccard ~ random, no greedy ranking policy can beat uniform by more than 2σ_F". This is publishable theory regardless of Phase 3a outcome.

These two run **in parallel**. Combined they are a 4-7 day Phase 3.

### Tier 2 — only if Tier 1 yields a positive

**Alt-C. External validity micro-experiment** *(1 day)*
Replicate `middle@B=2` (cleanest PASS cell) on MDLM backbone. If it holds, "small-B win is backbone-robust". If not, "ProSeCo-OWT-specific quirk". Cheap external-validity check.

**Alt-D. Phase 2c MAUVE F-swap on borderline cells** *(0.5 day)*
Run the 4 borderline cells (`middle@{3,4}`, `entropy_bot@{3,4}`) under MAUVE-F. Could promote NULL→PASS and extend "small-B win" claim from B=2 only to B≤4. Low risk.

### Tier 3 — actively rejected

**PRISM fine-tuning.** Rejected on:
- mismatch with the dominant explanation (combinatorial, not signal),
- mean_delta_oracle ceiling argument,
- top-MC-schedule scatter argument,
- scope creep relative to thesis statement,
- 2-4 week cost on a thesis-relevant timeline,
- integration risk with ProSeCo backbone.

**Backbone porting (MDLM full Phase 2b).** Rejected unless Tier 1 surfaces a positive result that needs external validation. Without a positive to validate, this is replication-of-negative — low scientific value, high HPC cost.

---

## A4. Final decision

**Decision: REJECT PRISM. Execute Tier-1 alternatives (Alt-A combinatorial baselines + Alt-B theory finalisation) in parallel, starting now.**

**Reasoning in one paragraph.** Phase 2b produces three independent lines of evidence (mean_delta_oracle ceiling, top-MC-vs-Δ_t Jaccard ≈ random, top-MC internal Jaccard ≈ bottom-MC) that all point to the same conclusion: the structural bottleneck is the *single-step greedy abstraction*, not the *signal quality*. PRISM addresses the latter and would not move the former. The cheap, decision-relevant experiments are a combinatorial search baseline (does *any* non-greedy scheme close the gap?) and a theory finalisation (formalise the rank-based calibration result and the negative-result corollary). Both fit within a week and produce thesis-grade output regardless of which way they land.

**Phase 3 question now being tested:**
> "Is the +0.45 oracle-vs-uniform headroom recoverable by *any* practical scheduling procedure that does not require ground-truth labels — or is it irreducible at the single-step granularity, making aggregate-signal corrector scheduling structurally bounded by uniform + 2σ_F?"

This is a *real* falsifiable question with cheap experiments and direct thesis impact. The PRISM question — "does a learned token-quality signal help?" — is neither cheap nor decision-critical given the data we already have.

---

## Provenance

- Phase 2b raw: `results/phase2b_proseco_owt/{policy_raw,mc_raw}.json`
- Phase 2b paired analysis: `results/phase2b/policy_comparison_paired.json`
- MC oracle bound: `results/phase2b/mc_oracle.json`
- Smoking-gun calculations: ad-hoc Python in this audit (rerun in `scripts/phase2b_smoking_guns.py` if needed for the writeup)
- Alternative plan: `docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md`
- Theory ledger: `docs/thesis/theory/THEORY_STATUS.md`
- Canonical research direction: `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
