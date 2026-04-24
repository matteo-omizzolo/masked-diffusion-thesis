# Assessment of memos and external analysis (skeptical audit)

## Scope and method

This audit treats all claims as hypotheses. Claims are classified using:

- **supported**
- **plausible but unproven**
- **speculative**
- **contradicted**
- **not yet assessable**

Primary evidence used: canonical thesis docs, mainline scripts, and JSON artifacts in `results/phase2b*` and `results/phase3a*`.

---

## Repo-grounded baseline (before memo assessment)

Evidence directly confirmed from artifacts:

1. Phase 3a search methods beat uniform at all tested B (CD and BS both PASS at B = 2,3,4,8) in `results/phase3a_proseco_owt/{cd_paired,bs_paired}.json`.
2. Oracle-gap closure at B = 2,3,4 is substantial (CD ~74–84%, BS ~49–64%) in `results/phase3a_proseco_owt/oracle_gap_closure.json`.
3. Phase 2b paired outcomes are mixed, with very few PASS cells and many FAIL/NULL cells; rank correlation A vs G decays with B in `results/phase2b/{policy_comparison_paired,rank_A_vs_G_phase2b}.json`.
4. Current mainline empirical evidence is ProSeCo-OWT-centric (`results/phase2b_proseco_owt`, `results/phase3a_proseco_owt`); cross-backbone replication is documented as parked in canonical docs.

These observations are **supported**.

---

## Memo 1 audit
`docs/future ideas/Deep Research Audit of Budgeted Informed Corrector Scheduling.md`

## Supported

- “Narrow framing (fixed predictor, fixed corrector, fixed budget scheduling) is defensible.”  
  **Status:** supported by canonical direction docs.
- “Greedy/separable rankers underperform while schedule-aware search succeeds on current mainline.”  
  **Status:** supported by phase2b/phase3a artifacts and canonical reports.
- “CD/BS search results should not be read as immediately deployable inference policies.”  
  **Status:** supported by method design (true-G feedback / rollout dependence) in scripts and docs.
- “External validity is not established yet.”  
  **Status:** supported; canonical docs explicitly note ProSeCo-OWT-only evidence and parked cross-backbone replication.

## Plausible but unproven

- “Biggest weakness is evaluation/benchmark mismatch (not only backend choice).”  
  **Status:** plausible but unproven from repo alone. Repo uses `neg_nll` reference scorer as main metric; whether this is *the* dominant weakness requires external comparative evidence not in-repo.
- “ProSeCo is too narrow as sole empirical anchor for broader 2026 claims.”  
  **Status:** plausible; repo supports single-backbone dependence. Strength of “too narrow” judgment depends on external target scope.

## Speculative / not yet assessable

- Claims about precise frontier dominance of specific external systems and benchmark standards.  
  **Status:** not yet assessable from repo evidence; requires independent literature verification.

## Premature / risky recommendations

- Immediate cross-backbone or benchmark pivots as “must-do now” without first extracting all information from existing Phase 2b raw schedule distributions.  
  **Risk:** medium. Could create scope creep before resolving in-repo open structural diagnostics.

## What is missing to justify stronger claims

1. Direct in-repo diagnostics quantifying schedule interaction structure (not only summary prose).
2. A documented metric-sensitivity check showing main conclusions survive alternate quality metrics.
3. At least one replication beyond ProSeCo-OWT (if generality claims are desired).

---

## Memo 2 audit
`docs/future ideas/Theoretical Frameworks for Budgeted Informed-Corrector Scheduling in Masked Diffusion Language Models.md`

## Supported

- “Open-loop schedule optimization is the right formalization for current fixed-schedule setup; adaptive/state-dependent version is a distinct extension.”  
  **Status:** supported by current scripts and protocol design (open-loop allocation dicts).
- “Greedy failure is evidence against separable stepwise ranking assumptions.”  
  **Status:** supported by phase2b/phase3a empirical pattern.
- “Use adaptive submodularity as a falsification lens rather than immediate positive theory.”  
  **Status:** plausible and consistent with observed greedy limitations.

## Plausible but unproven

- “Finite-horizon budgeted MDP is the most principled base theory for the thesis.”  
  **Status:** plausible but unproven in-repo; no formal in-repo derivation currently binds existing empirical estimators to that formulation.
- “Control-as-inference / SMC is the best imported approximation layer.”  
  **Status:** plausible but unproven for this codebase; no in-repo implementation or comparative evidence yet.
- “Belief propagation is suitable as an approximation over schedule factors.”  
  **Status:** plausible but unproven; depends on actually measured low-order sparsity/interaction structure.

## Speculative

- Any claim implying near-term feasibility or payoff of SMC/control-as-inference/BP without first estimating interaction order and approximation error on current artifacts.

## What is missing to justify stronger theoretical pivots

1. A formal bridge from current empirical objects (`G`, `A`, residuals, MC schedule sets) to a chosen control objective.
2. Interaction-order diagnostics (pairwise vs higher-order dominance) measured from existing schedule data.
3. Complexity-budget analysis for proposed algorithms relative to thesis constraints.

---

## External analysis summary audit
Input summary (user-provided) was assessed as hypotheses.

1. “Two memos are directionally strong.”  
   **Status:** plausible but unproven (depends on literature claims not independently re-verified here).
2. “Safest formalization is open-loop now, finite-horizon budgeted MDP as adaptive extension.”  
   **Status:** supported/plausible (supported for open-loop distinction; MDP extension plausible, not proven necessary yet).
3. “Adaptive submodularity better as falsification lens than positive theory.”  
   **Status:** plausible and consistent with current greedy-failure evidence.
4. “ProSeCo is defensible but too narrow as only empirical anchor.”  
   **Status:** supported for “defensible” and “single-anchor”; “too narrow” remains plausible but context-dependent.
5. “Biggest immediate weakness is evaluation/benchmark mismatch, not only backend choice.”  
   **Status:** plausible but unproven from repo-only evidence.
6. “Good next step: cross-backbone replication or stronger correction-specific benchmark layer.”  
   **Status:** plausible; not forced by current evidence if high-value in-repo structural diagnostics remain unfinished.
7. “Control-as-inference/SMC promising but not immediate.”  
   **Status:** plausible.
8. “Belief propagation only after evidence of sparse low-order interaction.”  
   **Status:** supported as methodological caution.

---

## Audit verdict

## On current ProSeCo mainline

**Verdict:** still defensible as current thesis mainline **for a narrow claim** (fixed-kernel, fixed-budget scheduling on this backend), but not sufficient for broad external generalization without additional evidence.

## On current theoretical direction

**Verdict:** current theorem-plus-honesty-ledger direction is coherent and appropriately cautious; however, bridging to a stronger formal control framing is still incomplete and should be incremental, not a hard pivot.

## On memo overreach

The main overreach risk in both memos is **premature framework escalation** (SMC/control-as-inference/BP) before extracting and formalizing interaction evidence already available in Phase 2b raw schedules.

---

## What to keep, reject, undecided

## Keep

- Ranker-vs-search distinction.
- Open-loop vs adaptive distinction.
- Caution against overclaiming deployability.
- Need for stronger structural diagnostics and possibly broader validation.

## Reject (for now)

- Any implied immediate pivot to heavyweight algorithmic frameworks without intermediate evidence steps.
- Any claim that current evidence already establishes broad cross-backbone generality.

## Undecided (needs evidence)

- Whether metric mismatch is the dominant immediate weakness.
- Whether cross-backbone replication should precede all in-repo structural analysis.

