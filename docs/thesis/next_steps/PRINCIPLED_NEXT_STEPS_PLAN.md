# Principled next-steps plan

## Problem statement and approach

Current evidence supports a narrow thesis claim: for fixed predictor, fixed informed corrector kernel, and fixed extra correction budget, scheduling quality is a schedule-level problem where greedy separable rankers underperform and schedule-aware search can recover substantial headroom (on ProSeCo-OWT).

The plan is to strengthen this claim **without over-pivoting**: first add rigorous structural diagnostics from existing artifacts, then decide whether broader empirical expansion is required.

---

## A. Audit verdict

## 1) Is current ProSeCo-based mainline defensible?
Yes, **as a narrow mainline**. It is not yet sufficient for broad cross-backbone conclusions.

## 2) Is current theoretical direction well chosen?
Partly. Current theorem framing + honesty-ledger is coherent. Missing piece is a tighter, evidence-linked formalization of schedule interactions and ranker failure mechanisms.

## 3) Do memos overreach?
Yes in places: they lean toward early large-framework pivots (MDP/SMC/BP/control-as-inference) before exhausting in-repo structural evidence extraction.

---

## B. Ranked open problems

1. **Missing structural diagnostics of schedule interactions** from existing Phase 2b/3a raw schedule data.
2. **Theory/experiment bridge gap** (formal object vs currently computed diagnostics).
3. **Metric robustness uncertainty** (mainline relies on one quality functional).
4. **Single-backbone dependence** (external validity open).
5. **No deployable scheduler claim boundary formalization** beyond prose.

---

## C. Candidate next-step options (compared)

## Option 1 — Stay with ProSeCo and do theory only
- Scientific value: medium  
- Tractability: high  
- Complexity: low  
- Thesis fit: high  
- Risk: medium (theory may outrun evidence)  
- Evidence produced: formal refinement, limited new empirical grounding

## Option 2 — Stay with ProSeCo, add stronger evaluation/diagnostics
- Scientific value: high  
- Tractability: high  
- Complexity: low-medium  
- Thesis fit: very high  
- Risk: low  
- Evidence produced: stronger mechanistic support from existing artifacts

## Option 3 — Cross-backbone replication on ProSeCo-LLaDA-SFT
- Scientific value: very high  
- Tractability: low-medium (depends on infra/checkpoint availability)  
- Complexity: high  
- Thesis fit: high  
- Risk: high (scope/time/compute)  
- Evidence produced: first external-validity anchor

## Option 4 — Pivot to general trajectory-control framing now
- Scientific value: medium-high  
- Tractability: medium  
- Complexity: medium-high  
- Thesis fit: medium (risk of broadening object)  
- Risk: medium-high  
- Evidence produced: cleaner framing, weaker immediate empirical grounding

## Option 5 — Build pairwise schedule-interaction model first
- Scientific value: high  
- Tractability: high  
- Complexity: medium  
- Thesis fit: very high  
- Risk: low-medium  
- Evidence produced: direct test of low-order vs higher-order interaction hypotheses

## Option 6 — Explore SMC/control-as-inference now (stretch)
- Scientific value: potentially high  
- Tractability: low  
- Complexity: high  
- Thesis fit: medium (premature)  
- Risk: high  
- Evidence produced: uncertain within thesis constraints

---

## D. Decisive recommendation

**Recommended staged sequence: Option 2 + Option 5 first; postpone 3/6.**

1. **Immediate (now):** add a reproducible combinatorial diagnostics layer over existing Phase 2b artifacts to quantify:
   - ranker-envelope mismatch signals,
   - overlap structure (oracle vs MC top schedules),
   - top-vs-bottom internal similarity,
   - within-seed vs between-seed variance contribution.
2. **Then:** use those diagnostics to tighten theory claims and identify whether pairwise interaction modeling is adequate.
3. **Only after that:** decide if cross-backbone replication is mandatory for thesis closure or future-work positioning.

Rationale: highest value per unit risk; improves rigor now; avoids premature heavy pivots.

---

## E. Software implementation plan

## Chosen implementation (this cycle)

Add a thesis-mainline diagnostics utility for combinatorial schedule structure:

1. **New analysis module**
   - `src/mdm_playground/analysis/combinatorial_diagnostics.py`
   - Functions for:
     - pairwise schedule Jaccard,
     - top-k overlap diagnostics (MC-top vs mean-delta-oracle),
     - top-k vs bottom-k internal overlap,
     - variance decomposition of schedule gains into within-seed / between-seed components.

2. **New script**
   - `scripts/analyze_combinatorial_diagnostics.py`
   - Inputs:
     - `results/phase2b_proseco_owt/mc_raw.json`
     - `results/phase2b_proseco_owt/policy_raw.json`
   - Output:
     - `results/phase2b/combinatorial_diagnostics.json`
   - Explicit schema and clear failure messages for missing/incompatible inputs.

3. **Test coverage**
   - `tests/test_combinatorial_diagnostics.py`
   - Lightweight unit tests for each core function and a small end-to-end synthetic case.

4. **Documentation touchpoint**
   - Keep docs coherent by referencing the new artifact path in next-step docs if needed.

## Expected artifacts

- Deterministic diagnostics JSON with all core structural measures.
- Tests passing on local test suite.

## Reproducibility and coherence constraints

- No new heavyweight dependencies.
- No changes to existing Phase 2b/3a pipelines.
- No generated logs/checkpoints committed.

