# Current Status — MSc Thesis

> **Current source of truth.** Updated 2026-05-05.
> Compact summary of what is established, what failed, what is open, and risks.

---

## What has been done (experiment phases)

### Phase 1 — Protocol A (signal calibration)
50 OWT trajectories × T = 64 steps, ProSeCo-OWT backbone.
Per-step marginal gain Δ_t and per-step signals (entropy H_t, inverse margin M_t^{-1},
quality mass Q_t) measured for every (seed, step). Spearman ρ(ψ, Δ_t) ≈ 0.10–0.15
(weak but positive). MC-oracle headroom over uniform = **+0.45 paired G** at B ∈ {2,3,4}.

### Phase 2b — Policy comparison + MC oracle
K = 30 paired seeds. 10 greedy signal-ranker policies × B ∈ {2, 3, 4, 8, 16}.
MC oracle (best-of-100 random schedules) at B ∈ {2, 3, 4}.
**Three smoking guns confirming the negative result for greedy rankers:**
1. `mean_delta_oracle` (cheating oracle ranker) saturates at B = 8 — enters NULL band.
2. Top-10 MC ∩ oracle Jaccard ≈ 1.2–1.3× random baseline across B ∈ {2,3,4}.
3. Top-10 MC internal Jaccard ≈ bottom-10 (schedules differ just as much within top vs across top/bottom).
Additivity constants: σ_ξ = 0.174/0.240/0.309 at B = 2/3/4. Spearman ρ(A,G) = 0.60/0.54/0.46.

### Phase 3a — Combinatorial search baselines
CD-G (coordinate descent, true-G feedback) and BS-AG (beam search, cheap-A pruning + true-G rollouts).
Both PASS at every B ∈ {2, 3, 4, 8}. Recovery:
- CD-G: 74–84 % of +0.45 MC-oracle headroom at B ∈ {2, 3, 4}.
- BS-AG: 49–64 % of +0.45 headroom at B ∈ {2, 3, 4}.
**Primary positive result.** PRISM pivot rejected: PRISM is in the ranker class.

### Cross-backbone (LLaDA-SFT bounded probe, K = 8)
T = 64, B ∈ {2, 4}, GPT-2 reference. Uniform-not-beaten transfers (Tier 3). MC-oracle
headroom does NOT transfer (CI includes 0 or negative at tested resolution). Three
non-discriminable hypotheses: corrector dominance (H1), protocol sparseness (H2),
reference mismatch (H3). Phase 3a NOT authorized on LLaDA-SFT (pre-registered no-go).

### Protocol C — Adaptive controller (CPU, OWT)
Bucketed-state conditioning z = (signal_quartile, phase(t)) on 12 buckets per signal.
ε̃ / ε ∈ [0.983, 0.986] — state conditioning shrinks calibration error by < 1.7 %.
Best after-uncertainty close ratio = +0.015 (entropy, B=2); negative at B ≥ 3.
**Honest negative.** Theorem A-ad lives in Appendix F only.

---

## What results are established (reliable)

| Result | Tier | Anchor |
|---|---|---|
| MC-oracle headroom over uniform = +0.45 at B ∈ {2,3,4} | T1 | `oracle_gap_closure.json` |
| Greedy rankers fail by B = 8 (all 10 policies) | T1 | `policy_comparison_paired.json` |
| CD-G recovers 74–84 % at B ∈ {2,3,4} | T1 | `oracle_gap_closure.json` |
| BS-AG recovers 49–64 % at B ∈ {2,3,4} | T1 | `oracle_gap_closure.json` |
| Theorem A, Refinements A′/A″, Negative-Result Corollary: formally proved | — | `research/candidate_theorems.md` |
| Uniform-not-beaten transfers to LLaDA-SFT at tested resolution | T3 | `cross_backbone/` |

---

## What failed or is deprecated

| Item | Verdict |
|---|---|
| Greedy signal rankers as primary method | Negative result — correct. Not a failure. |
| Protocol C (adaptive controller on OWT) | Honest negative. Appendix F only. |
| MC-oracle headroom on LLaDA-SFT | Does not transfer at tested resolution. |
| PRISM pivot | Rejected. PRISM is in the ranker class. |
| Theorem A L∞ form | Empirically vacuous at every B ∈ {4, 8, 16}. A″ rank-based form is load-bearing. |
| ReMDM-loop, MDLM Phase 1 | Archived. Backend issues and near-zero Δ_t. |
| Full-scale HPC new runs | Gated — pending theory scaffold + Phase 0 audit. |

---

## What is uncertain

- Whether results generalize beyond ProSeCo-OWT to other backbones (LLaDA-SFT probe was
  inconclusive; only one clean backbone tested).
- Whether BS-AG's performance at B = 4, 8 would hold at K = 100 seeds (K = 30 tested).
- Formal proof tightness of Refinement A′ (σ_ξ · √B/√2 mass-form: order-statistics
  derivation exists but depends on mixing/cancellation hypothesis).
- Non-vacuity of Theorem A L∞ form on any backbone tested.

---

## Current thesis risk

**Primary risk:** Single-backbone scope. All primary results are on ProSeCo-OWT. The
cross-backbone probe was inconclusive, not confirmatory. The thesis needs to clearly
scope this as a principled case study, not a universal claim.

**Secondary risk:** Theory-experiment gap. Theorem A (L∞ form) is empirically vacuous;
the narrative must pivot to Refinements A′/A″ as the empirically-anchored variants.

---

## Current phase

**Theory-first reassessment and Phase 0 reproducibility planning.**

The previous ProSeCo-OWT result remains the baseline: rankers fail, schedule search
works. The current aim is to reframe and extend this into a cleaner theory-first study
of marginal, interaction-aware, and online corrector timing.

Sequential gates:
1. Opus theory pass — formalize Theorem B, Proposition C, Theorem D.
2. Phase 0 reproducibility audit — reproduce ProSeCo-OWT baseline locally.
3. Interaction diagnostics — only after Phase 0 passes.
4. Pairwise scheduler — only if diagnostics show structure.
5. LaTeX writing — running in parallel once the theory scaffold is stable.

No full-scale new HPC experiments should be launched until the theory scaffold and
Phase 0 reproducibility audit are complete. See `docs/05_next_steps.md` for the
sequential action plan and `docs/06_theory_first_research_plan.md` for the full
theory-first programme.
