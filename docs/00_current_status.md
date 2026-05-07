# Current Status — MSc Thesis

> **Current source of truth.** Updated 2026-05-07.
> Compact summary of what is established, what failed, what is open, and risks.

---

## What has been done (experiment phases)

### Phase 1 — Protocol A (signal calibration)
50 OWT trajectories × T = 64 steps, ProSeCo-OWT backbone.
Per-step marginal gain Δ_t and per-step signals (entropy H_t, inverse margin M_t^{-1},
quality mass QM_t; raw key: `Q_t`) measured for every (seed, step). Spearman ρ(ψ, Δ_t) ≈ 0.10–0.15
(weak but positive). MC-oracle headroom over uniform = **+0.45 paired G** at B ∈ {2,3,4}.

### Phase 2b — Policy comparison + MC-oracle
K = 30 paired seeds. 10 greedy signal-ranker policies × B ∈ {2, 3, 4, 8, 16}.
MC-oracle (best-of-100 random schedules) at B ∈ {2, 3, 4}.
**Three diagnostics supporting the scoped ranker-class limitation:**
1. `mean_delta_oracle` (cheating oracle ranker) saturates at B = 8 — enters NULL band.
2. Top-10 MC ∩ oracle Jaccard ≈ 1.2–1.3× random baseline across B ∈ {2,3,4}.
3. Top-10 MC internal Jaccard ≈ bottom-10 (schedules differ just as much within top vs across top/bottom).
Additivity constants: σ_ξ = 0.174/0.240/0.309 at B = 2/3/4. Spearman ρ(A,G) = 0.60/0.54/0.46.

### Phase 3a — Combinatorial search baselines
CD-G (coordinate descent, true-G feedback) and BS-AG (beam search, cheap-A pruning + true-G rollouts).
Both PASS at every B ∈ {2, 3, 4, 8}. Recovery:
- CD-G: 74–84 % of +0.45 MC-oracle headroom at B ∈ {2, 3, 4}.
- BS-AG: 49–64 % of +0.45 headroom at B ∈ {2, 3, 4}.
**Primary positive result.** PRISM, used as a separable per-step score, falls in
the ranker class limited by the Empirical Ranker-Class Limitation (`candidate_theorems.md` §1.5); a non-separable PRISM
use is not pursued for this thesis but is not ruled out.

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
| Tested separable rankers do not recover MC-oracle headroom; mean-Δ̄ envelope enters no-detectable-gain band by B = 8 | T1 | `policy_comparison_paired.json` |
| CD-G recovers 74–84 % at B ∈ {2,3,4} | T1 | `oracle_gap_closure.json` |
| BS-AG recovers 49–64 % at B ∈ {2,3,4} | T1 | `oracle_gap_closure.json` |
| Theorem A (uniform proxy regret) proved; A′/A″ are diagnostics; Empirical Ranker-Class Limitation has formal + empirical parts | — | `research/candidate_theorems.md` §1, §1.5 |
| Uniform-not-beaten transfers to LLaDA-SFT at tested resolution | T3 | `cross_backbone/` |

---

## What failed or is deprecated

| Item | Verdict |
|---|---|
| Greedy signal rankers as primary method | Negative result — correct. Not a failure. |
| Protocol C (adaptive controller on OWT) | Honest negative. Appendix F only. |
| MC-oracle headroom on LLaDA-SFT | Does not transfer at tested resolution. |
| PRISM pivot | Not pursued as thesis pillar. Separable PRISM signals fall in the ranker class; non-separable use not ruled out. |
| Theorem A L∞ form | Empirically vacuous at B ≥ 4 in uniform form. Safe selected-schedule statement is the finite-pool corollary (Theorem A as B′(Q := A)). |
| Refinement A′ | **Demoted to additivity-scale diagnostic.** No longer presented as a regret refinement. |
| Refinement A″ | **Demoted to rankability diagnostic.** ε_R is not a theorem constant. |
| ReMDM-loop, MDLM Phase 1 | Archived. Backend issues and near-zero Δ_t. See abandoned-backend lessons table below. |
| Phase 1 interaction diagnostics | Gated — pending K=30 critical replication (Step 2d). Phase 0 complete. |

---

## Abandoned-backend lessons (compact)

| Backend / corrector | Result | Lesson | Status |
|---|---|---|---|
| MDLM heuristic corrector (Gibbs-style one-shot resample of all masked positions) | All Δ_t ≤ 0 in Phase 1 (job 478600) | Resampling 97 % masked positions with < 5 % context is a known-bad corrector design | Documented negative; not a scheduling platform |
| MDLM/ReMDM-loop on legacy framework (Phase 1 era) | Near-zero Δ_t; backend mismatch and Bug #1 (signals over wrong positions) | Signals must be computed over the same revisable set R_t the corrector acts on; backend choice is not interchangeable | Abandoned as scheduling platform; useful as negative control |
| LLaDA-SFT bounded probe (K=8, T=64, B ∈ {2,4}) | Uniform-not-beaten transfers; MC-oracle headroom does not transfer at tested resolution | Phase 3a-style schedule search is not authorised on this backbone without resolving metric/protocol issues | Pre-registered no-go for full Phase 3a |

Raw legacy result files were removed in the May 2026 cleanup; git history preserves them.

---

## What is uncertain

- Whether results generalize beyond ProSeCo-OWT to other backbones (LLaDA-SFT probe was
  inconclusive; only one clean backbone tested).
- Whether BS-AG's performance at B = 4, 8 would hold at K = 100 seeds (K = 30 tested).
- Whether the finite-pool corollary (Theorem A as B′(Q := A), §2.7) is non-vacuous
  on a candidate pool C_B with the planned no-leakage construction.
- Whether ζ_{B,C} − η_{B,C} (improvement of pairwise over additive on C_B) is
  statistically positive on ProSeCo-OWT under the Phase 1 uncertainty protocol.

---

## Current thesis risk

**Primary risk:** Single-backbone scope. All primary results are on ProSeCo-OWT. The
cross-backbone probe was inconclusive, not confirmatory. The thesis needs to clearly
scope this as a principled case study, not a universal claim.

**Secondary risk:** Theory-experiment gap. Theorem A (L∞ form) is empirically
vacuous; the operative selected-schedule statement is the finite-pool form
(Theorem A as B′(Q := A)). A′/A″ are diagnostics only.

---

## Current phase

**K=30 critical replication. Phase 0 complete (2026-05-07); K=30 gate open.**

The previous ProSeCo-OWT result remains the baseline: tested separable rankers
do not recover MC-oracle headroom, while true-G schedule search recovers much
of it. The current aim is to reframe and extend this into a cleaner theory-first
study of marginal, interaction-aware, and online corrector timing.

Sequential gates:
1. Opus theory pass ✅ — Theorem A baseline, Theorem B/B′ central, Diagnostic Framework C, A′/A″ as diagnostics, Empirical Ranker-Class Limitation; D/E optional.
2. Phase 0 reproducibility audit ✅ — PF1–PF8: 11/11 on HPC login node (CPU, slnode01, 2026-05-06). K=3 smoke on A100 (gnode01, job 489457) qualitatively matches prior results. K=30 critical replication gate is now open.
3. Interaction diagnostics — ⛔ blocked until K=30 replication passes.
4. Pairwise scheduler — only if diagnostics show structure.
5. LaTeX writing — running in parallel once the theory scaffold is stable.

K=30 replication is the immediate next HPC action: submit
`hpc/phase2b_proseco_owt.sbatch` then `hpc/phase3a_combinatorial.sbatch`.
Phase 1 interaction diagnostics remain blocked until K=30 passes. See
`docs/05_next_steps.md` for the sequential action plan and
`docs/06_theory_first_research_plan.md` for the full theory-first programme.
