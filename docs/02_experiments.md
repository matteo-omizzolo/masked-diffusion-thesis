# Experiments — Summary

> **Current source of truth.** Updated 2026-05-08.

---

## Phase Results Ledger

| Phase / protocol | Status / tier | Core result | Raw output folder | Gate / next action |
|---|---|---|---|---|
| **Phase 0 — reproducibility gate** | ✅ Gate passed (2026-05-07) | PF1–PF8: 11/11 on HPC CPU (slnode01). K=3 smoke on A100 (gnode01, job 489457) matches prior qualitative pattern. Not thesis evidence. | `results/phase0_smoke_d4edc92/` | Closed provenance. |
| **Phase 1 — Protocol A** | T1 (K=50) | MC-oracle headroom +0.45; signal ρ≈0.10–0.15 | `results/phase1_proseco_owt_full/` | Baseline for all downstream phases. |
| **Phase 2b — policy comparison** | T1 (K=30) | Canonical headroom +0.451/+0.441/+0.450; K=30 replication job 490106 headroom +0.385/+0.355/+0.380. Separable rankers fail in both. | `results/phase2b_proseco_owt/`, `results/phase2b/`; replication `results/phase2b_k30_rep_cf89e00/` | Empirical Ranker-Class Limitation confirmed; K=30 gate closed. |
| **Phase 3a — combinatorial search** | T1 (K=30) | CD-G 74–84 %, BS-AG 49–64 % recovery vs canonical +0.45 at B∈{2,3,4}; p<0.001. Primary positive result. | `results/phase3a_proseco_owt/` | Thesis primary result; K=30 gate closed. |
| **Cross-backbone (LLaDA-SFT)** | T3 (K=8) | Uniform-not-beaten transfers; MC headroom does NOT transfer. | `results/cross_backbone/` | Phase 3a pre-registered no-go. |
| **Protocol C — adaptive controller** | Honest negative | ε̃/ε ∈ [0.983, 0.986]; < 1.7 % improvement. | `results/protocol_c_owt/` | Appendix F only; no further work. |
| **Phase 1 (interaction diagnostics)** | ✅ Open | Sparse pair diagnostics (Gate 3a) followed by schedule-level validation (Gate 3b). | — | Next empirical gate. |

---

## Evidence tiers (ANALYSIS_SPEC)

| Tier | Meaning |
|---|---|
| T1 | 95 % BCa bootstrap CI excludes 0; K ≥ 30 seeds paired |
| T2 | Directional evidence; K < 30 or CI touches 0 |
| T3 | Preliminary / indicative; K < 15 or CI not computed |
| T4 | Heuristic / single-seed observation |

All Phase 2b / Phase 3a results are T1 (K = 30, BCa CI). Cross-backbone is T3 (K = 8).

---

## Backbone

**ProSeCo-OWT** (ProSeCo annealed-refinement corrector, MDLM backbone, OpenWebText, T = 64).
Quality functional F = −GPT-2 NLL on a 512-token reference window.
Checkpoint snapshot: `~/mdm/checkpoints/proseco_owt` on HPC. Staged on HPC via
`scripts/stage_proseco_owt.py`.

---

## Phase 1 — Protocol A (signal calibration)

**What:** Measure per-step Δ_t = F(y_t^{+1}) − F(y_base) and per-step signals
(H_t, M_t^{-1}, QM_t; raw key: `Q_t`) for each of 50 seeds × T = 64 steps.

**Verdict:** Signal-to-gain Spearman ρ ≈ 0.10–0.15. Weak but positive.
Low-gain T_low region confirmed at early steps (R_t ≈ ∅). MC-oracle headroom = +0.45.

**Raw results:** `results/phase1_proseco_owt_full/protocol_a/trajectory_*.json`

---

## Phase 2b — Policy comparison + MC-oracle

**What:** K = 30 paired seeds. 10 greedy signal-ranker policies × B ∈ {2, 3, 4, 8, 16}.
MC-oracle (best-of-100 random schedules) at B ∈ {2, 3, 4}.

**Key findings (all T1):**
- MC-oracle headroom U_B^{MC,100} = +0.45 paired G at B ∈ {2, 3, 4}; 95 % BCa CI
  excludes 0. Note: this is the best-of-100 random schedule pool oracle; the
  exhaustive (T choose B) oracle is unobservable.
- All 10 tested separable rankers do not recover MC-oracle headroom. The
  cheating `mean_delta_oracle` (marginal / time-profile oracle ranker, not a
  non-separable policy) saturates and enters the no-detectable-gain band by B = 8.
- Top-10 MC ∩ oracle Jaccard ≈ 1.2–1.3× random baseline (schedules do not concentrate
  on a small corner of the space).
- Top-10 MC internal Jaccard ≈ bottom-10 Jaccard (no coherent "best schedule" cluster).

**K=30 replication (job 490106, gnode01, completed 2026-05-07):**
Output `results/phase2b_k30_rep_cf89e00/` has 4/4 shards, seeds 42–71,
`policy_raw.json` with 1500 rows, and `mc_raw.json` with 9000 rows. Replicated
headroom is +0.385/+0.355/+0.380 at B = 2/3/4 (avg ≈ +0.37), below the
canonical +0.451/+0.441/+0.450 (avg ≈ +0.45) but with the same qualitative
conclusion. `mean_delta_oracle` recovers only 13–29 % of replicated headroom.

**Theorem A diagnostics (measured on 9000 MC rows, 30 seeds; A′/A″ are
diagnostics, *not* regret-bound constants — see `research/candidate_theorems.md`
§1.3):**

| B | σ_ξ (A′ scale) | R_B = ρ(A,G) (A″) | σ_Δ | (1−|ρ|)·σ_Δ |
|---|---|---|---|---|
| 2 | 0.174 | 0.601 [0.571, 0.628] | 0.176 | 0.070 |
| 3 | 0.240 | 0.542 [0.516, 0.572] | 0.202 | 0.092 |
| 4 | 0.309 | 0.462 [0.430, 0.492] | 0.220 | 0.118 |

The (1−|ρ|)·σ_Δ column is a rank-correlation diagnostic, not a theorem
constant; the rigorous selected-schedule statement is the finite-pool form
of Theorem A (§2.7 of `candidate_theorems.md`).

**Raw results:** `results/phase2b_proseco_owt/per_seed/`, `results/phase2b/`
Key files: `policy_comparison_paired.json`, `mc_oracle.json`, `combinatorial_diagnostics.json`,
`theorem_a_constants.json`.

---

## Phase 3a — Combinatorial search baselines

**What:** Non-greedy search procedures tested at K = 30 seeds, B ∈ {2, 3, 4, 8}.
Canonical run: job 479941 on gnode02, complete 2026-04-20, 30 seeds × 4 budgets
× CD-G + BS-AG, with 60 per-seed files plus `cd_raw.json` and `bs_raw.json`.
Failed duplicate job 490469 is harmless: it died in the sbatch preamble before
launching shards and wrote no files.

### CD-G — Coordinate descent, true-G feedback
- Each iteration: sample one (in, out) position swap; accept iff G improves.
- Per-cell budget: ≤ 65 true-G calls.
- Result: structural existence result (not deployable — uses true G for every decision).

### BS-AG — Beam search, cheap-A pruning + true-G rollouts
- Round 1: rank singletons by cheap A (Δ_t from Phase 1); rollout top-8 with true G.
- Rounds 2…B: rank extensions by cheap A; rollout top-8; keep best-G beam.
- Per-cell budget: 8 × B true-G rollouts.
- Result: more practical (O(B) G-calls); uses true G only for rollout scoring.

**Recovery rates vs canonical +0.45 MC-oracle headroom:**

| B | CD-G recovery | BS-AG recovery | Oracle CI |
|---|---|---|---|
| 2 | **78.9 %** | 64.1 % | [0.383, 0.528] |
| 3 | **74.1 %** | 57.1 % | [0.366, 0.519] |
| 4 | **84.3 %** | 48.8 % | [0.386, 0.520] |
| 8 | PASS (still > uniform) | PASS | oracle in NULL band |

**Recovery rates vs Phase 2b K=30 replication headroom (~+0.37):**

| B | CD-G recovery | BS-AG recovery |
|---|---|---|
| 2 | **93.1 %** | 75.8 % |
| 3 | **85.5 %** | 64.3 % |
| 4 | **98.2 %** | 56.2 % |

**Verdict:** Combinatorial search with G feedback is the right policy class on
this triple. The tested separable rankers are limited by the
**Empirical Ranker-Class Limitation** (`candidate_theorems.md` §1.5;
formal part for time-only / seed-averaged separable ψ; empirical part on
tested rankers). PRISM-as-separable-score is in this class; non-separable
PRISM uses are not ruled out and not pursued in this thesis.

**Raw results:** `results/phase3a_proseco_owt/per_seed/`, `cd_raw.json`,
`bs_raw.json`. The K=30 gate does not depend on a separate
`oracle_gap_closure.json` artifact.

---

## Cross-backbone probe — LLaDA-SFT (bounded, K = 8)

**What:** Bounded replication of Phase 2b on LLaDA-SFT backbone. T = 64, B ∈ {2, 4},
K = 8 seeds, GPT-2 reference.

**Verdict (T3):**
- Uniform-not-beaten transfers: all policies ≤ uniform (Tier 3 support).
- MC-oracle headroom does NOT transfer: paired CI [0, 0] at B = 4; [−4.07, −1.07] at B = 2.
- Three non-discriminable hypotheses: H1 corrector dominance, H2 protocol sparseness,
  H3 reference mismatch.
- **Phase 3a NOT authorized** on LLaDA-SFT (pre-registered no-go; reopening precondition
  `mc_oracle_minus_uniform.bootstrap_95_ci_lo > 0.05` not satisfied).

**Raw results:** `results/cross_backbone/proseco_llada_sft_bounded/`

---

## Protocol C — Adaptive controller (CPU, OWT)

**What:** Bucketed-state conditioning z = (signal_quartile × phase(t)) = 12 buckets per signal.
50 Phase 1 trajectories + 30 Phase 2b MC seeds. Additive surrogate with bucket-mean Δ_t.

**Verdict (honest negative):**
- ε̃ / ε ∈ [0.983, 0.986] — state conditioning shrinks ε by < 1.7 %.
- Best after-uncertainty close ratio = +0.015 (entropy, B=2); negative at B ≥ 3.
- Hamming distance from threshold schedule to best MC ≈ 2B (max possible): no overlap.
- Theorem A-ad: historical appendix/provenance item; not part of the active
  main theorem stack.
- No GPU or HPC used. No further adaptive-controller work authorized.

**Raw results:** `results/protocol_c_owt/protocol_c_summary.json`

---

## Scripts for each phase

| Phase | Runner | Analyzer |
|---|---|---|
| Phase 2b | `scripts/run_phase2b_proseco_owt.py` | `scripts/analyze_phase2b.py` |
| Phase 3a | `scripts/run_phase3a_combinatorial.py` | `scripts/analyze_phase3a.py` |
| Protocol C | `scripts/run_protocol_c_owt.py` | `src/mdm_playground/analysis/protocol_c.py` |
| Theorem A constants | `scripts/compute_theorem_a_constants.py` | — |
| Phase 2b diagnostics | `scripts/analyze_combinatorial_diagnostics.py` | — |
