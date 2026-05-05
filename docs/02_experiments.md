# Experiments — Summary

> **Current source of truth.** Updated 2026-05-05.
> Synthesized from Phase 3a results, cross-backbone results, Protocol C results,
> ANALYSIS_SPEC, and CANONICAL_EXPERIMENT_OVERVIEW (all archived after cleanup).

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
Checkpoint: `~/mdm/checkpoints/mdlm.ckpt` on HPC. Staged on HPC via `scripts/stage_proseco_owt.py`.

---

## Phase 1 — Protocol A (signal calibration)

**What:** Measure per-step Δ_t = F(y_t^{+1}) − F(y_base) and per-step signals
(H_t, M_t^{-1}, Q_t) for each of 50 seeds × T = 64 steps.

**Verdict:** Signal-to-gain Spearman ρ ≈ 0.10–0.15. Weak but positive.
Low-gain T_low region confirmed at early steps (R_t ≈ ∅). MC-oracle headroom = +0.45.

**Raw results:** `results/phase1_proseco_owt_full/protocol_a/trajectory_*.json`

---

## Phase 2b — Policy comparison + MC oracle

**What:** K = 30 paired seeds. 10 greedy signal-ranker policies × B ∈ {2, 3, 4, 8, 16}.
MC oracle (best-of-100 random schedules) at B ∈ {2, 3, 4}.

**Key findings (all T1):**
- MC-oracle headroom = +0.45 paired G at B ∈ {2, 3, 4}; 95 % BCa CI excludes 0.
- All 10 greedy rankers fail to match MC oracle. The cheating `mean_delta_oracle` ranker
  saturates and enters the NULL band by B = 8.
- Top-10 MC ∩ oracle Jaccard ≈ 1.2–1.3× random baseline (schedules do not concentrate
  on a small corner of the space).
- Top-10 MC internal Jaccard ≈ bottom-10 Jaccard (no coherent "best schedule" cluster).

**Theorem A constants (measured on 9000 MC rows, 30 seeds):**

| B | σ_ξ | ρ(A,G) | σ_Δ | ε_R = (1−|ρ|)·σ_Δ |
|---|---|---|---|---|
| 2 | 0.174 | 0.601 [0.571, 0.628] | 0.176 | 0.070 |
| 3 | 0.240 | 0.542 [0.516, 0.572] | 0.202 | 0.092 |
| 4 | 0.309 | 0.462 [0.430, 0.492] | 0.220 | 0.118 |

**Raw results:** `results/phase2b_proseco_owt/per_seed/`, `results/phase2b/`
Key files: `policy_comparison_paired.json`, `mc_oracle.json`, `combinatorial_diagnostics.json`,
`theorem_a_constants.json`.

---

## Phase 3a — Combinatorial search baselines

**What:** Non-greedy search procedures tested at K = 30 seeds, B ∈ {2, 3, 4, 8}.

### CD-G — Coordinate descent, true-G feedback
- Each iteration: sample one (in, out) position swap; accept iff G improves.
- Per-cell budget: ≤ 65 true-G calls.
- Result: structural existence result (not deployable — uses true G for every decision).

### BS-AG — Beam search, cheap-A pruning + true-G rollouts
- Round 1: rank singletons by cheap A (Δ_t from Phase 1); rollout top-8 with true G.
- Rounds 2…B: rank extensions by cheap A; rollout top-8; keep best-G beam.
- Per-cell budget: 8 × B true-G rollouts.
- Result: more practical (O(B) G-calls); uses true G only for rollout scoring.

**Recovery rates (oracle_gap_closure.json):**

| B | CD-G recovery | BS-AG recovery | Oracle CI |
|---|---|---|---|
| 2 | **0.79** | 0.64 | [0.383, 0.528] |
| 3 | **0.74** | 0.57 | [0.366, 0.519] |
| 4 | **0.84** | 0.49 | [0.386, 0.520] |
| 8 | PASS (still > uniform) | PASS | oracle in NULL band |

**Verdict:** Combinatorial search is the right policy class.
Greedy rankers are bounded by the ranker-class Negative-Result Corollary.
PRISM rejection refined: not "no structure exists" but "structure does not factor
through any separable per-step score."

**Raw results:** `results/phase3a_proseco_owt/per_seed/`, `oracle_gap_closure.json`

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
- Theorem A-ad: formally proved, promoted to Appendix F.
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
