# ProSeCo Experiment Options Audit

*Written: 2026-04-18.*
*Scope: what experiments are actually runnable with the proseco-owt checkpoint
and our Protocol A/B infrastructure, and which are thesis-relevant.*

---

## 1. What ProSeCo is and what it contributes

ProSeCo (Progressive Self-Correcting MDM; Kuleshov et al. 2025) trains a masked
diffusion backbone jointly with a corrector loss.  At inference, after each MDLM
predictor step, a "corrector loop" of K annealed forward passes refines the tokens
that have already been committed (unmasked) by the predictor.

The paper's headline result: with `corrector_steps=8`, ProSeCo improves perplexity
over standard MDLM sampling.  The improvement comes entirely from the backbone
learning, at training time, to respond to the corrector's perturbation structure.

For the thesis, ProSeCo is valuable because:
- The backbone IS co-trained → Δ_t should be non-trivially non-zero at mid-to-late
  trajectory steps, unlike MDLM on `mdlm.ckpt` (structural no-op) or MDLM-conf
  (weak signal).
- The corrector's action set (unmasked positions) grows monotonically over the
  trajectory — a natural hook for signal-adaptive scheduling.
- The paper's default schedule (`corrector_every_n_steps=1`) is uniform across all
  steps.  Our thesis asks: can signal-based scheduling outperform this uniform baseline?

---

## 2. The ProSeCo corrector configuration space

### 2.1 Knobs that exist in the ProSeCo repo

| Parameter | Controls | Default | Thesis relevance |
|-----------|---------|---------|----------------|
| `corrector_steps` | Depth per corrector application (inner loop count) | 0 (off) | FIX at 1 or 2; not a scheduling knob |
| `corrector_every_n_steps` | Frequency: apply corrector every N predictor steps | 1 | ProSeCo's *implicit* scheduling knob |
| `corrector_start_iter` | Burn-in: disable corrector before step N | 0 | Related to T_low / burn-in detection |
| `corrector_sampling` | Token selection within a loop: `argmax`, `sample`, `select_top_k` | `argmax` | Kernel-level; not a scheduling knob |
| `corrector_top_k` | Number of positions to update in `select_top_k` mode | 0 | Kernel-level; FIX at 0 (argmax) |
| `corrector_prior_is_argmax` | Init corrector from argmax vs sample | True | FIX at True |
| `nucleus_p` | Top-p filtering for token selection | 1.0 | FIX at 1.0 |

### 2.2 Three distinct corrector dimensions (thesis framework)

The thesis distinguishes three orthogonal aspects of any corrector:

| Dimension | ProSeCo knob | Thesis decision |
|-----------|-------------|----------------|
| **Kernel design** (how to correct one position) | `corrector_sampling`, argmax/topk | FIX: argmax (standard ProSeCo) |
| **Token selection within a loop** (which positions to correct in one application) | `corrector_top_k`, `select_top_k` | FIX: all unmasked positions (argmax mode) |
| **Trajectory-level scheduling** (WHEN to spend correction budget across the trajectory) | `corrector_every_n_steps`, `corrector_start_iter` | THIS IS THE THESIS VARIABLE |

The thesis contribution is entirely in the third dimension: signal-adaptive scheduling
replaces ProSeCo's fixed-frequency `corrector_every_n_steps=1` with a smarter budget
allocation guided by trajectory-level signals.

---

## 3. Possible experiments with proseco-owt

### Group A — directly thesis-relevant

| Experiment | Configuration | Goal |
|-----------|--------------|------|
| **A1: Protocol A calibration** | proseco-owt, corrector_steps=1, N=20, T=64 | Measure Δ_t per step; calibrate signals vs Δ_t |
| **A2: Protocol B additivity** | proseco-owt, corrector_steps=1, M=15, P=120 | Estimate η_B, γ; check Theorem A bound |
| **A3: Policy comparison** | proseco-owt, corrector_steps=1, B∈{4,8,16} | Compare signal-based vs uniform budget allocation |
| **A4: Full run N=50** | same as A1-A3 but N=50, M=30, P=300 | Tighten CIs; thesis-grade evidence |

### Group B — useful context but not the main experiment

| Experiment | Configuration | When useful |
|-----------|--------------|------------|
| **B1: corrector_steps sweep** | corrector_steps ∈ {1, 2, 4} | Only if A1 shows Δ_t is sensitive to depth; not needed for pilot |
| **B2: burn-in ablation** | vary corrector_start_iter | Confirms/denies T_low finding from Protocol A |
| **B3: MDLM-conf vs ProSeCo-OWT comparison** | both backends, same seeds | Cross-backend comparison of signal quality |

### Group C — distracting, do not explore now

| Experiment | Why skip |
|-----------|---------|
| corrector_sampling=sample sweep | Adds stochasticity; confounds signal calibration |
| select_top_k vs argmax within a loop | Token-selection, not scheduling — different thesis question |
| nucleus_p sweep | Orthogonal; no scheduling signal |
| LLaDA-SFT checkpoint | Much larger model; not on HPC; Phase 4 at earliest |
| corrector_every_n_steps sweep via ProSeCo's own config | Redundant with our Protocol A/B framework |

---

## 4. Why proseco-owt is the right checkpoint

The thesis question requires Δ_t > 0 at some trajectory steps.  This has been
empirically demonstrated only on the surrogate generator.  Real-checkpoint runs
produced:
- MDLM heuristic (all-masked resample): Δ_t ≤ 0 everywhere — harmful corrector
- ProSeCo on mdlm.ckpt: Δ_t ≡ 0 — structural no-op (not co-trained)
- MDLM-conf on mdlm.ckpt: Δ_t > 0 at 25/64 steps but signal Spearman ≈ 0.05

With `proseco-owt`, the backbone was trained with the corrector loss.  The
annealed refinement corrector should alter committed tokens non-trivially
(the backbone learned to respond to the corrector's specific perturbation pattern).
The signal (entropy / margin / quality_mass over committed tokens) should
therefore carry more predictive information about Δ_t than on mdlm.ckpt.

---

## 5. Best experimental plan

### Phase 1a: Pilot (now)
- **proseco-owt, corrector_steps=1, N=20, T=64, M=15, P=120, B∈{4,8,16}**
- Decision threshold: if Δ_t > 0 at ≥10/64 steps AND |Spearman| ≥ 0.10 for
  at least one signal → proceed to Phase 1b

### Phase 1b: Full run (if pilot passes)
- **proseco-owt, corrector_steps=1, N=50, T=64, M=30, P=300, B∈{4,8,16}**
- This is the primary thesis evidence base

### Phase 1c: Depth ablation (if Phase 1b is promising)
- **proseco-owt, corrector_steps=2, N=20** — one extra run to check if deeper
  correctors improve signal calibration or just scale Δ_t

### After Phase 1: Theory grounding
- Update Theorem A statement with proseco-owt ε values
- Write up Phase 1 results in ch7_experiments.tex

---

## 6. What NOT to do

- Do NOT run MDLM-conf at N=50 — Spearman is noise-level; bigger N won't help
- Do NOT sweep corrector_sampling or select_top_k — kernel design, not scheduling
- Do NOT run on LLaDA-SFT checkpoint — too large for HPC single-GPU setup
- Do NOT vary backbone temperature or nucleus_p — orthogonal to thesis question
- Do NOT implement ProSeCo's full Hydra config system — use our wrapper instead
