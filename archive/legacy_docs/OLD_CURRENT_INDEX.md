# CURRENT_INDEX — Thesis Navigation

**Last updated:** 2026-04-17  
**Phase:** Phase 1 — ProSeCo empirical backend (pilot pending GPU)

This document is the single entry point into the thesis. All active work
is reachable from here. Deprecated or archived material is not listed.

---

## Core thesis documents

| Document | Purpose |
|---|---|
| `docs/thesis_direction.md` | Research question, scope, non-goals |
| `research/candidate_theorems.md` | Theorem A (main), Propositions B/C, ranking |
| `research/proof_worklog.md` | Chronological derivation log (Entry 7 = latest) |
| `research/proof_ledger.md` | Provenance of every proof ingredient |
| `research/open_questions.md` | Active unresolved technical points |

---

## Experiment — Phase 1 (ProSeCo backend)

**Status as of 2026-04-17:**  
Surrogate sanity PASSED. Real pilot pending HPC GPU submission.

| Document | Purpose |
|---|---|
| `docs/experiments/proseco_experiment_definition.md` | Mathematical specification (Protocol A/B, signals, F) |
| `docs/experiments/proseco_protocol_mapping.md` | Code-level implementation of Protocol A/B |
| `docs/experiments/proseco_backend_audit.md` | Checkpoint decision, corrector mechanism, ProSeCo audit |
| `docs/experiments/proseco_validation_checklist.md` | Structural checks before/after GPU run |
| `docs/experiments/results/proseco_fast_sanity_run.md` | Surrogate sanity results (PASSED, 8.1s) |
| `docs/experiments/results/result_inventory.md` | Full run inventory (MDLM diagnostic + ProSeCo runs) |

### Key scripts and HPC

| File | Purpose |
|---|---|
| `src/mdm_playground/scheduling/backends/proseco.py` | ProSeCoGenerator (run_base, run_branch, run_with_schedule) |
| `scripts/run_phase1_proseco.py` | Protocol A/B runner (--surrogate for local, real for GPU) |
| `hpc/phase1_proseco.sbatch` | HPC job script (pilot active; sanity block commented) |

**To submit pilot:**
```bash
bash hpc/push.sh && sbatch hpc/phase1_proseco.sbatch
```

---

## MDLM heuristic — diagnostic baseline (NOT the main platform)

| Document | Purpose |
|---|---|
| `docs/experiments/results/result_inventory.md` | MDLM run 478600 — two bugs, negative result |
| `docs/experiments/phase1_interpretation.md` | Interpretation of MDLM diagnostic findings |

**Key finding:** MDLM Gibbs corrector has all Δ_t ≤ 0 (Bug #2: resamples all masked
positions) and invalid signals (Bug #1: signals over wrong positions). Documented as a
negative result. ProSeCo was adopted as the main backend.

---

## Theory stack summary

| Component | Status | Empirical hook |
|---|---|---|
| Theorem A: G(S*) − G(Ŝ) ≤ 2Bε + 2η_B | Proof complete (solid under assumptions) | ε (Protocol A), η_B (Protocol B) |
| Lemma A1: Oracle top-B optimal under additivity | Trivial | — |
| Lemma A2: Proxy gives 2Bε regret | Exchange argument | ε (Protocol A) |
| Proposition B: Burn-in gating benign | Plausible; verified by ProSeCo design | Δ_t profile (Protocol A) |
| Proposition C: η_B ≤ γB(B-1)/2 | Plausible | γ (Protocol B pairwise) |
| Stretch C2: E_fact contraction | Conjecture; demoted to appendix | — |

---

## Literature

| Document | Purpose |
|---|---|
| `docs/literature_map.md` | Full map of adjacent work (ProSeCo, PRISM, Zhao et al., etc.) |
| `docs/reading_plan.md` | Reading priority queue |
| `study/` | Papers by topic with notes |

---

## Thesis writing

| File | Status |
|---|---|
| `thesis/main.tex` | Entry point |
| `thesis/ch2_background_continuous.tex` | First draft done |
| `thesis/ch3_discrete.tex` | TODO |
| `thesis/ch4_scheduling.tex` | TODO (theory chapter — Theorem A, Propositions B/C) |
| `thesis/ch5_experiments.tex` | TODO (ProSeCo Phase 1 results) |

---

## HPC environment

- **Host:** `slogin.hpc.unibocconi.it` (user `3316152`)
- **Env:** `conda activate remdm311` (Python 3.11, PyTorch 2.10.0+cu128)
- **Checkpoint:** `~/mdm/checkpoints/mdlm.ckpt`
- **Known issues:** See `CLAUDE.md` §"Known HPC environment issues"

---

## Deprecated / archived

- `docs/md/` — older study documents (deprecation banners; do not use for thesis)
- `archive/` — previous research directions (step-sweep era)
- MDLM heuristic corrector (`backends/mdlm.py`) — kept for comparison; not the main platform
