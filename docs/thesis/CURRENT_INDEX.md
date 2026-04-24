> **STATUS:** CANONICAL (index only)
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Single authoritative entry point into the thesis repo. Pure index — pointers only. No numerical claims, no results summaries. For claims see the canonical documents linked below.

---

# CURRENT_INDEX — Thesis Navigation

This is the single authoritative entry point into the thesis repo. Everything
active is reachable from here. Archived material lives under `docs/archive/`
and is not listed in this index.

---

## 1. Thesis question

See `docs/thesis_direction.md` and `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
for the research-question framing, scope, and non-goals.

---

## 2. Canonical thesis documents (read in this order)

| # | Document | Purpose |
|---|---|---|
| 1 | `docs/thesis/CURRENT_INDEX.md` | This file — pure index |
| 2 | `docs/thesis_direction.md` | Top-level thesis direction and non-goals |
| 3 | `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` | Authoritative research direction, scope, theorem targets |
| 4 | `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | End-to-end explanation of the experiment (Protocol A/B) |
| 5 | `docs/thesis/experiments/ANALYSIS_SPEC.md` | Tier T1–T4 evidence rules, paired estimator, BCa CI |
| 6 | `docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md` | Direction audit justifying the Phase 3 pivot |
| 7 | `docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md` | Plan for Phase 3a + 3b (Half B merge target) |
| 8 | `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` | Phase 3a result report |
| 9 | `docs/thesis/theory/THEORY_STATUS.md` | Theorem A, Propositions B/C, Refinements A′/A″ |
| 10 | `docs/thesis/theory/NEXT_THEORY_STEPS.md` | Next theory tasks (Half B merge target) |
| 11 | `docs/thesis/experiments/CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md` | Cross-backbone replication implementation audit |
| 12 | `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` | Bounded LLaDA-SFT replication results (external-validity probe; Tier-3) |
| 13 | `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md` | Terminal go/no-go decision for Phase 3a on LLaDA-SFT |
| 14 | `docs/thesis/next_steps/ZANELLA_MEETING_WRITEUP.md` | Post-cross-backbone meeting brief (OWT mainline + bounded LLaDA-SFT probe) |

---

## 3. Theory (chronological record — keep consulting for provenance)

| Document | Purpose |
|---|---|
| `research/candidate_theorems.md` | Theorem A, Lemmas A1/A2, Propositions B/C, Stretch C2/C3 |
| `research/proof_worklog.md` | Entry-by-entry derivation log |
| `research/proof_ledger.md` | Provenance of every proof ingredient |
| `research/open_questions.md` | Active open questions |

---

## 4. Supporting documents

| Document | Purpose |
|---|---|
| `docs/literature_map.md` | Bibliography map for corrector scheduling and adjacent work |
| `docs/reading_plan.md` | Ordered reading queue with per-paper status |
| `docs/experimental_infrastructure.md` | Tier A/B backbones and HPC environment notes |

---

## 5. Active scripts

| Script | Purpose |
|---|---|
| `scripts/run_phase2b_proseco_owt.py` | Phase 2b paired K-seed evaluation and MC oracle |
| `scripts/analyze_phase2b.py` | Phase 2b aggregation and ranking analysis |
| `scripts/run_phase3a_combinatorial.py` | Phase 3a combinatorial scheduling baselines |
| `scripts/analyze_phase3a.py` | Phase 3a paired comparison and oracle-gap closure analysis |
| `scripts/stage_proseco_owt.py` | Reproducibility helper: stage the ProSeCo-OWT snapshot |
| `scripts/debug_proseco_owt_load.py` | CPU preflight for the staged ProSeCo-OWT backend |

Legacy Phase 1 scripts live in `archive/legacy_scripts/`.

---

## 6. Active HPC workflow

| File | Purpose |
|---|---|
| `hpc/phase2b_proseco_owt.sbatch` | Phase 2b paired sweep |
| `hpc/phase3a_combinatorial.sbatch` | Phase 3a CD-G + BS-AG combinatorial baselines |
| `hpc/push.sh` | rsync repo to HPC |
| `hpc/pull.sh` | rsync results back (broken on macOS; use ssh) |
| `hpc/setup_env.sh` | One-time conda env bootstrap |

Legacy Phase 1 sbatch files live in `archive/legacy_scripts/`.

HPC details: see root `CLAUDE.md` for host, user, checkpoint path, conda env,
and known environment issues.

---

## 7. Active backend code

| File | Purpose |
|---|---|
| `src/mdm_playground/scheduling/backends/proseco_owt.py` | ProSeCo-OWT staged snapshot loader |
| `src/mdm_playground/scheduling/backends/mdlm_conf.py` | Supporting backend for Phase 1 chronology |
| `src/mdm_playground/scheduling/backends/proseco.py` | Legacy ProSeCo on `mdlm.ckpt` |
| `src/mdm_playground/scheduling/backends/mdlm.py` | Legacy MDLM heuristic baseline |
| `src/mdm_playground/scheduling/signals.py` | Signal extraction (entropy, margin, quality mass) |
| `src/mdm_playground/scheduling/allocation.py` | Policies: uniform, top_B, burn_in_gated, front/back/middle |
| `src/mdm_playground/scheduling/gain.py` | `estimate_single_step_gain` |
| `src/mdm_playground/scheduling/evaluate.py` | `evaluate_schedule` |
| `src/mdm_playground/scheduling/surrogate.py` | Surrogate backend for CPU pipeline validation |

---

## 8. Main results files

Pointers only — claims and numbers live in the canonical experiment and
theory documents listed in §2.

| Path | Role |
|---|---|
| `results/phase3a_proseco_owt/` | Phase 3a paired outputs and oracle-gap closure |
| `results/phase2b_proseco_owt/` | Phase 2b raw paired policy rows and MC oracle samples |
| `results/phase2b/` | Phase 2b paired BCa confidence intervals and MC-oracle summary |
| `results/phase1_proseco_owt_full/` | Prerequisite chronology artifact for Phase 2b |
| `results/cross_backbone/proseco_llada_sft_bounded/` | Bounded LLaDA-SFT external-validity probe (Protocol A + Phase 2b raw + aggregates) |

Legacy Phase 1, Phase 2a, and old smoke/eval outputs are archived.

---

## 9. Thesis writing

| Path | Role |
|---|---|
| `thesis/main.tex` | Entry point |
| `thesis/chapters/abstract.tex` | |
| `thesis/chapters/ch1_introduction.tex` | |
| `thesis/chapters/ch2_background_diffusion.tex` | |
| `thesis/chapters/ch3_discrete_diffusion.tex` | |
| `thesis/chapters/ch4_masked_diffusion.tex` | |
| `thesis/chapters/ch5_informed_correctors.tex` | |
| `thesis/chapters/ch6_contribution.tex` | Theorem A / Propositions B/C write-up |
| `thesis/chapters/ch7_experiments.tex` | Phase 1–3a results write-up |
| `thesis/chapters/ch8_conclusion.tex` | |

Per-chapter drafting status lives in the chapter source files, not here.

---

## 10. Archived material

Archived material lives under `docs/archive/` with `ARCHIVED` banners:

| Path | Contents |
|---|---|
| `docs/archive/audits/` | Superseded audits (`EXPERIMENT_CRITICAL_AUDIT`, `THEORY_STRESS_TEST`, `proseco_backend_failure_audit`) |
| `docs/archive/phase1_era/` | Phase 1–2a era specs, audits, and interpretation |
| `docs/archive/chronicles/` | Historical experimental status chronicles (`RESULTS_STATUS`) |
| `docs/archive/operational/` | Operational snapshots (HPC sync, log cleanup, prompts) |
| `docs/archive/status/` | Dated LaTeX status snapshots |

Repository-level legacy (code, scripts, old results) lives under `archive/`.

---

## 11. What to read first (new reader or yourself in 3 weeks)

1. `docs/thesis/CURRENT_INDEX.md` (this file)
2. `docs/thesis_direction.md`
3. `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
4. `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md`
5. `docs/thesis/experiments/ANALYSIS_SPEC.md`
6. `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`
7. `docs/thesis/theory/THEORY_STATUS.md`

---

## 12. What to run next

See `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` (§ "Immediate next milestone")
and `docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md` for the binding
next-action spec. The bounded cross-backbone replication on LLaDA-SFT is closed
by `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md` (Phase 3a on that
backbone is **not** authorized under this decision; re-opening precondition is
stated in §6 of that file). No per-phase result summaries appear in this index.

---

## 13. Reorganization record

- `docs/thesis/maintenance/DOCS_REORGANIZATION_AUDIT.md` — full audit of docs state
- `docs/thesis/maintenance/DOCS_REORGANIZATION_PLAN.md` — executed reorganization plan
