# Repo Cleanup Inventory — 2026-05-05

> Generated as part of `repo-cleanup-compact-current-state` branch.
> This file classifies every active/semi-active doc before the cleanup.
> Raw results and source code are preserved in-place and NOT listed here
> unless explicitly reclassified.

---

## Classification Key

| Class | Meaning |
|---|---|
| `CURRENT_SOT` | Current source of truth — stays active |
| `MERGE_INTO` | Key info extracted into new compact doc, then archive |
| `ARCHIVE_HIST` | Old doc, superseded, or redundant — move to archive |
| `RAW_PRESERVE` | Raw experimental output — preserve in place |
| `KEEP_RUNNABLE` | Script needed for reproducibility |
| `LEGACY_CODE` | Old/debug script — move to scripts/legacy/ |

---

## Root-level docs

| File | Class | Disposition |
|---|---|---|
| `CLAUDE.md` | `CURRENT_SOT` | Update to point to new compact docs |
| `README.md` | `CURRENT_SOT` | Keep, minor update |
| `REPRODUCIBILITY.md` | `CURRENT_SOT` | Keep as-is |

---

## docs/thesis/ — Active canonical docs (all become ARCHIVE after merge)

| File | Class | Information preserved in |
|---|---|---|
| `docs/thesis/CURRENT_INDEX.md` | `MERGE_INTO` | `START_HERE.md`, `docs/README.md` |
| `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` | `MERGE_INTO` | `docs/01_research_direction.md` |
| `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | `MERGE_INTO` | `docs/02_experiments.md` |
| `docs/thesis/experiments/ANALYSIS_SPEC.md` | `MERGE_INTO` | `docs/02_experiments.md` §evidence tiers |
| `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` | `MERGE_INTO` | `docs/02_experiments.md` §Phase 3a |
| `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` | `MERGE_INTO` | `docs/02_experiments.md` §cross-backbone |
| `docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md` | `ARCHIVE_HIST` | Key decision already in CANONICAL_RESEARCH_DIRECTION |
| `docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md` | `ARCHIVE_HIST` | Executed; results in Phase3a doc |
| `docs/thesis/theory/THEORY_STATUS.md` | `MERGE_INTO` | `docs/03_theory.md` |
| `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` | `MERGE_INTO` | `docs/03_theory.md` |
| `docs/thesis/theory/NEXT_THEORY_STEPS.md` | `MERGE_INTO` | `docs/03_theory.md` §open tasks |
| `docs/thesis/theory/THEOREM_A_CONSTANTS.md` | `MERGE_INTO` | `docs/02_experiments.md` §constants |
| `docs/thesis/theory/MDM_THEORY_LANDSCAPE_POSITIONING.md` | `ARCHIVE_HIST` | Superseded by current direction |
| `docs/thesis/next_steps/POST_REASSESSMENT_DECISION.md` | `MERGE_INTO` | `docs/00_current_status.md`, `docs/05_next_steps.md` |
| `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_DECISION.md` | `MERGE_INTO` | `docs/00_current_status.md` §adaptive |
| `docs/thesis/next_steps/NEXT_STEP_REASSESSMENT.md` | `ARCHIVE_HIST` | Superseded by POST_REASSESSMENT_DECISION |
| `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md` | `ARCHIVE_HIST` | Superseded by POST_ADAPTIVE_CONTROLLER_DECISION |
| `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md` | `ARCHIVE_HIST` | Superseded by decisions |
| `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md` | `ARCHIVE_HIST` | Experiment completed; results in protocol_c_owt/ |
| `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md` | `ARCHIVE_HIST` | Decision closed; key outcome in current_status |
| `docs/thesis/next_steps/ZANELLA_MEETING_WRITEUP.md` | `ARCHIVE_HIST` | Historical; key points in current_status |
| `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md` | `ARCHIVE_HIST` | Superseded by POST_ADAPTIVE_CONTROLLER_DECISION |
| `docs/thesis/next_steps/ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md` | `ARCHIVE_HIST` | Old audit; superseded |
| `docs/thesis/next_steps/CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md` | `ARCHIVE_HIST` | Cross-backbone closed |
| `docs/thesis/next_steps/CROSS_BACKBONE_REPLICATION_PLAN.md` | `ARCHIVE_HIST` | Executed; closed |
| `docs/thesis/next_steps/INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md` | `ARCHIVE_HIST` | Historical audit |
| `docs/thesis/next_steps/INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` | `ARCHIVE_HIST` | Historical audit |
| `docs/thesis/next_steps/LARGE_MODEL_CONTINUATION_DECISION.md` | `ARCHIVE_HIST` | Historical decision |
| `docs/thesis/next_steps/NEXT_RESEARCH_DIRECTION_AUDIT.md` | `ARCHIVE_HIST` | Old direction audit |
| `docs/thesis/next_steps/NEXT_RESEARCH_DIRECTION_DECISION.md` | `ARCHIVE_HIST` | Old direction decision |
| `docs/thesis/next_steps/PHASE2B_RESUME_PLAN.md` | `ARCHIVE_HIST` | Executed; Phase 2b complete |
| `docs/thesis/next_steps/PRINCIPLED_NEXT_STEPS_PLAN.md` | `ARCHIVE_HIST` | Superseded by POST_REASSESSMENT_DECISION |
| `docs/thesis/maintenance/CLEANUP_LOG.md` | `ARCHIVE_HIST` | Old maintenance log |
| `docs/thesis/maintenance/DOCS_REORGANIZATION_AUDIT.md` | `ARCHIVE_HIST` | Old audit |
| `docs/thesis/maintenance/DOCS_REORGANIZATION_PLAN.md` | `ARCHIVE_HIST` | Old plan |

---

## docs/ — Active reference docs

| File | Class | Disposition |
|---|---|---|
| `docs/thesis_direction.md` | `MERGE_INTO` | `docs/01_research_direction.md` (scope section) |
| `docs/experimental_infrastructure.md` | `ARCHIVE_HIST` | Key info in docs/02_experiments.md; detailed reference archived |
| `docs/literature_map.md` | `ARCHIVE_HIST` | Reading mostly done; archive as historical reference |
| `docs/reading_plan.md` | `ARCHIVE_HIST` | Reading mostly done; archive as historical reference |
| `docs/future ideas/` | `ARCHIVE_HIST` | Future ideas — archive, not active |

---

## docs/archive/ — Already archived (pre-cleanup)

| Folder | Class | Disposition |
|---|---|---|
| `docs/archive/audits/` | `ARCHIVE_HIST` | Already archived — leave in place |
| `docs/archive/chronicles/` | `ARCHIVE_HIST` | Already archived — leave in place |
| `docs/archive/operational/` | `ARCHIVE_HIST` | Already archived — leave in place |
| `docs/archive/phase1_era/` | `ARCHIVE_HIST` | Already archived — leave in place |
| `docs/archive/status/` | `ARCHIVE_HIST` | Already archived — leave in place |

---

## archive/ (root) — Already archived

| Folder | Class | Disposition |
|---|---|---|
| `archive/` | `ARCHIVE_HIST` | Already archived — leave in place |

---

## research/ — Raw worklog (keep all active)

| File | Class | Disposition |
|---|---|---|
| `research/candidate_theorems.md` | `CURRENT_SOT` | Keep active — detailed theorem record |
| `research/proof_worklog.md` | `CURRENT_SOT` | Keep active — derivation log |
| `research/proof_ledger.md` | `CURRENT_SOT` | Keep active — provenance |
| `research/open_questions.md` | `CURRENT_SOT` | Keep active — open items |
| `research/adaptive_controller_research_notes.md` | `ARCHIVE_HIST` | Superseded by formal Theorem A-ad |

---

## results/ — Raw experiment outputs (all preserve)

| Folder | Class | Disposition |
|---|---|---|
| `results/phase1_proseco_owt_full/` | `RAW_PRESERVE` | Phase 1 Protocol A/B trajectories |
| `results/phase2b_proseco_owt/` | `RAW_PRESERVE` | Phase 2b per-seed raw data |
| `results/phase2b/` | `RAW_PRESERVE` | Phase 2b aggregated analysis outputs |
| `results/phase3a_proseco_owt/` | `RAW_PRESERVE` | Phase 3a CD-G + BS-AG results |
| `results/cross_backbone/` | `RAW_PRESERVE` | LLaDA-SFT bounded probe results |
| `results/protocol_c_owt/` | `RAW_PRESERVE` | Protocol C adaptive controller pilot |

---

## scripts/ — Organized by status

| File | Class | Disposition |
|---|---|---|
| `scripts/run_phase2b_proseco_owt.py` | `KEEP_RUNNABLE` | Phase 2b main runner — keep |
| `scripts/analyze_phase2b.py` | `KEEP_RUNNABLE` | Phase 2b analysis — keep |
| `scripts/run_phase3a_combinatorial.py` | `KEEP_RUNNABLE` | Phase 3a runner — keep |
| `scripts/analyze_phase3a.py` | `KEEP_RUNNABLE` | Phase 3a analysis — keep |
| `scripts/stage_proseco_owt.py` | `KEEP_RUNNABLE` | Reproducibility helper — keep |
| `scripts/run_protocol_c_owt.py` | `KEEP_RUNNABLE` | Protocol C runner — keep |
| `scripts/compute_theorem_a_constants.py` | `KEEP_RUNNABLE` | Constants computation — keep |
| `scripts/analyze_combinatorial_diagnostics.py` | `KEEP_RUNNABLE` | Phase 2b diagnostic — keep |
| `scripts/run_protocol_a_proseco_snapshot.py` | `LEGACY_CODE` | Phase 1 one-off snapshot; done |
| `scripts/debug_proseco_owt_load.py` | `LEGACY_CODE` | Debug script; environment issue fixed |
| `scripts/debug_proseco_llada_sft_load.py` | `LEGACY_CODE` | Debug script; LLaDA-SFT closed |
| `scripts/stage_proseco_llada_sft.py` | `LEGACY_CODE` | LLaDA-SFT staging; cross-backbone closed |
