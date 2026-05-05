# Migration Log — repo_cleanup_20260505

> Created 2026-05-05 on branch `repo-cleanup-compact-current-state`.
> Records every file action taken during the compact-repo cleanup.

---

## New files created

| New file | Purpose | Content source |
|---|---|---|
| `START_HERE.md` | 2-minute orientation | Synthesized from CURRENT_INDEX + CANONICAL_RESEARCH_DIRECTION + POST_REASSESSMENT_DECISION |
| `docs/README.md` | Doc structure guide + archive rule | New (replaces scattered index attempts) |
| `docs/00_current_status.md` | Established results, failures, risks | Synthesized from POST_REASSESSMENT_DECISION + POST_ADAPTIVE_CONTROLLER_DECISION + NEXT_STEP_REASSESSMENT |
| `docs/01_research_direction.md` | Thesis Q, scope, non-goals, contribution | Synthesized from CANONICAL_RESEARCH_DIRECTION + thesis_direction.md |
| `docs/02_experiments.md` | Phases, protocols, results, scripts | Synthesized from CANONICAL_EXPERIMENT_OVERVIEW + PHASE3A_COMBINATORIAL_RESULTS + CROSS_BACKBONE_REPLICATION_RESULTS + ANALYSIS_SPEC + THEOREM_A_CONSTANTS |
| `docs/03_theory.md` | Theorem stack, proof status, open gaps | Synthesized from THEORY_STATUS + ADAPTIVE_BUDGETED_CONTROLLERS + NEXT_THEORY_STEPS |
| `docs/04_results_index.md` | Raw results folder index | Synthesized from CANONICAL_EXPERIMENT_OVERVIEW + RESULTS_STATUS + INVENTORY |
| `docs/05_next_steps.md` | Action plan (write thesis) | Synthesized from POST_REASSESSMENT_DECISION §3–4 |
| `docs/archive/README.md` | Archive overview | New |
| `docs/archive/repo_cleanup_20260505/INVENTORY.md` | Full file classification | New |
| `docs/archive/repo_cleanup_20260505/MIGRATION_LOG.md` | This file | New |
| `docs/archive/repo_cleanup_20260505/ARCHIVE_MANIFEST.md` | Archived folder contents | New |
| `scripts/README.md` | Script index | New |

---

## Files archived (moved to docs/archive/repo_cleanup_20260505/)

### old_canonical_docs/
| Old path | Archive path | Reason |
|---|---|---|
| `docs/thesis/CURRENT_INDEX.md` | `old_canonical_docs/CURRENT_INDEX.md` | Superseded by START_HERE.md + docs/README.md |
| `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` | `old_canonical_docs/CANONICAL_RESEARCH_DIRECTION.md` | Superseded by docs/01_research_direction.md |
| `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | `old_canonical_docs/CANONICAL_EXPERIMENT_OVERVIEW.md` | Superseded by docs/02_experiments.md |

### old_experiment_docs/
| Old path | Archive path | Reason |
|---|---|---|
| `docs/thesis/experiments/ANALYSIS_SPEC.md` | `old_experiment_docs/ANALYSIS_SPEC.md` | Key info in docs/02_experiments.md |
| `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` | `old_experiment_docs/PHASE3A_COMBINATORIAL_RESULTS.md` | Key numbers in docs/02_experiments.md |
| `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` | `old_experiment_docs/CROSS_BACKBONE_REPLICATION_RESULTS.md` | Key numbers in docs/02_experiments.md |
| `docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md` | `old_experiment_docs/PHASE3_DIRECTION_AUDIT.md` | Decision already captured |
| `docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md` | `old_experiment_docs/PHASE3_ALTERNATIVE_PLAN.md` | Plan executed; superseded |

### old_theory_docs/
| Old path | Archive path | Reason |
|---|---|---|
| `docs/thesis/theory/THEORY_STATUS.md` | `old_theory_docs/THEORY_STATUS.md` | Superseded by docs/03_theory.md |
| `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` | `old_theory_docs/ADAPTIVE_BUDGETED_CONTROLLERS.md` | Key info in docs/03_theory.md |
| `docs/thesis/theory/NEXT_THEORY_STEPS.md` | `old_theory_docs/NEXT_THEORY_STEPS.md` | Key tasks in docs/03_theory.md + 05_next_steps.md |
| `docs/thesis/theory/THEOREM_A_CONSTANTS.md` | `old_theory_docs/THEOREM_A_CONSTANTS.md` | Constants table in docs/02_experiments.md |
| `docs/thesis/theory/MDM_THEORY_LANDSCAPE_POSITIONING.md` | `old_theory_docs/MDM_THEORY_LANDSCAPE_POSITIONING.md` | Superseded by current direction |
| `research/adaptive_controller_research_notes.md` | `old_theory_docs/adaptive_controller_research_notes.md` | Superseded by Theorem A-ad |

### old_next_steps/
| Old path | Archive path | Reason |
|---|---|---|
| `docs/thesis/next_steps/POST_REASSESSMENT_DECISION.md` | `old_next_steps/POST_REASSESSMENT_DECISION.md` | Terminal decision; captured in docs/00+05 |
| `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_DECISION.md` | `old_next_steps/POST_ADAPTIVE_CONTROLLER_DECISION.md` | Decision captured in docs/00_current_status.md |
| `docs/thesis/next_steps/NEXT_STEP_REASSESSMENT.md` | `old_next_steps/NEXT_STEP_REASSESSMENT.md` | Superseded by POST_REASSESSMENT_DECISION |
| `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md` | `old_next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md` | Superseded |
| `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md` | `old_next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md` | Superseded |
| `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md` | `old_next_steps/ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md` | Experiment complete |
| `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md` | `old_next_steps/POST_CROSS_BACKBONE_DECISION.md` | Decision captured |
| `docs/thesis/next_steps/ZANELLA_MEETING_WRITEUP.md` | `old_next_steps/ZANELLA_MEETING_WRITEUP.md` | Historical |
| `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md` | `old_next_steps/POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md` | Superseded |
| `docs/thesis/next_steps/ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md` | `old_next_steps/ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md` | Old audit |
| `docs/thesis/next_steps/CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md` | `old_next_steps/CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md` | Closed |
| `docs/thesis/next_steps/CROSS_BACKBONE_REPLICATION_PLAN.md` | `old_next_steps/CROSS_BACKBONE_REPLICATION_PLAN.md` | Executed |
| `docs/thesis/next_steps/INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md` | `old_next_steps/INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md` | Historical |
| `docs/thesis/next_steps/INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` | `old_next_steps/INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` | Historical |
| `docs/thesis/next_steps/LARGE_MODEL_CONTINUATION_DECISION.md` | `old_next_steps/LARGE_MODEL_CONTINUATION_DECISION.md` | Historical |
| `docs/thesis/next_steps/NEXT_RESEARCH_DIRECTION_AUDIT.md` | `old_next_steps/NEXT_RESEARCH_DIRECTION_AUDIT.md` | Old |
| `docs/thesis/next_steps/NEXT_RESEARCH_DIRECTION_DECISION.md` | `old_next_steps/NEXT_RESEARCH_DIRECTION_DECISION.md` | Old |
| `docs/thesis/next_steps/PHASE2B_RESUME_PLAN.md` | `old_next_steps/PHASE2B_RESUME_PLAN.md` | Executed |
| `docs/thesis/next_steps/PRINCIPLED_NEXT_STEPS_PLAN.md` | `old_next_steps/PRINCIPLED_NEXT_STEPS_PLAN.md` | Superseded |

### old_maintenance/
| Old path | Archive path | Reason |
|---|---|---|
| `docs/thesis/maintenance/CLEANUP_LOG.md` | `old_maintenance/CLEANUP_LOG.md` | Historical |
| `docs/thesis/maintenance/DOCS_REORGANIZATION_AUDIT.md` | `old_maintenance/DOCS_REORGANIZATION_AUDIT.md` | Historical |
| `docs/thesis/maintenance/DOCS_REORGANIZATION_PLAN.md` | `old_maintenance/DOCS_REORGANIZATION_PLAN.md` | Historical |

### old_reading_notes/
| Old path | Archive path | Reason |
|---|---|---|
| `docs/thesis_direction.md` | `old_reading_notes/thesis_direction.md` | Merged into docs/01_research_direction.md |
| `docs/experimental_infrastructure.md` | `old_reading_notes/experimental_infrastructure.md` | Key info in docs/02_experiments.md |
| `docs/reading_plan.md` | `old_reading_notes/reading_plan.md` | Reading mostly complete |
| `docs/literature_map.md` | `old_reading_notes/literature_map.md` | Reference; archived |

### old_plans/
| Old path | Archive path | Reason |
|---|---|---|
| `docs/future ideas/Deep Research Audit...md` | `old_plans/Deep Research Audit...md` | Premature; not on critical path |
| `docs/future ideas/Theoretical Frameworks...md` | `old_plans/Theoretical Frameworks...md` | Premature; not on critical path |

---

## Scripts moved to legacy

| Old path | New path | Reason |
|---|---|---|
| `scripts/run_protocol_a_proseco_snapshot.py` | `scripts/legacy/run_protocol_a_proseco_snapshot.py` | Phase 1 one-off; done |
| `scripts/debug_proseco_owt_load.py` | `scripts/legacy/debug_proseco_owt_load.py` | Debug; environment issue fixed |
| `scripts/debug_proseco_llada_sft_load.py` | `scripts/legacy/debug_proseco_llada_sft_load.py` | Debug; LLaDA-SFT closed |
| `scripts/stage_proseco_llada_sft.py` | `scripts/legacy/stage_proseco_llada_sft.py` | LLaDA-SFT staging; cross-backbone closed |

---

## Files NOT moved (preserved as-is)

| File / Folder | Class | Reason |
|---|---|---|
| `results/` | RAW_PRESERVE | All raw experiment outputs — never touch |
| `research/candidate_theorems.md` | CURRENT_SOT | Detailed theorem worklog — keep active |
| `research/proof_worklog.md` | CURRENT_SOT | Derivation entries — keep active |
| `research/proof_ledger.md` | CURRENT_SOT | Provenance — keep active |
| `research/open_questions.md` | CURRENT_SOT | Open items — keep active |
| `thesis/` | CURRENT_SOT | LaTeX chapters — untouched |
| `src/` | CURRENT_SOT | Source code — untouched |
| `notebooks/` | CURRENT_SOT | Background notebooks — keep |
| `hpc/` | CURRENT_SOT | HPC scripts — untouched |
| `external/` | CURRENT_SOT | Upstream repos — untouched |
| `docs/archive/` (pre-existing) | ARCHIVE_HIST | Already archived before this cleanup |
| `archive/` (root) | ARCHIVE_HIST | Already archived before this cleanup |
| `CLAUDE.md` | UPDATED | Updated to point to new entry points |
