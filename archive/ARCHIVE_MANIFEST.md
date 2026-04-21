# Archive Manifest

Last updated: 2026-04-21 (legacy-framework cleanup pass).

Everything below was moved — not deleted. Originals preserved under `archive/`.
Read the "Historical value?" column to decide whether you can safely ignore a
file or whether it still contains context worth revisiting.

## Legend

- **Safe to ignore?** "yes" = the file is superseded by a canonical doc; "no" = still useful.
- **Historical value?** "yes" = material provenance that documents how the direction evolved; "low" = one-off scratch.

---

## April 2026 cleanup pass (thesis-mainline cleanup)

### Docs — superseded by canonical thesis structure

| Original path | Archived path | Reason | Safe to ignore? | Historical value? |
|---|---|---|---|---|
| `docs/CURRENT_INDEX.md` | `archive/legacy_docs/OLD_CURRENT_INDEX.md` | Replaced by `docs/thesis/CURRENT_INDEX.md` | yes | yes (shows April-17 state) |
| `docs/md/correctors_deep_dive.md` (1264 lines) | `archive/legacy_docs/md/correctors_deep_dive.md` | Older deep-dive study doc; deprecated banner already in file | yes | yes (theoretical context) |
| `docs/md/research_directions.md` (1346 lines) | `archive/legacy_docs/md/research_directions.md` | March 2026 direction; superseded by `docs/thesis_direction.md` | yes | yes (March 2026 framing) |
| `docs/md/research_plan.md` | `archive/legacy_docs/md/research_plan.md` | Partially superseded; chapter structure still useful | no | yes |
| `docs/md/papers.md` | `archive/legacy_docs/md/papers.md` | Older paper list; see `docs/literature_map.md` | yes | low |
| `docs/pdf/background/` | `archive/legacy_docs/pdf/background/` | Empty background-PDF dir | yes | no |
| `docs/pdf/theory/` | `archive/legacy_docs/pdf/theory/` | Old PDF versions of deep-dive / research-direction docs | yes | low |
| `docs/phase2_changes_report.md` | `archive/legacy_results_notes/phase2_changes_report.md` | One-off change report; superseded by canonical experiment docs | yes | low |
| `docs/legacy_cleanup_log.md` | `archive/legacy_results_notes/legacy_cleanup_log.md` | Older cleanup log; this manifest supersedes it | yes | yes (pre-cleanup history) |
| `texput.log` | `archive/legacy_results_notes/texput.log` | Stray LaTeX run artifact | yes | no |

### Prompts and assessments — moved as a batch for provenance

| Original path | Archived path | Reason | Safe to ignore? | Historical value? |
|---|---|---|---|---|
| `docs/instructions/v1/MSc_thesis_direction_brief.md` | `archive/legacy_prompts/instructions/v1/…` | One-off prompt output | yes | yes (original brief) |
| `docs/instructions/v1/UPLOAD_INSTRUCTIONS.md` | `archive/legacy_prompts/instructions/v1/…` | Setup instructions | yes | no |
| `docs/instructions/v1/claude_code_math_analysis_prompt.md` | `archive/legacy_prompts/instructions/v1/…` | One-off prompt | yes | low |
| `docs/instructions/v1/claude_code_repo_update_prompt.md` | `archive/legacy_prompts/instructions/v1/…` | One-off prompt | yes | low |
| `docs/instructions/v2/gpt_pro_assessment.md` | `archive/legacy_prompts/instructions/v2/…` | GPT Pro v2 raw assessment | no | yes (seeds Theorem A) |
| `docs/instructions/v2/gpt_pro_experiment_design.md` | `archive/legacy_prompts/instructions/v2/…` | GPT Pro v2 experiment design | no | yes |
| `docs/instructions/v2/gpt_pro_theory_plan.md` | `archive/legacy_prompts/instructions/v2/…` | GPT Pro v2 theory plan | no | yes |
| `docs/gpt_pro_assessment_response.md` | `archive/legacy_prompts/gpt_pro_assessment_response.md` | Per-item response to GPT Pro v2 | no | yes |

### Proseco chronicle — chronological audit fragments, superseded by RESULTS_STATUS.md + HPC_NEXT_RUN_PLAN.md

| Original path | Archived path | Reason | Safe to ignore? | Historical value? |
|---|---|---|---|---|
| `docs/experiments/results/experiment_result_analysis.md` | `archive/legacy_results_notes/proseco_chronicle/…` | Initial MDLM analysis; superseded | yes | yes |
| `docs/experiments/results/fast_sanity_run.md` | `archive/legacy_results_notes/proseco_chronicle/…` | Surrogate sanity notes | yes | low |
| `docs/experiments/results/post_experiment_conclusion.md` | `archive/legacy_results_notes/proseco_chronicle/…` | Post-MDLM conclusion; superseded | yes | yes |
| `docs/experiments/results/post_fix_validation_checks.md` | `archive/legacy_results_notes/proseco_chronicle/…` | Fix validation notes | yes | low |
| `docs/experiments/results/pre_fix_pipeline_audit.md` | `archive/legacy_results_notes/proseco_chronicle/…` | Pre-fix pipeline audit | yes | low |
| `docs/experiments/results/pre_submission_proseco_check.md` | `archive/legacy_results_notes/proseco_chronicle/…` | Pre-submission preflight | yes | low |
| `docs/experiments/results/proseco_fast_sanity_run.md` | `archive/legacy_results_notes/proseco_chronicle/…` | ProSeCo surrogate sanity | yes | low |
| `docs/experiments/results/proseco_load_validation.md` | `archive/legacy_results_notes/proseco_chronicle/…` | Load validation | yes | low |
| `docs/experiments/results/proseco_preflight_checks.md` | `archive/legacy_results_notes/proseco_chronicle/…` | Preflight checks list | yes | yes (guard-rail history) |

### Kept in place — still canonical data references

| Path | Status | Reason |
|---|---|---|
| `docs/experiments/results/proseco_backend_failure_audit.md` | KEPT | Root-cause audit of jobs 478826/827/828 — cited by `RESULTS_STATUS.md` |
| `docs/experiments/results/result_inventory.md` | KEPT | Source of truth for MDLM 478600 + ProSeCo 478929; cited by canonical docs |
| `docs/experiments/entropy_proxy_experiment.md` | KEPT | Protocol A/B specification (still structural reference) |
| `docs/experiments/implementation_status.md` | KEPT | Scheduling-package API doc |
| `docs/experiments/phase1_interpretation.md` | KEPT | MDLM diagnostic interpretation |
| `docs/experiments/proseco_backend_audit.md` | KEPT | Backend audit reference |
| `docs/experiments/proseco_experiment_definition.md` | KEPT | Mathematical spec |
| `docs/experiments/proseco_import_audit.md` | KEPT | Import audit history |
| `docs/experiments/proseco_protocol_mapping.md` | KEPT | Code-level Protocol A/B mapping |
| `docs/experiments/proseco_validation_checklist.md` | KEPT | Active validation checklist |

### Documentation structure moves (not archive, just reorganisation)

| Original path | New path | Reason |
|---|---|---|
| `docs/maintenance/hpc_sync_inventory.md` | `docs/thesis/maintenance/hpc_sync_inventory.md` | Move under thesis/maintenance/ |
| `docs/maintenance/hpc_sync_report.md` | `docs/thesis/maintenance/hpc_sync_report.md` | Move under thesis/maintenance/ |

### Scripts / HPC / results — moved into the legacy framework bucket

| Original path | Archived path | Reason | Safe to ignore? | Historical value? |
|---|---|---|---|---|
| `src/mdm_playground/cli/` | `archive/legacy_framework/src/mdm_playground/cli/` | Generic multi-method inference CLI; not thesis mainline | yes | low |
| `src/mdm_playground/core/` | `archive/legacy_framework/src/mdm_playground/core/` | Generic utilities used by the old framework | yes | low |
| `src/mdm_playground/models/` | `archive/legacy_framework/src/mdm_playground/models/` | Adapter wrappers for RemeDi / ReMDM / PRISM | yes | yes |
| `src/mdm_playground/samplers/` | `archive/legacy_framework/src/mdm_playground/samplers/` | Shared sampler framework for the old project | yes | low |
| `src/mdm_playground/strategies/` | `archive/legacy_framework/src/mdm_playground/strategies/` | Old strategy abstraction for the generic framework | yes | low |
| `scripts/run_phase1_*.py` | `archive/legacy_framework/scripts/` | Phase 1 exploratory runners | yes | yes |
| `scripts/analyze_phase1.py` | `archive/legacy_framework/scripts/` | Phase 1 figure generation | yes | yes |
| `scripts/analyze_phase2a.py` | `archive/legacy_framework/scripts/` | Phase 2a offline re-analysis | yes | yes |
| `scripts/analyze_inverted_policies.py` | `archive/legacy_framework/scripts/` | Old inverted-policy experiment | yes | low |
| `scripts/debug_proseco_load.py` | `archive/legacy_framework/scripts/` | Legacy ProSeCo preflight | yes | low |
| `scripts/debug_mdlm_conf_load.py` | `archive/legacy_framework/scripts/` | Legacy MDLM-conf preflight | yes | low |
| `scripts/stage_owt_reference.py` | `archive/legacy_framework/scripts/` | MAUVE reference staging helper for the old branch | yes | low |
| `hpc/phase1_*.sbatch` | `archive/legacy_framework/hpc/` | Phase 1 job scripts | yes | yes |
| `hpc/submit.sh` | `archive/legacy_framework/hpc/` | Old multi-strategy submit wrapper | yes | low |
| `results/phase1_*` (except `results/phase1_proseco_owt_full/`) | `archive/legacy_framework/results/` | Phase 1 pilot outputs | yes | yes |
| `results/phase2a/` | `archive/legacy_framework/results/` | Phase 2a offline gate outputs | yes | yes |
| `results/full_eval/`, `results/sweep/`, `results/t1000_eval/`, `results/remdm_smoke/` | `archive/legacy_framework/results/` | Early step-sweep / smoke results | yes | low |
| `figures/phase1_*`, `figures/phase2a/` | `archive/legacy_framework/figures/` | Legacy figures from the old framework | yes | yes |
| `out/`, `err/` | `archive/logs/` | Root job logs moved out of the public main path | yes | low |

---

## Pre-April 2026 archive (already in place)

### `archive/legacy/` — 2025 / early 2026 artifacts

| Path | Notes |
|---|---|
| `archive/legacy/correctors_deep_dive.md` | 1260-line precursor to the April 2026 docs |
| `archive/legacy/correctors_deep_dive.pdf` | PDF version |
| `archive/legacy/research_directions.pdf` | PDF version of the March 2026 direction |

### `archive/old_directions/`

| Path | Notes |
|---|---|
| `archive/old_directions/research_directions_march2026.md` | March 2026 direction; superseded by the April 2026 thesis-direction doc |

### `archive/old_notes/`

| Path | Notes |
|---|---|
| `archive/old_notes/search_informed_correctors.md` | Early literature scan notes |
| `archive/old_notes/todo_march2026.md` | March todo list |

---

## HPC artifacts — intentionally NOT archived

The `out/`, `err/`, and `logs/` directories contain per-job stdout/stderr from
completed and failed HPC runs. These are preserved at the repo root (not under
archive/) because:

- They are tied to specific SLURM job IDs in the chronicle docs.
- They are evidence for the key-results tables in `docs/experiments/results/result_inventory.md` and `docs/thesis/experiments/RESULTS_STATUS.md`.
- They are small enough to keep in repo history without bloat.

Jobs referenced by the canonical docs:
- 478600 — MDLM heuristic pilot (diagnostic negative)
- 478826 / 478827 / 478828 — ProSeCo failed loads (root cause in `proseco_backend_failure_audit.md`)
- 478929 — ProSeCo pilot completed (structural no-op, all zeros)
- 478962 — MDLM-conf pilot completed (n_positive_delta_steps = 25/64; see `RESULTS_STATUS.md`)

Earlier `remdm_smoke_*` / `remdm_eval_*` / `remdm_sweep_*` jobs predate Phase 1
and can be ignored for the current experimental plan.
