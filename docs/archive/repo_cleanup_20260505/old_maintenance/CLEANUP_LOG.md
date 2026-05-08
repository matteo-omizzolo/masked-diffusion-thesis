> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


# Cleanup Log

## 2026-04-18 — Major cleanup and consolidation pass

### What was consolidated

Created `docs/thesis/` as the canonical thesis-docs subtree. New authoritative files:

- `docs/thesis/CURRENT_INDEX.md`
- `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
- `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md`
- `docs/thesis/theory/THEORY_STATUS.md`
- `docs/thesis/theory/NEXT_THEORY_STEPS.md`
- `docs/thesis/experiments/RESULTS_STATUS.md`
- `docs/thesis/experiments/HPC_NEXT_RUN_PLAN.md`
- `docs/thesis/experiments/PROSECO_PREFLIGHT_STATUS.md`
- `docs/thesis/maintenance/CLEANUP_LOG.md` (this file)

### What was archived

See `archive/ARCHIVE_MANIFEST.md` for the full move log. Groups:

- `docs/instructions/` (prompts v1/v2) → `archive/legacy_prompts/instructions/`
- `docs/md/` (older study docs) → `archive/legacy_docs/md/`
- `docs/pdf/` → `archive/legacy_docs/pdf/`
- `docs/phase2_changes_report.md` → `archive/legacy_results_notes/`
- `docs/gpt_pro_assessment_response.md` → `archive/legacy_prompts/`
- `docs/legacy_cleanup_log.md` → `archive/legacy_results_notes/`
- `docs/CURRENT_INDEX.md` → `archive/legacy_docs/OLD_CURRENT_INDEX.md`
- `docs/experiments/results/` 9 fragment files → `archive/legacy_results_notes/proseco_chronicle/`
- `docs/maintenance/*.md` → `docs/thesis/maintenance/`
- `texput.log` → `archive/legacy_results_notes/`

### What was kept in place

Top-level entry points kept at root: `README.md`, `CLAUDE.md`.

Docs kept in place (still canonical references):

- `docs/thesis_direction.md` (April 2026 direction — source for canonical research-direction file)
- `docs/implementation_plan.md` (source for HPC_NEXT_RUN_PLAN.md)
- `docs/experimental_infrastructure.md`
- `docs/literature_map.md`
- `docs/reading_plan.md`
- `docs/experiments/*.md` (protocol references)
- `docs/experiments/results/result_inventory.md` (canonical run inventory)
- `docs/experiments/results/proseco_backend_failure_audit.md` (root-cause audit)

Research log kept intact (chronological, untouched):

- `research/candidate_theorems.md`
- `research/proof_worklog.md`
- `research/proof_ledger.md`
- `research/open_questions.md`

Code, tests, external dependencies, HPC artifacts, results — all untouched.

### Notes on the MDLM-conf pilot discovery

While writing the new canonical experiment status docs, it was confirmed that
`results/phase1_mdlm_conf/` already contains a completed run from job 478962
(2026-04-17, N=20, T=64) with non-trivial results:

- n_positive_delta_steps = 25/64
- peak_mean_delta ≈ 0.093
- ε (entropy/margin/quality) RMS ≈ 0.222
- Spearman(s_t, Δ_t) ≈ 0.033 (signals barely predict Δ_t)
- η_B: 0.49 / 1.03 / 0.71 for B = 4 / 8 / 16
- γ_95 = 0.456
- All Theorem A bounds vacuous (2Bε + 2η_95 >> G_oracle)

This run is treated as a valid Phase 1 pilot — the first experiment in this
project that produced a non-trivial Δ_t profile. The low Spearman is the
next methodological problem to solve (see `HPC_NEXT_RUN_PLAN.md` and
`PROSECO_PREFLIGHT_STATUS.md`).

### Not deleted — nothing

No files were deleted in this pass. Everything is recoverable via
`archive/`. Re-instate anything by reversing the move listed in
`archive/ARCHIVE_MANIFEST.md`.

---

## 2026-04-18 — Signal-aligned MDLM-conf run (job 479257) complete

- `results/phase1_mdlm_conf_signal_aligned/` pulled from HPC. M2 fix confirmed in
  `protocol_a/trajectory_0.json` (n_action=20 at every step).
- `results/phase1_mdlm_conf/` (job 478962, M2-buggy) retained for reference — do not cite.
- `RESULTS_STATUS.md §9` written with full §7.1 PASS/FAIL table.
- `CURRENT_INDEX.md §8` updated to point to signal-aligned result path.
- `CURRENT_INDEX.md §12` updated: next action is proseco-owt staging (MDLM-conf Spearman
  remains noise-level; scaling N=50 would not change the conclusion).
