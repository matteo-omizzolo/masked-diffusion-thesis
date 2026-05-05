# Docs — Structure Guide

> **Current source of truth.** Updated 2026-05-05 as part of repo cleanup.

## Active documents (only these)

| File | Purpose |
|---|---|
| `../START_HERE.md` | 2-minute orientation — start here |
| `00_current_status.md` | What's done, what's established, what failed, risks |
| `01_research_direction.md` | Thesis Q, scope, non-goals, contribution |
| `02_experiments.md` | Phases, protocols, results, reliability classification |
| `03_theory.md` | Theorem stack, proof status, open gaps |
| `04_results_index.md` | Raw results index — folder → phase → status |
| `05_next_steps.md` | Sequential action plan — theory gates → Phase 0 → writing |
| `06_theory_first_research_plan.md` | **Active planning doc** — theory-first programme under feasibility assessment; not final thesis status |

Supporting (authoritative detailed records — do not summarize, leave as-is):
- `../research/candidate_theorems.md` — full theorem and proof record
- `../research/proof_worklog.md` — derivation entries
- `../research/proof_ledger.md` — provenance tracking
- `../research/open_questions.md` — unresolved technical questions
- `../CLAUDE.md` — HPC, environment, repo layout for Claude Code sessions

---

## RULE: Archive files are historical only

> **Only the files listed above are current sources of truth.**
> Files under `docs/archive/`, `archive/`, and `docs/thesis/` are historical
> and must NOT be used to infer current thesis status unless the user
> explicitly requests historical context. Archived files may contradict
> current docs because they predate key experimental results and decisions.

---

## Archive structure

```
docs/archive/
  README.md                         — archive overview
  repo_cleanup_20260505/
    INVENTORY.md                    — full classification of every file
    MIGRATION_LOG.md                — every move/archive action logged
    ARCHIVE_MANIFEST.md             — what is in each archive folder
    old_canonical_docs/             — former canonical docs (CANONICAL_*, CURRENT_INDEX)
    old_next_steps/                 — former next-steps decision docs
    old_theory_docs/                — former theory status docs
    old_experiment_docs/            — former experiment plans and audits
    old_plans/                      — old plans and implementation docs
    old_audits/                     — old audit docs
    old_operational/                — old operational / sync docs
    old_reading_notes/              — old reading plans and literature notes
    old_maintenance/                — old maintenance logs
  audits/                           — pre-cleanup audit archive (2026-04)
  chronicles/                       — pre-cleanup results chronicle (2026-04)
  operational/                      — pre-cleanup operational docs (2026-04)
  phase1_era/                       — pre-cleanup Phase 1 experiment docs
  status/                           — pre-cleanup status reports

archive/ (root)                     — legacy docs from before April 2026
```

---

## Script structure

```
scripts/
  README.md                         — script index
  run_phase2b_proseco_owt.py        — Phase 2b runner
  analyze_phase2b.py                — Phase 2b analysis
  run_phase3a_combinatorial.py      — Phase 3a runner
  analyze_phase3a.py                — Phase 3a analysis
  run_protocol_c_owt.py             — Protocol C runner (CPU, adaptive controller)
  stage_proseco_owt.py              — reproducibility helper
  compute_theorem_a_constants.py    — Theorem A constants
  analyze_combinatorial_diagnostics.py — Phase 2b combinatorial diagnostics
  legacy/                           — debug and one-off scripts (superseded)
```
