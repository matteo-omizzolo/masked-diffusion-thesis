# Second-Pass Audit — 2026-05-05

> Companion to INVENTORY.md and MIGRATION_LOG.md.
> Documents findings and actions for the second cleanup pass (root files,
> stale references, research worklog, navigation burden).

---

## Classification of every doc/instruction file

| File | Class | Action |
|---|---|---|
| `START_HERE.md` | `CURRENT_ENTRYPOINT` | Keep; minor stale-ref cleanup |
| `README.md` | `STALE_OR_CONTRADICTORY` | **Full rewrite** — all 6 "read first" links point to archived paths |
| `REPRODUCIBILITY.md` | `DUPLICATE_OF_ACTIVE_DOC` | **Archive** — commands overlap scripts/README.md; one path stale (`results/phase3a/`) |
| `CLAUDE.md` | `CURRENT_ENTRYPOINT` | Keep; remove stale `docs/md/correctors_deep_dive.md` reference |
| `docs/README.md` | `CURRENT_ENTRYPOINT` | Keep; minor cleanup |
| `docs/00_current_status.md` | `CURRENT_ACTIVE_DOC` | Keep |
| `docs/01_research_direction.md` | `CURRENT_ACTIVE_DOC` | Keep; remove stale provenance headers |
| `docs/02_experiments.md` | `CURRENT_ACTIVE_DOC` | Keep; remove stale provenance headers |
| `docs/03_theory.md` | `CURRENT_ACTIVE_DOC` | Keep; remove stale provenance headers |
| `docs/04_results_index.md` | `CURRENT_ACTIVE_DOC` | Keep; add reproducibility note |
| `docs/05_next_steps.md` | `CURRENT_ACTIVE_DOC` | Keep; add reassessment scaffold |
| `docs/archive/README.md` | `CURRENT_ACTIVE_DOC` | Keep |
| `research/candidate_theorems.md` | `CURRENT_SUPPORTING_REFERENCE` | Keep; add brief banner |
| `research/proof_worklog.md` | `CURRENT_SUPPORTING_REFERENCE` | Keep; add brief banner |
| `research/proof_ledger.md` | `CURRENT_SUPPORTING_REFERENCE` | Keep; add brief banner |
| `research/open_questions.md` | `STALE_OR_CONTRADICTORY` | **Replace** — 416 lines, most questions answered; refs stale paths |
| `scripts/README.md` | `RAW_RESULT_OR_CODE_INDEX` | Keep; minor improvement |
| `hpc/README.md` | `CURRENT_SUPPORTING_REFERENCE` | Keep; remove REPRODUCIBILITY.md reference |
| `archive/ARCHIVE_MANIFEST.md` | `ARCHIVE_HISTORICAL` | Add README to root archive/ explaining it's historical |
| `.claude/settings.local.json` | `CURRENT_SUPPORTING_REFERENCE` | Keep — permissions config |

---

## Stale references found in active files

| File | Stale reference | Action |
|---|---|---|
| `README.md` | `docs/thesis/CURRENT_INDEX.md` and 5 other archived paths | Rewrite README |
| `README.md` | `docs/thesis/` listed as "canonical documentation" | Rewrite README |
| `README.md` | `results/phase3a/` (folder is `results/phase3a_proseco_owt/`) | Rewrite README |
| `REPRODUCIBILITY.md` | `results/phase3a/` (stale path) | Archive REPRODUCIBILITY.md |
| `CLAUDE.md` | `docs/md/correctors_deep_dive.md` ("historically useful") | Remove line |
| `hpc/README.md` | "See REPRODUCIBILITY.md for setup" | Update to point to scripts/README.md |
| `docs/01_research_direction.md` | provenance header citing archived paths | Remove provenance header |
| `docs/02_experiments.md` | provenance header citing archived paths | Remove provenance header |
| `docs/03_theory.md` | provenance header citing archived paths | Remove provenance header |
| `docs/05_next_steps.md` | provenance header citing archived path | Remove provenance header |
| `research/open_questions.md` | `docs/experiments/entropy_proxy_experiment.md` (archived) | Replace file |
| `research/open_questions.md` | `docs/thesis/theory/THEORY_STRESS_TEST.md` (archived) | Replace file |
| `research/open_questions.md` | `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` (archived) | Replace file |
| `research/open_questions.md` | `docs/gpt_pro_assessment_response.md` (archived) | Replace file |

---

## Consistency check (active docs)

All active docs agree on:
- Thesis story: rankers fail, search works, combinatorial scheduling is the right class ✓
- No new HPC authorized ✓
- Theory complete (Theorem A + A′ + A″ + Corollary formally proved) ✓
- Critical path: LaTeX writing ✓
- Backbone: ProSeCo-OWT only ✓
- Archive rule: same in all docs ✓

Inconsistency fixed:
- `README.md` disagreed on entry points (pointed to deleted files). **Rewritten.**
- Provenance headers in docs/01-03+05 cited archived paths as if still active. **Removed.**
