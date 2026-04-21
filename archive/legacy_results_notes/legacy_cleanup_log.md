# Legacy Cleanup Log

**Date:** April 2026
**Reason:** Repository refocused from broad "informed correctors" framing to precise
"signal-adaptive corrector scheduling" thesis question.

---

## Archived Files

| File | Moved to | Reason |
|------|----------|--------|
| `docs/md/research_directions.md` | `archive/old_directions/research_directions_march2026.md` | Reflected earlier broad framing (3 directions, Riemann/MCMC unification). Content is historically useful but no longer aligned. Deprecation banner added to original. |
| `docs/md/correctors_deep_dive.md` | `archive/legacy/correctors_deep_dive.md` | 58KB deep dive on corrector mechanics from March 2026. Useful reference but superseded by the narrower scheduling focus. Deprecation banner added. |
| `docs/pdf/theory/correctors_deep_dive.pdf` | `archive/legacy/correctors_deep_dive.pdf` | PDF export of the above. |
| `docs/pdf/theory/research_directions.pdf` | `archive/legacy/research_directions.pdf` | PDF export of the old research directions. |
| `docs/agent_prompts/search_informed_correctors.md` | `archive/old_notes/search_informed_correctors.md` | One-time search prompt; no longer needed. |
| `tasks/todo.md` | `archive/old_notes/todo_march2026.md` | Old step-sweep-era TODO list (March 2026). Completed items and stale priorities. |

## Files with Deprecation Banners (Kept in Place)

| File | Action | Reason |
|------|--------|--------|
| `docs/md/research_directions.md` | Deprecation banner added | Still referenced; contains useful mathematical content on E_fact and spectral gaps |
| `docs/md/correctors_deep_dive.md` | Deprecation banner added | Contains detailed corrector mechanics that may be referenced |

## Files Kept As-Is

| File | Reason |
|------|--------|
| `docs/md/research_plan.md` | Partially superseded but contains valuable Gap B/C + E proof sketches and infrastructure status. Will be superseded by new docs over time. |
| `docs/md/papers.md` | Paper inventory follows `study/papers/` folder structure; still accurate. |
| `tasks/lessons.md` | HPC lessons learned; still relevant. |
| `notebooks/01_spectral_gap_and_mixing.ipynb` | Educational; may be useful for thesis background. |
| `notebooks/02_discretization_error_and_correctors.ipynb` | Educational; may be useful. |
| `study/` | All papers and notes preserved. |
| `external/` | All upstream repos preserved (remdm, mdlm, PRISM, sedd, remedi). |
| `src/mdm_playground/` | Code package preserved; may need refactoring later. |
| `scripts/` | Analysis scripts preserved. |
| `configs/` | Experiment configs preserved. |
| `hpc/` | HPC workflow scripts preserved. |
| `figures/` | Generated plots preserved (step-sweep results are archived but reproducible). |

## New Files Created

| File | Purpose |
|------|---------|
| `docs/thesis_direction.md` | Precise research question, scope, non-goals, open-question verdict |
| `docs/literature_map.md` | Categorized paper map with gap analysis |
| `docs/reading_plan.md` | Prioritized reading list with status tags |
| `docs/experimental_infrastructure.md` | Repos, checkpoints, setup status, manifest |
| `docs/implementation_plan.md` | Phased experiment roadmap with logging standards |
| `docs/legacy_cleanup_log.md` | This file |
| `research/proof_worklog.md` | Mathematical worklog |
| `research/candidate_theorems.md` | Candidate theorem statements |
| `research/proof_ledger.md` | Provenance tracking for proof ingredients |
| `research/open_questions.md` | Unresolved technical points |
| `README.md` | Rewritten for signal-adaptive corrector scheduling focus |

## Directories Created

| Directory | Purpose |
|-----------|---------|
| `archive/legacy/` | Historically useful but superseded material |
| `archive/old_directions/` | Abandoned thesis direction documents |
| `archive/old_notes/` | Stale notes and one-time prompts |
| `research/` | Mathematical work package |

## Not Deleted

Conservative policy: no files were permanently deleted. Everything was either kept in
place, deprecated in place with a banner, or copied to `archive/`. The originals of
deprecated files remain in their original locations for backward compatibility with any
existing cross-references.
