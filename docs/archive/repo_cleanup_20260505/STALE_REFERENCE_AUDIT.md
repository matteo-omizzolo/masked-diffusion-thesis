# Stale Reference Audit — Second Pass (2026-05-05)

> Records every stale reference found in active files and the action taken.

---

| File | Stale reference | Severity | Action |
|---|---|---|---|
| `README.md` | 6 links to `docs/thesis/CURRENT_INDEX.md`, `CANONICAL_*`, `RESULTS_STATUS.md`, etc. | CRITICAL — all dead links | **Full rewrite** |
| `README.md` | `docs/thesis/` listed as "canonical documentation" | HIGH | **Rewrite** |
| `README.md` | `results/phase3a/` (actual: `results/phase3a_proseco_owt/`) | HIGH | **Rewrite** |
| `REPRODUCIBILITY.md` | `results/phase3a/` (stale path) | MEDIUM | **Archived** |
| `REPRODUCIBILITY.md` | Duplicates `scripts/README.md` workflow | LOW | **Archived** |
| `CLAUDE.md` | `docs/md/correctors_deep_dive.md` ("historically useful") — path no longer exists | HIGH | **Removed**, replaced with `research/candidate_theorems.md` ref |
| `hpc/README.md` | "See REPRODUCIBILITY.md for setup" | MEDIUM | **Updated** to point to `scripts/README.md` |
| `docs/01_research_direction.md` | Provenance header citing archived paths | LOW | **Removed** header |
| `docs/02_experiments.md` | Provenance header citing archived paths | LOW | **Removed** header |
| `docs/03_theory.md` | Provenance header citing archived paths | LOW | **Removed** header |
| `docs/05_next_steps.md` | Provenance header citing archived path | LOW | **Removed** header |
| `research/open_questions.md` | `docs/experiments/entropy_proxy_experiment.md` (archived) | HIGH | **Replaced file** |
| `research/open_questions.md` | `docs/thesis/theory/THEORY_STRESS_TEST.md` (archived) | HIGH | **Replaced file** |
| `research/open_questions.md` | `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` (archived) | HIGH | **Replaced file** |
| `research/open_questions.md` | `docs/gpt_pro_assessment_response.md` (archived) | MEDIUM | **Replaced file** |
| `research/open_questions.md` | Most questions labelled "action required" but already answered | HIGH | **Replaced file** with compact current version |

---

## References that are harmless historical text (no action needed)

| File | Reference | Why harmless |
|---|---|---|
| `docs/README.md` | Mentions `docs/thesis/` in archive rule ("historical") | Correct — it's the archive warning |
| `START_HERE.md` | Mentions `docs/thesis/` in archive rule | Correct |
| `CLAUDE.md` | Mentions `docs/archive/`, `archive/` in archive rule | Correct |
| `archive/ARCHIVE_MANIFEST.md` | References old paths as "archived paths" | It's the archive manifest |
| `docs/archive/*` files | All reference old paths | They're archived docs |
