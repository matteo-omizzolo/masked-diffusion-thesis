# Results

Canonical result folders remain at their historical paths under `results/`.
They were not moved into `results/canonical/` because the docs, JSON indexes,
result configs, and scripts already point to those paths and preserving exact
provenance is more valuable than cosmetic relocation.

Most large local result directories are ignored by git. Treat
`results/EXPERIMENT_INDEX.json`, `results/BACKEND_FEASIBILITY_INDEX.json`, and
`docs/04_results_index.md` as the compact source of truth; inspect raw folders
only when reproducing a specific claim.

Machine-readable indexes:

- `results/EXPERIMENT_INDEX.json`
- `results/BACKEND_FEASIBILITY_INDEX.json`

Archived smoke/preflight/provenance material is under:

- `results/archive/2026-05-13_repo_cleanup/`

Future backend-validation smoke outputs should use:

- `results/backend_validation/informed_correctors/`

Do not write new outputs into canonical ProSeCo folders unless the run has been
explicitly approved as a canonical reproduction or extension.
