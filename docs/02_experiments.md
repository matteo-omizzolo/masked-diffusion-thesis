# Experiments

The machine-readable index is `results/EXPERIMENT_INDEX.json`.

## ProSeCo-OWT Case Study

| Family | Status | Scripts | Results |
|---|---|---|---|
| Headroom / Protocol A | complete canonical | `scripts/proseco/reproduction/` | `results/phase1_proseco_owt_full/protocol_a/` |
| Rankers/search/MC oracle | complete canonical | `scripts/proseco/reproduction/` | `results/phase2b_proseco_owt/`, `results/phase3a_proseco_owt/` |
| Pair interactions | complete canonical | `scripts/proseco/interactions/` | `results/phase1_interaction_diag_nogit/`, `results/phase1_schedule_validation_*` |
| Landscape/neighborhood | complete canonical | `scripts/proseco/landscape/` | `results/set_function_structure_0c39079/`, `results/schedule_landscape_geometry_0c39079/`, `results/phase4_schedule_neighborhood_0c39079/` |
| State predictability | complete canonical | `scripts/proseco/state_predictability/` | `results/state_predictability_*`, `results/tokenlevel_*_k30_0c39079/` |
| Saturation | complete canonical | `scripts/proseco/saturation/` | `results/saturation_structure_0c39079/` |
| Corrector strength | complete canonical | `scripts/proseco/corrector_strength/` | `results/corrector_strength_k30_0c39079/` |

## Backend Validation

| Family | Status | Scripts/HPC | Results |
|---|---|---|---|
| Backend feasibility audit | complete exploratory | audit only | `docs/08_backend_feasibility_audit.md`, `results/BACKEND_FEASIBILITY_INDEX.json` |
| informed-correctors/Text8 Stage 0 on Bocconi | complete environment smoke | `hpc/backend_validation/informed_correctors/stage0_env_smoke.sbatch` | local `results/backend_validation/informed_correctors/` |
| informed-correctors/Text8 Stage 1 on Bocconi | blocked | `hpc/backend_validation/informed_correctors/stage1_tiny_train.sbatch` | no useful checkpoint |
| informed-correctors/Text8 external GPU | active next gate | `docs/10_external_gpu_text8_fallback.md` | future external-GPU smoke output |
| PRISM LLaDA | not pursued | audit only | archived smoke provenance |

## Archived/Superseded

See:

- `docs/archive/2026-05-13_repo_cleanup/MANIFEST.md`
- `docs/archive/2026-05-13_aggressive_restructure/MANIFEST.md`

Canonical result folders were not moved.
