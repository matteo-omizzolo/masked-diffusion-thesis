# Aggressive Restructure Manifest: 2026-05-13

This manifest records the structural migration performed after the conservative
repo cleanup. Canonical result folders were not moved.

## Active Script Moves

| Original path | New path | Classification | Reason |
|---|---|---|---|
| `scripts/stage_proseco_owt.py` | `scripts/proseco/reproduction/stage_proseco_owt.py` | active utility | ProSeCo checkpoint/reproduction helper |
| `scripts/run_phase2b_proseco_owt.py` | `scripts/proseco/reproduction/run_phase2b_proseco_owt.py` | active canonical | Phase 2b reproduction |
| `scripts/analyze_phase2b.py` | `scripts/proseco/reproduction/analyze_phase2b.py` | active canonical | Phase 2b analysis |
| `scripts/run_phase3a_combinatorial.py` | `scripts/proseco/reproduction/run_phase3a_combinatorial.py` | active canonical | Phase 3a reproduction |
| `scripts/analyze_phase3a.py` | `scripts/proseco/reproduction/analyze_phase3a.py` | active canonical | Phase 3a analysis |
| `scripts/compute_theorem_a_constants.py` | `scripts/proseco/reproduction/compute_theorem_a_constants.py` | active utility | Theorem A constants |
| `scripts/analyze_combinatorial_diagnostics.py` | `scripts/proseco/reproduction/analyze_combinatorial_diagnostics.py` | active utility | Phase 2b/3a diagnostics |
| `scripts/run_protocol_c_owt.py` | `scripts/proseco/reproduction/run_protocol_c_owt.py` | provenance utility | Protocol C pilot reproduction |
| `scripts/run_phase1_interaction_diagnostics.py` | `scripts/proseco/interactions/run_phase1_interaction_diagnostics.py` | active canonical | Gate 3 pair diagnostics |
| `scripts/analyze_phase1_interactions.py` | `scripts/proseco/interactions/analyze_phase1_interactions.py` | active canonical | Gate 3 analysis |
| `scripts/validate_phase1_schedule_level_b2.py` | `scripts/proseco/interactions/validate_phase1_schedule_level_b2.py` | active canonical | B=2 schedule validation |
| `scripts/validate_phase1_schedule_level_b34.py` | `scripts/proseco/interactions/validate_phase1_schedule_level_b34.py` | active canonical | B=3/4 schedule validation |
| `scripts/analyze_set_function_structure.py` | `scripts/proseco/landscape/analyze_set_function_structure.py` | active canonical | Set-function diagnostics |
| `scripts/analyze_schedule_landscape_geometry.py` | `scripts/proseco/landscape/analyze_schedule_landscape_geometry.py` | active canonical | Landscape geometry |
| `scripts/run_schedule_neighborhood_diagnostics.py` | `scripts/proseco/landscape/run_schedule_neighborhood_diagnostics.py` | active canonical | Neighborhood diagnostics |
| `scripts/analyze_schedule_neighborhood_diagnostics.py` | `scripts/proseco/landscape/analyze_schedule_neighborhood_diagnostics.py` | active canonical | Neighborhood analysis |
| `scripts/analyze_state_predictability.py` | `scripts/proseco/state_predictability/analyze_state_predictability.py` | active canonical | Marginal state audit |
| `scripts/analyze_state_predictability_pair.py` | `scripts/proseco/state_predictability/analyze_state_predictability_pair.py` | active canonical | Pair state audit |
| `scripts/analyze_state_predictability_enriched.py` | `scripts/proseco/state_predictability/analyze_state_predictability_enriched.py` | active canonical | Enriched state audit |
| `scripts/extract_tokenlevel_features_proseco.py` | `scripts/proseco/state_predictability/extract_tokenlevel_features_proseco.py` | active canonical | Token-level feature extraction |
| `scripts/analyze_tokenlevel_state_predictability.py` | `scripts/proseco/state_predictability/analyze_tokenlevel_state_predictability.py` | active canonical | Token-level audit |
| `scripts/analyze_saturation_structure.py` | `scripts/proseco/saturation/analyze_saturation_structure.py` | active canonical | Saturation diagnostics |
| `scripts/run_corrector_strength_preflight.py` | `scripts/proseco/corrector_strength/run_corrector_strength_preflight.py` | active canonical | Corrector-strength runner; filename preserved for provenance |
| `scripts/analyze_corrector_strength_preflight.py` | `scripts/proseco/corrector_strength/analyze_corrector_strength_preflight.py` | active canonical | Corrector-strength analysis; filename preserved for provenance |

## HPC Moves

| Original path | New path | Classification |
|---|---|---|
| `hpc/phase0_smoke_k3.sbatch` | `hpc/proseco/reproduction/phase0_smoke_k3.sbatch` | closed provenance |
| `hpc/phase2b_proseco_owt.sbatch` | `hpc/proseco/reproduction/phase2b_proseco_owt.sbatch` | closed provenance |
| `hpc/phase3a_combinatorial.sbatch` | `hpc/proseco/reproduction/phase3a_combinatorial.sbatch` | closed provenance |
| `hpc/phase1_interaction_diagnostics.sbatch` | `hpc/proseco/interactions/phase1_interaction_diagnostics.sbatch` | closed provenance |
| `hpc/phase4_schedule_neighborhood_diagnostics.sbatch` | `hpc/proseco/landscape/phase4_schedule_neighborhood_diagnostics.sbatch` | closed provenance |
| `hpc/tokenlevel_k30.sbatch` | `hpc/proseco/tokenlevel/tokenlevel_k30.sbatch` | closed provenance |
| `hpc/corrector_strength_k30.sbatch` | `hpc/proseco/corrector_strength/corrector_strength_k30.sbatch` | closed provenance |

## Test Moves

Tests were grouped under `tests/proseco/{core,reproduction,interactions,landscape,state_predictability,saturation,corrector_strength}/`.

## Archived Generated Material

The bytecode archives below were originally retained as belt-and-braces
provenance of the restructure validation passes. They contained only
regenerable `.pyc` files (no scientific or reproducibility content) and were
subsequently removed on 2026-05-14 during the post-Stage-0 cleanup pass; the
table is kept here to document that they ever existed and that their content
is recoverable by re-running `pytest -q` on the corresponding source tree.

| Original path | Was archived to | Status (2026-05-14) | Reason removed |
|---|---|---|---|
| `scripts/__pycache__/` | `scripts/archive/2026-05-13_aggressive_restructure/__pycache__/` | removed | generated bytecode, regenerable |
| `tests/__pycache__/` | `tests/archive/2026-05-13_aggressive_restructure/__pycache__/` | removed | generated bytecode, regenerable |
| post-validation `scripts/**/__pycache__/` | `scripts/archive/2026-05-13_aggressive_restructure/post_validation_pycache/` | removed | generated bytecode, regenerable |
| post-validation `tests/**/__pycache__/` | `tests/archive/2026-05-13_aggressive_restructure/post_validation_pycache/` | removed | generated bytecode, regenerable |
| final-validation `scripts/**/__pycache__/` | `scripts/archive/2026-05-13_aggressive_restructure/final_validation_pycache/` | removed | generated bytecode, regenerable |
| final-validation `tests/**/__pycache__/` | `tests/archive/2026-05-13_aggressive_restructure/final_validation_pycache/` | removed | generated bytecode, regenerable |

## New Backend-Validation Files

- `scripts/backend_validation/informed_correctors/check_text8_training_feasibility.py`
- `scripts/backend_validation/informed_correctors/hollow_text8_stage1_config.py`
- `hpc/backend_validation/informed_correctors/stage0_env_smoke.sbatch`
- `hpc/backend_validation/informed_correctors/stage1_tiny_train.sbatch`
