# Repository Cleanup Manifest: 2026-05-13

This archive records files moved during the repo cleanup and research-state consolidation.
No canonical result folders were deleted. Active scripts and thesis files were left in place.

| Original path | New archive path | Reason | Referenced in docs? | Replacement exists? |
|---|---|---|---|---|
| `scripts/.DS_Store` | `scripts/archive/2026-05-13_repo_cleanup/.DS_Store` | macOS metadata, not part of reproducible research state | No | Not needed |
| `scripts/__pycache__/` | `scripts/archive/2026-05-13_repo_cleanup/__pycache__/` | generated Python bytecode cache | No | Regenerated automatically if needed |
| `hpc/.DS_Store` | `hpc/archive/2026-05-13_repo_cleanup/.DS_Store` | macOS metadata, not part of reproducible research state | No | Not needed |
| `hpc/tokenlevel_pilot_k10.sbatch` | `hpc/archive/2026-05-13_repo_cleanup/tokenlevel_pilot_k10.sbatch` | old K=10 token-level pilot superseded by K=30 token-level run | Historical/provenance only | `hpc/proseco/tokenlevel/tokenlevel_k30.sbatch` |
| `hpc/tokenlevel_preflight_k2.sbatch` | `hpc/archive/2026-05-13_repo_cleanup/tokenlevel_preflight_k2.sbatch` | completed preflight job superseded by K=30 token-level run | Historical/provenance only | `hpc/proseco/tokenlevel/tokenlevel_k30.sbatch` |
| `hpc/corrector_strength_preflight.sbatch` | `hpc/archive/2026-05-13_repo_cleanup/corrector_strength_preflight.sbatch` | completed preflight job superseded by K=30 corrector-strength run | Historical/provenance only | `hpc/proseco/corrector_strength/corrector_strength_k30.sbatch` |
| `results/backend_smoke_informed_correctors_8371dec/` | `results/archive/2026-05-13_repo_cleanup/backend_smoke_informed_correctors_8371dec/` | backend smoke artifact from feasibility audit, useful as provenance but not canonical ProSeCo result | Yes, backend feasibility audit | `docs/08_backend_feasibility_audit.md` summarizes outcome |
| `results/backend_smoke_prism_266d51e/` | `results/archive/2026-05-13_repo_cleanup/backend_smoke_prism_266d51e/` | backend smoke artifact from feasibility audit, useful as provenance but not canonical ProSeCo result | Yes, backend feasibility audit | `docs/08_backend_feasibility_audit.md` summarizes outcome |
| `results/tokenlevel_features_proseco_preflight_0c39079/` | `results/archive/2026-05-13_repo_cleanup/tokenlevel_features_proseco_preflight_0c39079/` | K=2 token-level preflight superseded by K=30 token-level feature/results folders | Yes, experiment log and index as archived provenance | `results/tokenlevel_features_proseco_k30_0c39079/` |
| `results/tokenlevel_state_predictability_preflight_0c39079/` | `results/archive/2026-05-13_repo_cleanup/tokenlevel_state_predictability_preflight_0c39079/` | K=2 token-level preflight superseded by K=30 token-level analysis | Yes, experiment log and index as archived provenance | `results/tokenlevel_state_predictability_k30_0c39079/` |
| `results/corrector_strength_preflight_0c39079/` | `results/archive/2026-05-13_repo_cleanup/corrector_strength_preflight_0c39079/` | small corrector-strength preflight superseded by K=30 corrector-strength analysis | Yes, experiment log and index as archived provenance | `results/corrector_strength_k30_0c39079/` |
