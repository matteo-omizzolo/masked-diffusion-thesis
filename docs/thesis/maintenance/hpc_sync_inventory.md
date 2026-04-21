# HPC Sync Inventory
**Date:** 2026-04-18  
**Remote:** `3316152@slogin.hpc.unibocconi.it:~/mdm/masked-diffusion-thesis`  
**Local:** `/Users/matteoomizzolo/masked-diffusion-thesis`

---

## Remote Directories Inspected

| Directory | Status | Notes |
|-----------|--------|-------|
| `docs/` | Synced | New `experiments/` subtree + `CURRENT_INDEX.md` |
| `src/` | Synced | New `scheduling/backends/` submodule |
| `scripts/` | Synced | 4 new scripts added on HPC |
| `hpc/` | Synced | 2 new sbatch files + README |
| `research/` | Synced | All 4 files updated on HPC |
| `figures/` | Synced | `phase1_pilot/` (6 PNGs) |
| `results/` | Synced | 4 new `phase1_*` experiment result sets |
| `out/` | Synced | 13 SLURM stdout files |
| `err/` | Synced | 13 SLURM stderr files |
| `tests/` | Synced | Empty (test .pyc files only, excluded) |
| `third_party/proseco/` | **Skipped** | Empty working tree on HPC (git repo cloned but files deleted/not checked out) |
| `external/` | Not synced | Already present locally; large; no changes |
| `archive/` | Not synced | Not modified; no new files |
| `study/` | Not synced | Not modified since last push |
| `thesis/` | Not synced | Not modified on HPC (LaTeX only on local) |
| `notebooks/` | Not synced | Not modified |
| `checkpoints/` | Not synced | Large binary files; not in HPC repo dir |

---

## Important Files Found on HPC (New Since Last Sync)

### docs/experiments/ (entirely new)
- `entropy_proxy_experiment.md` — Phase 1 experiment protocol A/B definition
- `implementation_status.md` — corrector scheduling implementation status
- `phase1_interpretation.md` — Phase 1 result interpretation
- `proseco_backend_audit.md`, `proseco_import_audit.md` — ProSeCo backend investigation
- `proseco_experiment_definition.md`, `proseco_protocol_mapping.md` — ProSeCo experiment setup
- `proseco_validation_checklist.md` — pre-run validation
- `results/` (11 experiment result docs) — full Phase 1 analysis chain

### src/mdm_playground/scheduling/backends/ (new)
- `mdlm.py` — MDLM backend adapter
- `mdlm_conf.py` — MDLM-conf backend adapter
- `proseco.py` — ProSeCo backend adapter

### scripts/ (new on HPC)
- `debug_mdlm_conf_load.py` — debugging script for MDLM-conf load
- `debug_proseco_load.py` — debugging script for ProSeCo load
- `run_phase1_mdlm_conf.py` — Phase 1 runner for MDLM-conf backend
- `run_phase1_proseco.py` — Phase 1 runner for ProSeCo backend

### hpc/ (new)
- `phase1_mdlm_conf.sbatch` — SLURM job for MDLM-conf Phase 1
- `phase1_proseco.sbatch` — SLURM job for ProSeCo Phase 1
- `README.md` — HPC workflow documentation

### results/ (new experiment sets)
- `phase1_mdlm_conf/` — 20 Protocol A trajectories + 45 Protocol B schedules + policy comparison
- `phase1_mdlm_conf_surrogate_sanity/` — sanity check run (5 traj + 6 schedules)
- `phase1_proseco/` — 20 Protocol A trajectories + 45 Protocol B schedules + policy comparison
- `phase1_proseco_surrogate_sanity/` — sanity check run

---

## What Was Synced

All lightweight repo files: docs, src (Python only), scripts, hpc sbatches, research markdown, figures (PNGs), results (JSON), SLURM logs.

## What Was Excluded and Why

| Excluded | Reason |
|----------|--------|
| `*.pyc`, `__pycache__/` | Compiled bytecode; regenerates automatically |
| `*.egg-info/` | Build artifacts |
| `results/**/generated_sequences.json` | Large; per gitignore |
| `results/**/samples_*.json` | Large; per gitignore |
| `*.pt`, `*.ckpt`, `*.pth`, `*.npz`, `*.safetensors` | Large binaries |
| `external/` | Already on local; large |
| `checkpoints/` | Large model weights; not in repo dir on HPC |
| `third_party/proseco/` | Empty on HPC disk (repo cloned but no working tree files present) |
| `docs/pdf/` | PDFs regeneratable from markdown; not modified on HPC |
| `.venv/` | Environment; not synced by design |
| `.claude/` | HPC-specific Claude config |
