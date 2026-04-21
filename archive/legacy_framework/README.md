# Legacy framework archive

This directory holds the older comparative framework that is no longer the
thesis mainline:

- legacy adapter / sampler / strategy packages
- phase 1 and phase 2a scripts
- phase 1 and smoke / eval HPC jobs
- stale exploratory results and figures
- old preflight and staging helpers

The current thesis path lives in:

- `scripts/run_phase2b_proseco_owt.py`
- `scripts/run_phase3a_combinatorial.py`
- `src/mdm_playground/scheduling/`
- `results/phase2b_proseco_owt/`
- `results/phase3a_proseco_owt/`

`results/phase1_proseco_owt_full/` remains at the repo root because Phase 2b
reuses it as a read-only prerequisite data artifact.
