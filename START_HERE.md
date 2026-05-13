# START HERE

This repository contains the MSc thesis project on finite-budget corrector
timing in masked diffusion language models. The ProSeCo-OWT line is a completed
case study: timing has headroom, marginal rankers fail, direct search over
`G(S)` succeeds, pair interactions are mostly redundant/saturating, and tested
state features do not improve prediction beyond geometry plus `A_pair`.

The active framing is:

> a diagnostic/statistical framework for finite-budget corrector timing regimes,
> with ProSeCo-OWT as a completed redundancy/saturation-driven case study and
> informed-correctors/Text8 as the proposed principled validation backend.

Do not overclaim ProSeCo generality. informed-correctors/Text8 is not yet
validated in this repo.

## Read Next

1. `docs/ACTIVE_RESEARCH_STATE.md`
2. `docs/00_current_status.md`
3. `docs/02_experiments.md`
4. `docs/08_backend_feasibility_audit.md`
5. `docs/09_informed_correctors_training_contingency.md`

## Active Architecture

- ProSeCo scripts: `scripts/proseco/`
- Backend-validation scripts: `scripts/backend_validation/`
- ProSeCo HPC jobs: `hpc/proseco/`
- Backend-validation HPC jobs: `hpc/backend_validation/`
- Tests: `tests/proseco/`
- Canonical result index: `results/EXPERIMENT_INDEX.json`

## Current Next Action

Author email in `docs/email_informed_correctors_authors.md` was sent
**2026-05-14**. Two parallel tracks:

- **Author track:** wait for reply; follow up after 7-10 days if none.
- **No-author-response track (current state, 2026-05-14):**
  - **Stage 0** ✅ passed (job 494221, gnode01): 14/14 imports OK, JAX sees
    GPU, HollowMD4 config loads, Text8 staged. Resolved the
    `remdm311`-lacks-JAX-ecosystem blocker by installing pinned versions
    (jax/flax/orbax/distrax compatible set + tensorflow-cpu + tf-keras +
    tensorboard + seaborn + wandb 0.26.1). See
    `docs/09_informed_correctors_training_contingency.md §Stage 0`.
  - **Stage 1** 🚫 blocked (jobs 494239 and 494245, gnode02):
    `ExitCode=120:0` ~3.5 min after `Using Hollow MD4`, no Python
    traceback. The Bocconi `stud` A100s run in MIG mode under CUDA-13
    driver while JAX 0.4.30 ships a cuda12 plugin. The
    `XLA_PYTHON_CLIENT_PREALLOCATE=false`/`MEM_FRACTION=0.5`/`JAX_PLATFORMS=cuda`
    workaround did not help. See CLAUDE.md known issue #14 +
    `docs/09 §Stage 1 — current blocker` for the deferred resolution
    paths (non-MIG queue request, `jax[cuda13]` wait, CPU fallback,
    author-track checkpoint).

Do not launch Stage 2 or full training without explicit approval. Stage 1
should be re-attempted only after one of the resolution paths is in place.

## Canonical ProSeCo Result Folders

- `results/phase1_proseco_owt_full/protocol_a/`
- `results/phase2b_proseco_owt/`
- `results/phase3a_proseco_owt/`
- `results/phase1_interaction_diag_nogit/`
- `results/phase1_schedule_validation_b2_0c39079/`
- `results/phase1_schedule_validation_b34_0c39079/`
- `results/set_function_structure_0c39079/`
- `results/schedule_landscape_geometry_0c39079/`
- `results/phase4_schedule_neighborhood_0c39079/`
- `results/state_predictability_0c39079/`
- `results/state_predictability_pair_0c39079/`
- `results/state_predictability_enriched_0c39079/`
- `results/tokenlevel_features_proseco_k30_0c39079/`
- `results/tokenlevel_state_predictability_k30_0c39079/`
- `results/saturation_structure_0c39079/`
- `results/corrector_strength_k30_0c39079/`
