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

**Working assumption (2026-05-14):** do **not** assume the informed-correctors
authors or the Bocconi HPC admins will reply in time. The active execution
plan is engineered to be self-sufficient.

**Primary path — clean Bocconi `ic_text8_jax13` env (`jax[cuda13]`):**

1. Set up the dedicated env on a Bocconi login node via
   `hpc/backend_validation/informed_correctors/setup_ic_text8_jax13.sh`.
2. Stage 0 (now includes a JIT compute probe that catches CUDA/driver/MIG
   issues at Stage 0 time, not Stage 1 time):
   `sbatch hpc/backend_validation/informed_correctors/stage0_env_smoke.sbatch`.
3. Stage 1 (tiny training-loop smoke) **iff Stage 0 passes**:
   `sbatch hpc/backend_validation/informed_correctors/stage1_tiny_train.sbatch`.
4. Both sbatches default to `CONDA_ENV=ic_text8_jax13`; the older
   `remdm311` env is reserved for ProSeCo/PyTorch work and is deprecated
   for Text8 training (CLAUDE.md issue #14).

**Fallback path — external GPU rental (only if Bocconi remains blocked):**
see `docs/10_external_gpu_text8_fallback.md` for machine recommendation,
setup, Stage 0/1 commands, ~$1.50 cost estimate, and stop criteria.

**Outgoing communications already sent:**

- Author email in `docs/email_informed_correctors_authors.md` — sent
  2026-05-14. Follow up after 7-10 days if no reply, but do **not** block
  on it.

**Historical state (2026-05-14, `remdm311` path — deprecated):**
Stage 0 eventually passed (job 494221) after installing JAX/Flax/TF into
`remdm311`; Stage 1 jobs 494239 and 494245 then failed with `ExitCode 120:0`
after `Using Hollow MD4` because JAX 0.4.30's cuda12 plugin crashed against
the A100 MIG + CUDA-13 driver on `stud`. Documented in CLAUDE.md issue #14
and `docs/09 §Stage 1`. The new `ic_text8_jax13` env uses `jax[cuda13]`,
which sidesteps this entirely.

Do not launch Stage 2 or full training without explicit approval.

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
