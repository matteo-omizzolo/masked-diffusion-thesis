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

**Bocconi path — exhausted (2026-05-14):** three independent env attempts
all hit structural blockers documented in CLAUDE.md known issue #14. The
remaining outcome: Stage 0 (job 494412) finally passes cleanly on
`ic_text8_jax13` with `jax[cuda12]` 0.10.0 — 13/13 imports, GPU compute
probe matches=True, no false-pass on CPU. But Stage 1 then hits an
upstream `informed-correctors` codebase + `tensorflow_probability` + JAX
0.10 API-removal incompatibility (`jax.interpreters.xla.pytype_aval_mappings`)
that requires patching multiple upstream modules. The cuda13 path on
Bocconi is blocked at Stage 0 because Bocconi exposes no CUDA-13 toolkit
module; the cuda12 path either crashes on MIG (older JAX) or breaks
upstream code at import time (newer JAX). See `docs/09 §Stage 1 — Bocconi
env path documented blocked` for full diagnostics.

**Active path — external GPU rental:** `docs/10_external_gpu_text8_fallback.md`
is the documented next-step. A fresh cloud GPU with a matching driver/toolkit
avoids all three Bocconi-specific constraints, costs ~$1.50 for Stage 0 +
Stage 1 + Stage 2, and is the cleanest path. The repo state (commits on
`codex/informed-correctors-backend-validation`) is ready to be cloned on a
rental machine, including the strengthened Stage 0 JIT compute probe and
the setup script.

**Outgoing communications already sent:**

- Author email in `docs/email_informed_correctors_authors.md` — sent
  2026-05-14. Follow up after 7-10 days if no reply, but **non-blocking**.

Do not launch Stage 2 or full training without explicit approval. Stage 1
should be retried only on the external-GPU fallback or after an authors'
checkpoint is shared.

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
