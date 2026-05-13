# Experiment Log

## ProSeCo-OWT

Status: complete case study.

The canonical ProSeCo sequence is:

1. Protocol A/headroom.
2. Phase 2b rankers, uniform schedules, MC oracle.
3. Phase 3a direct true-`G(S)` search.
4. Pair interactions and schedule-level validation.
5. Set-function, landscape, and neighborhood diagnostics.
6. Marginal, pair, enriched, and token-level state-predictability audits.
7. Saturation structure.
8. Corrector-strength analysis.

Interpretation: ProSeCo-OWT is a redundancy/saturation-driven regime. Direct
schedule search succeeds, while the tested marginal/state-predictive families do
not.

## Backend Feasibility

Status: complete exploratory audit.

- informed-correctors/Text8: pursue.
- PRISM LLaDA: do not pursue for this timing-validation gate unless conditions change.

The backend smoke folders are archived under
`results/archive/2026-05-13_repo_cleanup/`.

## informed-correctors/Text8 Prepared Stages

Status: prepared, not run.

- Stage 0: `hpc/backend_validation/informed_correctors/stage0_env_smoke.sbatch`
- Stage 1: `hpc/backend_validation/informed_correctors/stage1_tiny_train.sbatch`

No Stage 2 throughput benchmark or full training run has been launched.
