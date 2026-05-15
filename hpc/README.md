# HPC Workflow

No new long training or ProSeCo experiment is currently approved. The active
frontier is informed-correctors/Text8 backend validation.

## Layout

```text
hpc/
  proseco/
    reproduction/
    interactions/
    landscape/
    tokenlevel/
    corrector_strength/
  backend_validation/
    informed_correctors/
  legacy/
  archive/
```

## Closed ProSeCo Jobs

| Family | Path | Status |
|---|---|---|
| Phase 0 / Phase 2b / Phase 3a | `hpc/proseco/reproduction/` | closed provenance |
| Pair interactions | `hpc/proseco/interactions/` | closed provenance |
| Landscape neighborhood | `hpc/proseco/landscape/` | closed provenance |
| Token-level K=30 | `hpc/proseco/tokenlevel/tokenlevel_k30.sbatch` | closed canonical provenance |
| Corrector strength K=30 | `hpc/proseco/corrector_strength/corrector_strength_k30.sbatch` | closed canonical provenance |

## Backend Validation Jobs

| Stage | Path | Runtime | Purpose |
|---|---|---:|---|
| Stage 0 | `hpc/backend_validation/informed_correctors/stage0_env_smoke.sbatch` | minutes | imports, JAX/GPU, Text8 data path, config check |
| Stage 1 | `hpc/backend_validation/informed_correctors/stage1_tiny_train.sbatch` | <1 hour target | tiny HollowMD4 training-loop and checkpoint smoke |

These Bocconi jobs are retained as provenance and reusable scaffolding. As of
2026-05-14, Stage 0 passes on the dedicated env but Stage 1 is blocked by the
cluster/JAX/upstream dependency matrix. The active path is the external-GPU
runbook in `docs/10_external_gpu_text8_fallback.md`.

Stage 1 is not a useful checkpoint. No full Text8 training is approved by this
file.

## Cluster Details

| Item | Value |
|---|---|
| Host | `slogin.hpc.unibocconi.it` |
| User | `3316152` |
| Partition / QOS | `stud` / `stud` |
| GPU limit | up to 4 A100 per job |
| Repo path | `~/mdm/masked-diffusion-thesis` |
| Python env | conda `remdm311` unless overridden |

## Pre-Submit Rule

Before submitting any future sbatch job:

1. Confirm the job opens an approved research gate.
2. Run `git diff --check`.
3. Run relevant local tests or smoke checks.
4. Confirm the output directory is non-canonical unless the run is explicitly approved.
5. Record exact command, SHA, output folder, and pass/fail criteria.
