# hpc/backend_validation/informed_correctors

Runbook for the informed-correctors/Text8 backend-validation track on
Bocconi HPC. This is the **primary path** under the
no-author-no-admin-reply assumption; the external-GPU fallback lives in
`docs/10_external_gpu_text8_fallback.md`.

## Files

| File | Purpose |
|---|---|
| `setup_ic_text8_jax13.sh` | Idempotent one-shot env setup for the dedicated `ic_text8_jax13` conda env. Run **once** from a login node. |
| `stage0_env_smoke.sbatch` | GPU smoke that runs imports + `jax.devices()` + a JIT compute probe + Text8 path check on a compute node. |
| `stage1_tiny_train.sbatch` | Tiny HollowMD4 training-loop smoke (≤50 steps, batch_size=8, hidden_dim=128, 2 layers). |

## Sequence

```bash
# 0. Sync local edits to HPC (from your laptop, not a login node).
bash hpc/push.sh

# 1. Set up the dedicated env. Run from a Bocconi login node.
ssh 3316152@slogin.hpc.unibocconi.it
cd ~/mdm/masked-diffusion-thesis
bash hpc/backend_validation/informed_correctors/setup_ic_text8_jax13.sh

# 2. Stage 0 — environment + GPU compute smoke. Compute node.
sbatch hpc/backend_validation/informed_correctors/stage0_env_smoke.sbatch
# Wait for completion: squeue -u $USER
# Inspect: cat results/backend_validation/informed_correctors/stage0_env_smoke_*/feasibility.json

# 3. Stage 1 — tiny training-loop smoke. Compute node. ONLY if Stage 0 passed.
sbatch hpc/backend_validation/informed_correctors/stage1_tiny_train.sbatch
# Wait for completion: squeue -u $USER
# Inspect: ls results/backend_validation/informed_correctors/stage1_tiny_train_*/workdir/checkpoints/
```

## Stage 0 pass criteria

All six must hold:

- sbatch exits 0;
- every required import succeeds (14 packages — see `check_text8_training_feasibility.py::DEPENDENCIES`);
- `jax.devices()` returns at least one GPU;
- **JIT compute probe** (`(jnp.ones((64,64)) @ jnp.ones((64,64))).sum()`) runs and returns `64^3 = 262144.0`;
- HollowMD4 Text8 config loads (`model_type=hollow_md4`);
- Text8 zip exists at `$TEXT8_DATA_DIR/text8/text8.zip` (default: `~/mdm/data/text8_md4/text8/text8.zip`).

The compute probe is what distinguishes a healthy GPU/driver/JAX stack from
the "enumeration passes but compiled-kernel allocation crashes" failure
mode (CLAUDE.md issue #14).

## Stage 1 pass criteria

- sbatch exits 0;
- training starts (e.g. `Using Hollow MD4` reached) and completes the
  configured ≤50 steps;
- loss is finite (no NaN/Inf);
- an Orbax checkpoint is written under `$OUT_DIR/workdir/checkpoints/`;
- `external_repos/informed-correctors/` is **unchanged** (only the copied
  tree at `$OUT_DIR/upstream_copy/` is patched);
- wallclock is short (target < 30 minutes; sbatch time limit is 1 hour);
- no useful/full training has been launched.

## Do NOT run

- Stage 2 throughput benchmark — needs explicit approval.
- 1M-step full training — needs explicit approval and Stage 2 first.
- Any sbatch with elevated GPU/memory/time without a documented gate.

## Troubleshooting

- **Stage 0 fails on imports.** Re-run `setup_ic_text8_jax13.sh` from a login
  node. Check disk quota; the env is ~5-7 GB.
- **Stage 0 fails on `compute_probe`.** This is the canary for CUDA/driver
  mismatches. Capture the probe error from `feasibility.json` and consult
  CLAUDE.md issue #14. If the error persists, fall back to the external GPU
  plan in `docs/10_external_gpu_text8_fallback.md`.
- **Stage 1 fails with `ExitCode 120:0` after `Using Hollow MD4`.** This was
  the symptom of the cuda12-on-cuda13-MIG bug. With `ic_text8_jax13` (which
  uses `jax[cuda13]`) it should not recur. If it does, switch to the
  external-GPU fallback rather than fighting the env further.
- **`disk quota exceeded` during `pip install`.** Clear `~/.cache/pip`,
  `~/.conda/pkgs` (via `conda clean --all`), and consider removing the
  closed `~/mdm/checkpoints/proseco_llada_sft` checkpoint (15 GB; the
  cross-backbone line is closed per docs/00).

## Environment hygiene

- `ic_text8_jax13` is **JAX-only**; do not install PyTorch into it.
- `remdm311` is **PyTorch/ProSeCo only**; do not pip install JAX into it
  (it broke wandb/protobuf the first time around).
- The Text8 zip can be re-downloaded with the `setup_ic_text8_jax13.sh`
  script; it is idempotent.
- HPC sbatches always tag their output dirs with `$(git rev-parse --short HEAD)`
  via the `.hpc_git_sha` mechanism written by `hpc/push.sh`. A `-dirty`
  suffix means the worktree had uncommitted edits at push time.
