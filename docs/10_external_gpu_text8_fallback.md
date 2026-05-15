# External GPU Fallback — informed-correctors/Text8

This document describes the fallback path **if Bocconi HPC remains blocked**
on Stage 1 of the informed-correctors/Text8 backend-validation track (see
CLAUDE.md issue #14 and
`docs/09_informed_correctors_training_contingency.md §Stage 1`). It is the
last line of defence under the assumption that neither the
informed-correctors authors nor Bocconi HPC admins reply in time.

This plan is **not yet executed**. It is documented in advance so a future
session (or a future me) can pick it up without re-deriving the rationale.

## When to switch

The working assumption is that neither the informed-correctors authors nor
Bocconi HPC admins will reply in time — see START_HERE.md current-next-action.
The fallback trigger therefore depends only on the Bocconi clean-CUDA13
attempt itself, not on any external response:

Trigger the external-GPU fallback when **either** of these holds:

1. **Stage 0 fails on `ic_text8_jax13`.** The Stage 0 sbatch defaults to
   the `ic_text8_jax13` env (CUDA13 PJRT) and now runs a JIT compute
   probe; a failure means GPU+driver+JAX cannot agree. Inspect the
   `compute_probe` field of `feasibility.json`, attempt at most one
   targeted fix (env var, package pin), then escalate to this fallback.
2. **Stage 0 passes but Stage 1 fails** with a CUDA/driver/MIG-shaped
   error after `Using Hollow MD4` (the symptom of CLAUDE.md issue #14).
   Do not loop on Bocconi env tweaks here — switch.

Author and admin replies are **bonus**, not gating. If they arrive
between when this fallback is triggered and when it is executed, the
fallback can be cancelled, but they are not a reason to defer the
fallback.

A useful first-attempt cap before switching: spend **at most one
half-day of operator time** on Bocconi after Stage 1 first fails on the
clean CUDA13 env. After that, the rental cost (~$1.50) easily beats more
unpaid debugging.

## Justification

Stage 1 needs a single A100/H100/L40S-class GPU with a modern CUDA driver
for a tiny `(seq_len=256, batch=8, hidden=128, 2-layer)` HollowMD4 model.
The actual training-loop check runs in minutes — what we need is **clean
hardware without MIG partitioning**. Renting an hour of cloud GPU is far
cheaper than waiting weeks on either the authors or the cluster admins.

Once Stage 1 passes, Stage 2 (throughput benchmark) and the eventual
useful-checkpoint training run can in principle return to Bocconi, since
the bottleneck identified by Stage 1 will be the env, not the hardware.

## Recommended machine type

| Class | Examples | Why |
|---|---|---|
| **A100 80GB single-tenant** | Lambda Labs, RunPod, GCP `a2-highgpu-1g` | Closest hardware match to Bocconi; rules out hardware bugs. |
| **L40S 48GB** | Lambda Labs, RunPod | Cheaper; plenty of VRAM for Stage 1 + Stage 2; cuda13-driver options. |
| **H100 80GB** | RunPod, Lambda Labs, GCP `a3-highgpu-1g` | Overkill for Stage 1 but useful if we later need Stage 3+ throughput. |

Avoid: anything in MIG mode, anything with <16 GB VRAM, anything with
CUDA driver older than 12.0.

Pricing (mid-2026, indicative): A100 80GB ~$1.30–2.00/hr,
L40S ~$0.80–1.20/hr, H100 80GB ~$2.50–3.50/hr. Total expected spend for
Stage 0+1+2 round-trip: **under $10** — far below the cost of a paused
thesis week.

## Setup (one-shot, on the rented machine)

These commands assume an Ubuntu 22.04 image with NVIDIA drivers
pre-installed (the default for Lambda Labs / RunPod GPU instances).
Do **not** put credentials into this repo; configure SSH keys and any
cloud-API tokens via the rental provider's UI.

```bash
# Verify GPU is visible.
nvidia-smi

# Install miniconda3 if missing.
if ! command -v conda >/dev/null 2>&1; then
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Clone the thesis repo (read-only mirror is fine; we will push results
# back via scp, not git).
git clone https://github.com/matteo-omizzolo/masked-diffusion-thesis.git
cd masked-diffusion-thesis
git checkout codex/informed-correctors-backend-validation

# Stage the upstream informed-correctors checkout.
mkdir -p external_repos
git -C external_repos clone https://github.com/lindermanlab/informed-correctors
git -C external_repos/informed-correctors checkout 8371dec  # commit pinned in docs/08

# Set up the env and stage Text8 (same idempotent script we use on Bocconi).
bash hpc/backend_validation/informed_correctors/setup_ic_text8_jax13.sh
```

## Data staging

Text8 zip (~30 MB) is downloaded by `setup_ic_text8_jax13.sh` to
`$TEXT8_DATA_DIR/text8/text8.zip`. **No model checkpoints are required**
for Stage 0/1.

If you ever need ProSeCo-OWT for cross-checks, **do not** pull it via this
fallback path — that's a 16 GB download and outside the rules. ProSeCo
work stays on Bocconi.

## Stage 0 command (external machine)

The sbatch scripts assume SLURM; on a single-node external machine, invoke
the underlying Python directly:

```bash
conda activate ic_text8_jax13
mkdir -p logs results/backend_validation/informed_correctors
SHA="$(git rev-parse --short HEAD 2>/dev/null || cat .hpc_git_sha 2>/dev/null || echo nogit)"
OUT_DIR="results/backend_validation/informed_correctors/stage0_env_smoke_${SHA}_extgpu"
mkdir -p "$OUT_DIR"
python scripts/backend_validation/informed_correctors/check_text8_training_feasibility.py \
    --external-repo external_repos/informed-correctors \
    --data-dir "$HOME/data/text8_md4" \
    --check-imports --check-jax --check-compute --check-data \
    --write-report "$OUT_DIR/feasibility.json" 2>&1 | tee "logs/stage0_extgpu.log"
```

Pass criteria are identical to the Bocconi Stage 0 sbatch.

## Stage 1 command (external machine)

```bash
conda activate ic_text8_jax13
SHA="$(git rev-parse --short HEAD 2>/dev/null || cat .hpc_git_sha 2>/dev/null || echo nogit)"
OUT_DIR="results/backend_validation/informed_correctors/stage1_tiny_train_${SHA}_extgpu"
RUN_DIR="$OUT_DIR/upstream_copy"
mkdir -p "$OUT_DIR" "$RUN_DIR" logs

IC_REPO="$PWD/external_repos/informed-correctors"
TEXT8_DATA_DIR="$HOME/data/text8_md4"
TEXT8_WORD_VOCAB_PKL="$TEXT8_DATA_DIR/text8_word_vocab.pkl"

# Tiny placeholder vocab (matches what the sbatch creates).
if [ ! -f "$TEXT8_WORD_VOCAB_PKL" ]; then
    python -c "import pickle; pickle.dump({'the','of','and','to','in','a','is','that','for','it'}, open('$TEXT8_WORD_VOCAB_PKL','wb'))"
fi

# Copy + patch upstream tree (never touch external_repos/). This applies the
# same DATA_DIR and distrax/DiscretizedLogisticMixture guards used by the
# Bocconi sbatch.
python scripts/backend_validation/informed_correctors/prepare_text8_upstream_copy.py \
    --external-repo "$IC_REPO" \
    --run-dir "$RUN_DIR" \
    --data-dir "$TEXT8_DATA_DIR"

export PYTHONPATH="$RUN_DIR/text8:${PYTHONPATH:-}"
export WANDB_MODE=offline
export TEXT8_WORD_VOCAB_PKL
export IC_STAGE1_STEPS=50
export IC_STAGE1_BATCH_SIZE=8

python "$RUN_DIR/text8/md4/main.py" \
    --config="$PWD/scripts/backend_validation/informed_correctors/hollow_text8_stage1_config.py" \
    --workdir="$OUT_DIR/workdir" \
    --sharded=false 2>&1 | tee "logs/stage1_extgpu.log"
```

Pass criteria are identical to the Bocconi Stage 1 sbatch.

## Cost / time estimate

| Step | Wallclock | Cost (A100 @ $1.50/hr) |
|---|---|---|
| Env + data setup | 10 min | $0.25 |
| Stage 0 smoke | 2 min | $0.05 |
| Stage 1 tiny train | 10 min | $0.25 |
| Stage 2 throughput (1k steps, deferred) | 30 min | $0.75 |
| Total Stage 0+1+2 | ~1 hour | **~$1.50** |

A useful-checkpoint training run is **out of scope** for this fallback;
that decision belongs to the user with full evidence in hand.

## Stop criteria

Stop the external-GPU work and shut down the instance when:

- Stage 1 passes and an Orbax checkpoint is written; OR
- Stage 1 fails for a reason that is **not** CUDA/driver/MIG-shaped (e.g.
  an upstream md4 bug, a data-staging mistake, a config error). In that
  case, fixing the bug applies equally to Bocconi, so move back; OR
- One hour of wallclock has been spent without progress (sanity cap).

Do not let an external instance idle. Terminate immediately after results
are pulled.

## Bringing results back

```bash
# Tar + scp the SHA-tagged result dirs back to the local repo.
tar -czf /tmp/extgpu_stage01.tar.gz \
    results/backend_validation/informed_correctors/stage0_env_smoke_*_extgpu/ \
    results/backend_validation/informed_correctors/stage1_tiny_train_*_extgpu/ \
    logs/stage0_extgpu.log logs/stage1_extgpu.log
# From your laptop:
scp <user>@<external-host>:/tmp/extgpu_stage01.tar.gz /tmp/
cd /Users/matteoomizzolo/masked-diffusion-thesis && tar -xzf /tmp/extgpu_stage01.tar.gz
```

The `_extgpu` suffix preserves provenance and prevents collisions with
Bocconi outputs at the same SHA.

After unpacking, write a short interpretation under
`results/backend_validation/informed_correctors/stage1_tiny_train_<sha>_extgpu/INTERPRETATION.md`,
update `results/BACKEND_FEASIBILITY_INDEX.json` with the new entry, and
commit on `codex/informed-correctors-backend-validation`.

## Security / hygiene

- **No secrets in this repo.** SSH keys, cloud-API tokens, and W&B keys go
  in `~/.ssh/`, the rental provider's UI, or `~/.netrc` — never in tracked
  files.
- **No huge downloads.** Only Text8 (~30 MB) and the
  `external_repos/informed-correctors` clone (~1 MB) are required.
- **No useful checkpoint training.** Stage 1 is ≤50 steps with a tiny
  config; Stage 2 (deferred) is ≤5000 steps for benchmarking only.
- **Terminate the instance.** Do not leave external GPU instances running
  after results are pulled.
