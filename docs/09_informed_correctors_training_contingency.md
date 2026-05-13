# informed-correctors/Text8 Training Contingency

## A. Can We Train Without Author Reply?

Classification: **PROBABLY YES**.

The informed-correctors repo contains the key ingredients for Text8 HollowMD4
training: a Text8 HollowMD4 config, Text8 tokenizer/preprocessing logic,
HollowMD4 architecture, Gibbs/informed sampling code, optimizer settings, and
Orbax checkpointing/resume support.

It is not a clean one-command reproduction. The repo has practical gaps:

- `text8/md4/input_pipeline.py` hardcodes `DATA_DIR = "/root/md4/data_dir"`.
- `train.py` unconditionally opens `config.vocab_dir`, but the HollowMD4 Text8
  config does not define `vocab_dir`.
- W&B is imported and initialized; offline mode or a local setup is needed.
- Exact paper-quality evaluation targets/checkpoint quality are not documented.
- No public Text8 checkpoint was found.

These are engineering/configuration issues, not scientific blockers. Exact
paper reproduction still needs author clarification.

## B. Required Information

| Ingredient | Status | Notes |
|---|---|---|
| Text8 dataset source | present in repo | `input_pipeline.py` references `http://mattmahoney.net/dc/text8.zip`; training path expects staged zip. |
| Text8 preprocessing | present | 90M/5M/5M split, 27-character tokenizer, sequence length 256. |
| Vocabulary construction | partially missing | char vocab is implicit; `config.vocab_dir` word-vocab pickle for sample post-processing is missing from Hollow config. |
| Sequence length | present | `data_shape=(256,)`. |
| Diffusion steps/schedule | present | `timesteps=1000`, linear noise schedule, cosine sampling grid. |
| HollowMD4 architecture | present | `model_type="hollow_md4"`, `n_layers=36`, `hidden_dim=2048`, `num_heads=12`, mixed hollow layers. |
| Training objective | present | masked loss in HollowMD4/MD4 code. |
| Corrector type/hyperparameters | present but incomplete | Gibbs sampler implemented; config default sampler is `ancestral`, so exact reported corrector settings need clarification. |
| Optimizer/LR/batch | present | AdamW, LR sweep overrides to 5e-4, dropout 0.05, weight decay 0.03, batch 512, 4 microbatches. |
| Checkpoint save/resume | present | Orbax checkpoint manager with train state and Grain iterator. |
| Sampling command | present but not schedule-ready | `sampling.generate` uses `jax.lax.fori_loop`; schedule-specific timing needs a wrapper. |
| Evaluation metric | unclear | Native training logs loss and a word-validity `acc`; paper-quality bpc/loss target needs clarification. |
| Random seeds | present | config seed exists, default 0 in HollowMD4 config. |
| Hardware requirements | inferable | A100 should be compatible; throughput must be benchmarked. |
| Expected duration | uncertain | 1M steps in config; runtime requires Stage 2 benchmark. |

## C. Minimum Viable Training Stages

### Stage 0: Environment/JAX Smoke

Goal: import dependencies, verify JAX sees GPU, verify local upstream repo, verify
Text8 data path, and load the HollowMD4 Text8 config.

Command:

```bash
sbatch hpc/backend_validation/informed_correctors/stage0_env_smoke.sbatch
```

Expected runtime: minutes.

Pass criteria:

- all required imports succeed;
- JAX reports GPU device;
- `external_repos/informed-correctors/text8` exists;
- Text8 zip is staged at `$TEXT8_DATA_DIR/text8/text8.zip`;
- config loads.

#### Stage 0 environment setup (resolved 2026-05-14)

Initially, `remdm311` on the Bocconi cluster (built for PyTorch/ProSeCo)
did not contain the JAX ecosystem. Stage 0 job 494155 (2026-05-13)
correctly produced a feasibility report and exited 1. The remediation
below was applied on a login node and Stage 0 job 494221 (2026-05-13)
subsequently passed with all criteria green. Steps retained for future
sessions / re-building the env:

**Manual install (from a login node, since compute nodes have no outbound
internet — see CLAUDE.md known issue #12):**

```bash
ssh 3316152@slogin.hpc.unibocconi.it
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate remdm311

# JAX + Flax + ecosystem
pip install --upgrade jax jaxlib flax optax distrax orbax-checkpoint clu grain ml_collections absl-py

# TensorFlow (CPU only; the model itself runs on JAX/CUDA)
pip install --upgrade tensorflow tensorflow-datasets
```

Then stage the Text8 zip manually (no auto-download from compute nodes):

```bash
mkdir -p ~/mdm/data/text8_md4/text8
curl -L http://mattmahoney.net/dc/text8.zip -o ~/mdm/data/text8_md4/text8/text8.zip
```

After both steps, re-launch Stage 0 to confirm a clean pass before
proceeding to Stage 1. CUDA-enabled JAX (`jax[cuda12]`) may also be needed if
GPU enumeration fails on the JAX install above.

#### Stage 0 environment setup — resolution log (2026-05-14)

Stage 0 job 494221 on gnode01 (commit `ec580a8-dirty`) returned exit 0:0 in
1m32s with all pass criteria met: 14/14 imports succeed, JAX reports
`devices=['cuda:0']` with `local_device_count=1`, `model_type=hollow_md4`
config loads, Text8 data resolves at the preferred layout, no downloads.

Pinned versions on remdm311 needed to make the JAX/Flax stack co-exist with
distrax 0.1.8 (the latest published distrax):

- jax / jaxlib / jax-cuda12-* = 0.4.30
- flax = 0.8.5
- optax = 0.2.3
- chex = 0.1.86
- orbax-checkpoint = 0.5.20
- distrax = 0.1.8
- tensorflow-cpu = 2.21.0 (then full `tensorflow` 2.21.0 was pulled in as a
  dependency of `tf-keras`, which `tensorflow_probability` requires)
- tensorflow-datasets = 4.9.10
- tensorflow-probability latest
- tf-keras 2.21.0 (TFP requires it via lazy_loader validation)
- tensorboard 2.20.0 (clu.metric_writers requires it; not in tensorflow-cpu)
- wandb 0.26.1 (upgrade from 0.25.0 was required because the
  JAX-ecosystem install bumped protobuf to 7.x, which broke wandb 0.25.0)
- seaborn 0.13.2 (upstream `text8/md4/utils.py` imports it at module load)

Disk-quota note: the conda env grew to ~13G during these installs. The
~3.4G `~/.conda/pkgs` and `~/.cache/pip` caches were purged with
`conda clean -a && pip cache purge` to make headroom. If quota becomes
tighter, consider freeing space outside this env (e.g. the 15G
`~/mdm/checkpoints/proseco_llada_sft` checkpoint from the closed
cross-backbone line).

### Stage 1 — historical blocker (resolved-by-redirect 2026-05-14)

The remdm311-path Stage 1 attempts described below are kept as
provenance. The **active** Stage 1 path is the dedicated `ic_text8_jax13`
env (`jax[cuda13]`) set up via
`hpc/backend_validation/informed_correctors/setup_ic_text8_jax13.sh`.
That env uses cuda13 PJRT plugins, which match the Bocconi A100 driver
(580.95.05 / CUDA 13.0) and should avoid the MIG/cross-driver crash
documented below. If the new env's Stage 1 still fails, fall through to
the external-GPU fallback in `docs/10_external_gpu_text8_fallback.md`
rather than reverting to remdm311.

#### Original remdm311 Stage 1 failure log

Stage 1 job 494239 on gnode02 (after Stage 0 passed) entered the training
script, printed `Using Hollow MD4` from `utils.py:81`, and then died after
~3m30s of silent JIT compilation with `ExitCode=120:0`, no Python traceback,
no stdout, and an empty Orbax `checkpoints/` directory. SLURM accounting
shows `MaxRSS=5G` (well under the 48G request) so it is not an OOM kill.

Suspected cause: a CUDA 13.0 driver / JAX-cuda12 plugin compatibility issue
specific to the A100 80GB cards running in **MIG (Multi-Instance GPU)** mode
on the Bocconi `stud` partition (confirmed via `nvidia-smi` on gnode02:
`MIG M. Enabled`). JAX 0.4.30's GPU allocator may not negotiate cleanly
against MIG-partitioned A100s under driver 580.95.05 / CUDA 13.0. Stage 0
succeeds because `jax.devices()` returns the device handle but never
allocates compiled kernels.

Diagnostic next steps tried so far (2026-05-14):

1. **`XLA_PYTHON_CLIENT_PREALLOCATE=false` + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`
   + `JAX_PLATFORMS=cuda`** — added to the Stage 1 sbatch. Job 494245
   re-ran with these env vars set; **still failed identically** with
   ExitCode 120:0 after the same `Using Hollow MD4` log line and similar
   wallclock (3m47s). The workaround is left in place in the sbatch since
   it's harmless and may help on other cluster configurations.

Deferred diagnostic next steps (none auto-runnable, require user action):

2. **Ask cluster admins for a non-MIG queue or to disable MIG on `stud`
   nodes.** This is the cleanest fix if available.
3. **Wait for `jax[cuda13]` wheels** to be released (the cuda12 wheels are
   currently the only JAX flavor against CUDA 13 drivers; an official
   cuda13 wheel would skip the cross-version mismatch entirely).
4. **CPU fallback for the timing-diagnostics work.** Run the corrector
   scheduling primitives on a CPU-only JAX setup with a much smaller
   HollowMD4 config. Text8 char-level training would be slow but the
   scheduling primitives (Δ_t, G({s,t}), ξ, G(S)) don't need GPU.
5. **Author-track path.** The 2026-05-14 email requested a shareable
   Text8 HollowMD4 checkpoint. If the authors share one, the timing
   diagnostics can skip training entirely.

The Stage 1 sbatch and python-loop sampler design themselves are sound:
exit-120 happens deep inside JAX/CUDA after model instantiation, well past
any script-level concern. The `external_repos/` upstream clone was NOT
modified — Stage 1 only patches the copied tree at
`results/backend_validation/informed_correctors/stage1_tiny_train_<sha>/upstream_copy/`.

### Stage 1: Tiny Overfit / Training-Loop Smoke

Goal: run a small HollowMD4-family model for 10-100 steps, save a checkpoint,
and verify the training loop is operational. This is not a useful checkpoint.

Command:

```bash
sbatch hpc/backend_validation/informed_correctors/stage1_tiny_train.sbatch
```

Expected runtime: minutes to less than one hour.

Pass criteria:

- training starts;
- loss is finite;
- checkpoint is written;
- W&B runs offline or does not block;
- no external repo files are modified.

### Stage 2: Throughput Benchmark

Goal: run 1k-5k steps with the full or near-full HollowMD4 config and measure
steps/sec, memory, and projected time for 100k, 500k, and 1M steps.

Expected runtime: a few hours max.

Do not run until Stage 0/1 pass.

### Stage 3: Useful Checkpoint

Goal: train enough to produce non-degenerate samples and run a timing headroom
gate. A provisional target is 100k-250k steps, but the step count should be set
only after Stage 2 throughput and quality curves are available.

### Stage 4: Timing Headroom Gate

Goal: evaluate no-corrector, default-corrector, uniform-budget schedules, and
MC-oracle schedules using `Delta_t`, `G({s,t})`, `xi`, and `G(S)`.

Do not proceed until the checkpoint is non-degenerate.

## D. Good-Enough Checkpoint Criteria

A checkpoint is good enough for timing diagnostics only if:

- generated samples are not all blank, all mask, or trivially repetitive;
- training/eval loss improves over a random or untrained baseline;
- the corrector changes samples nontrivially;
- default correction improves the chosen utility over no correction;
- schedule budget has measurable headroom under paired seeds;
- metric variance is low enough for paired comparisons.

## E. If Training Fails

Fallbacks:

1. Ask the authors again with the exact missing item.
2. Use a smaller controlled Text8-like model to validate timing diagnostics.
3. Treat ProSeCo-OWT as the only empirical backend and make the external-validity limitation explicit.
4. Consider MDLM only if a cleaner public checkpoint and timing-controllable corrector are available.

## F. Current Missing Author Clarifications

- Shareable Text8 HollowMD4 checkpoint.
- Exact Text8 training config used for reported results.
- Word-vocab pickle or construction rule for `config.vocab_dir`.
- Exact sampling/evaluation commands.
- Corrector sampler and hyperparameters used in Text8 experiments.
- Expected loss/bpc/sample-quality target for a successful reproduction.
- Advice on schedule-masking corrector calls at selected time steps.
