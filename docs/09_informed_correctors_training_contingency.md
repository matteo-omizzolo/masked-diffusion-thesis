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

#### Stage 0 known blocker (job 494155, 2026-05-13)

The `remdm311` conda env on the Bocconi cluster was built for the
PyTorch/ProSeCo line and does not contain the JAX ecosystem. Job 494155
correctly produced a feasibility report and exited 1; the report shows the
following remediation steps are required before Stage 0 can pass:

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
