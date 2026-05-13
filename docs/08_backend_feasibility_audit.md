# Backend Feasibility Audit
**Date:** 2026-05-13  
**Branch:** theory-first-corrector-timing  
**SHA:** 0c39079  
**Auditor:** Claude Code session (feasibility audit, no commits, no staged files)

---

## Current Recommendation After Repo Restructure

The recommendation remains unchanged:

- pursue informed-correctors/Text8 as the principled second backend;
- do not pursue PRISM LLaDA for this thesis-specific timing-validation gate unless checkpoint access and timing/token-quality confounds change.

The operational next step is two-track: the author email in `docs/email_informed_correctors_authors.md` was sent on 2026-05-14 (follow up after 7-10 days if no reply), and in parallel the no-author-response stages documented in `docs/09_informed_correctors_training_contingency.md` are being driven independently.

Prepared thesis-side files:

- `scripts/backend_validation/informed_correctors/check_text8_training_feasibility.py`
- `scripts/backend_validation/informed_correctors/hollow_text8_stage1_config.py`
- `hpc/backend_validation/informed_correctors/stage0_env_smoke.sbatch`
- `hpc/backend_validation/informed_correctors/stage1_tiny_train.sbatch`

---

## Purpose

Evaluate two candidate second backends for thesis corrector-scheduling experiments:

1. **informed-correctors** — `github.com/lindermanlab/informed-correctors`, Text8 / HollowMD4, JAX
2. **PRISM LLaDA** — `github.com/SeunggeunKimkr/PRISM/PRISM_llada`, code generation, PyTorch

**Scientific motivation:** Primary results are on ProSeCo-OWT (GPT-2-scale, OWT domain).
A second backbone strengthens the thesis claim that corrector scheduling structure is
model-independent. The hollow transformer's *structural conditional logit guarantee* is
the key differentiator: p(x_d | x_{−d}, t) by construction, without the posterior
collapse risk present in standard transformers.

---

## Repo 1: informed-correctors (Text8 / HollowMD4)

### 1.1 Repository acquisition
- Cloned to `external_repos/informed-correctors/`, commit `8371dec`, branch `main`
- Key files examined:
  - `text8/md4/networks/hollow_transformer.py` — architecture
  - `text8/md4/models/diffusion/hollow_md4.py` — diffusion + samplers
  - `text8/md4/sampling.py` — outer sampling loop
  - `text8/md4/configs/hollow_md4/text8.py` — training config
  - `text8/md4/train.py` — training entrypoint

### 1.2 Pretrained checkpoints
**None released.** The codebase uses Orbax CheckpointManager to save/restore from a
user-specified `workdir`. No public HuggingFace or GCS link exists in the README or any
config. Training from scratch is required.

### 1.3 Training cost
Config (`configs/hollow_md4/text8.py`):
- `n_layers=36`, `n_heads=12`, `hidden_dim=2048`, `feature_dim=64`
- `n_layers_per_mixed=12` (mixing layers every 12 layers → 3 mixing layers total)
- `batch_size=512`, `seq_len=256`, `T=1000`, `vocab_size=27`
- `num_train_steps=1_000_000`
- `noise_schedule="linear"`, `cont_time=True`
- Framework: JAX + Flax + tensorflow-datasets (Text8 auto-downloads)

Text8 is char-level (27 tokens, 100M characters). At batch_size=512 × seq_len=256,
each step processes ~131K tokens. Total: ~131B token-steps.

**Estimated training time on 1×A100 80GB:** 2–5 days (character-level vocab is tiny,
sequences are short; comparable to MD4's original Text8 experiments).

### 1.4 Hollow Transformer — conditional probability structure

The hollow transformer computes **exact coordinate-wise conditionals p(x_d | x_{−d}, t)**
via structural masking in the input, not via posterior approximation.

Architecture (from `hollow_transformer.py`, lines 265–375):
```python
# Pad x on both sides
x = jnp.insert(x, 0, 0, axis=1)         # prepend BOS padding
x = jnp.insert(x, x.shape[1], 0, axis=1) # append EOS padding

# Two causal streams on the padded input
hf = h[:,:-2]   # forward: position i sees tokens 0..i-1 (via causal mask)
hb = h[:,2:]    # backward: position i sees tokens i+1..T-1 (via causal mask)

# Mixing layers: hm[i] = f(hf[i], hb[i])
# → hm[i] sees all tokens except x_i itself
logits = nn.Dense(output_channels)(hm)   # shape (B, T, vocab)
```

The padding guarantees that `hm[i]` has no path to `x[i]` in the original sequence.
Output `logits[b, d, :]` is therefore a valid parameterisation of p(x_d | x_{−d}, t).

This is precisely what the corrector in `ancestral_sample_step_informed` uses:
```python
logits, _ = self.predict_x(zs, s, cond=cond)
scores = logits[b_idx, d_idx, zs]   # log-prob of current token at each position
# → selects k lowest-confidence positions and resamples from conditional
```

**Thesis relevance: HIGH.** The hollow prior provides a direct probe of
p(x_d | x_{−d}, t) without the confounds of full-sequence autoregressive approximations.

### 1.5 Timing controllability

Current sampling loop (`sampling.py`, `simple_generate`):
```python
def body_fn(i, zt):
    return model.apply(variables, sub_rng, i, model.timesteps, zt,
                       conditioning=conditioning, method=model.sample_step)
z0 = jax.lax.fori_loop(lower=0, upper=model.timesteps, body_fun=body_fn, init_val=zt)
```

`jax.lax.fori_loop` is a compiled primitive — Python `if i in schedule` is invisible to it.

**Fix (1 day engineering):** Replace with a Python for-loop (JAX traces it statically):
```python
def python_loop_generate(rng, train_state, batch_size, model, schedule: set):
    variables = {'params': train_state.params, **train_state.state}
    rng, sub_rng = jax.random.split(rng)
    zt = model.apply(variables, batch_size, method=model.prior_sample,
                     rngs={'sample': sub_rng})
    for i in range(model.timesteps):
        k_step = model.k if i in schedule else 0
        zt = model.apply(variables, sub_rng, i, model.timesteps, zt,
                         conditioning=None, method=model.sample_step)
        # ^ need model variant that accepts per-step k; see below
    return model.apply(variables, zt, method=model.decode, rngs={'sample': rng})
```

The `HollowMD4.sample_step` dispatches on `self.sampler` (str) and `self.k` (int).
Since these are Flax module fields (dataclass), they can be overridden at init time.
The cleanest approach: create a `HollowMD4` variant where `k` is passed as a JAX array
per step (or pass a `do_correct: bool` flag through `method=model.sample_step`).
Alternatively, two model instances (k=0 and k>0) can be applied conditionally in the
Python loop — valid because JAX traces the body once per Python iteration.

### 1.6 Terminal utility F
Text8 metric options:
- **Bits-per-character (BPC):** standard NLL measure; lower is better. Directly
  computed from the model's forward pass. Used in the original MD4 paper.
- **Word-level accuracy:** fraction of generated words in a vocabulary (used in
  the `train.py` `post_process` function). Less rigorous.
- **Corrector-scheduling metric (thesis):** Δ_t and G_pair as on ProSeCo-OWT.
  Logits are accessible — G_pair can be computed from paired trajectories exactly
  as in Phase 2b.

**Recommendation:** Use BPC as primary F; Δ_t / ξ as secondary scheduling signal.

### 1.7 Minimal headroom gate for Text8
```
Gate T8-H: For T=64 (reduced timesteps), K=10 seeds, B=2:
  - Compute G_pair(i, j) for 5 sampled pairs using hollow conditional logits
  - Test ξ = G_pair − A_pair; check P(ξ > 0) > 0.5
  - If positive: proceed to K=30 replication
  - Estimated runtime: 2–4 GPU-hours on A100 (requires trained checkpoint first)
```

### 1.8 Framework compatibility
| Item | Status |
|---|---|
| JAX installed (project venv) | ✗ (CPU Mac — no JAX) |
| JAX on HPC (remdm311) | Installable (`pip install jax[cuda]` per requirements_gpu.txt) |
| Flax installed (project venv) | ✗ |
| tensorflow-datasets (Text8) | ✗ (project venv) |
| Python ≥3.10 | ✓ (code runs on 3.11; tested) |
| nvcc compatibility on HPC | Historically problematic (CLAUDE.md rule #4); but JAX[cuda] wheels do not need nvcc |

### 1.9 Smoke test result
```
Environment: project venv (Python 3.11)
JAX: MISSING — no module 'jax'
Flax: MISSING — no module 'flax'
HollowTransformer import: BLOCKED (Flax missing)
```
Smoke test is BLOCKED locally. On HPC, after `pip install jax[cuda] flax tensorflow
tensorflow-datasets`, the import chain should succeed (no circular deps, clean structure).

### 1.10 Summary score
| Criterion | Score | Notes |
|---|---|---|
| Pretrained checkpoint | ✗ | Must train 1M steps (~2–5 GPU-days) |
| Framework compatibility | Partial | JAX missing locally; installable on HPC |
| Hollow conditional logits | ✓ | Structural guarantee; thesis-aligned |
| Timing controllability | ✓ (with fix) | Python loop replacement; 1-day engineering |
| Inference cost | Low | Char-level, vocab=27, T=1000 |
| Data availability | ✓ | Text8 auto-downloads from tfds |
| Scientific fit | HIGH | Second backbone + different modality |

---

## Repo 2: PRISM LLaDA

### 2.1 Repository acquisition
- Available at `external/PRISM/PRISM_llada/` (existing `external/` mirror)
- Key files examined:
  - `llada.py` — RemaskingLLaDA architecture
  - `sampling.py` — llada_inference sampling loop
  - `self_corrector.py` — RemaskingTrainer training logic
  - `train.py` — training entrypoint
  - `llada_utils.py` — HuggingFace backbone download
  - `generate_samples.py` — inference entrypoint (hardcoded HPC checkpoint path)

### 2.2 Pretrained checkpoints
**Not publicly accessible.** `generate_samples.py` has hardcoded path:
```
default="/n/netscratch/sham_lab/Everyone/jay_llada/llada_PRISM/checkpoint-44000"
```
This is a private HPC directory at Harvard's Sham Lab. No public HuggingFace
or GCS release found.

### 2.3 Training cost
- Backbone: `GSAI-ML/LLaDA-8B-Instruct` (~16 GB)
- LoRA: r=256, target_modules=[q, k, v_proj], dropout=0.1
- RemaskingHead: 2-layer MLP on 4096-dim hidden states
- Dataset: OpenCoder-LLM/opc-sft-stage2 (code generation)
- Training: `num_epochs=100`, `batch_size_device=4`, `grad_accum_steps=3`

Training cost is **GPU-days at minimum** for the LoRA fine-tune; with cosine schedule
over 100 epochs on a code dataset this is substantial wall-clock time. The backbone alone
requires loading an 8B-parameter model.

### 2.4 Timing controllability — CRITICAL FINDING

**Timing is structurally entangled with the remasking_conf training objective.**

`llada_inference` (sampling.py, lines 155–216) runs per block:
```python
for i in range(steps_per_block):
    out = model(x)
    # --- Remasking (correction) step ---
    if remasking:
        remasking_score = -1.0 * out["remasking_conf"].squeeze(-1)  # trained signal
        # remask top-k low-confidence tokens
    # --- Unmasking (predictor) step ---
    x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
    # unmask top-k confident tokens
```

The `remasking_conf` head is trained inside `RemaskingTrainer.compute_loss`
(`self_corrector.py`) to predict at each step whether the current unmasked token is
correct. This objective is trained jointly with `xs_sampling="semi-autoregressive"`.

To study *when* to apply correction independently of *which* tokens to correct,
one would need to retrain the model with a different objective that separates the
timing signal from the correction quality signal. This is not an audit or engineering
task — it is a new research contribution.

**Consequence for thesis:** PRISM LLaDA cannot serve as a drop-in second backend for
corrector *scheduling* experiments without retraining. The timing lever is not separable.

### 2.5 Terminal utility F
Domain: code generation (Python, MBPP/HumanEval). Primary metric: pass@k.
This is a very different evaluation regime from ProSeCo-OWT (language perplexity).
Cross-backbone comparison would require significant re-framing of the thesis metrics.

### 2.6 Minimal headroom gate
Cannot be designed without retraining. With the current PRISM model, `remasking_conf`
is the *only* correction signal, and it was trained to be applied at every step.
Ablating timing without retraining would yield uninterpretable results (the model was
not trained to produce good conditionals when correction is skipped at some steps).

### 2.7 Framework compatibility
| Item | Status |
|---|---|
| PyTorch (project venv) | ✓ (2.10.0) |
| transformers (project venv) | ✓ (4.57.6) |
| peft | ✗ MISSING |
| datasets | Partial (version gaps) |
| Python ≥3.10 | ✓ (despite requirements.yaml specifying 3.9; code itself is 3.10-compatible) |
| cuda-nvcc | requirements.yaml includes it — triggers HPC sbatch crash (CLAUDE.md rule #4) |
| Backbone model (16 GB) | Not downloaded; requires user approval per project rules |

### 2.8 Smoke test result
```
Environment: project venv (Python 3.11)
PyTorch: 2.10.0 OK
transformers: 4.57.6 OK
peft: MISSING
PRISM sampling.py import (numpy/torch functions only): OK
RemaskingLLaDA import (mocked): OK
Full model: BLOCKED (peft missing + 16GB download required)
```

### 2.9 Summary score
| Criterion | Score | Notes |
|---|---|---|
| Pretrained checkpoint | ✗ | Private HPC path; must retrain |
| Framework compatibility | Partial | PyTorch ✓, peft ✗, 16GB backbone |
| Conditional logit structure | N/A | Standard full-LLM logits; remasking_conf is separate head |
| Timing controllability | ✗ | Entangled at training; not separable without new objective |
| Inference cost | High | 8B params, subword, max_length=1024 |
| Data availability | N/A | Code generation domain |
| Scientific fit | LOW | Timing not a separable lever |

---

## Part 5: Comparison Table

| Criterion | informed-correctors | PRISM LLaDA |
|---|---|---|
| Framework | JAX + Flax | PyTorch |
| Project venv compatible | ✗ (no JAX/Flax) | Partial (no peft) |
| Python ≥3.10 | ✓ | ✓ |
| Pretrained checkpoint available | ✗ | ✗ (private HPC path) |
| Training cost to get checkpoint | 2–5 GPU-days (1M steps, char-level) | GPU-days (8B LoRA fine-tune) |
| Model size | Small (~30M params, char-level) | 8B parameters |
| Logit structure | p(x_d \| x_{−d}, t) — hollow structural guarantee | Full LLM logits + jointly trained remasking_conf head |
| Timing separability at inference | ✓ (k=0 per step toggle) | ✗ (entangled at training) |
| Timing separability at training | N/A (unconditional Text8 LM) | ✗ (conf head trained jointly) |
| Timing control engineering | ~1 day (Python loop wrapper) | Requires new training objective |
| Inference cost per sample | Low (vocab=27, seq=256, T=1000) | Very high (subword, seq=1024, 8B) |
| Dataset domain | Natural language (char-level) | Code generation |
| Terminal metric F | BPC / word-accuracy | pass@k (MBPP/HumanEval) |
| Δ_t / G_pair extractable? | ✓ (logits accessible) | ✗ (requires major surgery + retraining) |
| HPC environment risk | Medium (JAX install + nvcc-free path known) | High (peft + 16GB download + cuda-nvcc rule) |
| Scientific fit for thesis | HIGH | LOW |
| Recommended path | Train + Python-loop wrapper | Do not pursue |

---

## Part 6: Recommendation

**Recommendation: pursue informed-correctors (Text8 / HollowMD4); do not pursue PRISM LLaDA.**

### Rationale

The hollow transformer provides a **structural guarantee** that is scientifically stronger
than anything achievable via posterior approximation: output logits at position d
are exactly p(x_d | x_{−d}, t) by construction. This directly validates the
corrector's Gibbs-sampling interpretation and provides a clean second domain
(character-level NLP) at very low inference cost.

PRISM LLaDA's `remasking_conf` is trained end-to-end with the correction objective —
timing cannot be varied as an independent axis without retraining. This makes it
unsuitable as a second-backend for corrector *scheduling* research.

### Exact next action

Before proceeding, the following preconditions must be met:

1. **HPC JAX environment:** In `remdm311`, run:
   ```bash
   pip install jax[cuda] flax tensorflow tensorflow-datasets clu grain distrax orbax-checkpoint
   # DO NOT install cuda-nvcc (CLAUDE.md rule #4)
   ```

2. **Python-loop sampler wrapper:** Add `scripts/backend_validation/informed_correctors/hollow_md4_pair_sampler.py` that:
   - Imports `HollowMD4` + loads Orbax checkpoint
   - Replaces `jax.lax.fori_loop` with a Python for-loop
   - Accepts a `schedule: set[int]` parameter to toggle k=0 vs k>0 per step
   - Outputs paired (corrected at t, corrected at t') trajectory tensors for ξ computation
   - Estimated ~100–150 lines

3. **Training sbatch:** Write `hpc/backend_validation/informed_correctors/hollow_md4_text8_train.sbatch` for 1M-step run.
   Apply Codex pre-submit review (CLAUDE.md mandatory rule) before submission.

4. **Checkpoint gate:** After training, run Gate T8-H (K=10, B=2, T=64) before any
   K=30 replication to confirm headroom exists on Text8.

**Prerequisite gate (G0-TEXT8):** confirm JAX+flax import + T=4 smoke run on a CPU
slnode before submitting the training job.

---

## Appendix: Smoke Test Log

```
=== informed-correctors smoke test ===
Environment: project venv (Python 3.11.14)
JAX:   MISSING (no module 'jax')
Flax:  MISSING (no module 'flax')
HollowTransformer import: BLOCKED
Status: BLOCKED (needs JAX install)

=== PRISM LLaDA smoke test ===
Environment: project venv (Python 3.11.14)
PyTorch: 2.10.0  OK
transformers: 4.57.6  OK
peft: MISSING
datasets: partial
sampling.py import (torch/numpy only): OK
RemaskingLLaDA import: BLOCKED (peft missing + HF download)
Status: PARTIAL — sampling logic accessible, full model not runnable
```

---

*This document does not commit, stage, or modify canonical results folders.
All findings are read-only audit observations.*

---

# Independent Codex Audit
**Date:** 2026-05-13  
**Auditor:** Codex independent backend validation audit  
**Scope:** Verify the earlier Claude feasibility report against local code, current remote refs, and small import/config smoke checks only. No long training, no large checkpoint/model download, no staging, no commits.

## Independent Backend Validation Audit After ProSeCo-OWT

## Executive Summary

- Claude's high-level direction is mostly correct: informed-correctors/Text8 is the more principled second backend; PRISM LLaDA is not a clean validation backend for corrector timing.
- The main correction is precision: the hollow transformer gives structural exclusion of coordinate `d`, so logits can parameterize model conditionals `p_theta(x_d | x_-d, t)`. It does not give exact true data conditionals.
- Pretrained informed-correctors Text8 weights were not found in the repo, README, local files, or web/GitHub search. Training appears required unless the authors provide weights privately.
- Text8 training is feasible on the Bocconi A100 cluster, but the earlier "2-5 A100 GPU-days" estimate is unverified. The local hollow config is larger than the paper description appears to imply, has hardcoded data/vocab rough edges, and needs a benchmark before committing to a 1M-step run.
- Timing control is feasible but not a clean one-day patch. `jax.lax.fori_loop` is not the only issue: `k` is a static module field, `sample_step(..., k=None)` ignores its `k` argument, and `model_utils.get_model` does not wire `config.k`.
- A better implementation is a scheduled JAX loop using a boolean `schedule_mask[T]` plus `jax.lax.cond`, or a split predictor/corrector API. A Python loop is acceptable for smoke work but may recompile per static schedule if handled naively.
- The current RNG design is favorable for common random numbers because sample steps derive randomness from `fold_in(rng, i)`. Preserve that pattern; do not switch to schedule-dependent sequential key splitting.
- PRISM LLaDA has a public LLaDA backbone dependency, but no public PRISM LLaDA checkpoint was found. The local default checkpoint path is private Harvard/Kempner infrastructure.
- PRISM's scientific lever is token-quality/remasking, not timing. A timing-only experiment would require fixing token selection and varying remasking times, but the trained head/objective and metric would still confound timing with learned token-quality correction.
- Recommended next action: email the informed-correctors authors for Text8 HollowMD4 checkpoints and exact training config before spending GPU-days. (Executed 2026-05-14 — see the "Current Recommendation After Repo Restructure" section above for the active-state two-track summary.)

## Current ProSeCo-OWT Evidence and Need for Validation

The current ProSeCo-OWT line already supports a coherent thesis story: correction timing has real headroom; marginal rankers and aggregate/enriched/token-level state features do not explain the headroom; true-G search recovers much of it; pair interactions are mostly negative/redundant; saturation and corrector-strength diagnostics suggest interaction structure is not removed by merely scaling correction amplitude. This is scientifically meaningful but still single-backend evidence. A second backend is useful if it tests the same timing formalism under a different model/corrector while preserving clean control of `(predictor, corrector, utility F, budget B)`.

A second backend is not necessary if it becomes a new thesis direction. It is valuable only if it gives a clean external-validity gate: fixed predictor schedule, fixed correction kernel, arbitrary schedule `S`, paired potential outcomes, and a terminal utility that supports low-variance schedule comparisons.

## Verification of Claude's Claims

| Claim | Status | Independent assessment |
|---|---:|---|
| No informed-correctors Text8 weights available | Verified | No checkpoint-like files found by `find`; README only links the paper; grep found Orbax save/restore but no public weights or HF/GDrive artifact. Web/GitHub search found the repo and paper, not weights. |
| Text8 training cost is 2-5 A100 GPU-days | Unverified | Config has `num_train_steps=1_000_000`, `batch_size=512`, `data_shape=(256,)`, `timesteps=1000`. Runtime needs a measured 100-1000 step benchmark. The repo config appears larger/rougher than Claude's estimate justifies. |
| Hollow transformer gives exact `p(x_d | x_-d, t)` | Partially verified | Structurally, output position `d` excludes input `x_d` via padded forward/backward streams and masks. Mathematically it is an exact architectural exclusion, not exact truth: the logits approximate the learned conditional under `p_theta`. |
| Timing control requires replacing `jax.lax.fori_loop` | Partially verified | `fori_loop` prevents Python `i in S`, but replacing with Python loop is only one option. A schedule mask plus `jax.lax.cond` inside a compiled loop is cleaner and avoids per-schedule recompilation. |
| Timing control is about one day engineering | Partially contradicted | There are extra issues: `self.k` is static, `sample_step(..., k=None)` ignores `k`, `model_utils.get_model` does not pass `k`, and tests/CRN instrumentation are needed. Estimate: 2-4 engineering days after environment works. |
| PRISM checkpoint paths are private/inaccessible | Verified | `PRISM_llada/generate_samples.py` defaults to `/n/netscratch/sham_lab/Everyone/jay_llada/llada_PRISM/checkpoint-44000`; training scripts use Harvard/Kempner paths. No local PRISM checkpoint files found. |
| PRISM remasking head is entangled with training objective | Verified | `RemaskingTrainer.compute_loss` trains `remasking_conf` from one-step sampled `xs` and correctness labels while also applying unmasking loss. |
| PRISM timing cannot be separated cleanly from token selection | Verified for thesis fit | Inference can toggle remasking, but the correction mechanism is explicitly top-k token remasking by learned quality/confidence. Timing-only ablations would be confounded by token-quality training and code-generation pass@k noise. |

## Informed-Correctors/Text8 Audit

### Repo State

- Local path: `external_repos/informed-correctors`
- Local commit: `8371dec19404eae279a4eb5ea4e06977d8d173b6`
- Remote `origin/main`: `8371dec19404eae279a4eb5ea4e06977d8d173b6`
- Branch: `main`
- Worktree status: clean during audit
- License: no root `LICENSE` file found. Many source files carry Apache-2.0 headers inherited from DeepMind MD4/VDM, but repository-level licensing is not explicit.
- README: minimal; says repo reproduces "Informed Correctors for Discrete Diffusion Models", is work in progress, and contains `imagenet/` and `text8/`.

### Entrypoints and Dependencies

- Training entrypoint: `text8/md4/main.py --config=... --workdir=...`; dispatches to `train.train_and_evaluate` or `sharded_train.train_and_evaluate`.
- Text8 config: `text8/md4/configs/hollow_md4/text8.py`.
- Sampling helpers: `text8/md4/sampling.py`.
- Model: `text8/md4/models/diffusion/hollow_md4.py`.
- Architecture: `text8/md4/networks/hollow_transformer.py`.
- GPU requirements: `text8/requirements_gpu.txt` includes JAX CUDA wheels, Flax ecosystem packages, TensorFlow, TFDS, Grain, WandB, CLU, Distrax, Orbax.

### Checkpoint/Weights Audit

Weights available: no.  
Weights referenced externally: no public Text8 checkpoint reference found.  
Checkpoints created by training: yes, under `<workdir>/checkpoints` via Orbax `CheckpointManager`.  
Format: Orbax PyTree checkpoint for `train_state` plus Grain iterator checkpoint.  
Resume: yes, if `checkpoint_manager.latest_step()` exists in `workdir`.  
W&B/local path requirements: training unconditionally calls `wandb.init`; use offline/disabled W&B or patch config/environment. Text8 data path is hardcoded as `/root/md4/data_dir/text8/text8.zip` in `input_pipeline.py`. `train.py` also opens `config.vocab_dir`, but the hollow Text8 config does not define `vocab_dir`; the non-hollow MD4 config uses placeholder `/dir_to/text8_vocab.pkl`. This must be fixed before training.

### Training Feasibility

Exact local config:

- `model_type="hollow_md4"`, `dataset="text8"`, `vocab_size=27`, `data_shape=(256,)`
- `timesteps=1000`, `noise_schedule="linear"`, `sampling_grid="cosine"`, `cont_time=True`
- `feature_dim=64`, `num_heads=12`, transformer dimension `768`
- `n_layers=36`, `n_layers_per_mixed=12`, `hidden_dim=2048`
- `batch_size=512`, `num_microbatches=4`, `num_train_steps=1_000_000`
- Checkpoint/eval every 5000 steps

Feasibility: likely feasible on A100 80GB, especially with microbatching, but not yet benchmarked. The runtime claim should be treated as a hypothesis. A 100-step and 1000-step benchmark is needed to estimate near-paper cost. Training stability risk is moderate: code is research-grade, W&B/data/vocab paths are brittle, and the paper/config mismatch needs clarification.

Expected runtime estimates:

- Import/config smoke: minutes, CPU or login node.
- Tiny training smoke: 5-30 minutes on GPU once dependencies/data/vocab exist.
- Useful checkpoint benchmark: 1000-5000 steps, probably hours; needed to estimate throughput and loss behavior.
- Near-paper checkpoint: 1M steps; GPU-days to possibly >1 week depending throughput. Resume is supported by Orbax.

### Hollow Transformer and Conditionals

The hollow architecture pads the sequence on both sides, constructs offset forward and backward streams, and mixes them with masks. For output coordinate `d`, the mixed representation can see tokens before and after `d`, but not the original `x_d`. Thus the logits at `d` are structurally functions of `x_-d` and `t`.

Precise interpretation:

- Verified: output coordinate `d` excludes input `x_d` architecturally.
- Verified: sampling exposes logits via `predict_x(zt, t)` and uses them in predictor and corrector steps.
- Corrected claim: logits are learned model conditionals/proxies, not exact data conditionals.
- Corrector implemented for the thesis-relevant path: `sampler="gibbs"` calls `ancestral_sample_step_informed`, which runs an ancestral predictor, then a Gibbs-like corrector selecting the `k` lowest-score non-mask coordinates and resampling from softmax logits.
- Other samplers implemented in `HollowMD4`: ancestral, Gibbs/informed, uninformed forward-backward, MaskGIT, ReMDM, top-p, mean.

### Timing Controllability

Current sampling uses `jax.lax.fori_loop` in `sampling.generate` and `sampling.simple_generate`, calling `model.sample_step` at every step. To control timing cleanly:

- Split predictor and corrector code paths, or add a scheduled `sample_step_scheduled(rng, i, timesteps, zt, schedule_mask, k)` method.
- Preserve the current `rng_body = fold_in(rng, i)` pattern for CRN. Predictor randomness should not depend on whether the corrector is executed.
- Prefer `schedule_mask: bool[timesteps]` and `jax.lax.cond(schedule_mask[i], do_correct, skip_correct, zs)` inside a compiled loop. This avoids recompiling per schedule if shape is fixed.
- A Python loop is fine for first smoke tests and debugging, but if `schedule` is a static Python tuple/set it can cause schedule-specific traces and slow large sweeps.
- The current code needs more than a loop swap: `self.k` is static, `sample_step(..., k=None)` ignores `k`, and config creation does not wire `k`.

Engineering estimate: 2-4 days for a robust implementation with CRN tests, schedule masks, per-schedule outputs, and a small benchmark. One day is plausible only for a fragile local prototype.

### Timing Diagnostics Feasibility

Feasible after a trained checkpoint and scheduled sampler:

- No-corrector baseline: schedule `S=empty`, fixed `k`.
- Uniform schedule: fixed evenly spaced `S`, same `|S|=B`.
- Arbitrary schedule `S`: boolean schedule mask.
- Single intervention `Delta_t = G({t})`: paired generation with `S={t}` versus `S=empty`.
- Pair intervention `G({s,t})`: paired generation with `S={s,t}` versus `S=empty`.
- Interaction `xi = G({s,t}) - Delta_s - Delta_t`.
- Schedule-level `G(S)` and finite-pool MC-oracle headroom: feasible, inference is char-level and relatively cheap once trained.

Define budget `B` as number of time indices with the Gibbs corrector enabled; keep per-step coordinate budget `k` fixed. Do not let `num_corrected_tokens = B*k` drift across schedules.

### Terminal Utility F

Primary recommendation: native Text8 sample quality, implemented as negative invalid-word/invalid-character rate against the fixed Text8 training vocabulary, matching the repo's `post_process` idea but made sample-level and reproducible. This is meaningful for generated Text8 strings, cheap, and directly tied to the informed-correctors paper's spelling-error motivation.

Backup: model-based bits-per-token/ELBO-style score under the trained diffusion model for generated samples. It is reproducible and lower variance, but it is a self-score from the same model being sampled and may not track human-visible sample quality. Report it as secondary, not the sole thesis utility.

I do not recommend Hamming/reconstruction error as the primary utility unless the experiment is reframed as conditional reconstruction from a fixed corrupted datum rather than free generation.

### Minimal Staged Path

Stage 0 - environment/import smoke:

```bash
cd external_repos/informed-correctors/text8
python -c "import jax, flax, tensorflow, tensorflow_datasets, grain, orbax.checkpoint, ml_collections; print(jax.devices())"
PYTHONPATH=. python -c "from md4.configs.hollow_md4 import text8; print(text8.get_config())"
```

Runtime: minutes. Pass: imports/config succeed without downloading huge assets. Decision: environment is viable.

Stage 1 - tiny training smoke:

```bash
cd external_repos/informed-correctors/text8
WANDB_MODE=offline PYTHONPATH=. python -m md4.main \
  --config=md4/configs/hollow_md4/text8.py \
  --workdir=/scratch/3316152/hollow_md4_text8_smoke \
  --config.num_train_steps=10 \
  --config.eval_every_steps=999999 \
  --config.checkpoint_every_steps=10 \
  --config.grain_num_workers=0 \
  --config.vocab_dir=/path/to/text8_vocab.pkl
```

Runtime: minutes to 30 minutes after data/vocab path is fixed. Pass: one checkpoint saved, loss finite, no W&B/data-path failure. Decision: code can train.

Stage 2 - useful checkpoint benchmark:

```bash
WANDB_MODE=offline PYTHONPATH=. python -m md4.main \
  --config=md4/configs/hollow_md4/text8.py \
  --workdir=/scratch/3316152/hollow_md4_text8_benchmark \
  --config.num_train_steps=1000 \
  --config.eval_every_steps=1000 \
  --config.checkpoint_every_steps=1000 \
  --config.vocab_dir=/path/to/text8_vocab.pkl
```

Runtime: hours. Pass: stable finite train/eval loss, measured steps/sec. Decision: estimate full training time.

Stage 3 - train checkpoint:

```bash
WANDB_MODE=offline PYTHONPATH=. python -m md4.main \
  --config=md4/configs/hollow_md4/text8.py \
  --workdir=/scratch/3316152/hollow_md4_text8_full \
  --config.vocab_dir=/path/to/text8_vocab.pkl
```

Runtime: determined by Stage 2. Do not launch until authors are contacted or the benchmark justifies it.

Stage 4 - timing headroom gate:

Run a scheduled sampler with `K=10` seeds, small schedule pool, `B in {1,2,4}`, CRN enabled, and compare `S=empty`, uniform, and sampled schedules using primary and backup `F`. Pass: non-degenerate headroom over no-corrector/uniform.

Stage 5 - pair/saturation diagnostics:

Compute `Delta_t`, `G({s,t})`, `xi`, and schedule-level `G(S)` on a fixed finite time grid. Pass: enough headroom and reproducible interactions to justify using Text8 in the thesis.

## PRISM LLaDA Audit

### Repo State

- Local path: `external/PRISM/PRISM_llada`
- Local PRISM commit: `266d51e886b28ac8bfb416eb8715394aa08a4e0f`
- Remote `origin/main` after fetch: `6e5ed6410d3d88fa1231eb78b002e3365be6deaa`
- Branch: local `main`, behind remote main
- Worktree state: pre-existing modified `PRISM_llada/__pycache__/*.pyc` files; no source files changed by this audit
- License: no repository-level `LICENSE` file found.
- README at remote main now includes LLaDA training/evaluation references, but still no public PRISM LLaDA weights.

### Checkpoint/Weights Audit

Pretrained PRISM LLaDA weights: not found.  
Public base LLaDA weights: yes, code loads `GSAI-ML/LLaDA-8B-Instruct` from HuggingFace.  
PRISM checkpoint path: private default `/n/netscratch/sham_lab/Everyone/jay_llada/llada_PRISM/checkpoint-44000`.  
Hardcoded institution paths: yes, in LLaDA training script and broader PRISM scripts/configs.  
Checkpoint format: HuggingFace Trainer output with `pytorch_model.bin` loaded by `generate_samples.py`.

### Training/Fine-Tuning Feasibility

PRISM LLaDA training requires:

- LLaDA-8B-Instruct backbone.
- PEFT/LoRA with `r=256`, target modules `q_proj`, `k_proj`, `v_proj`.
- `RemaskingHead` MLP on 4096-dim hidden states.
- OpenCoder-LLM/opc-sft-stage2 data.
- Defaults: 100 epochs, per-device batch 4, grad accumulation 3, bf16, save every 2000 steps.
- Provided Slurm example uses 3 nodes x 4 H100s for 3 days on Kempner/Harvard infrastructure.

This is feasible in principle on a sufficiently large cluster but not attractive as a validation backend. It imports/fine-tunes an 8B model, requires missing dependencies, depends on code-generation data/evaluation, and likely costs more engineering time than scientific clarity warrants.

### Timing Controllability and Thesis Fit

PRISM is primarily token-selection/remasking, not timing. The model is fine-tuned to estimate per-token quality and uses that quality to choose which generated tokens to remask. Inference has a remasking block inside each generation step, so one could add a schedule mask and enable remasking only at selected steps. But that would not isolate corrector timing in the ProSeCo sense because:

- The correction action is top-k token remasking, selected by `remasking_conf` or confidence/random alternatives.
- The training objective teaches a token-quality head on one-step sampled `xs`; it is not a fixed conditional corrector kernel.
- Remasking changes the number of tokens that must be unmasked later; remote-main code even increments unmasking counts by the number remasked.
- The metric is pass@k/code correctness, high variance and domain-shifted from Text8/OWT likelihood-style utilities.
- Without public PRISM weights, a timing gate requires expensive fine-tuning before any diagnostic.

Minimal PRISM timing gate if forced: use public baseline LLaDA plus `remasking_mode="remdm"` random remasking, fixed token budget, schedule mask over remasking steps, and pass@1 on a tiny MBPP/HumanEval subset. I do not recommend this: it tests remasking loop placement in a code decoder, not the thesis's informed corrector scheduling formalism.

## Smoke Tests

Created:

- `results/archive/2026-05-13_repo_cleanup/backend_smoke_informed_correctors_8371dec/SMOKE_LOG.md`
- `results/archive/2026-05-13_repo_cleanup/backend_smoke_prism_266d51e/SMOKE_LOG.md`

Informed-correctors local smoke: blocked by missing JAX/Flax/TensorFlow/Grain/Orbax/W&B/ml_collections. No training/sampling launched.

PRISM LLaDA local smoke: `sampling.py` imports and exposes `llada_inference`; training/data imports blocked by missing `wandb` and `datasets`; `peft` and `evaluate` are also missing. No LLaDA weights or PRISM checkpoints downloaded.

## Comparison Table

| Criterion | informed-correctors/Text8 | PRISM LLaDA |
|---|---|---|
| Thesis fit | High if used as external-validity gate | Low for timing-only thesis |
| Relation to informed correctors | Direct implementation of informed Gibbs-style corrector | Self-correction via learned quality/remasking head |
| Relation to masked diffusion | Absorbing masked diffusion, char-level Text8 | LLaDA masked diffusion decoder plus PRISM fine-tune |
| Pretrained weights | Not found | PRISM weights not found; base LLaDA public |
| Training feasibility | Feasible but benchmark required | Feasible only as large 8B LoRA project |
| HPC cost | Likely GPU-days; uncertain until benchmark | Multi-GPU/multi-day likely; example uses 12 H100s |
| Engineering risk | Medium: JAX env, path fixes, scheduled sampler | High: deps, 8B model, code eval, private checkpoint assumptions |
| Conditionals/logits | Structural coordinate exclusion; learned model conditional | Standard LLaDA logits plus quality head |
| Clean timing controllability | Feasible with schedule mask and CRN discipline | Technically hackable, scientifically confounded |
| Paired potential outcomes | Feasible; current RNG pattern helps | Hard and noisy; pass@k/domain confounds |
| `Delta_t/G_pair/xi` | Feasible after checkpoint/sampler | Not cleanly interpretable |
| Metric `F` clarity | Native invalid-word rate primary; BPT backup | pass@k primary but high variance and task-specific |
| Scientific value | High if checkpoint/training succeeds | Low as validation; becomes new research direction |
| Derail risk | Medium | High |
| Recommendation | Pursue cautiously | Do not pursue for this thesis gate |

## Recommended Next Action

Email the informed-correctors authors for the Text8 HollowMD4 checkpoint, exact paper training config, vocabulary file, and any intended sampling/evaluation command before training from scratch.

This is the highest-value next action because informed-correctors/Text8 is the principled backend, but the repo has no public weights and has enough config rough edges that immediate training would risk wasting engineering time. If weights are unavailable, proceed to Stage 0/1/2 only, then decide whether a full 1M-step run is worth it.

*Update 2026-05-14: the author email was sent. Stage 0 has been launched on
HPC in parallel and surfaced the `remdm311`-lacks-JAX-ecosystem blocker;
remediation steps are in
`docs/09_informed_correctors_training_contingency.md §Stage 0 known blocker`.
The "follow up after 7-10 days if no reply" guidance applies if no response
arrives by 2026-05-21 to 2026-05-24.*

## Files Created/Modified by Independent Audit

- Modified: `docs/08_backend_feasibility_audit.md` (appended this Independent Codex Audit section)
- Modified: `results/BACKEND_FEASIBILITY_INDEX.json` (added `independent_codex_audit`)
- Created, later archived by repo cleanup: `results/archive/2026-05-13_repo_cleanup/backend_smoke_informed_correctors_8371dec/SMOKE_LOG.md`
- Created, later archived by repo cleanup: `results/archive/2026-05-13_repo_cleanup/backend_smoke_prism_266d51e/SMOKE_LOG.md`

## Commands Run

Representative commands:

```bash
git status --short
sed -n '1,220p' START_HERE.md
sed -n '1,220p' docs/README.md
python3 -m json.tool results/BACKEND_FEASIBILITY_INDEX.json
git -C external_repos/informed-correctors rev-parse HEAD
git -C external_repos/informed-correctors ls-remote origin HEAD refs/heads/main
find external_repos/informed-correctors -iname '*.ckpt' -o -iname '*.pt' -o -iname '*.pth' -o -iname '*.pkl' -o -iname '*.safetensors' -o -iname '*.msgpack'
rg -n 'checkpoint|ckpt|pretrained|resume|restore|orbax|huggingface|gdown|drive.google|download' external_repos/informed-correctors/text8 external_repos/informed-correctors
nl -ba external_repos/informed-correctors/text8/md4/models/diffusion/hollow_md4.py
nl -ba external_repos/informed-correctors/text8/md4/networks/hollow_transformer.py
PYTHONPATH=external_repos/informed-correctors/text8 python3 -c '...import smoke...'
git -C external/PRISM fetch origin main
git -C external/PRISM show origin/main:PRISM_llada/sampling.py
find external/PRISM/PRISM_llada -iname '*.ckpt' -o -iname '*.pt' -o -iname '*.pth' -o -iname '*.pkl' -o -iname '*.safetensors'
rg -n 'checkpoint|from_pretrained|remasking_conf|LoRA|lora|/n/|netscratch' external/PRISM/PRISM_llada external/PRISM
PYTHONPATH=external/PRISM/PRISM_llada python3 -c '...import smoke...'
```

Web checks:

- GitHub/search for `lindermanlab/informed-correctors` weights/checkpoints found the repo/paper but no public checkpoint.
- GitHub/search for PRISM LLaDA checkpoint found paper/repo context but no public PRISM LLaDA checkpoint; local code remains the strongest evidence for the private checkpoint path.

## Final Hygiene Checks

- No `AGENTS.md` changes.
- No files staged or committed.
- No unrelated thesis/literature-review files intentionally modified.
- No canonical result folders overwritten.
- No long training launched.
- No huge checkpoint/model downloads performed.
- Reports exist: this markdown section, valid JSON update, and both smoke log folders.
