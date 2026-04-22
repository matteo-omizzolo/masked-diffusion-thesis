> **STATUS:** ARCHIVED
> **ARCHIVED ON:** 2026-04-22
> **SUPERSEDED BY:** `docs/experimental_infrastructure.md`
> **REASON:** Backend readiness audit resolved; current state is tracked in the infrastructure doc.

---

# ProSeCo Backend Audit

**Date:** 2026-04-17  
**Repository:** https://github.com/kuleshov-group/proseco  
**Paper:** "Learn from Your Mistakes: Self-Correcting Masked Diffusion Models" (arXiv:2602.11590)  
**Audit status:** Complete — implementation decision recorded below.

---

## 1. What ProSeCo checkpoint is available?

### Locally cached (HPC)

| Checkpoint | Location | Status |
|---|---|---|
| `mdlm.ckpt` | `/home/3316152/mdm/checkpoints/mdlm.ckpt` (~2.5 GB) | **In use** — validated |
| `kuleshov-group/mdlm-no_flashattn-fp32-owt` | `~/.cache/huggingface/hub/` | Cached, not used |
| `kuleshov-group/proseco-owt` | Not cached | Downloadable (~500 MB) |

### Implementation decision

The ProSeCo backend (`backends/proseco.py`) uses **`mdlm.ckpt`** (the Lightning checkpoint)
loaded via the `remdm` Diffusion class. This is the same model as the MDLM heuristic backend.

**What changes is the corrector strategy**, not the backbone weights. The ProSeCo corrector
(annealed iterative refinement from x̂_0) is implemented on top of the same MDLM backbone.

**Why not use `proseco-owt` (the co-trained checkpoint)?**
- `proseco-owt` was trained with a corrector loss that explicitly teaches the model to be
  better at correction. This would give stronger corrector quality.
- It is downloadable from HuggingFace (internet available on login node) but requires
  a separate loading path (`backbone=hf_dit`, ProSeCo's Diffusion class).
- For now: the corrector mechanism (annealed refinement from x̂_0) is meaningful with
  ANY masked diffusion backbone. The key thesis claim is about scheduling, not the checkpoint.
- **Upgrade path**: replace `checkpoint=mdlm.ckpt` with `proseco-owt` (HF model) for a
  stronger corrector. Document as experiment variant, not as thesis invalidation.

### Model specs (mdlm.ckpt)

- **Architecture:** DiT (Diffusion Transformer)
- **Vocab:** GPT-2 tokenizer (50,257 tokens + 1 mask = 50,258)
- **Hidden size:** 768, 12 heads, 12 blocks
- **Sequence length:** 1024 tokens
- **Training data:** OpenWebText
- **Noise schedule:** Loglinear

---

## 2. Corrector controls in the ProSeCo backend

All corrector scheduling is controlled via Python parameters to `ProSeCoGenerator`:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `corrector_steps` | int | 2 | Inner refinement iterations per corrector loop |
| `corrector_at_t` | int or None | None | Step at which to apply corrector (Protocol A) |
| allocation | dict | {} | Steps with correctors (Protocol B) |

The original ProSeCo config parameters (`corrector_every_n_steps`, `corrector_start_iter`,
`corrector_top_k`) are NOT used in our implementation. We control corrector placement
explicitly by step index, which is exactly what Protocol A and B require.

---

## 3. How the ProSeCo corrector works (code-level)

### Starting point

After predictor step `i`, `_ddpm_caching_update` returns `p_x0_cache`: the model's x0
probability predictions over all positions (shape `(B, L, V)`).

### Corrector loop (in `backends/proseco.py:_apply_corrector`)

```python
# Step 1: predicted clean x0 from current predictor
corrector_x = p_x0_cache.argmax(-1)         # (B, L) — full sequence, all positions

# Step 2: annealed refinement (tau decreasing from 1 to eps)
corrector_timesteps = linspace(1.0, eps, corrector_steps + 1)
for j in range(corrector_steps):
    tau = corrector_timesteps[j]
    sigma_tau, _ = model.noise(tau)
    log_p_corr = model.forward(corrector_x, sigma_tau)  # backbone call at noise tau
    corrector_x = log_p_corr.exp().argmax(-1)           # deterministic update

# Step 3: apply to UNMASKED positions only
unmasked = (x != mask_id)
x_new = x.clone()
x_new[unmasked] = corrector_x[unmasked]
```

### Key semantics

| Aspect | Value |
|---|---|
| Starting input | Full predicted x̂_0 = argmax(p_x0) — ALL positions |
| Corrector runs on | corrector_x (full sequence, not just unmasked) |
| Applied back to trajectory | UNMASKED positions only |
| Noise schedule direction | High τ (σ high) → low τ (σ low) — annealed refinement |
| Sampling mode | Argmax (deterministic) — avoids stochastic noise in gain measurements |
| NFEs per corrector loop | corrector_steps (2 by default) |

### What "one corrector loop at step t" means exactly

In Protocol A, "apply one corrector loop at step t" means:
1. Run predictor steps 0 … t (including step t). Get p_x0_cache from step t.
2. Apply `_apply_corrector(x, p_x0_cache)`:
   - x̂_0 = argmax(p_x0_cache) — predicted clean sequence
   - Run 2 refinement steps at τ ∈ {1.0, 0.5} (corrector_steps=2)
   - Apply result to unmasked positions in x
3. Continue predictor steps t+1 … T-1 from the corrected x.

This costs `corrector_steps = 2` extra backbone NFEs at step t.

### How ProSeCo corrector differs from MDLM heuristic

| Dimension | MDLM heuristic | ProSeCo corrector |
|---|---|---|
| Action set | MASKED positions | UNMASKED positions |
| Starting point | p(x_0 \| x_t) — full posterior | x̂_0 = argmax(predictor output) |
| Mechanism | One-shot Gibbs resample | Annealed iterative refinement |
| NFEs per loop | 1 (one forward pass) | corrector_steps (2 by default) |
| Early-step behavior | Destructive (resamples ~97% of tokens with <5% context) | Conservative (few unmasked positions → small effect) |
| Why principled | Samples from posterior | Refines committed tokens via correction noise schedule |

---

## 4. Parts of the scheduling pipeline reused unchanged

| Component | File | Reused? |
|---|---|---|
| Gain computation | `scheduling/gain.py` | ✓ Unchanged |
| Allocation policies | `scheduling/allocation.py` | ✓ Unchanged |
| Schedule evaluation | `scheduling/evaluate.py` | ✓ Unchanged |
| Signal module | `scheduling/signals.py` | ✓ Unchanged (not called directly; signals in backend) |
| Surrogate generator | `scheduling/surrogate.py` | ✓ Unchanged (used for pipeline validation) |
| Run script structure | `run_phase1_pilot.py` | Adapted → `run_phase1_proseco.py` |
| Analysis scripts | `analyze_phase1.py` | Will be adapted → `analyze_phase1_proseco.py` |
| HPC sbatch template | `hpc/phase1_pilot.sbatch` | Adapted → `hpc/phase1_proseco.sbatch` |
| Summary computation | inline in run scripts | Adapted (adds policy comparison, n_positive) |

---

## 5. What must be adapted for ProSeCo

| Item | Change |
|---|---|
| Backend | New file: `backends/proseco.py` (ProSeCoGenerator) |
| Corrector mechanism | Annealed refinement instead of Gibbs resample |
| Signal computation | Over UNMASKED positions (ProSeCo action set) — no extra NFE |
| run_branch efficiency | No signal recording in branches (only run_base records signals) |
| Policy comparison | Added to run script: all 10 policies compared explicitly |
| Summary metrics | Added: n_positive_delta, t_first_positive, oracle vs uniform comparison |
| Results directory | `results/phase1_proseco/` (separate from MDLM results) |

---

## 6. Dependency notes

All dependencies for the ProSeCo backend are satisfied by the existing `remdm311`
conda environment. No new packages are required. The ProSeCo backend imports from
`external/remdm/`, which is already on sys.path.

The ProSeCo backend does NOT require:
- Hydra / OmegaConf config files from ProSeCo's own configs/
- The `external/proseco/` directory at all
- WandB or MAUVE scoring
- PyTorch Lightning Trainer (uses Lightning's checkpoint loading only)
