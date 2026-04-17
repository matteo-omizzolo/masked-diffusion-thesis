# ProSeCo Repository Import Audit

**Date:** 2026-04-17  
**Repository:** https://github.com/kuleshov-group/proseco  
**Commit:** 69fe45d101d8367b0d1807466b409f71449a7c58  
**Paper:** "Learn from Your Mistakes: Self-Correcting Masked Diffusion Models" (arXiv:2602.11590)  

---

## Executive Summary

ProSeCo is the most mature open-source implementation of self-correcting masked diffusion models with tunable corrector scheduling. It is well-suited as the empirical backend for thesis experiments on signal-adaptive corrector scheduling because:

1. **Production-quality corrector implementation** with both training and inference paths
2. **Explicit corrector scheduling parameters** that directly encode uniform placement strategies
3. **Trained checkpoints** (`proseco-owt`) available on HuggingFace for reproducibility
4. **Modular architecture** allowing isolated replacement of corrector scheduling logic
5. **Comprehensive config system** (Hydra) for experiment parametrization

---

## Code Organization

```
external/proseco/
├── diffusion.py                    # Core sampling loop + corrector scheduling
├── main.py                         # Training entry point + checkpoint loading
├── noise_schedule.py               # Noise schedule abstraction
├── models/                         # Backbone architectures (UNet, DiT, Mamba)
├── configs/                        # Hydra config hierarchy
├── llada/                          # Evaluation with LLaDA SFT model
└── scripts/                        # Slurm templates
```

---

## 1. Sampling Code Location

**File:** `diffusion.py` (1800+ lines)

### Entry Point
- **Method:** `Diffusion.sample()` (lines 964–1010)
- **Signature:** Returns `(samples, NFEs_dict)` where NFEs_dict tracks function evaluations

### Diffusion Sampling Loop
- **Method:** `Diffusion._diffusion_sample()` (lines 1209–1450+)
- **Responsibility:** Iterates over denoising timesteps; embeds corrector scheduling
- **Key control flow:**
  ```python
  for i in range(self.config.sampling.steps):
      # Denoiser step
      xs, q_xs, cache = self._ddpm_denoise(...)
      
      # Corrector scheduling condition (CRITICAL):
      if (i + 1) % self.config.sampling.corrector_every_n_steps == 0 \
         and (i + 1) >= self.config.sampling.corrector_start_iter:
          # Apply corrector loop
  ```

### Corrector Application
- **Method:** `_corrector_denoise()` (nested loop within `_diffusion_sample`, lines 1313–1380+)
- **Responsibility:** Runs inner corrector refinement loop on masked sequence
- **Runs for:** `self.config.sampling.corrector_steps` iterations

---

## 2. Corrector Controls & Scheduling Parameters

All corrector scheduling logic is driven by **4 parameters** in `configs/config.yaml` (lines 44–56):

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `corrector_steps` | int | 0 | Max corrector inner iterations per loop |
| `corrector_every_n_steps` | int | 1 | Denoising step frequency: apply corrector every N steps |
| `corrector_start_iter` | int | 0 | Delay corrector until denoising step ≥ this (1-indexed) |
| `corrector_prior_is_argmax` | bool | True | Use argmax (vs. sampling) for corrector input |

### Uniform Placement Strategy
Current implementation encodes **uniform corrector placement**:
- Apply corrector at steps: `{k·corrector_every_n_steps : k=1,2,... and k·corrector_every_n_steps ≥ corrector_start_iter}`
- **Example:** `corrector_every_n_steps=2` → apply at steps {2, 4, 6, 8, ...}

### Training Parameters (separate scope)
These train the corrector network but **don't directly control inference scheduling**:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `corrector_training` | False | Enable corrector co-training |
| `use_weighted_corrector_loss` | True | Weight corrector loss by diffusion weighting |
| `corrector_loss_weight` | 0.0 | Scale factor for corrector loss |

---

## 3. Checkpoint Loading & Model Setup

**File:** `main.py` (lines 56–64)

### Function
```python
def _load_from_checkpoint(config, tokenizer, device='cuda'):
    if 'hf' in config.backbone:
        # Load from HuggingFace directly (no checkpoint needed)
        return diffusion.Diffusion(config, tokenizer=tokenizer).to(device)
    
    # Load from local checkpoint file
    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config, logger=False, map_location=device)
```

### Two Paths
1. **HuggingFace models** (e.g., `proseco-owt`)
   - Set `config.backbone=hf_dit`
   - Set `config.model.pretrained_model_name_or_path=kuleshov-group/proseco-owt`
   - Model weights auto-downloaded

2. **Local checkpoint** (PyTorch Lightning .ckpt files)
   - Set `config.eval.checkpoint_path=/path/to/model.ckpt`
   - Lightning handles model state restoration

---

## 4. Key Script & Config Locations

### Inference Configs
- **Default base config:** `configs/config.yaml` (full parameter reference)
- **Data configs:** `configs/data/` (e.g., `openwebtext-split.yaml`)
- **Noise schedules:** `configs/noise/` (loglinear, linear, ar, polynomial)

### Inference Entry Point
- **Script:** `main.py` with mode `mode: eval`
- **Sample generation:** Lines 1000–1200+ (inside Lightning `Trainer` callback)

### Evaluation Scripts
- **LLaDA evaluation:** `llada/generate.py` — applies corrector strategies on LLaDA SFT model
- **MAUVE scoring:** Integrated into main training loop (via `eval_utils.py`)

### Slurm Templates
- `scripts/train_*.sh` — training templates
- `scripts/eval_*.sh` — inference templates
- Example: `scripts/eval_generation.slurm`

---

## 5. Minimum Integration Surface for Protocol A/B

### To integrate Protocol A/B corrector scheduling:

**Required Changes:**

1. **Replace corrector scheduling logic in `_diffusion_sample`** (lines 1313–1318)
   - **Current:** Fixed modulo-based uniform schedule
   - **Target:** Signal-adaptive schedule decision based on:
     - Trajectory-level signals: entropy (σ_t), confidence margin, quality mass
     - Per-batch computed statistics
     - Adaptive threshold comparison

2. **Pass signals through sampling loop** (architecture change)
   - Extract logit entropy / confidence at each denoising step
   - Accumulate trajectory-level aggregate signals
   - Query Protocol A/B logic to decide corrector application
   - **Implementation pattern:**
     ```python
     # After _ddpm_denoise step, before corrector condition check
     entropy, conf_margin, qual_mass = compute_trajectory_signals(
         log_x_theta=log_x_theta,  # from cache
         t=t,
         xt=xt
     )
     should_correct = protocol_adaptive_schedule(
         entropy=entropy,
         conf_margin=conf_margin,
         qual_mass=qual_mass,
         trajectory_step=i,
         nfe_budget_remaining=...
     )
     ```

3. **Config-driven strategy selection** (minimal)
   - Add new config field: `sampling.schedule_strategy` ∈ {`uniform`, `entropy`, `margin`, `quality_mass`, `hybrid`}
   - Add optional config fields: `sampling.adaptive_threshold`, `sampling.signal_weight`

**Stable Integration Points (unlikely to change):**
- Model backbone interface (`Diffusion.backbone()` call)
- Token masking convention (`self.mask_index`)
- Denoiser cache structure (`cache` dict, keys `log_x_theta`, `log_x_theta_cond`)
- Output format (`samples, NFEs_dict`)

**Fragile Integration Points (likely to break on upgrades):**
- Exact timestep discretization (lines 1221–1223)
- NFE accounting heuristics (lines 1249–1250, 1336–1337)
- Guidance method dispatch (lines 1265–1306)

---

## 6. HuggingFace Checkpoint Availability

### Available Trained Models

**`kuleshov-group/proseco-owt`** (recommended for thesis)
- **Training Data:** OpenWebText (OWT), ~9B tokens
- **Model Size:** ~125M parameters (medium MDLM)
- **Noise Schedule:** Loglinear
- **Corrector:** Trained with corrector loss

**`kuleshov-group/proseco-llada-sft`**
- Same corrector as above
- Instruction-tuned via supervised fine-tuning (SFT)
- Use only if studying instruction-following robustness

### Loading in Code
```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("kuleshov-group/proseco-owt")
```

Or via ProSeCo:
```bash
# In main.py config:
backbone: hf_dit
model: hf
model.pretrained_model_name_or_path: kuleshov-group/proseco-owt
```

---

## 7. Dependency Analysis

### Critical Dependencies
- **PyTorch Lightning** 2.2.1+ (training/checkpoint ops)
- **Hydra** 1.3+ (config management)
- **Transformers** 4.38.2 (tokenizer, HF model loading)
- **Mamba-SSM** 2.0+ (if using dimamba backbone; optional)

### Known HPC Issues (from CLAUDE.md)
- **setuptools conflict:** Use `setuptools<70` (already documented in project)
- **NumPy 2.0:** Ensure `datasets>=2.21` (already documented)
- **flash_attn:** Not required; code falls back to PyTorch SDPA (already patched in `remdm`)

### Verification Command
```bash
cd external/proseco
pip install -e .  # Or: pip install torch lightning hydra-core transformers
```

---

## 8. Empirical Validation Strategy

### Baseline Reproduction (quick check)
1. **Load** `proseco-owt` checkpoint
2. **Run sampling** with `corrector_steps=4` at different `corrector_every_n_steps` ∈ {1, 2, 4}
3. **Measure:** MAUVE vs. NFE count (should match Figure 3 of ProSeCo paper)
4. **Expected:** Corrector every 2 steps ≈ corrector every 1 step in quality, ≈50% fewer NFEs

### Integration Test (after Protocol A/B integration)
1. **Replace** corrector scheduling condition with Protocol A/B logic
2. **Compare trajectories:**
   - Uniform `corrector_every_n_steps=2` (baseline)
   - Protocol A (entropy-based)
   - Protocol B (margin-based)
   - Protocol C (quality-mass-based)
3. **Metrics:** MAUVE, PPL, latency, NFE breakdown (denoiser vs. corrector)

---

## 9. Repository Cleanliness Notes

### Large Files (not fetched)
- Checkpoints: Provided via HuggingFace (no .ckpt in repo)
- Data: Streamed at runtime

### Git History
- 69 commits, clean history
- Main branch tracks latest paper version
- No unstaged changes

### Size
- ~200 KB code + configs
- Full `.git` is ~40 MB (normal for a mature repo)

---

## 10. Recommended Next Steps

1. **Verify imports:**
   ```bash
   cd external/proseco && python -c "import diffusion; print('OK')"
   ```

2. **Run minimal inference:**
   ```bash
   cd external/proseco
   python main.py mode=eval eval.checkpoint_path=... sampling.steps=32 sampling.corrector_steps=2
   ```

3. **Extract trajectory signals:**
   - Modify `_diffusion_sample` to log `log_x_theta` at each step
   - Compute entropy/margin/quality-mass post-hoc as a PoC

4. **Design Protocol A/B interface:**
   - Write standalone function `protocol_adaptive_schedule(entropy, margin, quality_mass, ...) -> bool`
   - Test against uniform placement baseline
   - Integrate into corrector scheduling condition

5. **Set up experiment tracking:**
   - Configure WandB project (ProSeCo has built-in integration)
   - Log NFE breakdown and quality metrics per strategy

---

## Appendix: File Reference

| File | Role | Integration Risk |
|------|------|------------------|
| `diffusion.py:_diffusion_sample()` | Sampling loop with scheduling | **HIGH** — core logic |
| `diffusion.py:_corrector_denoise()` | Corrector inner loop | **MEDIUM** — encapsulated |
| `configs/config.yaml` | Scheduling parameters | **LOW** — can extend |
| `main.py:_load_from_checkpoint()` | Checkpoint loading | **LOW** — stable API |
| `models/` | Backbone architectures | **LOW** — not modified |
| `llada/generate.py` | LLaDA evaluation | **LOW** — optional |

---

**Audit completed:** 2026-04-17  
**Next reviewer:** Thesis supervisor + Protocol A/B implementation team
