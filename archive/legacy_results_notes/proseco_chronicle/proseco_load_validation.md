# ProSeCo Backend Load Validation
*Run: 2026-04-17, before HPC resubmission*

## Validation command

```bash
python scripts/debug_proseco_load.py \
    --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --device cpu \
    --T 2
```

## Results (all 5 checks PASSED)

| Check | Result | Time |
|-------|--------|------|
| 1/5 Checkpoint config readable | PASS | 15.6s |
| 2/5 ProSeCoGenerator instantiates | PASS | 79.4s |
| 3/5 run_base (predictor + signals) | PASS | 6.1s |
| 4/5 run_branch (corrector loop) | PASS | 6.0s |
| 5/5 neg_nll finite | PASS | 0.0s |

Key outputs:
- `gen_ppl_eval_model_name_or_path = gpt2-large` (resolved from checkpoint)
- `mask_id = 50257`, `model.T = 0` (continuous-time SUBS confirmed)
- Final-step signals: entropy=0.0, margin=0.0, uf=1.0 (T=2 fully denoises in 2 steps)
- base neg_nll = -7.696, branch neg_nll = -7.696 (Δ=0 expected: T=2, fully unmasked after 2 steps)

## What this validates

- Checkpoint loads without ConfigAttributeError or TypeError
- `diffusion.Diffusion` instantiates with correct architecture (DIT, 768 hidden, cond_dim=128)
- `_ddpm_caching_update` runs without KeyError on `sampling.sampler` or `sampling.nucleus_p`
- Signal extraction handles `p_x0_cache = None` gracefully
- Corrector loop (annealed refinement, 1 step) runs without error
- `neg_nll` is finite and sign-correct

## What remains untested until the real pilot

- GPU execution on A100 (device=cuda, bfloat16 autocast active)
- Correct signal values at T=64 (T=2 collapses to trivial fully-unmasked state)
- Spearman correlation between signals and Δ_t over N=20 trajectories
- Protocol B schedule evaluation and η_B estimation
- Runtime at full scale (~3-5h on A100 for pilot)
