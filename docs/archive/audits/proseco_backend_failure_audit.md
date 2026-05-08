> **STATUS:** ARCHIVED
> **ARCHIVED ON:** 2026-04-22
> **SUPERSEDED BY:** `docs/experimental_infrastructure.md`
> **REASON:** Resolved incident; backend is now operational. Preserved for provenance.

---

# ProSeCo Backend Failure Audit
*Completed: 2026-04-17*

## 1. What the working MDLM backend (job 478600) actually did

The version of `backends/mdlm.py` that ran job 478600 contained the line:
```
[_load_diffusion_model] Reading checkpoint config from: /home/3316152/mdm/checkpoints/mdlm.ckpt
```

It loaded the checkpoint's `hyper_parameters['config']` — the full `OmegaConf.DictConfig` saved by Lightning at training time — and used it as the base config for `Diffusion.load_from_checkpoint`. This gave the model all required keys with the correct values from training.

## 2. How the current ProSeCo backend constructs its config

`backends/proseco.py` (pre-fix) used a hand-written minimal dict:
```python
cfg = omegaconf.OmegaConf.create({
    "eval": {
        "checkpoint_path": checkpoint_path,
        "disable_ema": False,
        "compute_generative_perplexity": False,
    },
    "model": {"length": 1024, "hidden_size": 768, "n_heads": 12, "n_blocks": 12, "dropout": 0.0},
    ...
})
```

This minimal config was also present in the current `backends/mdlm.py` (the refactored version written after job 478600 ran). Both backends shared the same structural bug.

## 3. Missing and mismatched keys

### Category A — Keys absent from training config but accessed unconditionally in diffusion.py `__init__`

| Key | Accessed at | Training config | Minimal config |
|-----|-------------|-----------------|----------------|
| `eval.gen_ppl_eval_model_name_or_path` | line 83 | `'gpt2-large'` | **MISSING** |
| `model.cond_dim` | DIT `__init__` line 371 | `128` | **MISSING** |
| `model.scale_by_sigma` | DIT `__init__` line 387 | `True` | **MISSING** |

### Category B — Keys that diffusion.py added after the checkpoint was trained

| Key | Accessed at | Needed value |
|-----|-------------|--------------|
| `T` | `__init__` line 115 | `0` (continuous time) |
| `subs_masking` | `__init__` line 116 | `False` |
| `sampling.sampler` | `_ddpm_caching_update` branch | `'mdlm'` |
| `sampling.nucleus_p` | `_ddpm_caching_update` | `1.0` (disable truncation) |

The training checkpoint's config doesn't have `T`, `subs_masking`, `sampling.sampler`, `sampling.nucleus_p` because those config keys were added to `diffusion.py` after the MDLM checkpoint was trained.

### Category C — Present in minimal config but wrong values or wrong key name

| Key | Minimal config | Training config | Notes |
|-----|----------------|-----------------|-------|
| `model.dropout` | `0.0` | `0.1` | Doesn't affect architecture |
| `optim.warmup_steps` | 2500 | uses `warmup` key | Only used in training, not inference |

### Category D — In training config but not needed for inference

`trainer.*`, `wandb.*`, `data.*`, `callbacks.*`, `strategy.*` — all present in training config but never accessed in inference mode. Safe to keep (lazy resolution).

## 4. Classification of required keys

**Truly required by checkpoint/model code (inference fails without them):**
- `eval.gen_ppl_eval_model_name_or_path` — read in `__init__` line 83; tokenizer loaded at line 133
- `model.cond_dim` — used by DIT backbone `TimestepEmbedder`
- `model.scale_by_sigma` — used by DIT backbone output layer
- `T` — read in `__init__` line 115; controls `_validate_configuration`
- `subs_masking` — read in `__init__` line 116
- `sampling.sampler` — required by `_ddpm_caching_update` branching (line 680)
- `sampling.nucleus_p` — required by `_ddpm_caching_update` nucleus check

**Required for signal extraction (ProSeCo-specific bug):**
- `p_x0` non-None for `_extract_signals` — `_ddpm_caching_update` returns `None` whenever `x` changes (i.e., most steps). Fixed separately.

## 5. Whether ProSeCo checkpoint expects same schema as MDLM

Yes — the ProSeCo backend uses the SAME checkpoint as the MDLM backend (`mdlm.ckpt`). The corrector strategy differs (ProSeCo annealed refinement vs MDLM Gibbs resample) but the backbone weights and config schema are identical.

## 6. Root cause classification

**Primary cause:** Config hydration regression. The old `_load_diffusion_model` read the checkpoint config; this logic was removed when the backends were refactored. The ProSeCo backend was written after the regression and copied the broken pattern.

**Secondary cause:** `diffusion.py` `__init__` accesses config keys unconditionally without defaults. Missing keys surface as `ConfigAttributeError` rather than being silently skipped.

**Additional bug (ProSeCo-specific):** `_run_loop` extracted signals from `p_x0_cache` which is `None` whenever `x` changes after a predictor step. This was always None in practice (tokens unmask at every step), causing `TypeError: 'NoneType' object is not subscriptable`.

## 7. The fix

1. Both backends now read `raw_ckpt["hyper_parameters"]["config"]` from the checkpoint, convert to a plain dict via `OmegaConf.to_container(resolve=False, throw_on_missing=False)`, add Category B keys via `setdefault`, override inference settings (eval flags, sampling.steps), and reconstruct as a fresh non-struct `OmegaConf.DictConfig`.

2. `_apply_corrector` signature changed to `(x, t, p_x0=None)` — computes `p_x0` via forward pass when cache is None.

3. `_run_loop` recomputes `p_x0_for_signals` via forward pass when `p_x0_cache` is None (all steps where x changed).
