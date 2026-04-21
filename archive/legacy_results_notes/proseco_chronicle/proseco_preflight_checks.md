# ProSeCo Backend Preflight Checks
*Created: 2026-04-17*

## Before every HPC submission: run the debug script

```bash
python scripts/debug_proseco_load.py \
    --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --device cpu --T 2
```

This takes ~2 min on CPU and verifies the full inference path end-to-end.  Exit code 0 = safe to submit.

## Guard rail 1 — Config key completeness

The `_load_diffusion_model` function in both backends reads the checkpoint's saved
config (`hyper_parameters['config']`) and adds missing inference keys.  The following
keys are always injected via `setdefault` (safe no-ops if they're already present):

- `T = 0` — continuous-time SUBS parameterization
- `subs_masking = False`
- `sampling.sampler = 'mdlm'`
- `sampling.nucleus_p = 1.0`

And the following are always overridden for inference:
- `eval.compute_generative_perplexity = False`
- `eval.generate_samples = False`
- `eval.compute_perplexity_on_sanity = False`

If `diffusion.py` adds new unconditional config accesses in a future patch, the debug
script will catch them before the 8-hour job slot is consumed.

## Guard rail 2 — p_x0 safety in corrector and signal extraction

`_apply_corrector(x, t, p_x0=None)` always accepts `t` as a fallback: if `p_x0` is
None (cache invalidated because x changed), it recomputes `p_x0 = model.forward(x, σ_t).exp()`.

`_run_loop` similarly recomputes `p_x0_for_signals` when `p_x0_cache` is None.

## Guard rail 3 — weights_only=False patch in both backends

Both backends apply the module-level monkey-patch:
```python
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
```

This is required for PyTorch ≥ 2.6 with old checkpoints containing numpy scalars.

## What to check after job submission

1. First 5 minutes: look for `[_load_diffusion_model] Reading checkpoint config` → confirms new code is running
2. Within 10 minutes: `[ProSeCoGenerator] Ready.` → model loaded, reference scorer loaded
3. Within 20 minutes: `Trajectory 1/20 (seed=42) ...` → Protocol A started
4. At ~30-60 min: first Δ values appearing → real data flowing
5. At ~3-5h: `Summary written to results/phase1_proseco/summary.json` → complete

Check with: `tail -50 out/phase1_proseco_<JOB_ID>.out`
