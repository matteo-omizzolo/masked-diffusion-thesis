# Current Status

## Bottom Line

The ProSeCo-OWT line is complete as a case study. The active research frontier
is informed-correctors/Text8 backend validation, with author contact plus a
no-author-response training contingency.

## ProSeCo-OWT Conclusion

The current conclusion is precise and modest:

- corrector timing has measurable headroom;
- marginal rankers do not recover that headroom;
- direct `G(S)` search recovers much of it;
- pair interactions are mostly redundant/negative;
- `A_pair`, phase, and geometry explain much of the structure;
- aggregate, enriched, and token-level state features do not add predictive
  signal under the tested models;
- saturation/sublinear composition is the best empirical interpretation.

This does not establish universality across masked diffusion models.

## Backend Validation Status

informed-correctors/Text8 remains the preferred backend because it is a cleaner
informed-corrector setting with HollowMD4 learned conditional logits. Public
Text8 weights were not found. The no-author-response audit says training is
probably feasible, but exact paper reproduction still needs clarification.

PRISM LLaDA is not currently pursued for thesis validation because its mechanism
is token-quality/remasking, not isolated corrector timing, and public PRISM
weights were not found.

## Current Actions

**Working assumption (2026-05-14):** do **not** assume the informed-correctors
authors or the Bocconi HPC admins will reply in time. The current execution
plan is self-sufficient.

1. Author email in `docs/email_informed_correctors_authors.md` — sent
   2026-05-14. Follow up after 7-10 days if no reply, but do not block on it.
2. **Bocconi path — exhausted (2026-05-14).** Three independent env attempts:
   - `jax[cuda12]==0.4.30` on remdm311: MIG crash on Stage 1 (jobs 494239,
     494245).
   - `jax[cuda13]` on `ic_text8_jax13`: Stage 0 fails because Bocconi has no
     CUDA-13 toolkit module (jobs 494409, 494410).
   - `jax[cuda12]==0.10.0` on `ic_text8_jax13`: Stage 0 **passes cleanly**
     with the new GPU compute probe (job 494412, 1m10s). Stage 1 then
     hits an upstream md4 + TFP + JAX-0.10 API-removal cascade (jobs
     494413–494416). Full diagnostic in CLAUDE.md known issue #14 and
     `docs/09 §Stage 1 — Bocconi env path documented blocked`.
3. **Active path — external GPU rental:** see
   `docs/10_external_gpu_text8_fallback.md`. A fresh cloud GPU with a
   matching driver/toolkit avoids all three Bocconi-specific constraints,
   costs ~$1.50 for Stage 0+1+2, and is the documented next step.
4. **`remdm311`** remains ProSeCo/PyTorch-only.

No long training is approved yet. Stage 2 + useful-checkpoint training
remain explicitly out of scope until Stage 1 passes cleanly on the
external-GPU rental (or until the authors share a checkpoint).
