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
2. **Primary path — Bocconi `ic_text8_jax13` (`jax[cuda13]`):**
   - Set up the dedicated env on a login node via
     `hpc/backend_validation/informed_correctors/setup_ic_text8_jax13.sh`
     (idempotent; safe to re-run).
   - Stage 0 sbatch defaults to `CONDA_ENV=ic_text8_jax13` and now runs a
     JIT compute probe in addition to import + enumeration checks (catches
     CUDA/driver mismatches at Stage 0 time).
   - Stage 1 sbatch also defaults to `ic_text8_jax13`. Tiny training loop
     only (≤50 steps, batch_size=8, hidden_dim=128, 2 layers).
3. **Fallback path — external GPU rental:** see
   `docs/10_external_gpu_text8_fallback.md`. Trigger only when the Bocconi
   clean-env path is exhausted; expected cost ~$1.50 for Stage 0+1+2.
4. **Deprecated path — `remdm311` for Text8 training:** historical Stage 0
   eventually passed (job 494221) but Stage 1 jobs 494239 and 494245 hit
   the cuda12-on-cuda13-MIG bug (CLAUDE.md issue #14). `remdm311` is now
   ProSeCo/PyTorch-only.

No long training is approved yet. Stage 2 + useful-checkpoint training
remain explicitly out of scope until Stage 1 passes cleanly.
