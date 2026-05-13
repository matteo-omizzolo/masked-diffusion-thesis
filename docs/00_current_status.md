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

1. Author email in `docs/email_informed_correctors_authors.md` — **sent
   2026-05-14**. Follow-up guidance: re-ping after 7-10 days if no reply.
2. **Stage 0** environment smoke — ✅ passed (job 494221, gnode01,
   2026-05-14). The initial blocker (`remdm311` lacked the JAX/Flax/TF
   ecosystem; Text8 not staged) was resolved by installing pinned versions
   on a login node and staging Text8. Steps retained in
   `docs/09_informed_correctors_training_contingency.md §Stage 0 environment setup`.
3. **Stage 1** tiny training-loop smoke — 🚫 currently blocked (jobs 494239
   and 494245). Exit 120:0 after `Using Hollow MD4`, no Python traceback;
   the Bocconi `stud` A100s run in MIG mode under CUDA-13 while JAX 0.4.30
   ships a cuda12 plugin. `XLA_PYTHON_CLIENT_PREALLOCATE=false`/
   `MEM_FRACTION=0.5`/`JAX_PLATFORMS=cuda` workaround did not help. See
   CLAUDE.md known issue #14 and
   `docs/09_informed_correctors_training_contingency.md §Stage 1 — current blocker`
   for the resolution paths (cluster admin / `jax[cuda13]` / CPU fallback /
   author-track checkpoint).

No long training is approved yet.
