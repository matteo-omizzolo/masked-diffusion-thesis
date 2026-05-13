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
2. In parallel, run Stage 0 informed-correctors/Text8 environment smoke. The
   first attempt (job 494155 on gnode01, 2026-05-13) surfaced a clean blocker:
   `remdm311` lacks the JAX/Flax/TF ecosystem and Text8 zip is not staged.
   Remediation steps are documented in
   `docs/09_informed_correctors_training_contingency.md §Stage 0 known blocker`.
3. Run Stage 1 tiny training-loop smoke **only** after Stage 0 passes
   cleanly, and **only** as a loop check.

No long training is approved yet.
