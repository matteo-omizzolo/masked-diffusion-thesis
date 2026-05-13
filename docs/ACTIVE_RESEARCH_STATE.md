# Active Research State

## Thesis Framing

The thesis studies finite-budget corrector timing regimes in masked diffusion
language models. The current contribution is a diagnostic/statistical framework
for deciding when timing is marginal, interaction-aware, saturated/redundant, or
best handled by direct schedule search.

## Completed ProSeCo-OWT Result

ProSeCo-OWT is a completed case study. The result is backend-specific:

- timing has real headroom;
- marginal rankers fail;
- true-`G(S)` search succeeds;
- pair interactions are mostly redundant/negative;
- geometry and `A_pair = Delta_s + Delta_t` explain much of the pair structure;
- aggregate, enriched, and token-level state features do not add held-out signal;
- saturation is sublinear rather than additive;
- increasing corrector strength rescales amplitude but does not remove the
  interaction structure.

## Active Backend Direction

informed-correctors/Text8 is the proposed principled validation backend. It is
preferred because HollowMD4 exposes architecturally valid learned conditional
logits through the hollow-transformer structure. This is not a claim that the
model gives true data conditionals.

PRISM LLaDA is not currently pursued for thesis validation because its timing
question is entangled with token-quality/remasking, and public PRISM checkpoints
were not found.

## Current Next Action

Author email in `docs/email_informed_correctors_authors.md` was sent
**2026-05-14**. Two parallel tracks:

- **Author track:** wait for reply; follow up after 7-10 days if none.
- **No-author-response track:** drive the staged path below.
  1. Stage 0 environment/data/config smoke. First attempt (job 494155,
     2026-05-13) surfaced the `remdm311`-lacks-JAX-ecosystem blocker.
     Remediation in `docs/09_informed_correctors_training_contingency.md`.
  2. Stage 1 tiny training-loop smoke only if Stage 0 passes.
  3. Stage 2 throughput benchmark before any full training.

## Canonical Result Folders

See `docs/04_results_index.md` and `results/EXPERIMENT_INDEX.json`.

## What Is No Longer Active

- searching for richer ProSeCo state features;
- PRISM LLaDA as the primary validation backend;
- old preflight/smoke folders as active evidence;
- archived docs as current thesis status.
