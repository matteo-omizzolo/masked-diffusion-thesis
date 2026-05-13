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

**Working assumption (2026-05-14):** do **not** assume the informed-correctors
authors or the Bocconi HPC admins will reply in time. The active path below
is self-sufficient.

- Author email sent 2026-05-14 (informational only; non-blocking).
- **Primary path — Bocconi `ic_text8_jax13` env (`jax[cuda13]`):**
  1. `bash hpc/backend_validation/informed_correctors/setup_ic_text8_jax13.sh`
     on a login node (idempotent).
  2. Stage 0 sbatch: now includes a JIT compute probe; defaults to the new
     env.
  3. Stage 1 sbatch (tiny training-loop only) defaults to the new env.
- **Fallback path — external GPU rental** if Bocconi remains blocked. See
  `docs/10_external_gpu_text8_fallback.md`.
- **Deprecated:** `remdm311` for Text8 training. It is ProSeCo/PyTorch-only
  and is known to fail Stage 1 under the cuda12-on-cuda13-MIG combination
  (CLAUDE.md issue #14).

## Canonical Result Folders

See `docs/04_results_index.md` and `results/EXPERIMENT_INDEX.json`.

## What Is No Longer Active

- searching for richer ProSeCo state features;
- PRISM LLaDA as the primary validation backend;
- old preflight/smoke folders as active evidence;
- archived docs as current thesis status.
