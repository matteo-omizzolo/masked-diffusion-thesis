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
- **Bocconi path: exhausted (2026-05-14).** Three independent env attempts
  ruled out the cluster. Stage 0 finally passes on `ic_text8_jax13` with
  `jax[cuda12]==0.10.0` (job 494412), but Stage 1 then hits an upstream
  `informed-correctors` + `tensorflow_probability` + JAX 0.10 API-removal
  cascade that can't be patched within the Stage 1 smoke scope. Full
  diagnostics in CLAUDE.md known issue #14 and
  `docs/09 §Stage 1 — Bocconi env path documented blocked`.
- **Active path — external GPU rental** per
  `docs/10_external_gpu_text8_fallback.md`. ~$1.50 round-trip for
  Stage 0+1+2; the strengthened Stage 0 GPU compute probe and the
  ic_text8_jax13 setup script transfer cleanly to a cloud GPU.
- **`remdm311`** remains ProSeCo/PyTorch-only.

## Canonical Result Folders

See `docs/04_results_index.md` and `results/EXPERIMENT_INDEX.json`.

## What Is No Longer Active

- searching for richer ProSeCo state features;
- PRISM LLaDA as the primary validation backend;
- old preflight/smoke folders as active evidence;
- archived docs as current thesis status.
