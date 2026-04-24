# Copilot instructions for `masked-diffusion-thesis`

This repository is an MSc thesis project on **budgeted scheduling of informed correctors in masked diffusion language models**. Treat it as a thesis-first research codebase, not a generic playground.

## Read first before major edits

1. `README.md`
2. `docs/thesis/CURRENT_INDEX.md`
3. `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
4. `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md`
5. `docs/thesis/experiments/RESULTS_STATUS.md`
6. `docs/thesis/theory/THEORY_STATUS.md`

## Mainline scope and paths

Primary mainline:

- `README.md`
- `docs/thesis/CURRENT_INDEX.md`
- `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
- `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md`
- `docs/thesis/experiments/RESULTS_STATUS.md`
- `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`
- `docs/thesis/theory/THEORY_STATUS.md`
- `scripts/run_phase2b_proseco_owt.py`
- `scripts/run_phase3a_combinatorial.py`
- `src/mdm_playground/scheduling/*`
- `results/phase2b_proseco_owt/*`
- `results/phase3a_proseco_owt/*`

Keep legacy/archive code separate and do not let it redefine thesis conclusions.

## Research principles

- Never assume a claim is correct just because it appears in a memo or doc.
- Always label claims as one of:
  - **supported by repo evidence**
  - **plausible but unproven**
  - **speculative**
- Prefer narrow, defensible claims over broad exciting ones.
- Keep the thesis object fixed unless explicitly asked to audit or pivot it.
- Fixed thesis object:
  - fixed predictor
  - fixed informed corrector kernel
  - fixed extra correction budget
  - core question: **when** to apply corrective refinement steps
- Distinguish carefully between:
  - open-loop schedule optimization
  - adaptive/state-dependent scheduling
  - remasking/self-correction papers
  - informed corrector scheduling
- Do not collapse these into the same thing.

## Current scientific direction

- Phase 2b: greedy/separable per-step rankers mostly fail.
- Phase 3a: schedule-aware search succeeds.
- Current interpretation: fixed-budget corrector allocation is a trajectory-level / schedule-level problem; greedy rankers are the wrong solution class.
- Keep interpretation precise:
  - do not overclaim universal impossibility
  - do not treat search results as deployable policies by default
  - distinguish structural existence results from practical schedulers

## Software principles

- Prefer small modular changes over sweeping rewrites.
- Keep code readable, typed where appropriate, and documented.
- Add tests for thesis-critical code.
- Do not introduce hidden dependencies.
- Make missing dependency errors explicit and actionable.
- Preserve reproducibility.
- Keep archive/legacy flows separated from the active mainline.
- Do not clutter the repo with generated artifacts, logs, checkpoints, PDFs, or scratch files.

## Operational guidance for Copilot

When asked for next steps:

1. assess evidence skeptically
2. write a coherent plan first
3. only then implement
4. clearly mark what is proven, empirical, heuristic, or speculative

## What not to do

- Do not treat PRISM, remasking, RemeDi, ReMDM, etc. as equivalent to informed corrector scheduling.
- Do not silently broaden the thesis question.
- Do not jump to heavy frameworks (SMC, control-as-inference, BP, etc.) without first justifying them.
- Do not remove canonical thesis docs.
- Do not break mainline scripts or mainline result paths.
- Do not commit sensitive files, logs, checkpoints, or local model snapshots.
