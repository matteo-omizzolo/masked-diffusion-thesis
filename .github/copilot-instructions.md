# Copilot instructions for `masked-diffusion-thesis`

This repository is an MSc thesis project on **budgeted scheduling of informed correctors in masked diffusion language models**. Treat it as a thesis-first research codebase, not a generic playground.

## Entry points (read these first)

1. `START_HERE.md` — 2-minute orientation
2. `docs/README.md` — active doc index
3. `docs/00_current_status.md` — what is established, what failed, what is open

Do NOT use `docs/archive/`, `archive/`, or `docs/thesis/` to infer current thesis
status — those are historical and may contradict current docs.

## Active doc paths

- `START_HERE.md`
- `docs/00_current_status.md`
- `docs/01_research_direction.md`
- `docs/02_experiments.md`
- `docs/03_theory.md`
- `docs/04_results_index.md`
- `docs/05_next_steps.md`
- `research/candidate_theorems.md` (theorem worklog)
- `research/proof_worklog.md` (derivation entries)
- `scripts/run_phase2b_proseco_owt.py`
- `scripts/run_phase3a_combinatorial.py`
- `results/phase2b_proseco_owt/`
- `results/phase3a_proseco_owt/`

## Current scientific direction

- Phase 2b: greedy/separable per-step rankers fail by B = 8.
- Phase 3a: CD-G and BS-AG schedule-search procedures recover 49–84 % of MC-oracle headroom.
- Current interpretation: fixed-budget corrector allocation is a combinatorial
  schedule-search problem; greedy rankers are the wrong solution class.
- **No new HPC runs authorized.** Critical path is LaTeX writing (ch3/4/5/7).

Keep interpretation precise:
- Do not overclaim universal impossibility.
- Do not treat CD-G/BS-AG as deployable inference-time schedulers (they use true G).
- Distinguish structural existence results from practical schedulers.

## Research principles

- Never assume a claim is correct just because it appears in a memo or doc.
- Always label claims as: **supported by repo evidence**, **plausible but unproven**, or **speculative**.
- Prefer narrow, defensible claims over broad exciting ones.
- Keep the thesis object fixed unless explicitly asked to audit or pivot it.
  - Fixed thesis object: fixed predictor, fixed corrector kernel, fixed NFE budget, question = when to apply corrector steps.
- Distinguish open-loop schedule optimization vs adaptive scheduling vs remasking.

## Software principles

- Prefer small modular changes over sweeping rewrites.
- Add tests for thesis-critical code.
- Preserve reproducibility — never delete raw results.
- Keep archive/legacy flows separated from the active mainline.

## What not to do

- Do not treat PRISM, remasking, RemeDi, ReMDM as equivalent to informed corrector scheduling.
- Do not silently broaden the thesis question.
- Do not jump to heavy frameworks without first justifying them.
- Do not break mainline scripts or mainline result paths.
- Do not commit sensitive files, logs, checkpoints, or local model snapshots.
