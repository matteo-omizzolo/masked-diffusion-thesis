# Theory-First Research Plan

This document is retained as active provenance for the May 2026 theory-first reassessment. The current operational state is summarized in `docs/ACTIVE_RESEARCH_STATE.md`.

## Status

The theory-first pass has served its purpose:

- the finite-budget intervention framework is the active framing;
- Theorem A is the proved marginal baseline;
- stronger negative-result claims have been replaced by a careful empirical ranker-class limitation;
- ProSeCo-OWT is now a completed redundancy/saturation-driven case study;
- informed-correctors/Text8 is the proposed principled validation backend.

## Completed Or Superseded Items

| Item | Current status |
|---|---|
| Reframe marginal ranker failure carefully | complete |
| Separate proved claims from empirical diagnostics | complete |
| Run aggregate state predictability diagnostics | complete, negative |
| Run pair/enriched/token-level predictability diagnostics | complete, negative |
| Run saturation structure diagnostics | complete |
| Run corrector-strength diagnostics | complete |
| Decide on second backend | complete exploratory audit; informed-correctors/Text8 recommended |

## Remaining Theory Work

- Write the theorem narrative in the thesis with modest assumptions and no universal ProSeCo claim.
- Present saturation/sublinear composition as an empirical diagnostic, not a theorem.
- Use informed-correctors/Text8 only as a proposed validation backend until a checkpoint exists and diagnostics are run.

## Current Operational Next Step

Email to informed-correctors authors sent 2026-05-14; awaiting reply. In
parallel, drive the external-GPU Stage 0 / Stage 1 sequence per
`docs/10_external_gpu_text8_fallback.md`. The Bocconi JAX path is documented
blocked; no full training is approved.
