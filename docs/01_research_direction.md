# Research Direction

## Refined Thesis Framing

The thesis is a diagnostic and statistical framework for finite-budget corrector timing regimes in masked diffusion language models.

The central question is:

> For a fixed predictor schedule and fixed corrector-placement budget `B`, when is corrector timing reducible to marginal signal ranking, when do interactions and saturation dominate, and when is direct schedule search needed?

The ProSeCo-OWT work is now best framed as a completed case study showing a redundancy/saturation regime. informed-correctors/Text8 is the proposed principled validation backend, but it is not yet trained or validated in this repo.

## Current Scientific Position

The ProSeCo evidence argues against rich state-predictive online scheduling as the next main path for this backend. State features at aggregate, enriched, and token levels did not add predictive power beyond simple geometry and `A_pair`.

The productive thesis contribution is therefore not "we found a universal online state predictor." It is:

- a finite-budget intervention framework for corrector timing;
- diagnostics for marginal gains, pair gains, and interaction residuals;
- empirical evidence that one major backend lies in a redundancy/saturation regime;
- search results showing that direct `G(S)` evaluation can recover headroom even when marginal rankers fail;
- a proposed validation path on a cleaner informed-correctors/Text8 backend.

## Scope

In scope:

- fixed predictor schedule;
- fixed corrector-placement budget;
- schedule-level utility `G(S)`;
- marginal interventions `Delta_t`;
- pair interventions `G({s,t})` and `xi = G({s,t}) - Delta_s - Delta_t`;
- saturation, redundancy, and schedule geometry diagnostics;
- backend validation using a principled informed-corrector implementation.

Out of scope:

- designing new token-selection policies;
- changing predictor schedules;
- proposing a new corrector kernel as the main contribution;
- claiming that ProSeCo-OWT results are universal;
- treating PRISM LLaDA as a clean validation backend without resolving its remasking/token-quality confounds.

## Backend Direction

informed-correctors/Text8 is the preferred second backend because it is conceptually close to locally informed discrete correctors and its hollow transformer gives structurally valid learned conditional logits. This should be stated carefully: the architecture excludes coordinate `d` from its own conditional prediction, but the learned conditional remains a model approximation, not the true data conditional.

PRISM LLaDA may be valuable for other questions, but it is not currently a clean answer to this thesis question because its central mechanism is learned token quality/remasking rather than isolated timing of a fixed corrector.
