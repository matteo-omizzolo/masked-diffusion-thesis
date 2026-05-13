# Theory

## Current Framework

The thesis theory is a finite-budget intervention framework for corrector timing.

For a fixed predictor schedule and fixed corrector-placement budget `B`, define a schedule utility `G(S)` and study:

- marginal intervention gain `Delta_t = G({t})`;
- pair gain `G({s,t})`;
- interaction residual `xi(s,t) = G({s,t}) - Delta_s - Delta_t`;
- schedule-level behavior `G(S)` under rankers, search, and oracle procedures.

This framing separates corrector timing from token selection, predictor schedule design, and corrector-kernel design.

## What Is Proved

The current proved baseline is Theorem A: a marginal/uniform proxy regret statement that formalizes when marginal schedules can be competitive and when the proxy can fail.

The former stronger negative-result framing has been replaced by a more careful empirical ranker-class limitation:

- a formal time-only part where assumptions are explicit;
- empirical evidence on the tested ProSeCo-OWT rankers;
- no universal impossibility claim.

## What Is Empirical

The main ProSeCo-OWT findings are empirical:

- marginal rankers fail on the tested backend and protocol;
- direct search over `G(S)` recovers much of the MC-oracle headroom;
- pair interactions are mostly redundant/negative;
- geometry and `A_pair = Delta_s + Delta_t` explain much of the pair structure;
- aggregate, enriched, and token-level state features do not add predictive signal;
- `G_pair` composes sublinearly as `A_pair` grows;
- stronger correctors rescale amplitude but do not remove the cross-time interaction pattern.

These support a redundancy/saturation interpretation for ProSeCo-OWT, not a theorem about all masked diffusion models.

## Saturation And Sublinear Composition

The current empirical picture is that high marginal gains do not combine additively. When `A_pair` is large, `G({s,t})` grows sublinearly and `xi` tends to become more negative. This is the best current explanation for why direct `G(S)` search can outperform marginal ranking even when pair interactions are not strongly synergistic.

## No ProSeCo Locally Balanced Theorem

There is no current theorem showing that ProSeCo-OWT implements an exact locally balanced corrector with tractable true conditionals. ProSeCo remains the empirical case study.

## Informed-Correctors/Text8 As Cleaner Backend

informed-correctors/Text8 is attractive because the hollow transformer gives architecturally valid learned conditional logits: coordinate `d` is excluded from the input used to predict coordinate `d`. This is a cleaner object for locally informed correctors than ProSeCo-OWT.

This must be stated precisely. The hollow transformer gives model conditionals/logits with structural exclusion, not true data conditionals. The model still has to be trained, and the learned conditional quality must be empirically checked.
