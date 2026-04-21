# gpt_pro_experiment_design.md

# Experiment design: is entropy a good proxy for one-loop marginal gain?

## Goal

Test whether trajectory signals such as entropy, inverse confidence margin, and quality mass predict the **marginal value of adding one corrector loop at predictor step `t`**.

The experiment has two linked purposes:

1. **Scientific:** identify whether entropy is a good proxy for one-loop marginal gain, and where it fails.
2. **Theoretical:** provide the empirical object needed by the recommended theorem route, namely the sequence of one-loop gains `Δ_t` and the degree to which signals rank them well.

---

## 1. Main principle

The key object is the **one-loop marginal gain** at step `t`.

Fix:

- a predictor schedule with `T` steps,
- a base masked diffusion model,
- a corrector mechanism,
- a random seed or deterministic sampling rule,
- a final evaluation objective `F`.

Let:

- `y_base` be the final sample from the baseline trajectory with **no added corrector loop**,
- `y_t^{+1}` be the final sample obtained by branching from state `z_t`, inserting **exactly one corrector loop** at step `t`, and then continuing the same predictor schedule with no additional corrector loops.

Define the one-loop marginal gain:

`Δ_t := F(y_t^{+1}) - F(y_base)`.

Everything in the experiment is organized around measuring `Δ_t`, its local token effects, and its relationship to trajectory signals.

---

## 2. Recommended platforms

## Primary platform

### ProSeCo-OWT

Use ProSeCo-OWT as the primary system because it is the cleanest public masked-diffusion text setup with explicit correction loops and public checkpoints.[^proseco_repo]

Why it is the best primary platform:

- direct support for correction loops,
- explicit inference controls for frequency / loop count / delayed start,
- public OWT checkpoint,
- no need to invent a corrector from scratch.

## Secondary platforms

### MDLM + ReMDM

Use MDLM / ReMDM as a secondary platform when you want:

- entropy computation utilities,
- alternative baselines,
- remasking-style comparators,
- a cleaner inference-only sandbox.[^mdlm_repo][^remdm_repo]

### PRISM

Use PRISM only if you can obtain or train the quality head. PRISM is the best way to test **quality mass** as a signal, but it is not needed for the first milestone.[^prism_repo]

## Practical recommendation

Phase 1 should be:

- **ProSeCo-OWT** for entropy and margin studies.

Phase 2, only if time permits:

- **PRISM-OWT** for quality mass.

---

## 3. Experimental protocols

Use **two complementary protocols**.

## Protocol A — reference-available denoising study

### Why

This gives exact token-level correctness. It is the cleanest way to measure whether a corrector improves the sample relative to known ground truth.

### Setup

- Draw a held-out text sequence `x` from OWT validation / test.
- Corrupt or mask it according to the forward process or a controlled masking pattern.
- Run the fixed predictor schedule to denoise.
- At each step `t`, branch the trajectory and insert one corrector loop.
- Continue to the end and compare both final outputs to the known clean target `x`.

### Best use

- token-level correctness,
- changed-token correctness,
- calibration of local signals,
- low-variance marginal-gain measurement.

## Protocol B — unconditional generation study

### Why

This tests actual generation quality in the setting that matters for the thesis narrative.

### Setup

- Start from the usual unconditional masked initialization.
- Run the baseline predictor trajectory.
- At each step `t`, branch and insert one corrector loop.
- Continue to the end and score final outputs with reference-free or distributional metrics.

### Best use

- external LM score changes,
- generative perplexity changes,
- schedule-level MAUVE,
- final validation that proxy-guided schedules matter in real generation.

## Thesis recommendation

Use **Protocol A for the main correlation study** and **Protocol B for the schedule-level quality study**.

That combination is much stronger than using only unconditional generation.

---

## 4. Trajectory states and branching procedure

For each input sequence or unconditional seed:

1. Run the baseline predictor-only trajectory and store `z_0, z_1, ..., z_T`.
2. At every step `t`, log the full signal set from `z_t`.
3. Create a branch from `z_t`.
4. Apply exactly **one** corrector loop to obtain `z_t'`.
5. Continue both baseline and branched trajectories to completion under the same continuation protocol.
6. Compute:
   - immediate token-change effects between `z_t` and `z_t'`,
   - final gain `Δ_t` between the completed outputs.

### Variance control

If the predictor or corrector is stochastic, reduce variance with one of these strategies:

- **common random numbers:** same continuation seed after branching,
- **deterministic decoding:** use argmax for the continuation study,
- **small rollout averaging:** average over `m = 2` or `4` continuations for a subset of inputs.

For the first thesis pass, deterministic or common-random-number continuation is preferable.

---

## 5. Signals to log

Log every signal at every predictor step `t`.

## A. Required signals

### 1. Average entropy over all currently predicted positions

`H_all(t) := mean_i H(p_theta(. | z_t, i))`

This is the broadest entropy measure.

### 2. Average entropy over revisable tokens

`H_rev(t) := mean_{i in R_t} H(p_theta(. | z_t, i))`

where `R_t` is the set of tokens the corrector is actually allowed to modify.

This is the most important entropy variant. It is more faithful to corrector value than full-trajectory entropy.

### 3. Confidence margin

For each position `i`, let `p_1(i)` and `p_2(i)` be the top-1 and top-2 token probabilities.

`M_all(t) := mean_i [p_1(i) - p_2(i)]`

### 4. Inverse confidence margin

`IM_all(t) := 1 - M_all(t)`

Also compute revisable-token versions.

### 5. Fraction unmasked

`u_t := (# currently unmasked tokens) / L`

This is essential for burn-in analysis.

### 6. Quality mass if available

If PRISM or another quality head is available,

`Q_t := mean_{i in R_t} [1 - q_phi(i | z_t)]`

This is the direct “how much low-quality mass remains?” signal.

## B. Strong optional signals

### 7. Entropy slope

`dH_t := H_rev(t) - H_rev(t-1)`

A flat or sharply declining entropy regime may be more informative than raw entropy alone.

### 8. Immediate instability signal

Measure how unstable the trajectory is just before step `t`, e.g.

- fraction of tokens that changed between `z_{t-1}` and `z_t`, or
- disagreement under two cheap stochastic predictor continuations.

### 9. Concentration statistics

- entropy variance across revisable tokens,
- maximum revisable-token entropy,
- low-margin token count.

These can matter if correction value is concentrated in a few positions.

---

## 6. Marginal-gain measurements

You asked specifically for token-change rate and quality improvement. Log both local and final effects.

## A. Immediate local effects of one corrector loop

### 1. Token-change rate

`TCR_t := (1 / |R_t|) * sum_{i in R_t} 1[z_t'(i) != z_t(i)]`

This is the direct measure of “how much the loop changed.”

Log also:

- `TCR_all(t)` over all positions,
- `TCR_rev(t)` over revisable positions.

### 2. Effective update rate

If the corrector is MH-style or otherwise has explicit acceptance / keep decisions, log the true acceptance rate.

If not, define an effective update rate as:

`EUR_t := fraction of revisable positions whose final token after the loop differs from the pre-loop token`.

### 3. Confidence or entropy drop after the loop

`ΔH_local(t) := H_rev(z_t') - H_rev(z_t)`

A useful corrector often lowers local uncertainty, though not always.

### 4. Changed-token usefulness (reference-available only)

If ground truth is known,

`CTU_t := (# changed tokens that became correct - # changed tokens that became incorrect) / |R_t|`

This is stronger than raw change rate.

## B. Final quality effects of one corrector loop

### 5. Final token-accuracy gain (Protocol A)

`ΔAcc_t := TokenAccuracy(y_t^{+1}, x) - TokenAccuracy(y_base, x)`

Also log changed-token accuracy gain.

### 6. Final sequence-level improvement (Protocol A)

Possible options:

- exact match rate if tasks allow it,
- normalized edit distance to ground truth,
- average negative log-likelihood of the target under an external scorer.

### 7. Final external-LM score gain (Protocol B)

Use a fixed external language model as scorer and log:

`ΔLM_t := Score_ext(y_t^{+1}) - Score_ext(y_base)`

This can be token-level average log-prob or negative perplexity.

### 8. Distributional quality (schedule-level only)

For full schedules, not per-sample one-loop branches, use:

- MAUVE,
- generative perplexity,
- repetition metrics,
- distinct-n / diversity diagnostics if feasible.

## C. Define a single marginal-gain scalar

For the main ranking analysis, you need a single scalar `Δ_t`.

### Recommended scalar for Protocol A

`Δ_t := ΔAcc_t`

This is the cleanest if ground truth is available.

### Recommended scalar for Protocol B

`Δ_t := ΔLM_t`

This is the cleanest reference-free scalar.

### Keep token-change separate

Do **not** collapse token-change rate into the same scalar. Treat it as a mechanistic diagnostic:

- `TCR_t` tells you whether the loop is acting,
- `Δ_t` tells you whether the action helps.

That distinction is important. High token-change with low quality gain means the loop is active but poorly targeted.

---

## 7. Correlation and ranking analysis

This section is the heart of the entropy-proxy study.

## A. Correlation analysis

For each signal `s_t` and gain `Δ_t`, compute:

### 1. Pearson correlation

Measures linear association.

### 2. Spearman correlation

Measures ranking quality. This is more important than Pearson for scheduling.

### 3. Within-sequence and pooled analyses

Do both:

- **within-sequence:** correlation across `t` for each sample, then average,
- **pooled:** correlation across all `(sample, t)` pairs.

Within-sequence is closer to the scheduling problem.

## B. Top-`k` ranking quality

For each budget `B`, compare:

- the oracle top-`B` steps by measured one-loop gain,
- the top-`B` steps induced by each signal.

Compute:

- top-`B` overlap,
- precision@`B`,
- recall@`B`,
- normalized discounted cumulative gain if you want a smoother ranking metric.

## C. Budgeted regret relative to oracle

Under the additive single-loop approximation, define the oracle regret of signal `s` at budget `B` by:

`Regret_B(s) := sum_{t in topB(Δ)} Δ_t - sum_{t in topB(s)} Δ_t`

This is the most important analysis number, because it directly matches the recommended theorem.

Compute regret for:

- raw entropy,
- entropy on revisable tokens,
- burn-in-gated entropy,
- inverse margin,
- quality mass if available,
- uniform baseline.

## D. Burn-in diagnostics

Plot and analyze:

- `Δ_t` versus `u_t`,
- `H_t` versus `u_t`,
- `TCR_t` versus `u_t`,
- `Δ_t / TCR_t` versus `u_t`.

This tells you whether early entropy is merely “high activity” or actually “high useful gain.”

---

## 8. Budgeted scheduling evaluation

After the one-loop study, evaluate full schedules at fixed total budget.

## A. Schedules to compare

At minimum:

1. **Uniform**
2. **Front-loaded**
3. **Back-loaded**
4. **Middle-loaded**
5. **Entropy-proportional**
6. **Entropy over revisable tokens**
7. **Burn-in-gated entropy**
8. **Inverse-margin-based**
9. **Quality-mass-based** if available
10. **Oracle top-`B` schedule using measured one-loop gain**

## B. Budgets

Use at least:

- `B in {T/8, T/4, T/2}`

If compute is tight, start with `T/8` and `T/4`.

## C. Important design detail

The oracle schedule should be treated as an **approximate oracle**, because it uses single-loop gains and ignores interactions. That is fine as long as you say so explicitly.

## D. Realized full-schedule evaluation

For each schedule and budget, run actual generation and compare final metrics. This is what tells you whether the proxy signal is practically useful beyond the one-loop approximation.

---

## 9. Metrics

## Protocol A — reference-available metrics

### Token-level

- token accuracy to clean target,
- changed-token correctness,
- Hamming error / edit distance,
- negative log-likelihood of the target under an external scorer if desired.

### Sequence-level

- exact reconstruction rate if meaningful,
- normalized edit distance,
- average per-sequence accuracy gain.

## Protocol B — unconditional generation metrics

### Per-sample / per-sequence

- external LM average log-prob,
- generative perplexity if available,
- repetition rate,
- length-normalized scoring.

### Batch-level

- MAUVE,
- diversity metrics,
- distributional quality summary.

## Practical diagnostics

Regardless of protocol, always log:

- token-change rate,
- effective update rate,
- entropy drop after the loop,
- fraction unmasked,
- wall-clock overhead per extra loop.

---

## 10. Interaction diagnostic for the additivity assumption

This is important and should be included.

Choose a small set of step pairs `(t1, t2)` and measure:

- `G({t1})`,
- `G({t2})`,
- `G({t1, t2})`.

Define pairwise interaction:

`I(t1, t2) := G({t1, t2}) - G({t1}) - G({t2})`.

Then summarize:

- mean absolute interaction,
- max interaction,
- interaction relative to average one-loop gain.

### Interpretation

- small interactions support the approximate-additivity theory route,
- large interactions mean the theorem should be stated more cautiously and the oracle schedule interpreted as approximate.

This diagnostic directly answers Q5 from the notes.

---

## 11. Concrete ablation plan

## Stage 1 — cheapest high-value study

Model: **ProSeCo-OWT**  
Data: 200 to 500 held-out sequences for Protocol A, 200 unconditional seeds for Protocol B  
Signals: entropy, revisable entropy, inverse margin, fraction unmasked  
Branching: one-loop branches at every `t`  
Outputs: `TCR_t`, `ΔAcc_t` or `ΔLM_t`, correlations, regret curves

This stage alone can already answer whether entropy is promising.

## Stage 2 — budgeted schedules

Evaluate the full schedule family at `B in {T/8, T/4, T/2}`.

## Stage 3 — quality mass extension

If PRISM is available, add quality mass and compare it to entropy / margin.

## Stage 4 — interaction and diminishing-returns diagnostics

- pairwise interaction study,
- multi-loop `k in {0,1,2,4,8}` at selected steps,
- fit geometric vs concave vs nonparametric gain curves.

---

## 12. What outcomes would mean

## Outcome A — entropy is a useful proxy

Evidence:

- Spearman correlation between entropy-based signal and `Δ_t` is consistently positive,
- entropy top-`B` overlap with oracle is high,
- entropy schedule regret is materially below uniform,
- realized entropy schedules outperform uniform across budgets.

Interpretation:

Entropy is a viable signal for scheduling, and the thesis can say so.

## Outcome B — entropy works only after burn-in

Evidence:

- raw entropy has weak or noisy correlation with `Δ_t`,
- early high-entropy steps show high `TCR_t` but low `Δ_t`,
- burn-in-gated entropy sharply improves regret and full-schedule results.

Interpretation:

This is arguably the most interesting likely outcome. It supports the burn-in thesis strongly.

## Outcome C — inverse margin or quality mass beats entropy

Evidence:

- margin / quality mass shows better rank correlation and lower regret,
- margin / quality schedules consistently beat entropy schedules.

Interpretation:

The thesis should pivot from “entropy-adaptive” to “signal-adaptive,” with entropy as a baseline rather than the main hero.

## Outcome D — all signals fail

Evidence:

- low correlations across the board,
- proxy schedules close to uniform,
- oracle gains do not vary much over `t`.

Interpretation:

This is still a valid result. It would suggest that either:

- timing matters less than expected, or
- aggregate trajectory signals are too crude, and one needs a learned or local signal.

That would still support a good thesis if framed honestly.

---

## 13. Minimal implementation checklist

1. Instrument ProSeCo sampling to save `z_t` and logits.
2. Implement one-loop branching from saved `z_t`.
3. Compute and store:
   - entropy,
   - revisable entropy,
   - margin,
   - inverse margin,
   - `u_t`,
   - optional `Q_t`.
4. Compute immediate effects:
   - token-change rate,
   - effective update rate,
   - entropy drop.
5. Continue trajectories and compute final `Δ_t`.
6. Build ranking / regret analysis scripts.
7. Build full-schedule evaluation scripts.
8. Add pairwise interaction diagnostics.

---

## 14. Recommended first milestone

If you only do one experiment first, do this:

> On ProSeCo-OWT, measure for every step `t` the one-loop token-change rate and the one-loop final-quality gain, then test whether revisable-token entropy and burn-in-gated entropy rank those gains better than raw entropy and uniform scheduling.

That single milestone directly decides whether the entropy route is alive.

---

## References consulted

- ProSeCo repository and public checkpoints. https://github.com/kuleshov-group/proseco
- Schiff et al., “Learn from Your Mistakes: Self-Correcting Masked Diffusion Models.” arXiv:2602.11590. https://arxiv.org/abs/2602.11590
- MDLM repository and `mdlm-owt` checkpoint. https://github.com/kuleshov-group/mdlm
- ReMDM repository. https://github.com/kuleshov-group/remdm
- Wang et al., “Remasking Discrete Diffusion Models with Inference-Time Scaling.” arXiv:2503.00307. https://arxiv.org/abs/2503.00307
- PRISM repository. https://github.com/SeunggeunKimkr/PRISM
- Kim et al., “Fine-Tuning Masked Diffusion for Provable Self-Correction.” arXiv:2510.01384. https://arxiv.org/abs/2510.01384

[^proseco_repo]: ProSeCo GitHub README exposes direct corrector-sampling parameters and released `proseco-owt` / `proseco-llada-sft` checkpoints.
[^mdlm_repo]: MDLM GitHub README exposes the public `kuleshov-group/mdlm-owt` checkpoint and standard sampling / evaluation entry points.
[^remdm_repo]: ReMDM GitHub README states that it adds MAUVE computation, entropy computation, several remasking strategies, and predictor-corrector baselines on top of MDLM.
[^prism_repo]: PRISM GitHub README exposes OWT fine-tuning and evaluation scripts, plus a theory-backed per-token quality head.
