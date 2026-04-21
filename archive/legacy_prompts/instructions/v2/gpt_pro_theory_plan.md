# gpt_pro_theory_plan.md

# Theory work plan: from tractable theorem to stretch goal

## Thesis-theory objective

The theory should support the following claim:

> Under a fixed predictor schedule and fixed corrector budget, the problem of placing corrector loops can be reduced to a **proxy-based ranking problem over one-loop marginal gains**, and simple signals are useful only insofar as they induce a low-regret schedule relative to the oracle ordering.

This is the best realistic theory target because it:

- is directly tied to the main experiment,
- does not require a global contraction theorem,
- is strong enough to be meaningful,
- and is finishable in MSc time.

---

## 1. The recommended theorem stack

## Main theorem

### Theorem A — Proxy-schedule regret under approximate additivity

#### Objects

- Fixed predictor schedule with steps `t = 1, ..., T`
- Final evaluation objective `F`
- One-loop marginal gain
  `Δ_t := F(y_t^{+1}) - F(y_base)`
- Actual gain from a size-`B` schedule `S`
  `G(S) := F(y_S) - F(y_base)`
- Proxy score `s_t`

#### Assumptions

1. **Binary placement for the theorem:** at most one extra corrector loop per step.
2. **Approximate additivity:** for all `S` with `|S| = B`,
   `|G(S) - Σ_{t in S} Δ_t| <= η_B`.
3. **Proxy approximation or calibrated proxy:** there exists a monotone calibration map `ψ` such that
   `|Δ_t - ψ(s_t)| <= ε` for all `t`.

#### Claim

Let `S*_B` be the optimal size-`B` schedule for the true gain `G`, and let `Ŝ_B` be the top-`B` schedule by `ψ(s_t)`. Then

`G(S*_B) - G(Ŝ_B) <= 2B ε + 2 η_B`.

This theorem says: if the proxy predicts one-loop gains well enough and the multi-step gain is close enough to additive, then the proxy-induced schedule is near-optimal.

#### Why this is the right main theorem

- It matches the exact empirical pipeline.
- It turns the thesis into a clean theory-plus-measurement project.
- It absorbs the additivity issue instead of hiding it.
- It only needs assumptions that can be stress-tested empirically.

---

## Supporting lemma 1

### Lemma A1 — Oracle top-`B` is optimal under exact additivity

Assume exact additivity:

`G(S) = Σ_{t in S} Δ_t`.

Then the optimal size-`B` schedule is the set of the `B` largest `Δ_t`.

#### Role

This is the formal backbone of Candidate 1. It should be proved quickly and cleanly.

#### Importance

Not publishable by itself, but essential as the base case for the main theorem.

---

## Supporting lemma 2

### Lemma A2 — Proxy approximation implies additive-regime regret bound

Assume exact additivity and `|Δ_t - ψ(s_t)| <= ε`.

Then top-`B` by `ψ(s_t)` has regret at most `2 B ε` relative to oracle top-`B` by `Δ_t`.

#### Proof idea

Standard exchange argument:

- every selected element can overestimate gain by at most `ε`,
- every omitted oracle element can underestimate by at most `ε`,
- summing over `B` positions gives `2Bε`.

#### Role

This is the cleanest “proxy quality controls schedule quality” statement.

---

## Supporting proposition

### Proposition B — Burn-in exclusion / gating proposition

Let `E_tau := { t : u_t < tau }` be an early region with low unmasked fraction.
Assume there exists `δ >= 0` such that for all `t in E_tau`,

`Δ_t <= δ`,

and there are at least `B` steps outside `E_tau` with gain strictly larger than `δ`.

Then any budget-`B` schedule that places a corrector loop inside `E_tau` is suboptimal relative to one that reallocates that loop to a sufficiently larger-gain step outside `E_tau`.

#### Why this is better than the current Candidate 3

It proves the practical burn-in idea without requiring a universal mutual-information monotonicity theorem.

#### Role

This should become a corollary / proposition that justifies burn-in-gated entropy or burn-in-gated proxies.

---

## Optional supporting proposition

### Proposition C — Near-optimality under bounded pairwise interaction

Define pairwise interaction error by

`I(t, t') := G({t, t'}) - Δ_t - Δ_{t'}`.

If interactions are uniformly bounded, e.g.

`|I(t, t')| <= κ`

and higher-order interaction is controlled so that `η_B <= c κ B(B-1)`, then the additive oracle remains near-optimal up to that interaction budget.

#### Role

This proposition gives a cleaner bridge from the theory to the additivity diagnostic experiment.

#### Practical value

High. It lets you say exactly what it means for additivity to be “good enough.”

---

## Empirical-only comparator

### Confidence margin / quality mass comparison

Signals based on inverse margin or quality mass should remain **empirical comparators**, not theorem centers.

Why:

- they may win empirically,
- but they rely on calibration or learned heads,
- and the interesting theory is the general schedule-regret framework, not the specific signal.

---

## Stretch route only

### Candidate 2 / contraction route

Keep a contraction-based theorem only as a stretch appendix or stylized-model exercise.

It is not the right first target because the current rigorous Gibbs contraction result is under strong log-concavity for random-scan Gibbs, not the real masked-diffusion text setting.[^ascolani]

---

## 2. First three derivations to attempt

## Derivation 1 — Formalize the objects exactly

Write down, precisely and without rhetoric:

1. baseline trajectory,
2. one-loop branch at step `t`,
3. final objective `F`,
4. one-loop marginal gain `Δ_t`,
5. multi-step gain `G(S)`.

### Deliverable

A two-page note with exact notation and no theorem yet.

### Why this is first

Because all later ambiguity comes from not separating `Δ_t` from `G(S)`.

---

## Derivation 2 — Prove Lemma A1 and Lemma A2

Do the exact-additivity case first.

### Deliverable

A clean theorem note containing:

- Lemma A1 (oracle top-`B` under exact additivity),
- Lemma A2 (regret bound under sup-norm proxy error),
- discussion of why this is not enough yet.

### Why this is second

This is the shortest route to a real theorem and clarifies what the experiment must estimate.

---

## Derivation 3 — Prove the burn-in exclusion proposition

State and prove Proposition B with the weakest clean assumption you can manage.

### Recommended form

Avoid MI monotonicity. Use a simple low-gain-region assumption.

### Deliverable

A short proposition with proof and a sentence explaining that the assumption is empirical / model-side rather than universal.

### Why this is third

It gives the thesis an interpretable practical insight early, even before the full experiment is done.

---

## 3. The first theorem to try to close

The first theorem to actually close should be:

## **Lemma A2 / Theorem A in the exact-additive version**

That is:

- exact additivity first,
- proxy error to regret second,
- approximate additivity extension third.

Do **not** begin with the full approximate-additivity statement if it will slow you down.

### Recommended order

1. exact-additive oracle lemma,
2. exact-additive regret theorem,
3. approximate-additivity extension,
4. burn-in proposition,
5. optional interaction corollary.

---

## 4. Assumptions to verify, weaken, or mark empirical

## Assumptions to keep explicit

### A. Fixed predictor schedule

Keep this fixed. Do not let predictor optimization leak into the thesis.

### B. Binary placement in the first theorem

Start with one extra loop per step. This keeps the theorem sharp and experimentally meaningful.

### C. Approximate additivity

This should be explicit and empirically checked.

## Assumptions to weaken if possible

### D. Uniform sup-norm proxy error

The clean `|Δ_t - ψ(s_t)| <= ε` assumption is easy to prove with, but can be softened later to:

- average-case error,
- ranking error,
- inversion count,
- isotonic calibration residual.

### E. Strict burn-in threshold

If the thresholded gate looks too crude, replace it with a monotone gate and prove only the threshold version as a special case.

## Assumptions to avoid as theorem foundations

### F. Strong log-concavity

Avoid using this as a real masked-text assumption.

### G. Stationarity of the corrector target across predictor time

Avoid assuming the corrector is converging to one fixed target distribution over the whole trajectory.

### H. Exact independence across steps

Replace with approximate additivity or bounded interaction.

---

## 5. Likely dead ends

These are the paths most likely to waste time.

## Dead end 1 — Full geometric contraction in the real MDM setting

Why it is dangerous:

- assumption mismatch,
- unclear stationary target,
- likely technical derailment.

## Dead end 2 — Universal theorem that entropy is the right signal

Why it is dangerous:

- even the thesis notes already contain plausible counterexamples,
- PRISM suggests a more direct quality signal may exist,[^prism]
- the actual question is comparative, not absolute.

## Dead end 3 — Universal mutual-information monotonicity theorem

Why it is dangerous:

- not needed for a good thesis,
- easy to get trapped in formalism that does not improve the final result.

## Dead end 4 — Overfitting theory to ProSeCo details

Why it is dangerous:

- the theorem should be about corrector scheduling at the level of abstract objects,
- not about specific hyperparameters in one codebase.

---

## 6. Fastest path to a defensible result

Here is the fastest defensible theory path.

## Step 1

Write the exact formal definitions of:

- `Δ_t`,
- `G(S)`,
- `Regret_B`,
- proxy score `s_t`,
- burn-in gate.

## Step 2

Prove the exact-additive oracle and regret lemmas.

## Step 3

Run the one-loop marginal-gain experiment.

From that experiment, estimate:

- how well each signal predicts `Δ_t`,
- how much interaction error exists,
- whether there is a low-gain burn-in region.

## Step 4

Use the empirical findings to finalize the approximate-additivity theorem statement and the burn-in proposition.

## Step 5

Only after this, revisit whether a small contraction appendix is worth adding.

That order is critical. Theory should be informed by the measured structure of `Δ_t`, not the other way around.

---

## 7. A concrete staged roadmap

## Stage I — closeable theorem

### Goal

Finish a mathematically correct result quickly.

### Output

- Lemma A1
- Lemma A2

### Status target

This should be fully written before any ambitious contraction attempt.

## Stage II — thesis-grade main theorem

### Goal

Add approximate additivity and schedule regret.

### Output

- Theorem A
- formal statement of `η_B`
- interaction diagnostic linked to theory

## Stage III — practical interpretability theorem

### Goal

Add burn-in proposition.

### Output

- Proposition B
- recommended gated scheduler class

## Stage IV — stretch material

### Goal

Only if time remains:

- diminishing returns / multi-loop water-filling,
- stylized contraction in a toy model,
- stronger ranking-based regret bounds.

---

## 8. What the final theory chapter should look like

A strong theory chapter would have this structure:

1. **Formal setup**  
   fixed predictor schedule, corrector loop, budget, objective.
2. **Oracle scheduling in the additive surrogate**  
   top-`B` lemma.
3. **Proxy-induced scheduling and regret**  
   main theorem.
4. **Burn-in exclusion and gated proxies**  
   proposition / corollary.
5. **Limitations and empirical validation plan**  
   approximate additivity, nonstationarity, signal choice.
6. **Optional appendix**  
   contraction diagnostics or stylized model.

That chapter is coherent, rigorous, and properly scaled to an MSc thesis.

---

## 9. Final recommendation

If you want the best balance of rigor, novelty, and finishability, do this:

- **Main theorem:** proxy-based schedule regret under approximate additivity.
- **Main proposition:** burn-in exclusion / gating.
- **Main experiment:** measure one-loop marginal gain and test whether entropy, margin, or quality mass ranks it well enough to beat uniform.
- **Stretch goal only:** contraction-based analysis.

That is the direction most likely to produce a thesis that is both mathematically honest and actually finishable.

---

## References consulted

- Ascolani, Lavenant, and Zanella, “Entropy contraction of the Gibbs sampler under log-concavity.” arXiv:2410.00858. https://arxiv.org/abs/2410.00858
- Lavenant and Zanella, “Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations.” arXiv:2510.25544. https://arxiv.org/abs/2510.25544
- Chen, Cong, and Li, “Optimal Inference Schedules for Masked Diffusion Models.” arXiv:2511.04647. https://arxiv.org/abs/2511.04647
- Kim et al., “Fine-Tuning Masked Diffusion for Provable Self-Correction.” arXiv:2510.01384. https://arxiv.org/abs/2510.01384

[^ascolani]: Ascolani, Lavenant, and Zanella, arXiv:2410.00858, abstract.
[^prism]: Kim et al., arXiv:2510.01384, abstract.
