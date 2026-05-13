# Theory Connections for Fixed-Budget Corrector Scheduling

> Research note. Generated 2026-05-07.
>
> This file is background/provenance for the theory-first programme. It is not
> a new active status document. Current status remains in `START_HERE.md` and
> `docs/README.md`; formal theorem statements remain in
> `research/candidate_theorems.md`.

## Purpose

This note asks which existing theories can inform the thesis problem:

> For a fixed masked-diffusion predictor, informed corrector, quality
> functional `F`, and corrector-placement budget `B`, when should the
> corrector be applied along the generation trajectory?

The conclusion is that no existing theory appears to give an off-the-shelf
optimal timing rule for masked-diffusion correctors. The most useful route is
to borrow lenses:

1. finite-horizon budgeted control for the normative optimum;
2. set-function optimization for offline schedule selection;
3. adaptive submodularity / bandit indices for cheap greedy or online policies;
4. SMC / Feynman-Kac / Particle Gibbs for the idea of corrective
   rejuvenation;
5. Kalman and sensor scheduling for uncertainty-triggered intervention;
6. value-of-information / metareasoning for deciding whether an extra
   computation is worth spending.

The active thesis framework is exactly in this spirit: Theorem A formalizes
the marginal/additive regime; Theorem B/B' formalizes interaction-aware
surrogate selection; Diagnostic Framework C classifies the observed regime.

## 0. Current Thesis Object

Let a masked diffusion model run for `T` predictor steps. A schedule

```text
S subset {0, ..., T-1}, |S| = B
```

specifies when to spend the `B` available corrector calls. The objects in
`candidate_theorems.md` are:

- `G(S)`: true paired quality gain from applying the corrector at schedule `S`.
- `Delta_t`: marginal gain from correcting only at time `t`.
- `A(S) = sum_{t in S} Delta_t`: additive / marginal surrogate.
- `xi_{t,t'} = G({t,t'}) - Delta_t - Delta_{t'}`: pairwise interaction.
- `Q(S) = A(S) + sum_{t<t', t,t' in S} xi_{t,t'}`: pairwise surrogate.

The mathematical question is not whether a corrector is useful in the
abstract. It is whether the timing problem is:

- no-op: no headroom over uniform placement;
- marginal/rankable: top-`B` per-step scores suffice;
- interaction-driven: low-order schedule interactions matter;
- higher-order/search-driven: schedule search works but simple surrogates do
  not explain it;
- online-decision: a cheap state-conditioned controller can spend budget
  competitively during generation.

## 1. Diffusion Predictor-Corrector Theory

### Source landscape

Continuous score-based diffusion has a principled predictor-corrector sampler:
Song et al. formulate sampling through reverse-time SDEs whose drift depends on
the time-dependent score, and introduce predictor-corrector samplers to reduce
discretization error during reverse dynamics
([Song et al., 2021](https://arxiv.org/abs/2011.13456)).

Discrete and masked diffusion provide the generative substrate:

- D3PMs define structured diffusion processes over discrete state spaces and
  include absorbing-state transitions that connect to mask-based generation
  ([Austin et al., 2021](https://arxiv.org/abs/2107.03006)).
- SEDD extends score ideas to discrete data by estimating ratios of the data
  distribution ([Lou et al., 2023](https://arxiv.org/abs/2310.16834)).
- MDLM provides a simplified masked diffusion language-model objective and
  efficient samplers ([Sahoo et al., 2024](https://arxiv.org/abs/2406.07524)).

Recent masked-diffusion self-correction work motivates the corrector itself:

- ProSeCo trains a model to perform both unmasking and correction, allowing
  already generated tokens to be revised and improving the quality-efficiency
  tradeoff ([Schiff et al., 2026](https://arxiv.org/abs/2602.11590)).
- PRISM learns per-token quality scores for inference-time self-correction in
  MDMs ([Kim et al., 2025](https://arxiv.org/abs/2510.01384)).

Adaptive timestep scheduling in diffusion models is adjacent but not identical:

- Chen et al. study adaptive time-stepping schedules for diffusion models,
  optimizing a sampling error bound and using a greedy adjustment for discrete
  diffusion ([Chen et al., 2024](https://proceedings.mlr.press/v244/chen24c.html)).
- Align Your Steps optimizes continuous diffusion sampling schedules for
  solvers/datasets rather than using fixed handcrafted schedules
  ([Sabour et al., 2024](https://research.nvidia.com/index.php/publication/2024-07_align-your-steps-optimizing-sampling-schedules-diffusion-models)).
- AdaDiff learns instance-specific step usage policies to trade generation
  quality against inference time ([Zhang et al., 2024](https://arxiv.org/abs/2311.14768)).
- Optimal Stepsize for Diffusion Sampling proposes a dynamic-programming
  framework for step-size optimization in image diffusion
  ([Pei et al., 2025](https://arxiv.org/abs/2503.21774)).
- Plan for Speed studies scheduling in masked diffusion language models, but
  for token unmasking / parallelization rather than corrector-placement budget
  ([Luxembourg et al., 2025](https://arxiv.org/abs/2506.19037)).

### What we can borrow

The continuous score-based literature gives the analogy:

```text
predictor step: follow approximate reverse dynamics
corrector step: reduce local sampler error or improve local consistency
```

The adaptive timestep literature gives a useful methodological pattern:

```text
fixed budget of expensive model calls
choose where to spend them along a trajectory
optimize an error or quality objective
```

### What it does not give

These papers generally schedule denoising timesteps, solver calls, or unmasking
positions. They do not directly answer:

```text
given a fixed corrector budget B, which denoising times should receive
corrector calls?
```

That is the thesis gap.

## 2. Finite-Horizon Budgeted Control / Multiple Stopping

### Core idea

The exact online problem is a finite-horizon budgeted control problem. At time
`t`, with current trajectory state `x_t` and remaining budget `b`, the ideal
policy compares:

```text
skip:    continue with state x_{t+1}, budget b
correct: apply corrector, continue with budget b-1
```

A Bellman recursion would have the form:

```text
V_t(x_t, b) =
  max {
    E[V_{t+1}(x_{t+1}, b) | skip],
    E[V_{t+1}(x'_{t+1}, b-1) | correct]
  }.
```

This is essentially a multiple-stopping or finite-horizon resource-allocation
problem. Classical optimal stopping theory gives the general view of stopping
decisions in stochastic processes, including Markovian and martingale methods
([Peskir and Shiryaev, 2006](https://link.springer.com/book/10.1007/978-3-7643-7390-0)).
POMDP/MDP theory similarly frames partially observed sequential decisions, but
exact solution is computationally hard in rich state spaces
([White, 1991](https://link.springer.com/article/10.1007/BF02204836)).

### Thesis use

This is the normative target:

```text
optimal online corrector timing = budgeted dynamic programming
```

The state `x_t` is a long partially masked token sequence, so exact dynamic
programming is infeasible. The thesis can use this framework to justify:

- why the online problem is hard;
- why cheap state summaries `z_t` are needed;
- why Theorem D is only appendix-grade unless we validate a compact state
  abstraction.

### Possible theorem direction

If a compact state `z_t = phi(x_t, t, b)` yields a uniformly accurate value
approximation

```text
|V_t(x_t, b) - Vhat_t(z_t, b)| <= delta,
```

then the induced online policy loses at most `O(T delta)` against the optimal
budgeted policy. This is the current spirit of Theorem D.

## 3. Set-Function Optimization and Submodularity

### Core idea

Offline schedule selection is a cardinality-constrained set-function problem:

```text
maximize G(S) subject to |S| = B.
```

If `G` were monotone submodular, greedy selection would be provably near
optimal. Nemhauser, Wolsey, and Fisher established classical approximation
results for greedy maximization of monotone submodular functions under
cardinality constraints
([Nemhauser et al., 1978](https://www.researchgate.net/publication/242914003_An_Analysis_of_Approximations_for_Maximizing_Submodular_Set_Functions-I)).
Later work extends this landscape to matroid constraints and non-monotone
settings ([Calinescu et al., 2011](https://epubs.siam.org/doi/10.1137/080733991);
[Buchbinder et al., 2014](https://epubs.siam.org/doi/10.1137/1.9781611973402.106)).

Adaptive submodularity extends this idea to policies that observe uncertain
outcomes after each action: if the objective is adaptively submodular, adaptive
greedy is competitive with the optimal policy
([Golovin and Krause, 2011](https://authors.library.caltech.edu/28643/)).

### Thesis use

This is the most directly useful borrowed theory.

Additivity, submodularity, and complementarity give a clean language for
rankers versus search:

- Additive `G`: `G(S) approx A(S)`. Marginal ranking should work.
- Submodular `G`: marginal returns diminish. Greedy can be justified under
  the right assumptions, but not necessarily by static per-step scores.
- Complementary / supermodular interactions: some corrections are useful only
  jointly. Greedy/separable rankers can fail.
- Non-submodular, non-monotone `G`: search or richer surrogates may be needed.

The current `xi_{t,t'}` terms are exactly a diagnostic for pairwise
departures from additivity:

```text
xi_{t,t'} > 0: complementarity
xi_{t,t'} < 0: redundancy / diminishing returns
```

### Experiment directions

For candidate schedules `S` in a no-leakage pool `C_B`, measure:

- additive residual: `G(S) - A(S)`;
- pairwise residual: `G(S) - Q(S)`;
- pairwise sign / magnitude `xi_{t,t'}`;
- empirical diminishing returns:

```text
G(S union {t}) - G(S) >=? G(S union {t,u}) - G(S union {u})
```

These diagnostics can classify the schedule objective as near-additive,
submodular-like, complementary, or higher-order.

### Connection to active theorem stack

- Theorem A is the additive/marginal case.
- Theorem B/B' is the pairwise surrogate case.
- Diagnostic Framework C is the regime taxonomy.

## 4. SMC, Feynman-Kac, and Rejuvenation

### Core idea

Sequential Monte Carlo approximates a sequence of distributions with weighted
particles that are propagated and resampled over time. Del Moral's
Feynman-Kac particle-system theory gives the broad mathematical foundation for
interacting particle approximations
([Del Moral, 2004](https://link.springer.com/book/10.1007/978-1-4684-9393-1)).
Del Moral, Doucet, and Jasra describe SMC samplers that approximate a sequence
of distributions with weighted random samples and can make MCMC algorithms
interact for optimization and Bayesian estimation
([Del Moral et al., 2006](https://academic.oup.com/jrsssb/article/68/3/411/7110641)).
Particle MCMC combines SMC and MCMC to build high-dimensional proposals
([Andrieu et al., 2010](https://academic.oup.com/jrsssb/article/72/3/269/7076437)).

### Thesis analogy

SMC has a familiar pattern:

```text
propagate particles
monitor degeneracy
resample / rejuvenate when needed
```

For masked diffusion correctors:

```text
propagate masked trajectory
monitor uncertainty / error accumulation
apply corrector when trajectory quality appears degraded
```

The analogy is strongest if we view a ProSeCo-style corrector as a
rejuvenation kernel: it revises already generated tokens to reduce accumulated
errors.

### What to borrow

SMC suggests that correction timing should be signal-driven by degeneracy
diagnostics, not necessarily by fixed time:

- effective sample size in SMC -> entropy / margin / quality mass / revisable
  set size in MDM;
- particle impoverishment -> early wrong commitment / low-quality committed
  tokens;
- rejuvenation kernel -> informed corrector.

### Limitation

The current thesis pipeline does not maintain weighted particle populations.
Therefore SMC theory should be used as motivation and language, not as a direct
proof engine.

## 5. Particle Gibbs and Ancestor Sampling

### Core idea

Particle Gibbs with ancestor sampling combines SMC and MCMC to update
high-dimensional, highly correlated latent trajectories. The ancestor-sampling
step improves mixing even with relatively few particles
([Lindsten et al., 2014](https://jmlr.org/papers/v15/lindsten14a.html)).

### Thesis analogy

Masked diffusion generation can lock in early choices. Correctors revise parts
of the token trajectory after additional context appears. This resembles
trajectory rejuvenation:

```text
Particle Gibbs: refresh latent trajectory conditional on observations.
ProSeCo-style correction: refresh generated tokens conditional on current
masked context.
```

### Thesis role

Particle Gibbs is useful as a conceptual cousin:

- it supports the idea that trajectory-level correlation and path dependence
  are central;
- it gives language for why local independent updates may mix badly;
- it motivates correction as a trajectory-revision move.

It should not be a main theorem source unless the thesis explicitly builds a
particle approximation, which is not on the critical path.

## 6. Kalman Filtering, Sensor Scheduling, and Active Sensing

### Core idea

Kalman filtering is the classical theory of recursive state estimation.
Kalman's original paper derives recursive updates and a covariance equation
for optimal linear filtering under Gaussian assumptions
([Kalman, 1960](https://cir.nii.ac.jp/crid/1360855570047666048)).

Sensor scheduling asks when or which sensors to use under a resource
constraint. Recent control literature connects sensor selection/scheduling for
Kalman filtering with submodularity and greedy algorithms. For example,
sensor scheduling in linear dynamical systems studies choosing measurements
under energy constraints to minimize estimation error, with submodularity
results in some cases
([Automatica 2015](https://www.sciencedirect.com/science/article/pii/S0005109815003489)).
Other work shows both positive submodularity results and limitations for
Kalman sensor selection
([Tzoumas et al., 2016](https://www.researchgate.net/publication/282266868_Sensor_Placement_for_Optimal_Kalman_Filtering_Fundamental_Limits_Submodularity_and_Algorithms);
[Summers et al., 2016](https://www.sciencedirect.com/science/article/pii/S0005109816305337)).

### Thesis analogy

Kalman/sensor scheduling says:

```text
spend scarce measurement resources when they most reduce state uncertainty.
```

Corrector scheduling says:

```text
spend scarce correction resources when they most improve final sample quality.
```

This motivates uncertainty-triggered online policies:

- high entropy;
- low margin;
- large revisable set;
- phase-specific uncertainty;
- sharp uncertainty changes.

### Limitation

Kalman theory is linear-Gaussian. Masked diffusion over tokens is discrete,
nonlinear, and model-learned. Therefore the analogy is useful for feature
design and experimental framing, but not for directly proving optimal timing.

## 7. Restless Bandits, Gittins Indices, and Whittle Indices

### Core idea

Bandit-index theory studies dynamic allocation of limited effort across
competing projects. Gittins indices solve classical discounted multi-armed
bandits under specific assumptions
([Gittins and Jones, 1979](https://academic.oup.com/biomet/article-pdf/66/3/561/632080/66-3-561.pdf);
[Whittle, 1980](https://academic.oup.com/jrsssb/article/42/2/143/7027598)).
Restless bandits generalize the setting: passive arms also evolve. Whittle's
index is a Lagrangian-relaxation heuristic for such problems; modern reviews
summarize conditions and applications
([Nino-Mora, 2023](https://www.mdpi.com/2227-7390/11/7/1639)).

### Thesis analogy

Corrector scheduling can be viewed as an index policy:

```text
I_t = value of applying correction at current time/state
correct if I_t is high enough relative to budget pressure
```

For token-level correction, each token or region could be an arm. For this
thesis, the arms are more naturally time opportunities along one trajectory.

### Thesis role

Bandit-index theory is useful for the online extension:

- derive a cheap per-step priority index;
- include remaining budget and remaining horizon;
- make the controller deployable: one score per step, no rollout search.

### Limitation

Index optimality requires strong separability/indexability assumptions. Those
assumptions may fail precisely because the current evidence suggests schedule
interactions. Therefore bandit indices are best treated as a cheap policy
class to test, not as guaranteed optimal.

## 8. Value of Information and Rational Metareasoning

### Core idea

Value-of-information theory quantifies whether information is worth its cost.
Howard's information-value theory and later decision-analysis work formalize
the benefit of information through downstream decision improvement
([Howard, 1966](https://colab.ws/articles/10.1109/TSSC.1966.300074)).
Rational metareasoning applies the same idea to computation: a computation is
worth doing if its expected improvement in external action exceeds its cost
([Russell and Wefald, 1991](https://www.sciencedirect.com/science/article/pii/000437029190015C)).

### Thesis analogy

A corrector call is an expensive computation. It should be spent when its
expected value exceeds the opportunity cost of saving it:

```text
correct now if
  E[final quality improvement from correction now]
  >
  E[value of saving the budget for future steps].
```

This is a very clean language for online corrector scheduling because it
separates:

- benefit: expected improvement in final `F`;
- cost: one unit of corrector budget / extra NFE;
- opportunity cost: fewer future corrections.

### Cheap implementation direction

Train a small model for expected value of correction:

```text
I_t = E[G(S union {t}) - G(S) | z_t, b, T-t]
```

where `z_t` contains cheap features already available during generation. The
policy applies a budget-aware threshold to `I_t`.

## 9. MCMC Mixing, Spectral Gaps, and Corrector Kernels

### Core idea

If the corrector is viewed as a Markov kernel over currently revisable token
states, Markov chain theory offers concepts such as detailed balance, spectral
gap, contraction, and mixing time. These tools are already adjacent to the
project's proof worklog.

### Thesis use

This theory can answer questions about the corrector kernel itself:

- does a correction step move the local token distribution toward a better
  conditional?
- when is a corrector action likely to be ineffective because the revisable
  set is empty or nearly empty?
- how should multiple consecutive corrector applications behave?

### Limitation

This is more about **how** the corrector kernel works than **when** to spend
corrector budget. It can support side lemmas, but the main timing framework
should remain schedule-selection / budgeted-control.

## 10. Practical Online Scheduler Design

The online policy should be cheap:

```text
runtime cost = base generation + B corrector calls + tiny scoring overhead
```

It should not evaluate candidate schedules with extra model rollouts.

### Candidate online policies

#### 10.1 Budgeted threshold policy

Score the current state with a scalar uncertainty signal:

```text
I_t = f(z_t)
```

Correct if `I_t` exceeds a threshold depending on remaining budget and
remaining time:

```text
correct if I_t >= tau[b_remaining, phase(t)].
```

Pros:

- cheapest;
- easy to explain;
- directly linked to Kalman/SMC uncertainty-trigger logic.

Cons:

- may collapse back into the separable ranker class;
- ignores selected schedule history.

#### 10.2 Learned value-of-correction index

Train a small supervised model offline:

```text
I_t = f(z_t, b_remaining, T-t)
```

where labels estimate the gain from applying a correction at `t`. Possible
models:

- logistic regression;
- ridge regression;
- shallow MLP;
- gradient-boosted trees.

Features should be cheap and already available:

- normalized time `t/T`;
- remaining budget `b`;
- remaining steps `T-t`;
- entropy over revisable positions;
- inverse margin;
- quality mass / PRISM-style score if available;
- revisable-set size `|R_t|`;
- phase bucket;
- change in uncertainty since the previous step.

Pros:

- deployable;
- budget-aware;
- testable on held-out seeds.

Cons:

- still mostly marginal unless history features are added;
- can overfit without strict seed splits.

#### 10.3 Pairwise-aware online index

If Phase 1 validates pairwise structure, use:

```text
score(t | S_selected) =
  Delta_hat(z_t)
  + sum_{s in S_selected} xi_hat(phase(s), phase(t), distance(s,t), z_s, z_t).
```

The pairwise term should be low-dimensional, e.g. phase-pair bins, not a full
`T x T` learned table unless there is enough data.

Pros:

- directly connected to `Q(S)`;
- can express redundancy/complementarity with already selected corrections.

Cons:

- only justified after schedule-level validation of `Q(S)`;
- more fragile statistically.

#### 10.4 Rollout-free receding-horizon approximation

Use the learned index to simulate future score distributions cheaply, not by
running the MDM. For example, use empirical phase histograms to reserve budget:

```text
spend now if I_t is above the expected top-b threshold among remaining phases.
```

Pros:

- approximates dynamic programming without rollouts;
- remains cheap.

Cons:

- needs calibration on held-out trajectories.

## 11. How This Maps to the Current Experimental Gates

### Gate 2d: K=30 critical replication

Purpose:

```text
reconfirm that the current code path reproduces the baseline:
MC-oracle headroom exists; tested separable rankers do not recover it;
search procedures recover substantial headroom.
```

This does not prove interactions. It reopens the evidence base needed to test
interactions.

### Gate 3a: sparse pair diagnostics

Purpose:

```text
measure xi_{t,t'} and detect whether interactions are non-negligible or
structured by phase/distance/signal.
```

This can support an interaction hypothesis but does not by itself validate
pairwise scheduling.

### Gate 3b: schedule-level validation

Purpose:

```text
test whether Q(S) predicts G(S) better than A(S) on a no-leakage candidate pool.
```

Decision quantities:

```text
eta_{B,C} = error of A(S)
zeta_{B,C} = error of Q(S)
R_B = Spearman(A(S), G(S))
P_B = Spearman(Q(S), G(S))
```

Proceed toward pairwise scheduling only if:

```text
zeta_{B,C} < eta_{B,C}
and/or
P_B > R_B
```

with uncertainty accounted for.

### Gate 4: pairwise or online scheduler

Purpose:

```text
convert diagnostics into a policy and test held-out performance.
```

Deployable success requires Level 3:

```text
feature-conditioned scheduler beats tested separable rankers on held-out seeds
without true-G leakage or extra rollout evaluation.
```

## 12. Recommended Thesis Positioning

The most defensible positioning is:

> Fixed-budget corrector timing is a finite-horizon budgeted schedule-selection
> problem. Classical dynamic programming gives the normative optimum but is
> intractable for masked-token trajectories. We therefore study tractable
> set-function surrogates: additive marginal scheduling, pairwise interaction
> scheduling, and search-based scheduling. SMC/rejuvenation and Kalman/sensor
> scheduling motivate uncertainty-triggered online rules; submodular and
> adaptive-submodular theory clarifies when greedy timing could be justified;
> value-of-information theory gives the budget-aware online decision rule.

This supports the current theorem stack:

- Theorem A: if additivity and calibration hold, marginal timing is justified.
- Empirical Ranker-Class Limitation: tested separable rankers fail on
  ProSeCo-OWT, subject to K=30 replication.
- Theorem B/B': if `Q` approximates `G` better than `A`, interaction-aware
  selection is justified on a fixed candidate pool.
- Diagnostic Framework C: classify the observed model/corrector/metric triple
  into no-op, marginal, interaction-driven, higher-order/search-driven, or
  online-decision regimes.

## 13. What Not to Overclaim

- Do not claim Kalman, SMC, Particle Gibbs, or bandit theory directly solves
  MDM corrector timing.
- Do not claim `Q(S)` is useful until schedule-level validation shows it
  predicts `G(S)` better than `A(S)`.
- Do not treat MC-oracle as the exhaustive optimum; it is a best-of-`N`
  random-schedule pool oracle.
- Do not treat K=3 smoke outputs as thesis evidence.
- Do not build an online controller before K=30 replication and interaction
  diagnostics unless it is explicitly appendix/prototype work.

## 14. Source Map

Diffusion / masked diffusion:

- Song et al., "Score-Based Generative Modeling through Stochastic
  Differential Equations" ([arXiv:2011.13456](https://arxiv.org/abs/2011.13456)).
- Austin et al., "Structured Denoising Diffusion Models in Discrete
  State-Spaces" ([arXiv:2107.03006](https://arxiv.org/abs/2107.03006)).
- Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the
  Data Distribution" ([arXiv:2310.16834](https://arxiv.org/abs/2310.16834)).
- Sahoo et al., "Simple and Effective Masked Diffusion Language Models"
  ([arXiv:2406.07524](https://arxiv.org/abs/2406.07524)).
- Schiff et al., "Learn from Your Mistakes: Self-Correcting Masked Diffusion
  Models" ([arXiv:2602.11590](https://arxiv.org/abs/2602.11590)).
- Kim et al., "Fine-Tuning Masked Diffusion for Provable Self-Correction"
  ([arXiv:2510.01384](https://arxiv.org/abs/2510.01384)).
- Luxembourg et al., "Plan for Speed: Dilated Scheduling for Masked Diffusion
  Language Models" ([arXiv:2506.19037](https://arxiv.org/abs/2506.19037)).

Adaptive diffusion schedules:

- Chen et al., "Adaptive Time-Stepping Schedules for Diffusion Models"
  ([PMLR UAI 2024](https://proceedings.mlr.press/v244/chen24c.html)).
- Sabour et al., "Align Your Steps: Optimizing Sampling Schedules in Diffusion
  Models" ([NVIDIA / ICML 2024](https://research.nvidia.com/index.php/publication/2024-07_align-your-steps-optimizing-sampling-schedules-diffusion-models)).
- Zhang et al., "AdaDiff: Adaptive Step Selection for Fast Diffusion Models"
  ([arXiv:2311.14768](https://arxiv.org/abs/2311.14768)).
- Pei et al., "Optimal Stepsize for Diffusion Sampling"
  ([arXiv:2503.21774](https://arxiv.org/abs/2503.21774)).

Optimization / control:

- Peskir and Shiryaev, "Optimal Stopping and Free-Boundary Problems"
  ([Springer](https://link.springer.com/book/10.1007/978-3-7643-7390-0)).
- White, "A survey of solution techniques for the partially observed Markov
  decision process" ([Springer](https://link.springer.com/article/10.1007/BF02204836)).
- Nemhauser, Wolsey, and Fisher, "An Analysis of Approximations for Maximizing
  Submodular Set Functions" ([ResearchGate mirror](https://www.researchgate.net/publication/242914003_An_Analysis_of_Approximations_for_Maximizing_Submodular_Set_Functions-I)).
- Calinescu et al., "Maximizing a Monotone Submodular Function Subject to a
  Matroid Constraint" ([SIAM](https://epubs.siam.org/doi/10.1137/080733991)).
- Buchbinder et al., "Submodular Maximization with Cardinality Constraints"
  ([SIAM](https://epubs.siam.org/doi/10.1137/1.9781611973402.106)).
- Golovin and Krause, "Adaptive Submodularity: Theory and Applications in
  Active Learning and Stochastic Optimization"
  ([CaltechAUTHORS](https://authors.library.caltech.edu/28643/)).

Sequential Monte Carlo / trajectory rejuvenation:

- Del Moral, "Feynman-Kac Formulae"
  ([Springer](https://link.springer.com/book/10.1007/978-1-4684-9393-1)).
- Del Moral, Doucet, and Jasra, "Sequential Monte Carlo Samplers"
  ([JRSSB](https://academic.oup.com/jrsssb/article/68/3/411/7110641)).
- Andrieu, Doucet, and Holenstein, "Particle Markov Chain Monte Carlo Methods"
  ([JRSSB](https://academic.oup.com/jrsssb/article/72/3/269/7076437)).
- Lindsten, Jordan, and Schoen, "Particle Gibbs with Ancestor Sampling"
  ([JMLR](https://jmlr.org/papers/v15/lindsten14a.html)).

Filtering / active sensing:

- Kalman, "A New Approach to Linear Filtering and Prediction Problems"
  ([CiNii / DOI](https://cir.nii.ac.jp/crid/1360855570047666048)).
- "Submodularity and greedy algorithms in sensor scheduling for linear
  dynamical systems" ([Automatica](https://www.sciencedirect.com/science/article/pii/S0005109815003489)).
- Tzoumas, Jadbabaie, and Pappas, "Sensor Placement for Optimal Kalman
  Filtering" ([ResearchGate mirror](https://www.researchgate.net/publication/282266868_Sensor_Placement_for_Optimal_Kalman_Filtering_Fundamental_Limits_Submodularity_and_Algorithms)).
- "Sensor selection for Kalman filtering of linear dynamical systems"
  ([Automatica](https://www.sciencedirect.com/science/article/pii/S0005109816305337)).

Bandits / indices / value of computation:

- Gittins and Jones, "A dynamic allocation index for the discounted
  multiarmed bandit problem" ([Biometrika](https://academic.oup.com/biomet/article-pdf/66/3/561/632080/66-3-561.pdf)).
- Whittle, "Multi-Armed Bandits and the Gittins Index"
  ([JRSSB](https://academic.oup.com/jrsssb/article/42/2/143/7027598)).
- Nino-Mora, "Markovian Restless Bandits and Index Policies: A Review"
  ([Mathematics](https://www.mdpi.com/2227-7390/11/7/1639)).
- Howard, "Information Value Theory"
  ([IEEE metadata](https://colab.ws/articles/10.1109/TSSC.1966.300074)).
- Russell and Wefald, "Principles of metareasoning"
  ([Artificial Intelligence](https://www.sciencedirect.com/science/article/pii/000437029190015C)).

## 15. Methodology Note

This note was produced by searching for primary papers and publisher pages in
five clusters:

1. diffusion / masked diffusion / correctors;
2. adaptive diffusion sampling schedules;
3. finite-horizon control, optimal stopping, POMDPs;
4. submodular/adaptive-submodular optimization;
5. SMC, Particle Gibbs, Kalman filtering, bandits, and value of information.

The synthesis labels direct facts from sources separately from thesis-specific
inferences. Any claim about ProSeCo-OWT empirical behavior should be checked
against the active result docs and K=30 replication, not inferred from this
background note.
