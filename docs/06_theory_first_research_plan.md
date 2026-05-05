> **ACTIVE PLANNING DOCUMENT — theory-first reassessment.**
> This is not final thesis status. It defines the next research phase:
> theory scaffold → Phase 0 reproducibility → interaction/online timing experiments.
> Feasibility of the full programme is under assessment.
> Current thesis baseline: ProSeCo-OWT shows ranker failure and search success (see `START_HERE.md`).

# Theory-First Research Plan: When to Apply Informed Correction in Masked Diffusion LMs

**Purpose.** This document is a detailed plan for the next research phase of the masked-diffusion thesis. It is written for Claude Code. The goal is not to restart the thesis, but to re-aim it toward a cleaner, more mathematically rigorous contribution around **when informed correction should be applied**.

**Core principle.** The theory must come first. Experiments should not be exploratory fishing runs. Each experiment must verify, falsify, or calibrate a clearly stated theorem, assumption, prediction, or regime diagnostic.

---

## 0. Context and motivation

The current repo already contains a cleaned documentation structure and a first complete experimental/theoretical story:

- ProSeCo-OWT shows nonzero headroom over uniform corrector placement.
- Greedy/separable per-step rankers fail by budget B = 8.
- Search procedures over schedules, CD-G and BS-AG, recover substantial oracle headroom.
- The current Theorem A gives a proxy-regret bound for marginal top-B scheduling under additivity and proxy calibration.
- The original L-infinity version of Theorem A is empirically too pessimistic/vacuous.
- Protocol C, a simple bucketed adaptive controller, was an honest negative.
- The current story is coherent but narrow: "rankers fail; search works."

We now want a stronger thesis direction:

> **When should informed correctors be applied during masked diffusion generation?**
>
> The thesis should characterize when corrector timing is reducible to marginal signal ranking, when it is interaction-driven, and whether a budget-aware online controller can approximate the offline schedule-selection problem.

The key shift is from a purely empirical conclusion to a theory-first program:

1. Define formal policy classes for corrector timing.
2. Prove what each class can and cannot capture.
3. Derive diagnostic quantities that distinguish regimes.
4. Run experiments specifically to test those diagnostics.
5. Only then develop or evaluate new schedulers.

---

## 1. Non-negotiable research principles

Claude Code should follow these throughout.

### 1.1 Theory before experiments

Before launching any new HPC experiment, write a short theory/specification document that answers:

- What theorem, proposition, or hypothesis motivates this experiment?
- What assumptions are being tested?
- What quantities must be measured?
- What result would support the theory?
- What result would falsify or weaken the theory?
- What decision gate follows from the result?

### 1.2 No unstructured exploratory runs

Do not run new experiments merely because they might be interesting. Every run must be tied to a named research question and a decision gate.

### 1.3 Reproducibility first

Before extending the project, reproduce the existing ProSeCo-OWT results enough to verify that:

- G(S), A(S), Delta_t, F, and the NFE budget are consistently defined.
- Pairing/common random numbers are implemented correctly.
- Existing Phase 2b and Phase 3a conclusions are qualitatively reproducible.

### 1.4 Minimalism

Do not create many new status documents. Prefer one compact new planning document and one theory document. Keep the repo easy to navigate.

Recommended new active files, if needed:

```text
research/theory_corrector_timing.md
experiments/ or docs/06_research_plan.md only if the repo already has such a convention
```

If creating a new doc would duplicate existing docs, update the existing doc instead.

### 1.5 Honest negative results are allowed

A failed scheduler can still be a thesis contribution if it falsifies a clean theory or classifies a regime. Do not hide negative outcomes.

---

## 2. Proposed new thesis framing

### 2.1 Working title

**When to Correct? Interaction-Aware and Budget-Aware Timing of Informed Correctors in Masked Diffusion Language Models**

### 2.2 Main research question

> Given a masked diffusion language model with an informed corrector and a fixed extra NFE budget, when should correction steps be applied along the denoising trajectory?

### 2.3 Refined research questions

**RQ1 — Usefulness.** Is there measurable headroom over uniform corrector placement?

**RQ2 — Marginality.** Can the headroom be explained by per-step marginal gains or aggregate trajectory signals?

**RQ3 — Interaction.** If marginal ranking fails, are pairwise or schedule-level interactions responsible?

**RQ4 — Regime.** Across model/corrector pairs, do we observe different timing regimes: no-op, rankable, interaction-driven, or chaotic?

**RQ5 — Online control.** Can an online budgeted controller using current trajectory state approximate the offline schedule-selection problem?

**RQ6 — Generality.** Does the interaction-driven behavior persist beyond ProSeCo-OWT?

---

## 3. Mathematical framework

This section is the first task. Do not begin new HPC experiments until this framework is written in `research/theory_corrector_timing.md` or an equivalent active theory file.

### 3.1 Objects

Let the predictor generate a trajectory of states:

```text
Z_0, Z_1, ..., Z_T
```

At each step t, a corrector can be applied or skipped. A schedule is:

```text
S subset {1, ..., T}, with |S| <= B
```

where B is the fixed corrector budget.

Let F be a trajectory-level quality functional, e.g. negative GPT-2 NLL:

```text
F(y) = quality of final sample y
```

Let:

```text
G(S) = F(y^S) - F(y^base)
```

be the joint gain from applying correctors at schedule S.

Let:

```text
Delta_t = G({t})
```

be the one-step marginal gain.

Let:

```text
A(S) = sum_{t in S} Delta_t
```

be the additive/marginal surrogate.

Let trajectory signals be:

```text
s_t = (H_t, M_t^{-1}, Q_t, u_t, phase_t, etc.)
```

where H_t is aggregate entropy, M_t^{-1} inverse margin, Q_t quality mass proxy, u_t unmasked fraction, and phase_t = t/T.

---

## 4. Theory package to develop

The contribution should not be one theorem only. It should be a small theorem stack connecting policy classes to experiments.

## 4.1 Theorem A — marginal proxy scheduling baseline

This theorem already exists. Keep it as the baseline.

**Statement.** If:

1. schedules are binary, |S| = B;
2. gains are approximately additive:

```text
|G(S) - A(S)| <= eta_B
```

3. the proxy is calibrated:

```text
|Delta_t - psi(s_t)| <= epsilon
```

then the top-B proxy schedule satisfies:

```text
G(S_B*) - G(S_hat_B) <= 2 B epsilon + 2 eta_B.
```

**Role in the thesis.** Theorem A answers: when should marginal/signal ranking work?

**Experiments that test it.**

- Estimate proxy error epsilon or rank-based epsilon_R.
- Estimate additivity slack eta_B.
- Measure whether top-B signal schedules beat uniform.

**Expected interpretation.** If epsilon and eta_B are large, the theorem predicts that marginal rankers need not work. This matches current ProSeCo-OWT evidence.

---

## 4.2 Theorem B — pairwise surrogate regret

This should become the main new mathematical object if interactions are central.

### Definition

Define a pairwise surrogate:

```text
Q(S) = sum_{t in S} Delta_t + sum_{t < t', t,t' in S} xi_{t,t'}.
```

The pairwise interaction is:

```text
xi_{t,t'} = G({t,t'}) - Delta_t - Delta_{t'}.
```

Assume:

```text
|G(S) - Q(S)| <= zeta_B
```

for all schedules |S| = B.

Let S_Q be the exact maximizer of Q over |S| = B.

Let S_hat be a schedule returned by an optimizer with optimization gap omega_B:

```text
Q(S_Q) - Q(S_hat) <= omega_B.
```

### Theorem B1 — exact pairwise surrogate

Under the above assumptions:

```text
G(S_B*) - G(S_hat) <= 2 zeta_B + omega_B.
```

### Theorem B2 — estimated pairwise surrogate

If we only have an estimated surrogate Q_hat satisfying:

```text
|Q_hat(S) - Q(S)| <= alpha_B
```

for all |S| = B, and S_hat maximizes Q_hat up to optimization gap omega_B, then:

```text
G(S_B*) - G(S_hat) <= 2 zeta_B + 2 alpha_B + omega_B.
```

### Proof sketch

Use the standard comparison chain:

```text
G(S*) <= Q(S*) + zeta_B
       <= Q(S_Q) + zeta_B
       <= Q(S_hat) + omega_B + zeta_B
       <= G(S_hat) + omega_B + 2 zeta_B.
```

For Q_hat, add two surrogate-estimation errors.

### Role in the thesis

This theorem answers: when is interaction-aware scheduling justified?

### Experiments that test it

- Estimate xi_{t,t'}.
- Estimate zeta_B = |G(S) - Q(S)| on held-out schedules.
- Estimate alpha_B for fitted Q_hat.
- Evaluate pairwise-scheduler regret relative to MC oracle / search baselines.

### Falsification

The pairwise model is not sufficient if:

- zeta_B is large;
- Q_hat has poor held-out correlation with G;
- pairwise schedules do not beat marginal rankers or uniform.

---

## 4.3 Proposition C — interaction-driven failure of separable rankers

This should be a clean construction showing why per-step ranking can fail even if the problem has exploitable structure.

### Proposition C1 — separable ranker impossibility under pure complementarity

Construct a schedule-gain function G over T steps such that:

```text
Delta_t = G({t}) = c
```

for all t, so every separable ranker based only on Delta_t is indifferent, but there exists a pair or subset S* with large positive interaction:

```text
G(S*) >> G(S)
```

for most other schedules of the same size.

Then no separable per-step score psi_t can identify S* unless the interaction information is encoded into psi_t externally.

### Role

This proposition gives a rigorous explanation for why marginal scheduling is the wrong policy class when interactions dominate.

### Experimental test

Measure whether real ProSeCo-OWT resembles this construction:

- weak marginal signal-to-gain correlation;
- low concentration of top schedules around marginal-oracle picks;
- high interaction residuals;
- search succeeds while rankers fail.

---

## 4.4 Theorem D — budgeted online controller abstraction

This is the Direction 6 component. It should be theory-first but limited.

### Setup

At each time t, define an observable state:

```text
z_t = phi(Z_t, t, b_t)
```

where b_t is remaining corrector budget.

Examples:

```text
z_t = (phase_t, H_t, M_t^{-1}, Q_t, u_t, b_t)
```

An online policy pi chooses:

```text
a_t in {0,1}
```

with total budget constraint:

```text
sum_t a_t <= B.
```

Let V_t^*(z,b) be the optimal value-to-go under the compressed state abstraction, and let pi_hat be the policy induced by estimated value function V_hat.

### Theorem D1 — value approximation regret

Assume the compressed-state Bellman approximation error is bounded:

```text
|V_t^*(z,b) - V_hat_t(z,b)| <= beta
```

for all reachable states and budgets. Then the greedy policy induced by V_hat has regret bounded by:

```text
G(S_online^*) - G(S_pi_hat) <= C_T beta
```

where C_T is a horizon-dependent constant. If the policy only makes B correction decisions, derive a sharper bound proportional to B if possible.

### Theorem D2 — abstraction error decomposition

Decompose beta into:

```text
beta <= estimation error + state-aliasing error + finite-sample error.
```

The key empirical quantity is **state aliasing**:

> Do trajectories with the same compressed state z_t have similar future correction value?

### Role

This theorem answers: when can "when to correct" be solved online using current trajectory diagnostics?

### Experimental test

- Estimate value/advantage by phase-signal-budget buckets.
- Measure within-bucket variance of correction advantage.
- If within-bucket variance is large, simple online control cannot work.
- If within-bucket variance is small, learn an online controller and test it.

### Important warning

Protocol C already showed that simple bucketed-state conditioning barely improved proxy calibration on OWT. Therefore Theorem D should be used as a falsifiable framework, not assumed to produce a positive result.

---

## 4.5 Regime definitions

Define diagnostic quantities for each model/corrector pair and budget B.

### Usefulness

```text
U_B = G(S_oracle_B) - G(S_uniform_B)
```

or MC-oracle approximation if true oracle is unavailable.

### Rankability

```text
R_B = rho(A(S), G(S))
```

where rho can be Spearman correlation over sampled schedules.

### Interaction strength

```text
I_B = sigma(G(S) - A(S)) / sigma(A(S))
```

or a robust version using median absolute deviation.

### Pairwise sufficiency

```text
P_B = rho(Q_hat(S), G(S))
```

on held-out schedules.

### Search closure

```text
C_B = (G(S_method) - G(S_uniform)) / (G(S_MC_oracle) - G(S_uniform)).
```

### Online sufficiency

```text
O_B = (G(S_online) - G(S_uniform)) / (G(S_MC_oracle) - G(S_uniform)).
```

### Regime classification

| Regime | U_B | R_B | I_B | P_B/O_B | Interpretation |
|---|---:|---:|---:|---:|---|
| No-op | low | irrelevant | low | low | Corrector timing does not matter |
| Rankable | high | high | low | not needed | Marginal scheduling works |
| Interaction-driven | high | low/moderate | high | high P_B | Pairwise/search scheduling works |
| Online-decision | high | moderate | moderate/high | high O_B | State/budget controller works |
| Chaotic | high | low | high | low | Headroom exists but is hard to predict |

The experiments should classify ProSeCo-OWT and any additional model/corrector pair into this table.

---

## 5. Theory-to-experiment mapping

Before coding, create a table like this in the active plan.

| Theory item | Assumption / prediction | Measured quantity | Experiment | Possible outcome |
|---|---|---|---|---|
| Theorem A | Additivity + proxy calibration makes rankers good | epsilon, eta_B, rho(A,G) | Phase 0/1 replication | Supported or falsified |
| Proposition C | Interactions can defeat separable rankers | xi, Jaccard, ranker failure | Phase 1 interaction map | ProSeCo resembles or does not resemble construction |
| Theorem B | Pairwise Q explains G | zeta_B, rho(Q,G), alpha_B | Phase 1/2 pairwise surrogate | Pairwise model works or fails |
| Theorem D | Online state captures future correction value | bucket variance, online regret | Phase 4 controller | Online controller works or fails |
| Regime map | Different correctors induce different regimes | U_B, R_B, I_B, P_B, O_B | Phase 3 model comparison | Generality or limitation |

---

# 6. Experimental plan

## Phase 0 — Reproducibility and definition audit

### Purpose

Verify that previous results are real before extending them.

### Research questions

- RQ0.1: Can we reproduce the qualitative Phase 2b and Phase 3a conclusions?
- RQ0.2: Are G(S), A(S), Delta_t, F, and the NFE budget consistently defined?
- RQ0.3: Are common random numbers / paired seeds correctly implemented?
- RQ0.4: Are model checkpoint paths, tokenizers, reference metric, and corrector settings correct?

### Required checks

1. Read and document the code path for:
   - base generation;
   - branch generation;
   - scheduled generation;
   - corrector invocation;
   - F scoring;
   - random seed control.
2. Add assertions or tests for:
   - same seed gives same base output;
   - schedule S = empty equals base;
   - repeated schedule evaluation is deterministic if expected;
   - applying one corrector at t matches Protocol A branch definition;
   - B budget counts NFEs consistently.
3. Run a smoke test:
   - K = 3 seeds;
   - T = 64;
   - B in {2,4};
   - policies: uniform, mean_delta_oracle, CD-G, BS-AG.
4. Run a critical replication:
   - K = 30 if feasible;
   - T = 64;
   - B in {2,3,4,8};
   - uniform, marginal rankers, MC oracle best-of-100, CD-G, BS-AG.

### Models

- ProSeCo-OWT only.

### Expected outcome

Qualitatively reproduce:

- MC-oracle headroom at B in {2,3,4};
- greedy rankers fail by B = 8;
- CD-G and BS-AG beat uniform.

### Decision gate

- If results reproduce: proceed to Phase 1.
- If not: stop and debug before any new theory or experiments.

---

## Phase 1 — Pairwise interaction diagnostics on ProSeCo-OWT

### Purpose

Determine whether schedule interactions are structured enough to support Theorem B.

### Research questions

- RQ1.1: Is xi_{t,t'} structured or noise?
- RQ1.2: Are interactions local in time, phase-based, or long-range?
- RQ1.3: Are interactions complementary or redundant?
- RQ1.4: Can simple features predict xi_{t,t'}?
- RQ1.5: Does interaction structure explain ranker failure?

### Definitions

Seed-wise interaction:

```text
xi_i(t,t') = G_i({t,t'}) - Delta_i(t) - Delta_i(t').
```

Mean interaction:

```text
bar_xi(t,t') = mean_i xi_i(t,t').
```

### Phase 1a — sparse stratified pairwise map

Run:

- K = 30 paired seeds.
- T = 64.
- P = 256 sampled pairs per seed, stratified by:
  - phase pair: early/early, early/middle, early/late, middle/middle, middle/late, late/late;
  - temporal distance: short, medium, long;
  - marginal-gain buckets if available.

### Phase 1b — dense mean map

Run:

- K = 8 or 10 seeds.
- All T choose 2 = 2016 pairs.
- Purpose: visualization and structural analysis, not final claims.

### Analysis

Produce:

- heatmap of bar_xi(t,t');
- heatmap of P(xi_i(t,t') > 0);
- phase-pair table of mean and std interaction;
- interaction vs |t-t'|;
- interaction vs marginal Delta_t, Delta_t';
- low-rank approximation diagnostics;
- regression/R^2 for predicting xi from simple features.

### Expected outcomes

**Best case:** xi has stable phase or distance structure.

**Medium case:** xi is noisy but coarse phase buckets explain some variance.

**Bad case:** xi has no stable structure.

### Decision gate

- If pairwise structure exists: proceed to Phase 2.
- If no structure exists: emphasize regime diagnostics and consider online/controller only as a falsification study.

---

## Phase 2 — Interaction-aware scheduler

### Purpose

Test whether a pairwise surrogate scheduler can outperform separable rankers without using true G feedback at inference.

### Research questions

- RQ2.1: Can a pairwise surrogate schedule beat uniform and marginal rankers?
- RQ2.2: How much MC-oracle headroom does it recover?
- RQ2.3: Is phase-pair structure enough, or are signal-conditioned interactions needed?
- RQ2.4: Does the method still help at B = 8, where rankers fail?

### Schedulers

#### Baselines

- uniform;
- random average;
- front/middle/back;
- entropy top-B;
- margin top-B;
- quality-mass top-B;
- mean_delta_oracle;
- MC oracle best-of-100;
- CD-G;
- BS-AG.

#### Pairwise scheduler S1 — phase-pair model

Bucket t into K_phase phase bins, e.g. 4 or 8.

```text
Q_hat(S) = sum_{t in S} Delta_hat_t
           + lambda sum_{t<t', t,t' in S} xi_hat_{bucket(t),bucket(t')}.
```

Tune lambda on training seeds only.

#### Pairwise scheduler S2 — distance/phase model

```text
xi_hat(t,t') = a_{bucket(t),bucket(t')} + c * log(1 + |t-t'|).
```

#### Pairwise scheduler S3 — feature regression

Use linear/ridge regression features:

- phase_t, phase_t';
- entropy_t, entropy_t';
- margin_t, margin_t';
- quality_t, quality_t';
- unmasked_fraction_t, unmasked_fraction_t';
- |t-t'|;
- products/differences.

Do not use a large neural network initially.

#### Pairwise scheduler S4 — low-rank model

Fit a low-rank symmetric interaction matrix:

```text
xi_hat(t,t') = v_t^T v_t'.
```

Use only if Phase 1b suggests low-rank structure.

### Optimization over schedules

For each Q_hat, select schedules using:

- greedy add-one;
- beam search width 8 or 16;
- local swap search.

Track optimization gap if possible by comparing optimizers.

### Train/test split

- Train/calibrate on 15 seeds.
- Evaluate on 15 held-out seeds.
- If promising, rerun final K = 30 evaluation.

### Metrics

- paired gain over uniform;
- BCa bootstrap CI;
- closure ratio against MC oracle;
- comparison to mean_delta_oracle;
- rho(Q_hat(S), G(S)) on held-out sampled schedules;
- zeta_B estimate: |G(S) - Q(S)|;
- alpha_B estimate: |Q_hat(S) - Q(S)| if Q is available;
- true-G calls required at inference.

### Expected outcomes

**Best case:** pairwise scheduler beats rankers and recovers 20-50% of oracle headroom without true-G inference calls.

**Medium case:** pairwise surrogate predicts G better than A but does not robustly beat uniform.

**Bad case:** pairwise surrogate fails; result becomes evidence that interactions are not stable/compressible.

### Decision gate

- If successful: pairwise interaction-aware timing becomes the main contribution.
- If failed: retain as negative result; proceed to regime map and online-control diagnostics.

---

## Phase 3 — Regime map across model/corrector pairs

### Purpose

Avoid overclaiming from one backbone. Determine whether different corrector mechanisms induce different timing regimes.

### Research questions

- RQ3.1: Is ProSeCo-OWT uniquely interaction-driven?
- RQ3.2: Do heuristic or remasking-style correctors fall into no-op/rankable regimes?
- RQ3.3: Does the corrector mechanism determine whether timing is useful?
- RQ3.4: Are results stable across quality metrics?

### Candidate model/corrector pairs

#### M1 — ProSeCo-OWT

Primary model. Use for all main results.

#### M2 — ReMDM-conf or ReMDM-loop

Use if infrastructure is stable. This is a strong secondary target because ReMDM is already relevant to inference-time scaling and remasking.

Need to clarify whether the mechanism is best described as corrector scheduling, remasking, or a comparison baseline.

#### M3 — MDLM-conf partial resample

Use as a controlled heuristic corrector. This is less publishable but useful for a mechanism comparison.

#### M4 — ProSeCo-LLaDA-SFT

Only use if metric/protocol issues can be fixed. Existing bounded probe was inconclusive.

#### M5 — PRISM

Optional feasibility branch only. See Section 8.

### Minimal regime protocol for each new model/corrector

Do not run a full K = 30 experiment immediately.

Run first:

- K = 10 seeds;
- T = 64;
- B in {2,4};
- Protocol A-lite: Delta_t and signals;
- Policy-lite: uniform, middle, entropy, mean_delta_oracle, random/MC-20;
- Interaction-lite: P = 128 sampled pairs.

Compute:

- U_B usefulness;
- R_B rankability;
- I_B interaction strength;
- preliminary pairwise sufficiency P_B;
- qualitative timing regime.

### Promotion criteria

Promote to K = 30 only if:

- U_B is nontrivial;
- outputs are not degenerate;
- metric appears meaningful;
- timing differences are not pure noise.

### Expected outcomes

Possible final regime map:

- ProSeCo-OWT: interaction-driven.
- ReMDM-conf: rankable or interaction-driven.
- MDLM-conf: heuristic/noisy or no-op/harmful.
- LLaDA-SFT: inconclusive or no-op under current metric.

---

## Phase 4 — Budgeted online controller

This is the Direction 6 component. It should be theory-driven and limited.

### Purpose

Test whether corrector timing can be approximated as an online finite-horizon budgeted decision problem.

### Research questions

- RQ4.1: Can an online policy using current trajectory state compete with offline schedules?
- RQ4.2: Is ranker failure due to lack of budget/future awareness?
- RQ4.3: Does adding remaining-budget state improve over top-B ranking?
- RQ4.4: Is state aliasing too large for simple online control?

### State

```text
z_t = (phase_t, H_t, M_t^{-1}, Q_t, u_t, b_t)
```

Optional:

- previous correction indicator;
- last correction time;
- phase bucket;
- recent trend in entropy or margin.

### Actions

```text
a_t = 1 correct
 a_t = 0 skip
```

with total budget constraint:

```text
sum_t a_t <= B.
```

### Policy classes

#### Online P1 — budget-aware threshold

Correct if:

```text
score(z_t) > lambda_{phase,budget}
```

#### Online P2 — dynamic programming on buckets

Discretize:

```text
phase x signal_quantile x remaining_budget
```

Estimate value-to-go from training seeds.

#### Online P3 — one-step lookahead

Correct if estimated immediate advantage plus future value exceeds skip value.

### Train/evaluate

- Train on calibration seeds.
- Evaluate on held-out seeds.
- Compare to uniform, marginal rankers, pairwise scheduler, BS-AG, MC oracle.

### Key diagnostic: state aliasing

For each bucketed state z, estimate variance of realized correction advantage.

If within-state variance is high, the state abstraction is too poor and online control should fail.

### Expected outcomes

**Best case:** budget-aware online controller beats marginal rankers.

**Medium case:** online controller helps less than pairwise scheduler but gives an interpretable result.

**Bad case:** online controller fails, confirming that current state summaries are insufficient. Keep as appendix or negative result.

---

# 7. PRISM decision

## 7.1 Is PRISM worth using?

PRISM is relevant because it provides learned per-token quality scores for self-correction/remasking. However, it is not automatically central to this thesis.

### Pros

- Conceptually strong quality-signal baseline.
- Theoretically aligned with quality estimation.
- Could provide a better signal than entropy/margin.

### Cons

- No reliable pretrained weights are assumed available.
- Fine-tuning may consume significant time.
- PRISM is primarily about token quality / which tokens to revise, whereas this thesis is about when to spend correction budget.
- If used only as a separable per-step quality signal, it may remain inside the ranker class already shown to be limited.

## 7.2 Recommended PRISM policy

Do not make PRISM mandatory.

Use PRISM in one of three ways:

### PRISM-Lite: literature-only

Use PRISM to motivate why quality signals are natural but insufficient for temporal scheduling.

### PRISM-Feasibility: one-week implementation check

Allow at most 1-2 weeks to answer:

- Can PRISM code run?
- Are weights available?
- Can a small quality head be trained cheaply?
- Can it produce a per-step quality-mass signal on ProSeCo/MDLM trajectories?

If no: stop.

### PRISM-Signal: optional quality signal

If feasibility succeeds, include PRISM quality mass as one extra signal in Protocol A and ranker baselines.

Do not fine-tune PRISM as a main thesis pillar unless explicitly approved by the supervisor.

---

# 8. What not to do

Do not:

1. Run full K = 30 on a new model before Protocol A-lite shows nonzero useful Delta_t.
2. Run dense pairwise maps before sparse maps show structure.
3. Train neural schedulers before linear/phase-pair models are tested.
4. Spend a month on PRISM without pretrained weights or a very clear feasibility result.
5. Launch LLaDA-SFT Phase 3a without fixing metric/protocol issues.
6. Try to prove a general neural corrector contraction theorem unless the supervisor explicitly requests it.
7. Create many new status documents.
8. Interpret any single-backbone result as universal.

---

# 9. Timeline to September

Assume the thesis is due in September. Use this as a rough plan.

## May

- Finalize theory-first framework.
- Write Theorem B and Theorem D statements/proofs.
- Perform Phase 0 reproducibility audit.
- Run sparse interaction diagnostics.

## June

- Complete dense/sparse interaction maps.
- Build and evaluate pairwise surrogate schedulers.
- Decide whether pairwise scheduling becomes the main contribution.
- Start one secondary-model regime probe.

## July

- Complete secondary-model regime map.
- Run limited online controller if still justified.
- Freeze all experiments by end of July.

## August

- Write thesis chapters.
- Prepare figures/tables.
- Integrate theory and experiments.
- Supervisor review.

## September

- Final revisions.
- Polish LaTeX.
- Submit.

---

# 10. Expected final thesis outcomes

## Outcome A — strongest

Pairwise interactions are structured, pairwise scheduler beats rankers, and at least one second model supports the regime framework.

**Thesis claim:** Corrector timing can be interaction-driven; marginal rankers fail, but interaction-aware scheduling partially recovers oracle headroom.

## Outcome B — still strong

Pairwise interactions explain ProSeCo-OWT but do not generalize cleanly.

**Thesis claim:** ProSeCo-OWT is an interaction-driven corrector-timing regime; generality remains open.

## Outcome C — diagnostic thesis

Pairwise and online schedulers fail, but diagnostics show why.

**Thesis claim:** Corrector timing headroom exists but is not compressible into simple marginal, pairwise, or low-dimensional online policies under the tested setup.

## Outcome D — online success

Budget-aware online controller beats rankers.

**Thesis claim:** Corrector timing can be approximated as a budgeted online decision problem under suitable state abstraction.

Any of A-C is thesis-viable if written honestly. A is the target.

---

# 11. Immediate tasks for Claude Code

Do these in order.

## Task 1 — Create theory document

Create or update:

```text
research/theory_corrector_timing.md
```

Include:

- formal problem setup;
- Theorem A recap;
- Theorem B pairwise surrogate regret;
- Proposition C separable ranker failure construction;
- Theorem D online controller abstraction;
- regime diagnostics;
- theorem-to-experiment mapping table.

Keep it mathematically precise. Mark assumptions clearly.

## Task 2 — Update active docs minimally

Update only if necessary:

- `START_HERE.md`
- `docs/01_research_direction.md`
- `docs/03_theory.md`
- `docs/05_next_steps.md`

Do not create a new maze of docs. Mention that the new research phase is theory-first and exploratory, replacing the previous "no new HPC" planning assumption.

## Task 3 — Write pre-registration for Phase 0 and Phase 1

Create a compact experiment specification, preferably one file:

```text
docs/06_theory_first_experiment_plan.md
```

or update `docs/05_next_steps.md` if adding a file is against repo policy.

Include:

- Phase 0 reproducibility audit;
- Phase 1 interaction diagnostics;
- exact metrics;
- decision gates;
- expected outcomes;
- kill criteria.

## Task 4 — Implement tests before runs

Add or verify tests/assertions for:

- deterministic base generation;
- empty schedule equals base;
- single-correction schedule equals Protocol A branch;
- budget accounting;
- F scoring consistency;
- common random numbers.

## Task 5 — Run Phase 0 smoke

Run K=3 smoke on ProSeCo-OWT.

Only after it passes, prepare K=30 replication.

## Task 6 — Prepare Phase 1 interaction map code

Implement sparse stratified pair sampling and analysis.

Do not run dense all-pair maps until sparse diagnostics justify it.

---

# 12. Final instruction to Claude Code

Your priority is not to produce more files or more experiments. Your priority is to build a mathematically coherent thesis:

1. state a theory;
2. derive predictions;
3. design experiments that can falsify those predictions;
4. run only the experiments that are necessary;
5. interpret outcomes honestly.

Keep the repo simple. Keep active docs compact. Do not let the project become messy again.
