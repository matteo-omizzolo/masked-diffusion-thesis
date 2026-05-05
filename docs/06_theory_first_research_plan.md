> **ACTIVE PLANNING DOCUMENT — theory-first reassessment.**
> This is not final thesis status. It defines the next research phase:
> theory scaffold → Phase 0 reproducibility → interaction/online timing experiments.
> Feasibility of the full programme is under assessment.
> Current thesis baseline: ProSeCo-OWT shows ranker failure and search success (see `START_HERE.md`).
>
> **Status (2026-05):** Theorem stack formalized. Formal definitions, theorems, proofs,
> and theory-to-experiment map live in `research/candidate_theorems.md` §0–§7.
> §3 and §4 of this document are now compact pointers to that file; §6 is the
> experiment plan; §7–§9 cover PRISM, what-not-to-do, and timeline.

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

The full formal setup — trajectory and schedules, marginal gain Δ_t, additive
surrogate A(S), pairwise interaction ξ_{t,t'}, pairwise surrogate Q(S), separable
rankers, online state z_t, randomness conventions — is in
`research/candidate_theorems.md` §0.

This section in the plan is intentionally a pointer; duplicating the formalism
in two places creates drift.

---

## 4. Theory package (theorem stack)

Formal statements, assumptions, proofs, and falsifiers live in
`research/candidate_theorems.md`. Summary of the stack:

| § | Object | Status | Role |
|---|---|---|---|
| §1 | Theorem A — marginal proxy regret: G(S_B^*) − G(Ŝ_B) ≤ 2Bε + 2η_B | Proved | Baseline; tested by (A2)/(A3) diagnostics |
| §1 | Refinements A′ (variance form), A″ (rank form) | Proved | Empirically anchored variants of Theorem A |
| §1 | Negative-Result Corollary (separable-ψ envelope) | Proved | Documents ranker class failure on ProSeCo-OWT |
| §2 | Theorem B — exact pairwise regret: G(S_B^*) − G(Ŝ) ≤ 2ζ_B + ω_B | Proved | **Central new theorem** |
| §2 | Theorem B — estimated pairwise regret: G(S_B^*) − G(Ŝ_Q̂) ≤ 2ζ_B + 2α_B + ω_B | Proved (constant 2 derived; not 4) | Operational form for held-out evaluation |
| §3 | Proposition C — regime diagnostics (U_B, R_B, I_B, P_B, C_B) | Definition + classification protocol | Regime taxonomy / framework |
| §4 | Theorem D — online controller: V_1 − 𝔼[F | π̂] ≤ 2Tδ | Proof sketch (standard ADP) | Optional / appendix unless Phase 4 promotes |
| §5 | Proposition E — burn-in exclusion via L_F-Lipschitz F | Proof sketch | Optional / side lemma |

The thesis backbone is **A → B → C**. D and E are appendix unless empirical
results promote them.

---

## 5. Theory-to-experiment mapping

Before coding, create a table like this in the active plan.

| Theory item | Assumption / prediction | Measured quantity | Experiment | Possible outcome |
|---|---|---|---|---|
| Theorem A (A2) | |G − A| ≤ η_B | η_B from Phase 0/2b residuals | Theorem A non-vacuous | Move to Theorem B |
| Theorem A (A3) | |Δ − ψ| ≤ ε; rankers ≈ A top-B | ε, ε_R, ρ(A,G) Phase 0 | Marginal regime II | Negative-Result Corollary; → B |
| Theorem A util. | 2Bε + 2η_B < ranker headroom | plug-in vs measured headroom | Theorem A operative | Bound is structural, not predictive |
| Theorem B (B2) | ζ_B < η_B; P_B > R_B | Phase 1 sparse pairwise (ζ_B, ρ(Q,G)) | Interaction regime III | Higher-order / chaotic (regime IV) |
| Theorem B (B3) | |Q − Q̂| ≤ α_B; held-out Q̂ ≈ Q | Phase 1/2 train/test split | Pairwise scheduler buildable | Surrogate undersampled / optimizer bound |
| Theorem B util. | G(Ŝ_Q̂) > G(rankers) | Phase 2 held-out evaluation | **Theorem B is central result** | CD-G/BS-AG remain comparison only |
| Proposition C | Diagnostics stable at K=30 | U_B, R_B, I_B, P_B, C_B with BCa CI | Diagnostic framework valid | Single-backbone case study |
| Theorem D | Compact z_t admits small ‖V−V̂‖_∞ | Phase 4 (only if reached) | Online controller in main | Stays appendix (Protocol C generalizes) |
| Proposition E | L_F·|R_t|/D bounds Δ_t at burn-in | Phase 0 audit of R_t and L_F | Exclusion lemma in main | Side remark only |

Detailed table (with exact theorem references): `research/candidate_theorems.md` §7.

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

Assume the thesis is due in September. The plan is sequential and gated; if any
gate fails, do **not** burn time on later gates — cut online controller (Theorem D)
and regime-map (Phase 3) before cutting interaction diagnostics (Phase 1).

## May (current)

- Theorem stack formalized in `research/candidate_theorems.md` (DONE 2026-05).
- Phase 0 reproducibility smoke (K=3 on ProSeCo-OWT). **Gate.**
- If smoke passes, begin sparse pairwise diagnostics design (no full HPC yet).

## June

- Phase 1 sparse pairwise ξ_{t,t'} diagnostics on ProSeCo-OWT.
- Compute ζ_B, ρ(Q,G), I_B; classify regime.
- If P_B > R_B and ζ_B < η_B: build pairwise surrogate scheduler (start of Phase 2).
- Otherwise: classify ProSeCo-OWT as regime IV (chaotic) and document.

## July

- Phase 2 held-out evaluation of pairwise scheduler.
- Optional: one secondary-model regime probe (Phase 3) — only if Phase 1/2 leaves
  budget. Default is to **skip** this and consolidate.
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

### Stop conditions

- If Phase 0 smoke does not reproduce baseline qualitatively → **stop**, debug,
  do not start Phase 1.
- If Phase 1 shows ζ_B ≥ η_B (no pairwise improvement) → **stop** Phase 2,
  document as regime IV, write thesis as case study.
- If Phase 2 held-out scheduler does not beat rankers → **stop**, document
  honest negative, focus thesis on Theorem A + Negative-Result Corollary
  + diagnostic framework.
- Theorem D (online controller, Phase 4) is the **first** thing to cut if time
  is tight. It is appendix-only by default.

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

# 11. Status and next concrete tasks

**Status — 2026-05.** The theorem stack is formalized.

Done:
- Formal problem setup (`research/candidate_theorems.md` §0).
- Theorem A statement and proof (§1).
- Theorem B exact-Q form, statement and proof (§2.1).
- Theorem B estimated-Q̂ form, statement and proof with corrected constant 2α_B (§2.2).
- Proposition C regime taxonomy and diagnostics (§3).
- Theorem D statement and proof sketch with honest 2Tδ constant (§4).
- Proposition E burn-in exclusion sketch (§5).
- Theory-to-experiment map (§7).

### Next concrete tasks (sequential, gated)

1. **Phase 0 smoke** (K=3, ProSeCo-OWT). Confirm baseline reproducibility.
   This is the gate before any Phase 1 design work.

2. **Pre-flight tests for the existing scripts**: deterministic base generation;
   empty schedule equals base; single-correction equals Protocol A; budget
   accounting; F scoring consistency; CRN.

3. **Phase 1 sparse pairwise ξ_{t,t'} estimator** — design only. Do not run
   on HPC until Phase 0 smoke passes.

4. **Hold the cut decisions**: if any gate fails, cut Theorem D / Phase 4 first,
   then Phase 3 secondary backbone, then PRISM, *before* cutting Phase 1.

---

# 12. Final instruction to Claude Code

Your priority is not to produce more files or more experiments. Your priority is to build a mathematically coherent thesis:

1. state a theory;
2. derive predictions;
3. design experiments that can falsify those predictions;
4. run only the experiments that are necessary;
5. interpret outcomes honestly.

Keep the repo simple. Keep active docs compact. Do not let the project become messy again.
