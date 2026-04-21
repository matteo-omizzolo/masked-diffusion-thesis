# MSc Thesis Direction Brief

## Purpose of this brief

This note is meant to be uploaded to Claude Code together with the repository. Its job is to give Claude an accurate, up-to-date framing of the thesis, so the repo can be cleaned, refocused, and turned into a realistic MSc research workspace.

The goal is **not** to keep a vague thesis topic around masked diffusion language models. The goal is to narrow the work to a question that is:

1. genuinely open enough,
2. feasible with public code and checkpoints,
3. defensible as an MSc thesis with both theory and experiments,
4. careful about distinctions that are often conflated in this literature.

---

## Updated thesis direction

### Preferred high-level title

**Signal-adaptive corrector scheduling for masked diffusion language models**

### Realistic core research question

> **For a fixed predictor schedule and fixed corrector NFE budget in masked diffusion language models, can aggregate trajectory signals—entropy, confidence margin, or quality mass—predict the marginal value of a corrective refinement loop well enough to outperform uniform corrector placement?**

This is the main question the repo should now optimize for.

### Why this framing is better than the old one

The older framing focused too narrowly on **entropy-adaptive informed-corrector scheduling**. That was a good starting intuition, but it is too brittle as a thesis target because:

- raw entropy alone may not be the best signal,
- some adjacent papers already use signal-based remasking or token-level quality scores,
- the thesis should not hinge on proving that entropy itself is optimal,
- the strongest open gap is broader: **trajectory-level allocation of correction effort under a fixed budget**.

So the thesis should be framed around **signal-adaptive scheduling**, with entropy as the first candidate signal rather than the only one.

---

## Precise scope boundaries

Claude should preserve these distinctions everywhere in the documentation.

### 1. Informed correctors
Inference-time corrective updates applied to already committed or current states, using a signal such as logits, confidence, entropy, score, or quality estimates.

### 2. Remasking methods
Methods that revisit tokens by remasking and regenerating them within the denoising process. These are adjacent, but not identical to informed correctors.

### 3. Predictor / unmasking schedule optimization
Work on the predictor path: when and how many tokens to unmask, what order to unmask, how to set block size, whether to adapt the number of revealed tokens, and similar decisions.

### 4. Token-selection policies
Rules about **which** token(s) to update, correct, unmask, or remask.

### 5. Corrector scheduling
Rules about **when** to spend corrective refinement steps across the denoising trajectory, and how to allocate a fixed global corrector budget over time.

The thesis should focus on **(5)**, while possibly using ideas from (1)–(4) as signals, proxies, or baselines.

---

## Open-question verdict

### Bottom line
The question is **open in the precise formulation above**, but **not open in a vague, blanket sense**.

### What is already addressed in nearby work

- Some papers already study **self-correction** or **corrective loops** in masked diffusion language models.
- Some papers already use **quality or confidence signals** to decide what to revisit.
- Some papers already optimize **predictor schedules**, including entropy-aware unmasking.
- Some papers already study non-uniform scheduling in nearby theory settings such as Gibbs sampling.

### What remains open

What still appears open is the following narrower problem:

> Given a fixed predictor schedule and a fixed corrector compute budget, how should corrective refinement steps be allocated across diffusion time, and can trajectory-level signals justify a better-than-uniform allocation?

That gap appears real because the closest existing papers provide either:

- empirical heuristics,
- token-level correction/remasking signals,
- predictor-side schedule theory,
- or corrector-kernel design,

but not a principled trajectory-level corrector allocator for masked diffusion language models.

### Safe claim to make

Use this wording in docs and notes:

> The literature contains strong adjacent work on self-correction, remasking, token-quality scoring, and predictor/unmasking schedule design, but there does not yet appear to be a principled theory-plus-experiments study of **trajectory-level fixed-budget corrector allocation** in masked diffusion language models.

### Claim to avoid

Do **not** claim that “nobody studied when to correct” or that “the area is completely untouched.” That would be too strong and inaccurate.

---

## Most relevant papers and what they solve

This section is intentionally selective. It is not meant to be the full reading list.

### Central papers

#### 1. Learn from Your Mistakes: Self-Correcting Masked Diffusion Models (ProSeCo, 2026)
- Closest masked-diffusion-LM paper to the thesis question.
- Trains models to both unmask and correct already-generated tokens.
- Exposes correction schedule knobs such as frequency, loop count, and start time.
- Empirically studies corrector-budget allocation.
- Gap relative to the thesis: does not derive a principled trajectory-level scheduler or prove that a signal-adaptive schedule should beat uniform placement.

#### 2. Fine-Tuning Masked Diffusion for Provable Self-Correction (PRISM, 2025)
- Adds a per-token quality head to masked diffusion models.
- Uses quality scores to remask low-quality clean tokens.
- Important because it gives a **signal** with theoretical backing.
- Gap relative to the thesis: theory is about the quality head and self-correction mechanism, not about globally allocating a fixed corrector budget across time.

#### 3. Informed Correctors for Discrete Diffusion Models (2025)
- Closest direct paper on informed correctors in discrete diffusion.
- Important for defining the corrector object and why informed correctors matter.
- Gap relative to the thesis: not a masked-diffusion-LM scheduling paper, and not about trajectory-level budget allocation.

### Central adjacent papers

#### 4. Remasking Discrete Diffusion Models with Inference-Time Scaling (ReMDM, 2025)
- Strong inference-only baseline on top of pretrained masked diffusion models.
- Very important experimentally.
- Includes remasking and corrector-style baselines, entropy tooling, and a practical code path.
- Gap relative to the thesis: mainly remasking, not explicit corrector scheduling theory.

#### 5. Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations (2025)
- Probably the best proof template for this thesis.
- Gives dimension-free relative-entropy bounds and an information-profile view for predictor scheduling.
- Gap relative to the thesis: focuses on the predictor schedule rather than corrector allocation.

#### 6. Optimal Inference Schedules for Masked Diffusion Models (2025)
- Sharp theoretical paper on optimal unmasking schedules.
- Useful to understand what “optimal schedule” means in masked diffusion.
- Gap relative to the thesis: solves the wrong schedule problem for this thesis, namely the predictor path rather than corrector timing.

#### 7. Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking (EB-Sampler, 2025)
- Key adjacent paper if the thesis starts from entropy.
- Uses entropy to decide how many and which tokens to unmask.
- Includes a useful error-decomposition viewpoint.
- Gap relative to the thesis: entropy is used for unmasking, not for allocating corrective loops over time.

### Secondary but relevant papers

#### 8. Don’t Settle Too Early: Self-Reflective Remasking for Diffusion Language Models (RemeDi, 2025/2026)
- Confidence-guided remasking framework.
- Important for positioning signal-based revisiting methods.
- Not the same as fixed-budget corrector scheduling.

#### 9. Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models (DUS, 2025)
- Predictor-side schedule paper.
- Useful as an adjacent scheduling reference and as a warning not to conflate schedule types.

#### 10. Learning Unmasking Policies for Diffusion Language Models (2025)
- Learned inference policy paper.
- Good for optional future extensions, not the main thesis target.

### Theory papers most likely to matter

#### 11. Adapting the Gibbs Sampler (2018)
- Strong analogy for non-uniform scan allocation and update-effort optimization.
- Useful for the schedule-allocation lens.

#### 12. Informed Proposals for Local MCMC in Discrete Spaces (Zanella, 2017)
- Probably the best MCMC-side theory reference for “informed” local moves in discrete spaces.
- Useful if the corrector is formalized as a local informed kernel.

#### 13. Entropy Contraction of the Gibbs Sampler under Log-Concavity (2026 preprint)
- Useful only as a proof template if a contraction-style theorem is pursued.
- Probably not the main theory foundation, because assumptions are much cleaner than real text distributions.

---

## Recommended thesis formulation

### Thesis statement

The thesis should be framed as a **theory-plus-experiments study of trajectory-level corrector allocation**.

### Recommended main question

> For a fixed predictor schedule and a fixed corrector budget, can a signal-adaptive scheduler allocate corrective refinement loops more effectively than a uniform scheduler in masked diffusion language models?

### Candidate signals

The first three signals to test should be:

1. **Aggregate entropy** over revisable tokens.
2. **Confidence margin** or related logit sharpness measure.
3. **Quality mass** if a PRISM-style or analogous quality signal is available.

### Recommended hypothesis ordering

1. Uniform corrector placement is not optimal under a fixed budget.
2. Pure entropy-proportional placement may help, but may fail early in the trajectory.
3. Burn-in-gated or middle-phase-weighted signal-adaptive schedules may be stronger than naive entropy-only scheduling.
4. Quality-like signals may outperform entropy if available.

### Non-goals

The thesis does **not** need to prove:

- universal optimality of entropy scheduling,
- a complete theory of masked diffusion sampling,
- or a learned end-to-end optimal policy over all inference choices.

The thesis does **not** need to redesign the full model training objective.

---

## The most realistic theory target

A good MSc-level theorem is not “entropy-proportional scheduling is always optimal.”

A realistic theorem target is something like:

- fix a predictor schedule,
- define a residual error proxy after each predictor step,
- assume one corrector loop reduces that residual by a diminishing-return amount depending on a trajectory signal,
- derive the optimal or near-optimal budget allocation under those assumptions,
- then test whether entropy, confidence margin, or quality mass predicts the marginal gain well enough in real models.

That is rigorous enough to be meaningful, but modest enough to be realistic.

### Likely mathematical routes

1. **Residual error reduction view**
   - Define the expected gain from one corrector loop at time `t`.
   - Allocate budget where marginal gain is largest.
   - Under monotonicity assumptions, justify a signal-proportional or threshold-based rule.

2. **Information-profile / factorization-error view**
   - Borrow the structure from predictor scheduling theory.
   - Treat correctors as targeting dependence or factorization error left behind by the predictor.
   - Show why the best schedule may be concentrated in a middle or late-middle phase.

3. **Non-uniform scan Gibbs analogy**
   - View the corrector as a local informed update kernel.
   - Use non-uniform scan allocation ideas to justify non-uniform effort across time.

### Important warning

The theory should stay honest about assumptions. The repo docs should explicitly separate:

- proved statements,
- heuristic arguments,
- empirical findings,
- and conjectures.

---

## Practical experimental infrastructure

This is the practical shortlist that the repo should be reorganized around.

### Tier A: must-have

#### MDLM
- Role: base masked diffusion language model.
- Value: small/medium public checkpoint; clean open baseline.
- Why important: many adjacent works build on or reuse it.
- Suggested use: baseline sampling and instrumentation.
- Repo: `https://github.com/kuleshov-group/mdlm`
- Checkpoint: `kuleshov-group/mdlm-owt`

#### ReMDM
- Role: practical inference-time scaling and remasking framework.
- Value: useful code path for entropy diagnostics, remasking baselines, and inference experiments.
- Suggested use: fast experimentation platform if a direct corrector setup is not yet ready.
- Repo: `https://github.com/kuleshov-group/remdm`

#### ProSeCo
- Role: best direct platform for corrector scheduling experiments.
- Value: explicit correction-loop controls and public checkpoints.
- Suggested use: primary experimental platform for corrector-scheduling ablations.
- Repo: `https://github.com/kuleshov-group/proseco`
- Checkpoints: `kuleshov-group/proseco-owt`, `kuleshov-group/proseco-llada-sft`

### Tier B: strongly useful

#### PRISM
- Role: quality-signal and self-correction platform.
- Value: best route if the thesis extends beyond entropy to learned quality signals.
- Suggested use: second-stage experiments after entropy and confidence baselines exist.
- Repo: `https://github.com/SeunggeunKimkr/PRISM`

#### dLLM
- Role: broader diffusion LM framework.
- Value: useful for generalization and future portability.
- Suggested use: optional common framework if the repo grows beyond one backbone.
- Repo: `https://github.com/ZHZisZZ/dllm`

### Tier C: validation targets if compute allows

#### LLaDA
- Role: larger-scale validation.
- Value: strong external masked diffusion LM target with public code and weights.
- Suggested use: only after small-model results are stable.
- Repo: `https://github.com/ML-GSAI/LLaDA`

#### Dream
- Role: larger diffusion LM validation target.
- Value: useful if the method is inference-only and easy to port.
- Suggested use: optional late-stage external validation.
- Repo: `https://github.com/DreamLM/Dream`

---

## Recommended implementation order

The repo plan should be updated to follow this sequence.

### Phase 1: establish a small-model, reproducible baseline

1. Set up MDLM.
2. Reproduce baseline generation and evaluation on OpenWebText.
3. Log per-step trajectory diagnostics.
4. Add simple schedule baselines:
   - uniform,
   - front-loaded,
   - back-loaded,
   - middle-loaded.

### Phase 2: direct corrector scheduling experiments

1. Set up ProSeCo with the small OWT checkpoint.
2. Keep the predictor schedule fixed.
3. Compare fixed-budget corrector allocations.
4. Add signal-adaptive baselines:
   - entropy-proportional,
   - entropy-thresholded,
   - burn-in-gated entropy,
   - confidence-margin-based.

### Phase 3: stronger signals

1. Add PRISM or a lighter proxy quality signal.
2. Compare entropy vs confidence vs quality mass.
3. Evaluate whether the best signal predicts actual marginal gain.

### Phase 4: optional external validation

1. Port the best scheduler to a larger public model.
2. Validate whether gains are architecture-specific or robust.

---

## Experimental questions the repo should track

Claude should update the repo so these become explicit tracked questions.

1. Under fixed extra NFE budget, does any non-uniform corrector schedule beat uniform?
2. Does the best schedule differ by total budget?
3. Are early high-entropy regions misleading because context is incomplete?
4. Does a burn-in gate improve entropy-based scheduling?
5. Does confidence margin outperform entropy?
6. Does a quality signal outperform both entropy and confidence?
7. Can the marginal value of a correction loop be predicted from trajectory diagnostics?

---

## Metrics and outputs to standardize

The repo should standardize logging for:

- total predictor NFEs,
- total corrector NFEs,
- schedule allocation over time,
- per-step entropy,
- per-step confidence margin,
- per-step quality mass if available,
- fraction of tokens changed by each correction loop,
- generation quality metrics already used by the chosen codebase,
- compute/quality tradeoff curves.

It should also maintain a compact experiment table that makes fixed-budget comparisons easy.

---

## Documentation changes Claude should make

### The repo should gain a clean core set of docs

Suggested target files:

- `README.md` — short summary of the thesis focus and repo layout.
- `docs/thesis_direction.md` — precise research question, scope, non-goals, and open-question verdict.
- `docs/literature_map.md` — categorized paper map.
- `docs/experimental_infrastructure.md` — repos, checkpoints, and setup status.
- `docs/reading_plan.md` — next papers to read, tagged by status.
- `docs/implementation_plan.md` — concrete experiment roadmap.
- `docs/legacy_cleanup_log.md` — what was archived or deleted and why.
- `docs/proof_worklog.md` or `research/proof_worklog.md` — ongoing derivations and theorem attempts.

### Old documentation should be handled carefully

Old docs that are still historically useful but no longer aligned should be:

- marked as **deprecated**,
- moved into `archive/legacy/` if appropriate,
- cross-linked from the new thesis direction doc if they contain useful historical context.

Files that are clearly stale and add confusion can be removed, but only if they are obviously obsolete or duplicated.

---

## Safe cleanup policy

Claude should apply a conservative cleanup policy.

### Safe to archive

- superseded plans,
- abandoned thesis directions,
- duplicate note files,
- outdated summaries that conflate remasking and corrector scheduling,
- rough brainstorming documents that are no longer needed in the main root.

### Safe to delete

- generated artifacts that can be reproduced,
- empty placeholders,
- obsolete scratch files,
- duplicated exports,
- dead files that are clearly not referenced anywhere.

### Do not delete without strong justification

- reading notes,
- code experiments that may still matter,
- data-processing scripts,
- proofs or derivations,
- results folders unless reproducible and clearly obsolete.

The cleanup log should say what was archived, what was deleted, and why.

---

## Reading-plan update requirement

Claude should update the reading plan by checking what has already been read in the repo.

It should:

1. scan notes, summaries, bibliographies, PDFs lists, and reading trackers,
2. infer which papers are already read or partially read,
3. avoid rebuilding a reading plan from scratch if the information already exists,
4. create a prioritized next-reading list focused on the current thesis formulation.

The updated plan should tag papers as one of:

- `read`,
- `partially read`,
- `to read next`,
- `optional`,
- `background only`.

It should also tag each paper by function:

- `corrector design`,
- `corrector scheduling`,
- `remasking`,
- `predictor scheduling`,
- `theory`,
- `experiments`,
- `background`.

---

## Mathematical worklog requirement

Claude should begin a research worklog in parallel with the repo cleanup.

That worklog should:

- attempt candidate theorem paths,
- keep assumptions explicit,
- cite proof ingredients precisely,
- separate borrowed ideas from novel ones,
- record failed approaches and counterexamples,
- and maintain a running correctness status.

The worklog should be exploratory, but rigorous.

---

## Target final thesis shape

The repo should now be organized around this target thesis shape:

### Theory contribution
A modest but meaningful theorem or proposition about fixed-budget corrector allocation under explicit assumptions.

### Experimental contribution
A strong empirical comparison of uniform vs signal-adaptive corrector schedules on open masked diffusion language models with public checkpoints.

### Positioning contribution
A clear distinction between:

- informed correctors,
- remasking,
- predictor schedules,
- token-selection policies,
- and corrector scheduling.

That distinction is part of the value of the thesis and should be visible in the docs.

---

## What Claude should not do

- Do not re-expand the thesis into a generic survey of diffusion language models.
- Do not let the thesis question drift into “all possible inference policies.”
- Do not present remasking papers as if they already solved corrector scheduling.
- Do not present predictor-schedule theory as if it already solved corrector scheduling.
- Do not overclaim that the open gap is completely untouched.
- Do not anchor the thesis too tightly to one signal if experiments may favor another.

---

## Preferred default wording for the repo

Use wording close to the following:

> This repository studies signal-adaptive allocation of corrective refinement steps in masked diffusion language models. The central question is whether aggregate trajectory diagnostics such as entropy, confidence margin, or quality mass can predict the marginal value of a correction loop well enough to outperform uniform corrector placement under a fixed compute budget.

