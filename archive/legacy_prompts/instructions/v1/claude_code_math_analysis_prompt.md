# Prompt for Claude Code: Start the Mathematical Analysis Work

You are working inside an MSc thesis repository on masked diffusion language models. The repository has been refocused around the following question:

> **For a fixed predictor schedule and fixed corrector NFE budget in masked diffusion language models, can aggregate trajectory signals—entropy, confidence margin, or quality mass—predict the marginal value of a corrective refinement loop well enough to outperform uniform corrector placement?**

Your task here is **not** to merely summarize papers. Your task is to **start doing the mathematical work** needed for the thesis in a way that is rigorous, traceable, and iterative.

I do not want you to lock yourself into one rigid theorem too early. I want you to explore several routes and try to find the most elegant one. At the same time, I want your work to be auditable: I need to know what is borrowed, what is adapted, what is novel, and what is still uncertain.

Treat this as the start of a real research worklog.

---

## Main research target

You should explore whether one can justify a non-uniform corrector allocation rule under a fixed budget by relating the **marginal value of a correction loop** to aggregate trajectory diagnostics such as:

- entropy,
- confidence margin,
- low-quality mass,
- or another well-justified proxy.

The output does not need to be a finished theorem immediately. But it should become a serious, structured attempt at one.

---

## What I want you to produce

Create a mathematical work package in the repo, for example under `research/` or `docs/`, containing a small set of high-value files. Use good judgment, but something like the following would be ideal:

1. `research/proof_worklog.md`
   - chronological research log,
   - derivation attempts,
   - insights,
   - failed directions,
   - corrections.

2. `research/candidate_theorems.md`
   - several candidate theorem statements,
   - assumptions,
   - plausibility notes,
   - difficulty ranking.

3. `research/proof_ledger.md`
   - provenance tracking for proof ingredients,
   - what comes from which paper,
   - what has been adapted,
   - what appears novel.

4. `research/open_questions.md`
   - unresolved technical points,
   - fragile assumptions,
   - places where empirical evidence may be needed to guide theory.

You can choose a slightly different file structure if it is clearly better, but keep it compact and useful.

---

## Provenance and honesty rules

This is very important.

Whenever you use a known idea, proof step, decomposition, inequality, or modeling choice from prior work, mark it explicitly.

Use tags like:

- `[Borrowed from <paper>]`
- `[Adapted from <paper>]`
- `[Analogy to <paper>]`
- `[Novel attempt]`
- `[Needs verification]`
- `[Potential gap]`
- `[Empirically motivated only]`

I want to be able to track the origin of each argument.

Do not blur borrowed reasoning into “our derivation” unless the step is actually new.

---

## Correctness tracking

I also want you to track your confidence in each derivation.

For each candidate theorem, lemma, or proof sketch, add a status such as:

- `solid`,
- `plausible but incomplete`,
- `heuristic only`,
- `likely false`,
- `blocked by assumption gap`.

Whenever you suspect a hidden assumption, state it.
Whenever a step may be invalid, isolate it and flag it.
Whenever a route looks elegant but brittle, say so.

The worklog should be rigorous enough that I can later inspect it and see where the reasoning is actually sound.

---

## Suggested mathematical directions

These are suggestions, not hard constraints. Feel free to iterate toward something more elegant if a better route appears.

### Direction A: residual-error reduction model

Try to formalize a residual error quantity after predictor step `t`, then model the gain from one corrector loop at time `t` as a function of that residual and a signal `s_t`.

Possible shape:

- fixed predictor schedule,
- budget `sum_t b_t <= B`,
- one corrector loop at time `t` reduces residual by a gain `g_t` or by a multiplicative factor,
- repeated loops have diminishing returns,
- optimal allocation spends budget where marginal gain is highest.

Then ask:

- under what assumptions is entropy a valid proxy for marginal gain?
- when is confidence margin better?
- when would low-quality mass dominate both?
- does a burn-in gate naturally emerge?

### Direction B: information-profile / factorization-error route

Borrow structure from predictor-scheduling theory in masked diffusion models and ask whether corrector loops primarily reduce a dependence or factorization error left by the predictor.

Then explore whether:

- the value of a corrector loop concentrates in a middle phase,
- entropy is only a proxy for a deeper dependence error,
- a schedule can be derived from an upper bound rather than directly from empirical signal values.

### Direction C: non-uniform scan Gibbs analogy

Model a corrector as a local informed update kernel and study whether non-uniform allocation of update effort over time can be justified by analogies with non-uniform scan Gibbs.

This route may not map perfectly to masked diffusion, but it could still give an elegant allocation principle.

### Direction D: burn-in-gated scheduling

Given that early uncertainty may reflect missing context rather than correctable error, explore whether any reasonable theory naturally suggests:

- zero or low correction in the very early phase,
- a ramp-up period,
- concentration of budget in mid or late-mid stages.

This may turn out to be more defensible than naive entropy-proportional scheduling.

---

## Useful literature anchors

As you work, keep track of which papers are relevant to which proof route.

Especially important:

- ProSeCo
- PRISM
- Informed Correctors for Discrete Diffusion Models
- ReMDM
- Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations
- Optimal Inference Schedules for Masked Diffusion Models
- EB-Sampler
- Adapting the Gibbs Sampler
- Informed Proposals for Local MCMC in Discrete Spaces

Do not assume they solve the target problem. Use them as anchors, templates, or cautionary references as appropriate.

---

## What I want the worklog to include

### 1. A clean problem formalization

Define clearly:

- predictor schedule,
- corrector loop,
- budget,
- aggregate signal,
- objective,
- what “outperform uniform” means mathematically or empirically.

### 2. At least three candidate theorem paths

Even if one ends up strongest, I want several candidate routes compared.

For each route, include:

- theorem idea,
- assumptions,
- expected payoff,
- risk of being vacuous or too stylized.

### 3. A set of candidate lemmas

Identify the small technical results that may be easier to prove first, such as:

- monotonicity of marginal gain under assumptions,
- diminishing returns of repeated correction loops,
- proxy ordering results,
- or budget-allocation optimality under simple concavity assumptions.

### 4. Counterexamples and failure modes

Please actively try to break naive formulations.

For example:

- situations where entropy is high but correction is not yet useful,
- cases where a confidence proxy is miscalibrated,
- scenarios where the best schedule is not monotone in a single signal.

This is not wasted effort; it improves the final theorem.

### 5. A bridge to experiments

Whenever theory suggests a measurable proxy, note how it could be tested in code.

The theoretical work should stay connected to what is feasible experimentally.

---

## Tone and style

Think like a careful research collaborator.

I do not need polished textbook exposition yet. I need real mathematical progress with explicit uncertainty and provenance.

Please prefer:

- concise definitions,
- explicit assumptions,
- transparent derivation steps,
- self-critique,
- and honest distinction between theorem, conjecture, and heuristic.

---

## A good first milestone

A very good early milestone would be:

1. a clean formalization of fixed-budget corrector allocation,
2. two or three candidate gain models,
3. one provisional theorem or proposition under stylized assumptions,
4. one clear argument for why a burn-in-gated or middle-weighted schedule may be more defensible than naive full-trajectory entropy proportionality,
5. and a list of the exact empirical quantities needed to test those claims.

---

## Final reminder

Please let yourself iterate toward the most elegant solution rather than forcing one too early.

But every time you use an argument from a paper, tag it.
Every time you make a novel move, tag it.
Every time you are unsure, flag it.

I want a work product that is mathematically serious and easy to audit later.

