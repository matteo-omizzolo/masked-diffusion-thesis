# Prompt for Claude Code: Refocus the Thesis Repo and Clean It Up

You are working inside a thesis repository for an MSc project on masked diffusion language models. Your job is to **refocus the repo around the right thesis question**, clean and simplify the repo, update the documentation, preserve useful historical material, remove genuinely obsolete clutter, and bootstrap the experimental infrastructure.

You have been given a separate brief called `MSc_thesis_direction_brief.md`. Treat that brief as the authoritative high-level research context.

## Main goal

Refactor this repository so that it becomes a clean, focused workspace for the following thesis direction:

> **For a fixed predictor schedule and fixed corrector NFE budget in masked diffusion language models, can aggregate trajectory signals—entropy, confidence margin, or quality mass—predict the marginal value of a corrective refinement loop well enough to outperform uniform corrector placement?**

This is now the primary thesis question.

Do not keep the repository centered on a broader or vaguer “informed correctors” direction if the docs can be made sharper.

---

## What to optimize for

1. A thesis question that is genuinely open and realistically feasible.
2. Documentation that is clear, non-redundant, and aligned with current literature.
3. A cleaned repo where obsolete material is archived or removed.
4. A practical experimental setup centered on public code and checkpoints.
5. A living research plan that includes both theory and experiments.
6. A worklog structure that lets the student track what is borrowed from prior papers and what is novel.

---

## Important conceptual constraints

Keep these distinctions explicit everywhere in the repo:

1. **Informed correctors**
2. **Remasking methods**
3. **Predictor / unmasking schedule optimization**
4. **Token-selection policies**
5. **Corrector scheduling**

Do not conflate them.

In particular:

- do not present remasking as identical to corrector scheduling,
- do not present predictor/unmasking schedule theory as if it already solves corrector scheduling,
- do not overclaim that the target question is untouched,
- and do not overstate entropy-only scheduling as the final thesis before checking whether confidence or quality signals are stronger.

---

## Operating style

Work iteratively and use judgment.

I do **not** want a mechanical rewrite of every file. I want you to inspect the repo, understand what already exists, infer what has been read or attempted, and then reorganize the project so it reflects the most realistic and defensible thesis direction.

Be proactive, but conservative when deleting files.

---

## What you should do first

### 1. Inventory the repository

Read the repository structure and classify:

- current README and landing docs,
- literature notes,
- past thesis direction docs,
- experimental notes,
- code experiments,
- old plans,
- PDFs lists / reading trackers / Zotero exports / bibliographies,
- scratch files,
- obsolete artifacts,
- infrastructure / scripts / environment files.

### 2. Infer what has already been read

Before writing a fresh reading plan, scan the repo for evidence of papers already read or partially read.

Examples of places to inspect:

- reading notes,
- annotated bibliographies,
- markdown notes,
- PDF filenames,
- TODO files,
- notebooks,
- literature summaries,
- citation databases,
- planning documents.

Then build the new reading plan on top of that existing state rather than ignoring it.

### 3. Classify files by action

For each major file or folder, classify it into one of:

- `keep as-is`,
- `update`,
- `move to archive/legacy`,
- `delete`,
- `unclear / needs manual review`.

Keep a log of these decisions.

---

## Main documentation tasks

Create or update a focused documentation set. Use existing files if they already serve the purpose well; otherwise create new ones.

### Core docs to produce

1. `README.md`
   - concise repo overview,
   - current thesis question,
   - repo structure,
   - where to start.

2. `docs/thesis_direction.md`
   - exact research question,
   - scope,
   - open-question verdict,
   - what is and is not being claimed,
   - non-goals,
   - why this is a realistic MSc direction.

3. `docs/literature_map.md`
   - organized by categories,
   - especially preserving the distinctions between correctors, remasking, predictor scheduling, token selection, and theory,
   - brief summary of what each central paper actually solves,
   - what gap remains.

4. `docs/reading_plan.md`
   - updated using the papers already read in this repo,
   - tags such as `read`, `partially read`, `to read next`, `optional`, `background`,
   - reading order optimized for the new thesis question.

5. `docs/experimental_infrastructure.md`
   - practical shortlist of usable public repos and checkpoints,
   - what has already been obtained or cloned,
   - what is realistic for thesis-scale experiments,
   - what is Tier A / Tier B / Tier C.

6. `docs/implementation_plan.md`
   - short- and medium-term experiment roadmap,
   - baselines,
   - scheduler variants,
   - logging and metrics,
   - order of implementation.

7. `docs/legacy_cleanup_log.md`
   - what was archived,
   - what was deleted,
   - why.

8. `research/proof_worklog.md` or `docs/proof_worklog.md`
   - active mathematical worklog,
   - theorem attempts,
   - assumptions,
   - provenance tracking.

You may add a few more well-justified docs if they improve clarity, but do not explode the number of files.

---

## Repo cleanup policy

Use good judgment. The repo should become noticeably cleaner.

### Archive rather than delete when in doubt

Move historically useful but no-longer-central material to something like:

- `archive/legacy/`
- `archive/old_directions/`
- `archive/old_notes/`

Add a short deprecation banner at the top of archived markdown docs where useful.

### Delete only when safe

Delete files that are clearly:

- duplicated,
- empty,
- autogenerated and reproducible,
- dead scratch artifacts,
- obsolete exports,
- or stale clutter with no research value.

### Be conservative with

- derivations,
- reading notes,
- experiment outputs,
- scripts,
- notebooks,
- bibliographies,
- and anything that might encode past thought.

If a file is questionable, prefer archive over deletion.

---

## Reframing tasks

Update the repo so the central direction becomes:

### New thesis framing

**Signal-adaptive corrector scheduling for masked diffusion language models**

### Preferred core question

> For a fixed predictor schedule and fixed corrector NFE budget in masked diffusion language models, can aggregate trajectory signals—entropy, confidence margin, or quality mass—predict the marginal value of a corrective refinement loop well enough to outperform uniform corrector placement?

### Position this carefully

The docs should explain that:

- the problem is open in this precise formulation,
- nearby papers exist and matter,
- the real gap is trajectory-level fixed-budget corrector allocation,
- a good MSc version aims for a modest theorem plus feasible experiments,
- and entropy is a candidate signal, not necessarily the only or best one.

### Also add a short “why not the old framing?” note

Briefly explain why a pure “entropy-adaptive informed-corrector scheduling” framing is too narrow and why the broader signal-adaptive formulation is safer and more realistic.

---

## Literature handling tasks

Use the brief plus anything found in the repo to update the literature documentation.

### Central papers that should definitely appear

Make sure the new literature map and reading plan treat these as central or central-adjacent:

- ProSeCo
- PRISM
- Informed Correctors for Discrete Diffusion Models
- ReMDM
- Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations
- Optimal Inference Schedules for Masked Diffusion Models
- EB-Sampler
- RemeDi
- Adapting the Gibbs Sampler
- Informed Proposals for Local MCMC in Discrete Spaces

Add other papers only if they are genuinely useful to this thesis direction.

### Reading plan task

Do not blindly create a generic reading list.

Instead:

1. detect what has already been read,
2. identify what is missing for the current thesis formulation,
3. produce a prioritized next-reading plan.

Use tags for:

- status,
- relevance,
- function,
- whether the paper is for theory or experiments.

---

## Experimental infrastructure tasks

Start getting the necessary experimental infrastructure into shape.

### Preferred priority order

**Tier A**
- MDLM
- ReMDM
- ProSeCo

**Tier B**
- PRISM
- dLLM

**Tier C, optional later**
- LLaDA
- Dream

### What to do

1. Create a clean `external/`, `third_party/`, or similarly named area if the repo does not already have one.
2. Add a machine-readable manifest of external repos and checkpoints.
3. If network access is available and it is practical, clone the most important repos.
4. Pin commit hashes where possible.
5. Add setup instructions or scripts.
6. Record what works, what is missing, and what likely requires heavy reimplementation.

### Aim

The repo should make it clear which stack is the main thesis path:

- first small-model reproducible experiments,
- then direct corrector-scheduling experiments,
- then stronger signals,
- and only later optional larger-model validation.

### Strong suggestion

Prefer a setup that starts from **MDLM / ReMDM / ProSeCo** rather than scattering effort across too many frameworks.

---

## Experimental plan to encode in docs

Please convert the thesis question into a concrete plan.

### Phase 1
- establish a reproducible masked-diffusion baseline,
- instrument trajectory diagnostics,
- define fixed-budget scheduler baselines.

### Phase 2
- compare uniform vs front-loaded vs back-loaded vs middle-loaded corrector schedules,
- keep predictor schedule fixed,
- evaluate at multiple corrector budgets.

### Phase 3
- add signal-adaptive schedules:
  - entropy-proportional,
  - entropy-thresholded,
  - burn-in-gated entropy,
  - confidence-margin-based,
  - quality-mass-based if available.

### Phase 4
- if feasible, validate the strongest scheduler on a second model family.

Also define the logging and metrics that need to be tracked consistently.

---

## Mathematical worklog tasks

In parallel with the repo cleanup, start a mathematical worklog.

I do **not** want you to lock into a theorem too early. I want you to explore elegant routes, but rigorously.

### Start from this guiding question

Can one justify a non-uniform fixed-budget corrector allocation rule by modeling the marginal value of a corrective refinement loop as a function of trajectory diagnostics such as entropy, confidence margin, or quality mass?

### Explore, but keep the work structured

Maintain a worklog with at least these sections:

- candidate theorem statements,
- assumptions ledger,
- proof sketches,
- where each proof ingredient comes from,
- what appears novel,
- failed or doubtful routes,
- concrete next steps.

### Provenance requirement

Every time you borrow or adapt a proof idea from another paper, mark it explicitly.

Use tags like:

- `[Borrowed from ...]`
- `[Adapted from ...]`
- `[Novel attempt]`
- `[Needs verification]`
- `[Potential gap]`

This matters a lot. I want a worklog that is both intellectually honest and easy to audit later.

### Correctness requirement

Please self-audit the derivations.

Whenever you are unsure, say so explicitly. Keep a list of doubtful steps and unresolved assumptions.

Do not hide uncertainty behind polished prose.

---

## Deliverables I expect in the repo after your pass

At minimum, I want:

1. a cleaned and updated README,
2. updated thesis direction docs,
3. a literature map centered on the correct open gap,
4. an updated reading plan that respects what has already been read,
5. an experimental infrastructure document,
6. a concrete implementation plan,
7. a legacy cleanup log,
8. a mathematical proof worklog,
9. a small external-repo bootstrap structure,
10. archived or removed obsolete material.

If there are obvious code-level changes that help structure the work, make them too, but documentation and repo focus come first.

---

## Constraints on style and judgment

- Be clear and concrete.
- Prefer fewer, better documents over many fragmented ones.
- Be conservative with deletion.
- Preserve useful history but move it out of the critical path.
- Use current, up-to-date sources if you need to verify anything external.
- Do not overclaim.
- Keep the repo focused on text / masked diffusion / discrete diffusion.
- Avoid vision-diffusion digressions unless directly useful for theory.

---

## Suggested starting sequence

1. inspect repo,
2. infer current state and what has already been read,
3. classify files,
4. rewrite top-level framing docs,
5. build focused literature and reading docs,
6. archive/remove obsolete material,
7. bootstrap external infrastructure,
8. start proof worklog,
9. leave the repo in a clean, navigable state.

Take initiative and make the repo look like a serious MSc thesis workspace rather than a loose collection of notes.

