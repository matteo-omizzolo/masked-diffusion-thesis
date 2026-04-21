# gpt_pro_assessment.md

# Assessment of the current theory direction

## Scope and bottom line

I read the four attached notes first and treated them as the primary internal record of the thesis direction. I then checked the closest current primary sources on masked diffusion schedule theory, self-correction, and informed correctors to decide what is realistic for an MSc thesis.

**Bottom line:** the current notes are strongest when they treat corrector scheduling as a **budgeted ranking / allocation problem over time**. They are weakest when they try to import a full **geometric contraction theorem** for masked-diffusion correctors from Gibbs-sampler theory. The best MSc route is **not** to make geometric contraction the center of the thesis. The best route is to prove a **proxy-based scheduling theorem under approximate additivity**, then validate the key assumption experimentally by measuring one-loop marginal gain.

That route is novel enough, experimentally aligned with the actual question, and much more finishable than a direct contraction theorem.

---

## 1. Audit of the attached documents

## 1.1 `open_questions.md`

### What is solid

- The document asks the right seven questions.
- The priority ordering is mostly good: novelty relative to ProSeCo, the contraction question, and the entropy-proxy question really are the critical issues.
- The questions are phrased skeptically rather than as hidden claims, which is exactly right for a thesis note.

### What is fragile

- Q1 is framed as if the fate of the thesis depends on geometric contraction. I do **not** think it should. That question matters only if Candidate 2 remains the main theorem target.
- Q2 correctly says entropy is mainly an empirical question, but it should be made even sharper: the thesis does **not** need entropy to be globally good. It only needs some signal to be good enough to beat uniform at fixed budget.
- Q5 treats additivity as a possible follow-up limitation. I would upgrade it: additivity is the **main modeling approximation** of the first theorem, and the theory should explicitly absorb that via an interaction term or near-additivity assumption.

### What is likely wrong or too ambitious

- Q1 implicitly suggests that a clean stationary-kernel contraction theorem might be reachable for the real text setting. That is likely too ambitious. The current Gibbs contraction paper proves entropy contraction for **random-scan Gibbs** under strong log-concavity, and notes that outside strong log-concavity exponential rates degrade; that is too far from high-dimensional masked text generation to be the core theorem target.[^ascolani]

### What should be dropped or reframed

- Do **not** let Q1 be a blocker for thesis commitment.
- Reframe Q1 from “must prove contraction” to “should we keep contraction as a stretch appendix or empirical diagnostic?”

### What should become central

- Q2 and Q5 should become central: is a signal predictive of one-loop marginal gain, and is approximate additivity good enough for schedule design?

---

## 1.2 `proof_ledger.md`

### What is solid

- The ledger is already doing something many theses fail to do: it distinguishes borrowed, adapted, and novel ingredients.
- The unverified-claims table is useful and should be kept.
- The explicit novelty warning about ProSeCo is healthy.

### What is fragile

- The ledger is not yet precise enough at the **theorem level**. It tracks paper-level provenance, but not exact proposition-level provenance.
- It does not yet separate:
  1. definitions,
  2. assumptions,
  3. borrowed results,
  4. conjectures,
  5. experimentally testable claims,
  6. disproved ideas.

### What is likely wrong

- The ledger states that Ascolani–Lavenant–Zanella gives per-step KL contraction for **systematic-scan** Gibbs. The arXiv abstract instead states the result for **random-scan Gibbs**, with extensions to Metropolis-within-Gibbs and Hit-and-Run.[^ascolani] That is not a small wording issue; it shows the exact borrowed result has not yet been pinned down precisely enough.

### What should be added

I would extend the ledger schema. For every proof ingredient, track:

- exact source theorem / proposition number,
- exact assumptions,
- exact borrowed statement,
- what is changed in the adaptation,
- why the adaptation is plausible,
- whether the adaptation is only heuristic,
- what empirical test is meant to stress it,
- whether a counterexample is known.

Also add two new tags:

- `[Incorrect as stated]`
- `[Definition only]`

At the moment the ledger is good enough to prevent blatant overclaiming, but not yet good enough to prevent **assumption drift**.

---

## 1.3 `proof_worklog.md`

### What is solid

**Entry 1** is strong. The formalization with a fixed predictor schedule, a corrector budget, and a trajectory-level allocation problem is the right abstraction.

**Entry 2** contains the most valuable thesis intuition: **high entropy early is not the same thing as high corrector value**. That is the right reason to test burn-in gating.

**Entry 3** is a good warm-up theorem sketch, but only as a lemma.

### What is promising

- The “marginal gain” framing is the right organizing variable.
- The shift from “is entropy optimal?” to “does a signal rank gain well enough to beat uniform?” is exactly the right thesis question.
- The burn-in observation is practically important and highly testable.

### What is underspecified

- `R_t` and `Δ_t` are introduced abstractly, but the actual thesis needs a **concrete operational definition** of one-loop marginal gain.
- The objective should distinguish:
  - **oracle single-loop gain** `Δ_t := F(output with one loop at t) - F(baseline output)`, and
  - **multi-loop schedule gain** `G(S)` for a set or multiset of corrected steps.

Without that distinction, the additive assumption and the experiment do not line up cleanly.

### What is likely wrong or too brittle

- Entry 5 is too optimistic when it says “if the contraction factor can be related to the signal, the theorem writes itself.” That step is exactly the hard part, and in the real masked-text setting it is the least believable part.
- The claim that mutual information monotonicity in unmasked fraction should be proved as a general theorem is also too ambitious. It is a nice intuition, but it is not necessary for a good thesis.

### What should be dropped

- Drop the idea that the first main theorem should directly prove `E_fact^{corrected}(t, k_t) = E_fact(t) ρ(t)^{k_t}` in the real MDM setting.

### What should become the main route

- Keep Entry 1.
- Keep the burn-in idea from Entry 2 / Entry 4.
- Replace Entry 5 as the main target with a theorem about **proxy-ranked scheduling under approximate additivity**.

---

## 1.4 `candidate_theorems.md`

## Candidate 1 — Optimal Binary Corrector Allocation

### Verdict

**Keep, but demote to lemma.**

This is correct under its assumptions and useful as a starting point, but too obvious to be the thesis centerpiece. Its value is not in the theorem itself; its value is that it reduces the empirical question to **ranking marginal gains**.

## Candidate 2 — Factorization-Error Contraction Under Correctors

### Verdict

**Do not make this the main theorem. Keep only as stretch goal / appendix route.**

Why:

- It depends on transferring entropy-contraction logic from Gibbs sampling under strong log-concavity to masked diffusion correctors in text, which is highly nontrivial and may simply fail in the needed generality.[^ascolani]
- It assumes a per-step stationary or well-targeted contraction picture that is much cleaner than the real predictor-corrector trajectory.
- Even if a stylized version is true, it is not the fastest path to a defensible MSc result.

## Candidate 3 — Burn-In Gated Allocation Dominates Naive Entropy

### Verdict

**Keep the intuition, but change the theorem statement.**

The current statement relies on a broad mutual-information monotonicity claim that is not necessary and may be annoying to prove cleanly. A better version is a simple low-gain-region proposition:

> If early steps form a region where one-loop gain is uniformly small, then any schedule that spends budget there is weakly dominated by one that reallocates those loops to sufficiently larger-gain later steps.

That gives you a clean burn-in theorem without needing a universal MI theorem.

## Candidate 4 — Confidence Margin as Alternative Proxy

### Verdict

**Empirical only.**

This is worth testing, and it may even outperform entropy in the late regime, but it is not a good main theorem target.

---

## 2. Answers to the open questions

## Q1 — Does geometric contraction hold for masked diffusion correctors?

### Verdict

**Not a realistic main-theorem target for this MSc thesis.**

### Reasoning

The closest rigorous source is Ascolani–Lavenant–Zanella, which proves entropy contraction for **random-scan Gibbs** under strong log-concavity, and notes weaker, polynomial behavior outside strong log-concavity.[^ascolani] Your setting is discrete text generation with model-approximate conditionals and a nonstationary predictor-corrector trajectory. That is too large a gap to bet the thesis on.

Even if some local or stylized contraction statement exists, it is unlikely to be the fastest path to a clean result. The mismatch is not only technical; it is conceptual:

- the target distribution changes with predictor step,
- the corrector may not be an exact Gibbs kernel for a fixed joint target,
- text distributions are not naturally handled by strong log-concavity assumptions.

### Affects

- Candidate 2 directly.
- Entry 5 in the worklog.

### Status

**Postpone as stretch goal. Not a blocker.**

### What to do instead

Use contraction empirically as a **diagnostic**: fit diminishing-returns curves or approximate per-step contraction factors from data, but do not anchor the main theorem on them.

---

## Q2 — Is entropy a good proxy for marginal gain?

### Verdict

**Probably not globally. Plausibly useful after burn-in, and likely weaker than a learned quality signal if one is available.**

### Reasoning

There are three distinct possibilities:

1. **Raw full-trajectory entropy is too crude.** Early in the trajectory entropy can be high because context is missing, not because correction is valuable.
2. **Entropy over revisable tokens may be materially better.** If the corrector only acts on already-decoded or otherwise revisable positions, aggregate entropy should be restricted to those positions.
3. **Quality mass may beat entropy if available.** PRISM explicitly learns per-token quality scores in the same forward pass and uses them for inference-time self-correction.[^prism] That is a more direct proxy for “error worth fixing” than entropy.

So the right empirical question is **not** “is entropy theoretically correct?” It is “does entropy rank one-loop marginal gains well enough to beat uniform, and if not, does burn-in gating or another signal fix the failure?”

### Affects

- The main experiment.
- The choice between entropy, inverse margin, and quality mass.
- The final thesis claim.

### Status

**Central empirical question. Not a theoretical blocker.**

---

## Q3 — How sensitive is the optimal schedule to the total budget `B`?

### Verdict

**Likely moderately sensitive, and you should assume budget dependence until proven otherwise.**

### Reasoning

Even in the simplest additive model, top-`B` selection depends on `B`. Once diminishing returns or repeated loops enter, the schedule should shift with budget. ProSeCo already explores budget allocation through the frequency of corrector application and the number of steps per loop, and explicitly studies how to allocate a corrector budget empirically.[^proseco_appendix] That is indirect evidence that schedule quality is budget-dependent.

### Affects

- Experimental design: must test multiple budgets.
- Theory: the theorem should compare schedules at fixed `B`, not claim a single universal schedule shape.

### Status

**Manageable. Must be tested, but not a blocker.**

---

## Q4 — Does ProSeCo already subsume the thesis direction?

### Verdict

**No, but it narrows the novelty claim sharply.**

### Reasoning

ProSeCo is the closest masked-diffusion paper. It trains a model to both unmask and correct, interleaves corrective refinement with generation, exposes sampling controls such as `corrector_every_n_steps`, `corrector_steps`, and `corrector_start_iter`, and discusses how frequency and loop length allocate a corrector budget.[^proseco][^proseco_repo][^proseco_appendix]

What I did **not** find in the abstract, repo, or surfaced appendix summary is a **signal-adaptive trajectory scheduler** that uses entropy, confidence, or quality to allocate a fixed correction budget over time. So ProSeCo does not subsume your question.

But it **does** kill any novelty claim of the form:

- “nobody studied when to correct,” or
- “nobody explored corrector-budget allocation.”

Your defensible novelty claim is narrower:

> ProSeCo studies correction loops and fixed hyperparameter schedules empirically; this thesis studies **signal-adaptive corrector budget allocation** and seeks a **principled ranking / regret formulation** for it.

### Affects

- Positioning section of the thesis.
- The proof ledger novelty claim.

### Status

**Critical for positioning, but not a blocker.**

---

## Q5 — Is the additive gains assumption realistic?

### Verdict

**Not literally realistic, but acceptable as a first-order theorem model if interaction is surfaced rather than hidden.**

### Reasoning

Correcting at step `t` changes later states, so exact additivity is false in general. But it can still be a useful approximation if pairwise and higher-order interactions are not too large at the budgets you care about.

The theory should not say “assume independence” and move on. It should say something closer to:

> Let `G(S)` be the actual gain from correcting a set `S` of steps. If `G(S)` is approximately additive in the sense that `|G(S) - Σ_{t in S} Δ_t|` is small, then top-`B` schedules based on single-loop gains are near-optimal.

That is much more defensible, because the interaction error can be **measured experimentally**.

### Affects

- Candidate 1.
- The recommended replacement theorem.
- Experiment design: add an interaction diagnostic.

### Status

**Manageable assumption. Make it explicit and test it.**

---

## Q6 — What is the right formal definition of “corrector loop”?

### Verdict

**Use a time-local fixed-noise Markov-kernel definition.**

### Recommended definition

At predictor step `t`, a corrector loop is a Markov kernel `K_t(· | z_t)` on the state space at the **same predictor time / noise level**, which:

1. takes the current state `z_t` as input,
2. updates some subset of token positions using model-informed conditional information,
3. may change already-decoded tokens,
4. does **not** advance the predictor schedule or change the global noise level.

This definition cleanly includes:

- Gibbs-style or MH-within-Gibbs token resampling,
- ProSeCo-style correction loops,
- other fixed-time informed corrective kernels.

It excludes pure predictor actions and keeps remasking conceptually distinct when remasking changes the effective predictor state rather than refining at fixed time.

### Affects

- Thesis framing.
- Theorem statements.
- Experimental implementation.

### Status

**Not a blocker. Lock this definition early.**

---

## Q7 — Can the contraction factor `ρ(t)` be estimated empirically?

### Verdict

**Yes as a diagnostic, but it should not be the centerpiece.**

### Reasoning

You can run `k in {0,1,2,4,8}` corrector loops at fixed steps and fit either:

- a geometric model,
- a concave diminishing-returns model,
- or a simple nonparametric curve.

That tells you whether “contraction-like” behavior is present. But if the curves are not geometric, that does not hurt the main thesis. It only hurts Candidate 2.

### Affects

- Diagnostic experiment.
- Whether Candidate 2 is worth keeping as a stretch appendix.

### Status

**Useful but secondary.**

---

## 3. Audit of the proof process

## Provenance tracking

### Is it adequate?

**Adequate as a first version, but not yet research-grade enough for a theory chapter.**

### Why

It already distinguishes borrowed vs adapted vs novel. That is good. But it still allows hidden slippage in three ways:

1. paper-level citations are too coarse,
2. assumptions are not attached tightly enough to borrowed statements,
3. failed ideas are not yet preserved as explicit dead ends.

### What to change

For each formal claim in the thesis notes, add a table with:

- **Claim ID**
- **Type**: definition / theorem / lemma / assumption / conjecture / empirical hypothesis
- **Source**: exact paper + theorem number if borrowed
- **Borrowed text summary**
- **What changes in this thesis**
- **Assumptions introduced here**
- **Current confidence**
- **How it will be validated**
- **Counterexample status**

That will make the ledger actually protective against accidental overclaiming.

---

## Are the current tags sufficient?

**Not quite.**

Current tags are good but incomplete. Add:

- `[Definition]`
- `[Incorrect as stated]`
- `[Conjecture]`
- `[Refuted / abandoned]`
- `[Validated empirically]`
- `[Depends on calibration]`
- `[Depends on approximate additivity]`

---

## Is the worklog branching in the right directions?

**Yes, but one branch is too expensive.**

### Good branches

- residual-error allocation,
- entropy failure modes,
- burn-in gating,
- explicit identification of the empirical ranking problem.

### Overly expensive branch

- direct contraction / information-profile transfer as the main theorem.

### Where effort should go next

1. formalize one-loop marginal gain `Δ_t`,
2. formalize actual multi-loop schedule gain `G(S)`,
3. prove an oracle / regret theorem under approximate additivity,
4. design the branch experiment that measures `Δ_t` directly,
5. only then revisit contraction as a diagnostic or stylized appendix.

---

## Which ingredients are safe, plausible, novel, or too speculative?

## Safely borrowed

- Lavenant–Zanella decomposition and information-profile viewpoint for predictor scheduling.[^lz]
- ProSeCo’s existence proof that trained self-correcting masked diffusion with correction loops is practical, plus exposed inference hyperparameters for budget allocation.[^proseco][^proseco_repo]
- PRISM’s existence proof that a learned per-token quality signal can be obtained and used at inference time.[^prism][^prism_repo]
- The generic resource-allocation / top-`B` optimization idea.

## Adapted but plausible

- Treating one-loop corrector placement as a budgeted ranking problem.
- Using approximate additivity rather than exact independence.
- Burn-in gating as a practical scheduler improvement.

## Genuinely novel enough for an MSc thesis

- The framing of **trajectory-level corrector scheduling as proxy-based budget allocation under a fixed predictor schedule**.
- The use of **measured one-loop marginal gain** as the target quantity a signal must predict.
- A theorem that turns schedule design into a **ranking / regret problem**, then validates the required assumptions experimentally.

## Too speculative to build the thesis around

- universal geometric contraction for masked-diffusion correctors,
- a universal entropy-to-gain proportionality law,
- a universal MI monotonicity theorem for the real masked-diffusion text setting.

---

## 4. The most promising theorem target

## Recommendation

The best main theorem is **not** one of the four candidates exactly as written.

### Best main theorem target

## **Modified main theorem: proxy-based top-`B` scheduling under approximate additivity**

### Formal shape

Let:

- `F` be the final evaluation objective for a generated sequence,
- `Δ_t := F(output with one corrector loop inserted at step t) - F(baseline output)` be the **one-loop marginal gain**,
- `G(S)` be the actual gain from inserting corrector loops at a set `S` of steps.

Assume:

1. **Fixed predictor schedule**.
2. **Binary budget for the theorem**: at most one extra loop per step.
3. **Approximate additivity**:
   `|G(S) - Σ_{t in S} Δ_t| <= η_B` for all `|S| = B`.
4. A proxy signal `s_t` admits a calibrated score `ψ(s_t)` with approximation error
   `|Δ_t - ψ(s_t)| <= ε` for all `t`.

Then the schedule `Ŝ_B` that selects the top-`B` steps by `ψ(s_t)` has regret bounded by a function of `ε` and `η_B`; in the cleanest sup-norm version,

`G(S*_B) - G(Ŝ_B) <= 2Bε + 2η_B`,

where `S*_B` is the optimal size-`B` schedule.

This is the right theorem shape because:

- it is mathematically modest but nontrivial,
- it directly matches the experiment,
- it does not require global contraction,
- it makes the real scientific question explicit: can entropy / margin / quality make `ε` small enough to beat uniform?

## What the existing candidates become

- **Candidate 1** → lemma: oracle top-`B` is optimal in the exactly additive binary case.
- **Candidate 2** → stretch appendix / abandoned unless a much simpler stylized setting emerges.
- **Candidate 3** → proposition or corollary: burn-in gating helps when an early region has uniformly low gain.
- **Candidate 4** → empirical comparator.

---

## 5. Best overall research path

## Recommended direction: hybrid theorem + empirical validation route

### Core claim

For a fixed predictor schedule and fixed corrector budget, corrector scheduling can be formulated as a **ranking problem over one-loop marginal gains**. A proxy signal is useful precisely insofar as it ranks those gains well enough that the induced top-`B` schedule improves over uniform placement.

### Exact mathematical object to optimize

- **Oracle single-loop quantity:** `Δ_t`
- **Schedule objective:** `G(S)`
- **Proxy quality:** calibration / ranking error of `s_t` against `Δ_t`
- **Schedule regret:** `Regret_B(s) := G(S*_B) - G(Ŝ_B(s))`

### Minimum assumptions

- fixed predictor schedule,
- binary placement theorem first,
- approximate additivity instead of exact independence,
- measurable one-loop marginal gain.

### What to prove first

1. oracle top-`B` lemma under exact additivity,
2. regret bound under approximate additivity and proxy error,
3. burn-in exclusion proposition.

### What to leave empirical

- whether entropy, margin, or quality mass best predicts `Δ_t`,
- how large interaction terms actually are,
- whether burn-in gating materially helps,
- whether contraction-like diminishing returns exist in practice.

### What becomes the thesis narrative

A good narrative is:

1. **Problem:** corrector loops are valuable but budgeted.
2. **Theory:** under a tractable model, scheduling is a proxy-ranking problem with explicit regret.
3. **Experiment:** measure one-loop gain directly and test which signals predict it.
4. **Result:** entropy may work only after burn-in; margin or quality mass may be better; signal-adaptive schedules can beat uniform at fixed budget.

That is a strong, realistic MSc thesis.

---

## 6. What should be dropped, postponed, or reframed

## Drop as main-route commitments

- “main theorem = geometric contraction of masked-diffusion correctors”
- “must prove universal entropy-proportional optimality”
- “must prove MI monotonicity in the real MDM setting”

## Postpone to stretch goals

- empirical fitting of `ρ(t)` and diminishing-returns curves,
- stylized contraction theorem in a toy model,
- stronger multi-loop water-filling theorem once binary case is finished.

## Reframe immediately

Reframe the thesis from:

> Is entropy-proportional scheduling optimal?

to:

> Can simple aggregate signals rank one-loop marginal gains well enough to induce a low-regret corrector schedule under a fixed budget?

That formulation is both stronger scientifically and safer mathematically.

---

## Boxed recommendation

> **Best theorem target**  
> A **proxy-based top-`B` scheduling theorem under approximate additivity**, with Candidate 1 as a lemma and burn-in gating as a proposition/corollary.
>
> **Best experiment**  
> A branch-based one-loop marginal-gain study on a masked diffusion text model, measuring per-step token-change rate and final-quality improvement, then comparing entropy, inverse margin, and quality mass as predictors of gain.
>
> **Best thesis formulation**  
> “For a fixed predictor schedule and fixed corrector NFE budget, can aggregate trajectory signals predict one-loop marginal gain well enough to induce a lower-regret corrector schedule than uniform placement?”
>
> **Biggest trap to avoid**  
> Building the thesis around a full geometric-contraction theorem for real masked-diffusion text correctors. It is the prettiest route, but currently the least robust and the most likely to stall.

---

## References consulted

- Lavenant, H., and Zanella, G. “Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations.” arXiv:2510.25544. https://arxiv.org/abs/2510.25544
- Chen, S., Cong, K., and Li, J. “Optimal Inference Schedules for Masked Diffusion Models.” arXiv:2511.04647. https://arxiv.org/abs/2511.04647
- Ascolani, F., Lavenant, H., and Zanella, G. “Entropy contraction of the Gibbs sampler under log-concavity.” arXiv:2410.00858. https://arxiv.org/abs/2410.00858
- Schiff, Y. et al. “Learn from Your Mistakes: Self-Correcting Masked Diffusion Models.” arXiv:2602.11590. https://arxiv.org/abs/2602.11590
- ProSeCo repository. https://github.com/kuleshov-group/proseco
- PRISM paper. “Fine-Tuning Masked Diffusion for Provable Self-Correction.” arXiv:2510.01384. https://arxiv.org/abs/2510.01384
- PRISM repository. https://github.com/SeunggeunKimkr/PRISM
- Wang, G. et al. “Remasking Discrete Diffusion Models with Inference-Time Scaling.” arXiv:2503.00307. https://arxiv.org/abs/2503.00307
- ReMDM repository. https://github.com/kuleshov-group/remdm
- Zhao, Y. et al. “Informed Correctors for Discrete Diffusion Models.” arXiv:2407.21243. https://arxiv.org/abs/2407.21243

[^ascolani]: Ascolani, Lavenant, and Zanella, “Entropy contraction of the Gibbs sampler under log-concavity,” arXiv:2410.00858, abstract.
[^lz]: Lavenant and Zanella, “Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations,” arXiv:2510.25544, abstract.
[^proseco]: Schiff et al., “Learn from Your Mistakes: Self-Correcting Masked Diffusion Models,” arXiv:2602.11590, abstract.
[^proseco_repo]: ProSeCo GitHub README, sampling parameters `corrector_every_n_steps`, `corrector_steps`, `corrector_start_iter`, and released `proseco-owt` / `proseco-llada-sft` checkpoints.
[^proseco_appendix]: Surfaced arXiv appendix summary for ProSeCo notes that Appendix D.3 / Figure 8 studies corrector-budget allocation via frequency and number of steps per loop.
[^prism]: Kim et al., “Fine-Tuning Masked Diffusion for Provable Self-Correction,” arXiv:2510.01384, abstract.
[^prism_repo]: PRISM GitHub README, including theory-backed per-token quality head and OWT / LLaDA fine-tuning and evaluation scripts.
