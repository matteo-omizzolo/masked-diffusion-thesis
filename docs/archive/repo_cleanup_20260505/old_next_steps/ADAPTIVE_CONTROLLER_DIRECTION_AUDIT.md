> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


> **STATUS:** AUDIT (Phase 1 of adaptive-controller research study)
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Skeptical pre-literature audit of whether "adaptive state-dependent budgeted controllers" is a justified extension of the current open-loop corrector-scheduling thesis. Written before any new literature search. Answers the six audit questions; does not pick frameworks.
> **COMPANION DOCS:** `INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` (Supported/Plausible/Speculative taxonomy), `ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md` (premature-framework-escalation risk), `PRINCIPLED_NEXT_STEPS_PLAN.md` (recommends Options 2+5, postpones SMC/control-as-inference), `LARGE_MODEL_CONTINUATION_DECISION.md` (Phase 2b-only bounded GO).

---

# Adaptive Controller Direction Audit — Phase 1

## Purpose

The **current** thesis object is a deterministic **open-loop** schedule:

    S ⊆ {1, ..., T},   |S| = B,   a_t = 𝟙[t ∈ S]

chosen **once** from a trajectory-level signal vector (s_1, ..., s_T) or from a joint combinatorial score A(S). Theorem A bounds the regret of a calibrated-proxy top-B selector Ŝ_B against the open-loop oracle S_B* by 2Bε + 2η_B.

The **proposed extension** is a state-dependent **adaptive** policy:

    π : (z_t, b_t) ↦ a_t ∈ {0, 1},   subject to Σ a_t ≤ B,

where z_t is the masked-diffusion state at trajectory step t and b_t = B − Σ_{τ < t} a_τ is the remaining budget. The policy reads the state online and decides locally.

This is a **strictly larger** hypothesis class: every open-loop S is representable as a state-blind adaptive policy. Adaptive is therefore "mathematically harmless" as an abstraction, but whether it is **empirically justified**, **theoretically tractable**, and **thesis-appropriate** is a different question.

This audit answers:

1. Is adaptive a natural extension of the current open-loop thesis?
2. What empirical evidence from the repo supports moving in that direction?
3. What evidence is still missing?
4. What are the dangers of moving too quickly to adaptive control?
5. Does the thesis need adaptive control for a strong conclusion, or is it optional next-stage work?
6. What exact research question should the adaptive extension ask?

All answers are written **before** Phase 2 literature search. They therefore reflect only the repo's current evidence and the existing audit chain.

---

## 1 — Naturalness as an extension

### 1.1 Open-loop as a special case

Formally, the current thesis object is a special case of the proposed extension with π ignoring z_t. So **every open-loop result is recoverable** in the adaptive framing. In that purely abstract sense, the extension is "natural."

### 1.2 But naturalness ≠ warrant

Three observations blunt the naturalness argument:

- **(a) Theorem A's proof does not transfer unchanged.** The 2Bε + 2η_B bound uses a swap argument over a fixed subset S_B* of size B. In the adaptive case the oracle is an optimal policy π*, not an optimal subset, and ε, η_B are no longer the correct calibration/additivity constants. Adaptive proxy-regret requires Bellman-style machinery or a different regret decomposition. This is not catastrophic but it is not a free extension either — it is a new theorem, not a corollary.

- **(b) The empirical infrastructure is open-loop.** Protocol A measures per-step Δ_t; Protocol B measures G(S) for sampled subsets; Phase 3a's CD-G / BS-AG search within subsets of size B. None of this measures Δ_t conditioned on z_t features, nor does it measure the conditional value function V(z_t, b_t). Converting infrastructure costs real engineering.

- **(c) The governance decision in `INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` (2026-04-21) explicitly classifies the MDP / SMC / control-as-inference direction as "plausible but premature; treat as future-work, not active research direction for this thesis."** That verdict is one day old at time of writing and has not been countermanded by new evidence.

**Naturalness verdict.** `[Mathematically natural as generalization]` `[Operationally costly]` `[Blocked by standing governance decision as thesis-core direction, but not blocked as future-work section]`.

---

## 2 — Empirical evidence in the repo that supports moving adaptive

### 2.1 Evidence FOR adaptive (directly or indirectly)

- **E1. High within-seed variance at small B.** `combinatorial_diagnostics.json` reports within-seed shares 0.691 / 0.645 / 0.619 at B ∈ {2, 3, 4}. ≈62% of G(S) variance at B = 4 is explained by MC-stochasticity within a fixed trajectory, not by between-trajectory differences. **Strength:** weak-to-moderate. Consistent with "best S depends on MC realization," which would argue for reading the realized z_t. But also consistent with "G itself has irreducible MC noise that no policy can remove" — the two are observationally equivalent from this statistic alone. See §3.2 for what would disambiguate.

- **E2. Top-k / bottom-k MC Jaccard ≈ random baseline.** `combinatorial_diagnostics.json` reports `ratio_topk_vs_random` = 1.20 / 1.18 / 1.30 at B ∈ {2, 3, 4}. The schedules that produced the best G on a given MC sample barely overlap with each other (≈20–30% above random). **Interpretation:** on fresh MC samples there is no stable open-loop winner; different realizations prefer different schedules. This is a genuine hint that per-realization information matters. **Strength:** moderate.

- **E3. Ranker class already failed open-loop.** Phase 3a's ranker-class negative (`PHASE3A_COMBINATORIAL_RESULTS.md`) shows that greedy per-step rankers using entropy/margin/quality signals do not beat uniform. Theorem A's Negative-Result Corollary is `[Validated empirically]` for the ranker class. This makes adaptive state-blind rankers unattractive **unless** one can argue that reading z_t adds information the aggregate signals miss. That argument is at present conjectural.

- **E4. Search-class success depends on an expensive oracle.** CD-G achieves 74–84% oracle closure and BS-AG achieves 49–64%, both using G (or an AG ranker + G-rollout) at inference. If one wants a deployable inference-time scheduler, either (i) a cheap open-loop surrogate for G must be discovered, or (ii) an amortised adaptive policy that approximates S_B* without querying G must be trained. Option (ii) is the adaptive motivation on **deployability** grounds, not quality grounds.

### 2.2 Evidence AGAINST adaptive (or limiting its expected upside)

- **A1. Open-loop combinatorial search already reaches 74–84% of the mean-delta-oracle closure.** The remaining 16–26% is an upper bound on what any adaptive policy could add **if** the oracle is itself a meaningful upper bound (it is mean-field, so adaptive could in principle beat it — but not by much in expectation if the pairwise interaction ξ is not dominated by state-dependent structure).

- **A2. Aggregate signals (H_t, 1−M_t, Q_t) do not rank Δ_t well already.** Spearman(signal, Δ_t) ≈ −0.19 to +0.1 on Phase 2b (per `PHASE3A_COMBINATORIAL_RESULTS.md` and surrounding notes). A state-dependent policy that reads the same signals has no a priori reason to do better than an open-loop one: the signals themselves are weak, not just their aggregation. The adaptive framing only helps if the policy reads **finer-grained** state features (individual logit distributions, context structure), which is a substantially harder learning problem with its own sample-complexity penalty.

- **A3. Mean-delta-oracle S_A* is informative but not tight.** The open-loop oracle we currently approximate is mean-field top-B of Δ̄_t. The true S_B* = argmax_{|S|=B} G(S) is not computed at scale (Q12 in `open_questions.md`). Adaptive cannot be clearly better than open-loop until open-loop's ceiling is measured. Right now we're comparing adaptive-future to a proxy of the open-loop ceiling, not to the ceiling itself.

- **A4. Theorem A is corrector-kernel-agnostic by design, not state-dependent by design.** The current theory's key methodological claim is portability across kernels (MDLM/ProSeCo-OWT/ProSeCo-LLaDA-SFT). Moving to adaptive risks re-specialising the theory to a specific z_t representation, narrowing the contribution rather than broadening it.

- **A5. Thesis scope is explicitly bounded to scheduling (not policy design).** `docs/thesis_direction.md` §3 distinguishes five concepts: informed correctors, remasking, predictor schedules, token-selection, corrector scheduling. The thesis targets #5. An adaptive policy π(z_t, b_t) blurs the boundary toward #4 (token-selection can emerge from state-dependent decisions) and toward learned control more generally. That is scope creep.

- **A6. Phase 2b variance decomposition does not isolate state-dependence.** Within-seed variance can be caused by (i) corrector stochasticity downstream of a deterministic z_t, (ii) predictor stochasticity in future trajectory, (iii) MC noise in F. None of (i–iii) is removable by reading z_t. We currently cannot partition within-seed variance into the "adaptive-removable" vs "adaptive-irreducible" components.

### 2.3 Evidence verdict

| Claim | Strength | Evidence |
|---|---|---|
| Per-realization schedule preference exists | **Moderate** | E2 top-10/bottom-10 MC Jaccard ≈ 1.2× random |
| The residual gap above open-loop search is large | **Weak** | E1 alone can't disambiguate adaptive-removable vs irreducible |
| Aggregate signals read adaptively would beat open-loop | **Weak-to-negative** | A2 signals are weak unconditionally |
| An amortised adaptive policy could be **cheaper** than BS-AG | **Moderate** | E4 — deployability argument |
| An adaptive policy could be **more accurate** than CD-G | **Speculative** | A1 — little headroom unless fine-grained z_t features help |

**Overall: evidence supports "adaptive is a reasonable next-stage question," not "adaptive is necessary now."**

---

## 3 — What evidence is still missing

Moving to adaptive before the following are measured would be premature.

### 3.1 Missing empirical measurements (ranked by importance)

1. **True G(S_B*) — Q12 in `open_questions.md`.** Without the true open-loop oracle, the gap "adaptive vs open-loop" is unestimable. Phase 2b MC upper envelope is a lower bound on G(S_B*); for small B (B=2, 3, 4) exhaustive enumeration is feasible on a handful of trajectories (C(64, 4) ≈ 635k, minutes of CPU).

2. **State-conditional Δ_t(z_t) beyond the time index.** Protocol A gives Δ_t at each t, averaged over the realized z_t at that step. It does **not** cross-tabulate Δ_t against low-dimensional state summaries (unmasked fraction u_t, committed-set statistics, etc.). A single heatmap Δ_t × u_t would show whether the within-t variance of Δ_t is explained by u_t — a cheap prerequisite for any adaptive argument.

3. **Variance partitioning with an oracle adaptive policy.** Even an oracle "pick the top-B steps in this MC realization" computed post hoc per seed would tell us the upper bound on what **any** adaptive policy can achieve on this data. If oracle-adaptive only narrowly beats oracle-open-loop, the adaptive direction is bounded above empirically. This is a small offline analysis on existing Phase 2b MC data.

4. **Calibration of a state-aware proxy ψ(z_t).** If the trivial state feature u_t (unmasked fraction) correlates with Δ_t better than aggregate signals do, that would be first genuine evidence that state-dependent information beats trajectory-level information. No such analysis exists in the repo.

5. **Search-class gap on LLaDA-SFT.** The currently-running Phase 2b on LLaDA-SFT will produce K=8 paired cells; if Phase 3a is eventually run on LLaDA-SFT it would give the CD-G / BS-AG closure on a second backbone. That closure number is the "open-loop ceiling" against which adaptive must be compared on an independent platform.

### 3.2 Missing theoretical pieces

- **Adaptive proxy-regret inequality.** No analogue of Theorem A exists for adaptive policies in the repo. At minimum one needs a Bellman-decomposition regret of the form

      V*(z_0, B) − V^π̂(z_0, B)  ≤   f(ε_adaptive, η_adaptive, B),

  where ε_adaptive measures calibration of the state-action value function Q̂ and η_adaptive measures the deviation between the true MDP dynamics and the factorised/Markov approximation used by π̂. Until such a statement is written even once, the adaptive direction is a programme, not a theorem.

- **Identification of the state abstraction.** Is z_t the full masked sequence? The committed subset and a coarse context summary? The full latent posterior? Different abstractions give different regret decompositions. The memos gesture at this but no concrete choice is made.

- **Budget-depletion structure.** b_t is deterministic given past actions; this makes the problem a finite-horizon constrained MDP in a specific benign form (one resource, monotone depletion). Conditions under which dynamic-programming approximation is tractable (e.g., action-sparsity, short effective horizon) have not been identified.

### 3.3 Missing experimental infrastructure

- Δ_t conditioned on z_t (§3.1.2 above) requires no new runs — just reanalysis of existing Phase 2b trajectory JSONs.
- Oracle-adaptive vs oracle-open-loop (§3.1.3) likewise uses existing MC data.
- Any actual adaptive **policy learning** would require new runs (offline RL from Phase 2b triplets (z_t, a_t, Δ_t), or online rollouts). That is real infrastructure, not reanalysis.

---

## 4 — Dangers of moving too quickly to adaptive

### 4.1 Premature framework escalation

This is the danger most explicitly flagged in the existing audits.

- **`ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md`** explicitly warns of "premature framework escalation" — moving to MDP / SMC / Particle Gibbs machinery before the open-loop-search story is complete.
- **`INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md`** classifies the adaptive direction as "plausible but premature" and recommends treating it as a future-work section.
- **`PRINCIPLED_NEXT_STEPS_PLAN.md`** recommends Option 2 (stronger diagnostics on existing artefacts) + Option 5 (pairwise interaction model), postponing adaptive/SMC to Option 6.

Three audits, three convergent warnings. Overriding them requires new evidence or new reasoning; neither is in hand.

### 4.2 Theorem loss

Theorem A is currently the thesis's main theoretical result. It is:

- proved under explicit assumptions,
- decomposed (A′ / A″ refinements, B / C propositions),
- empirically anchored (ε, η_B, γ measured on Phase 2b),
- cleanly corrector-agnostic.

An adaptive pivot risks relegating Theorem A to "intermediate result" and introducing a weaker, less mature adaptive result as the headline. That is a real thesis-quality risk at the T-6-month stage of an MSc thesis, not an abstract risk.

### 4.3 Scope creep into token-selection / policy learning

The thesis explicitly disavows policy learning (§Non-Goals in `thesis_direction.md`: "Derive a learned end-to-end optimal policy over all inference choices"). Adaptive control, done seriously, is learned control. The boundary between "signal-adaptive scheduling" and "learned scheduling policy" is easily crossed by accident.

### 4.4 Sample-complexity blowup

An open-loop top-B selector needs calibration of T scalar signals against T scalar gains — K seeds × T = O(K·T) samples suffice. An adaptive Q̂(z, a) function over even a coarse z-abstraction needs O(K·T·|Z|) samples where |Z| can be very large. On K=8 / K=30, this is simply infeasible to fit without severe pooling assumptions, at which point the adaptive policy collapses back toward an open-loop one anyway.

### 4.5 Opportunity cost on the running Phase 2b job

Phase 2b on LLaDA-SFT is running now (Jobs 481264 + 481265). Its results will either strengthen or weaken the open-loop-search story on a second backbone. If the Phase 2b result is a weak NULL, re-framing the thesis as adaptive to "save" the narrative is the kind of post hoc pivot that reviewers and supervisors rightly penalise. Better: let Phase 2b finish, report PASS / NULL / FAIL honestly per the existing contract (`LARGE_MODEL_CONTINUATION_DECISION.md`), then decide whether adaptive is a next-stage extension.

### 4.6 Loss of portability claims

The current thesis narrative rests on "ranker-class fails, search-class partially succeeds, across at least one backbone family." That claim is corrector-kernel-agnostic and generalises cleanly. An adaptive-policy claim is tied to a z-abstraction choice and a function-class choice — two more points of configuration that must be carried through every experiment.

---

## 5 — Does the thesis need adaptive control for a strong conclusion?

### 5.1 Three-part contribution without adaptive

The thesis's current three-part contribution (per `CANONICAL_RESEARCH_DIRECTION.md` and `CURRENT_INDEX.md`) is defensible without adaptive:

1. **Theory.** Theorem A (proxy-regret bound) + Refinements A′ (variance form) + A″ (rank form) + Propositions B, C. All open-loop, all proved under explicit assumptions, all anchored in measurable constants (ε, η_B, γ).

2. **Experiment.** Phase 2a (ε, η_B calibration). Phase 2b (ranker-class negative, paired). Phase 3a (search-class positive, CD-G / BS-AG closure). Optional Phase 2b-LLaDA-SFT (external validity probe, bounded K=8, running now).

3. **Positioning.** The five-way distinction (informed correctors vs remasking vs predictor schedules vs token-selection vs corrector scheduling) + ProSeCo audit showing the thesis provides the theoretical framework ProSeCo lacks.

None of these pieces requires adaptive control.

### 5.2 What adaptive would add (if included prematurely)

- A weaker theorem (adaptive proxy-regret; not currently proved).
- A new experimental axis (Q̂ learning) that needs its own design, infrastructure, and calibration.
- A broader scope that reviewers may find more impressive but also more diffuse.

### 5.3 What adaptive legitimately offers (as next-stage work)

- A principled deployability story (inference-time policy that does not need G).
- A framework to explain the 16–26% residual gap between CD-G and the open-loop oracle.
- A bridge to the broader diffusion-control literature (Feynman–Kac/SMC, Particle Gibbs, budgeted MDPs).

**Verdict for the thesis itself.** **Optional next-stage work.** The thesis's strong-conclusion path is open-loop. The adaptive extension is an appropriate **Future Work chapter** or a **short appendix** that sketches the generalisation and the open empirical questions. Moving it to thesis-core now would degrade, not improve, the overall contribution.

### 5.4 What changes this verdict

Three events, individually sufficient, would force a reconsideration:

1. **Phase 2b NULL on LLaDA-SFT with no obvious methodological cause.** If the open-loop story fails to replicate, the adaptive direction becomes a legitimate rescue candidate — but only if preceded by a fresh design-of-experiments thought (see §6 below).
2. **Direct empirical measurement (from existing data) that state-conditional Δ_t(z_t) has substantially lower variance than marginal Δ_t.** This would turn §3.1.2 into positive evidence and shift the balance.
3. **A closed-form or near-closed-form adaptive proxy-regret theorem appearing in the literature for masked diffusion or close analogue.** Phase 2 literature search (next phase of this study) will determine whether such a result already exists.

---

## 6 — Exact research question the adaptive extension should ask (if pursued)

### 6.1 Badly-posed versions to avoid

- "Does adaptive beat open-loop?" — too vague; every adaptive policy class beats some open-loop baseline somewhere.
- "Can we train a policy π(z_t, b_t)?" — yes, trivially; policy-fitting is not research.
- "Is there a contraction bound for adaptive correctors?" — the contraction question is already covered (and demoted to stretch) in Theorem A's ecosystem; no reason to re-open it here.
- "Is adaptive optimal?" — optimality is framework-dependent and empirically unverifiable at thesis scale.

### 6.2 Well-posed framings

Each of the following is defensible as a research question. They are listed in order of increasing scope.

**(Q-A, narrowest).** Given the Phase 2b empirical fact that the best budget-B schedule is MC-realization-specific (top-k/bottom-k Jaccard ≈ random), does there exist a cheap state-dependent scoring function ψ(z_t) such that the online policy π̂ "select top-B by ψ as b_t depletes" closes a measurable fraction of the gap between open-loop search (CD-G/BS-AG) and the true G(S_B*) oracle?

- **Testable object:** ψ : state features → scalar, benchmarked by expected G under rollout.
- **Smallest valid experiment:** offline reanalysis of Phase 2b MC data + simple ψ fits (linear in u_t and signal features).
- **What it delivers:** a quantified answer to "how much headroom does adaptive buy on the data we already have?"
- **What it does not deliver:** a new theorem, unless a calibrated ψ comes with a regret bound.

**(Q-B, moderate).** Under what assumptions on (i) the state-action value Q(z, a) factorisation, (ii) a calibrated state-dependent proxy ψ̂(z_t, b_t), and (iii) the approximation error between the true finite-horizon constrained MDP and a tractable surrogate, does an online greedy policy π̂(z_t, b_t) := arg max_{a ∈ {0, 1}} ψ̂(z_t, b_t, a) achieve a regret bound analogous to Theorem A?

- **Deliverable:** an adaptive Theorem A' under explicit assumptions, empirically anchored via the same ε/η/γ discipline.
- **Risk:** the adaptive assumptions may be harder to verify than the open-loop ones, weakening the bound.

**(Q-C, broadest — not recommended for this thesis).** For masked diffusion language models, what is the smallest-complexity controller class Π such that some π ∈ Π approximates the optimal finite-horizon budgeted path-control policy to within a measurable loss, and what normative framework (budgeted-MDP / Feynman–Kac / control-as-inference) is tightest for bounding that loss?

- **Deliverable:** a normative survey + empirical benchmark.
- **Risk:** PhD-sized.

### 6.3 Recommended framing

**If** adaptive is pursued as a thesis next-stage item (Future Work chapter or short appendix), the correct framing is **Q-A** — narrowest, data-bound, reanalysis-driven. It makes a falsifiable claim on existing data before any new theory is written.

**If** adaptive is pursued as a post-thesis research programme, **Q-B** is the right level.

**Q-C is explicitly out of scope.**

---

## 7 — Summary and recommendation

| Question | Answer |
|---|---|
| Is adaptive a natural extension? | **Yes, mathematically.** Also yes operationally, but only as generalisation — the open-loop proof does not transfer unchanged. |
| Is adaptive empirically justified by existing repo evidence? | **Suggestive, not sufficient.** E1, E2 give moderate hints; A1–A6 substantially constrain the expected upside. |
| Is evidence missing? | **Yes, substantially.** True G(S_B*), state-conditional Δ_t(z_t), oracle-adaptive upper bound — none are measured. All three are cheap offline reanalyses, not new runs. |
| Is moving quickly dangerous? | **Yes.** Premature framework escalation; theorem loss; scope creep; sample-complexity blowup; opportunity cost on Phase 2b; loss of portability. |
| Does the thesis need adaptive? | **No.** Thesis conclusion is strong without adaptive. Adaptive is appropriate as Future Work chapter or short appendix. |
| If pursued, what question? | **Q-A** (narrow, data-bound, reanalysis-driven) — not Q-B or Q-C. |

### 7.1 Operational recommendation for Phase 1 of this research study

1. **Do not pivot** the thesis to adaptive now.
2. **Proceed with Phase 2** (deep literature search on adaptive-controller frameworks) strictly under **research-study discipline**: the purpose is to identify 2–4 strong frameworks for a Future Work chapter, not to find a new thesis direction.
3. **Phase 3 (mathematical adaptation)** should work in Q-A's direction: state the adaptive analogue of Theorem A for the **narrowest** useful problem (binary actions, finite horizon, scalar budget), and identify what would have to hold empirically for the statement to be non-vacuous.
4. **Phase 4 (framework selection)** should pick **one normative + one algorithmic** framework, with explicit acknowledgement that the picks are for Future Work positioning, not for thesis-core proof.
5. **Phase 5 (write-up)** should place the output in `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` as a **Future Work** essay, clearly marked as such, with the caveat banner matching this audit.
6. **Phase 6 (recommendation)** should be unambiguous: **next-stage, not current-thesis**, unless Phase 2b produces a catastrophic NULL.

### 7.2 What would change this recommendation

- Phase 2b on LLaDA-SFT completes with a statistically clean FAIL and the core open-loop story cannot be recovered by Protocol A diagnostics alone.
- A preprint in the literature search (Phase 2) supplies a closed-form adaptive proxy-regret bound for masked-diffusion scheduling that can be cleanly applied to this thesis's data.
- A cheap offline reanalysis (Δ_t(z_t) heatmap at trivial z-features) shows that state-conditional variance is 10× smaller than marginal variance — i.e., adaptive is clearly useful on this data.

Until one of these three happens, **the audit holds: adaptive is next-stage work, not the thesis's current mainline.**

---

## Appendix A — Cross-reference map

| Audit question | Primary evidence document | Secondary |
|---|---|---|
| Thesis scope & non-goals | `docs/thesis_direction.md` §3, §6 | `CANONICAL_RESEARCH_DIRECTION.md` |
| Current theory object | `research/candidate_theorems.md` (Theorem A) | `THEORY_STATUS.md` |
| Phase 2b within-seed variance | `results/phase2b/combinatorial_diagnostics.json` | `INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md` §3.1 |
| Phase 3a ranker-negative + search-positive | `PHASE3A_COMBINATORIAL_RESULTS.md` | `THEORY_STATUS.md` (Negative-Result Corollary) |
| True S_B* open question | `research/open_questions.md` Q12 | `THEORY_STRESS_TEST.md` §8 |
| Existing "adaptive/MDP/SMC" verdict | `INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` | `ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md`, `PRINCIPLED_NEXT_STEPS_PLAN.md` |
| Existing memos being audited | `docs/future ideas/Theoretical Frameworks for Budgeted Informed-Corrector Scheduling in Masked Diffusion Language Models.md` | `docs/future ideas/Deep Research Audit of Budgeted Informed Corrector Scheduling.md` |
| Phase 2b running contract | `LARGE_MODEL_CONTINUATION_DECISION.md`, `PHASE2B_RESUME_PLAN.md` | `CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md` |

## Appendix B — Provenance

- **Prior beliefs at start of this audit:** the user's initial prompt framed adaptive as the "most promising extension." That framing is **not accepted by this audit**; the audit's independent conclusion is "most promising next-stage extension," not "most promising thesis pivot."
- **Skeptical stance:** this document was written before any new literature search (Phase 2), specifically to avoid literature-induced framework advocacy contaminating the audit.
- **Dependency on standing audits:** the 2026-04-21 INDEPENDENT_AUDIT was read and its classifications are adopted where relevant; divergences are explicit (this audit is slightly more sympathetic to Q-A-level pursuit than the standing audit).
- **What this audit does not cover:** specific framework comparisons (Phase 4), specific proofs (Phase 3, Phase 5), recommendation on framework number (Phase 6). Those are later phases.
