> TECHNICAL WORKLOG.
> Current thesis status starts at `START_HERE.md`.
> Current theory-first plan: `docs/06_theory_first_research_plan.md`.
> Theorem stack: `research/candidate_theorems.md` §0–§7.

# Open Questions — Current

**Updated:** 2026-05-11 (post Gate 6).
The direction is locked around statistical correction timing:
estimate correction value, test state predictability, and use interaction /
search diagnostics to explain failures of simple timing rules. Gate 6 is a
held-out state-predictability negative on ProSeCo-OWT; online scheduler work is
not justified from current features.

---

## Active open questions — theory-first programme

### OQ-T1 — Theorem B at which level of analysis?

Theorem B can be instantiated at three levels (`candidate_theorems.md` §2.4):
Level 1 diagnostic (uses true counterfactual G({t,t'})), Level 2 population
(optimises Q̄ for one global schedule), Level 3 feature-conditioned (Q̂_i
from observable features, no true G_i at inference). Gate 3b answer so far:
useful at B=2 for one held-out estimator, but not useful for the tested
deployable low-dimensional Q models at B ∈ {3,4}. The thesis can claim a
deployable scheduler **only at Level 3**, and the current evidence does not
authorize that claim.

Status: Phase 1 settles this as an honest negative for cheap pairwise
composition at B=3/4. A new Theorem-B scheduler attempt would need a new spec,
not a continuation of the current gate.

### OQ-T2 — Estimating ζ_{B,C}, α_{B,δ}, ω_B, κ_B without leakage; no-leakage pool construction

Constants split into orthogonal sources of error:
- ζ_{B,C} — pairwise approximation bias on candidate pool C_B.
- α_{B,δ} — surrogate estimation error on C_B (high probability over training
  pool); use leave-one-seed-out.
- ω_B — optimizer gap, reported by the optimizer.
- κ_B — pool-vs-MC-pool gap (estimable only against the MC pool, never
  against the exhaustive (T choose B) oracle).

**No-leakage pool construction (Theorem B′ data-dependence caveat).** The
candidate pool C_B must be **fixed independently of held-out evaluation G**.
Acceptable: random sampling (MC pool); beam candidates from training-seed
Δ̂, ξ̂; Q̂-greedy candidates from training-seed Q̂. **Not acceptable**:
pools selected using held-out G or by an optimizer with G-feedback on test
seeds. If C_B depends on training data, evaluate on held-out seeds; if it
depends on test G, the bound does not apply.

Status: a clean local protocol now exists for B=2 and B=3/4. The open part is
not protocol correctness; it is whether denser pair measurements or richer
surrogates can overcome the B=3/4 failure. Do not spend more HPC on this
without a new theory/spec.

### OQ-T2b — What set-function structure does G(S) have?

After Gate 3b, the central technical question is no longer whether sparse
pairwise residuals exist, but what kind of set function G(S) is:

- approximately monotone or non-monotone;
- diminishing-returns / weakly submodular, or genuinely non-submodular;
- high curvature, so greedy still works only with degraded guarantees;
- locally smooth under one-swap schedule neighborhoods;
- dominated by higher-order residuals that cannot be recovered by summed
  pairwise penalties.

Tests should use existing Phase 2b / Phase 3a / Gate 3b artifacts first. Only
after local diagnostics should any new HPC job be proposed.

Status: Gate 4 and Gate 5 are complete. True-G search should be reported as
an empirical probe showing that exploitable structure exists, not as the next
method to implement. Exact Gate 5 DR triples are mildly positive on average,
but the evidence is region-dependent and insufficient for a model-agnostic
submodularity claim.

### OQ-T2c — Is Bayesian schedule optimization more principled than greedy search?

Bayesian optimization is appealing because G(S) is expensive and schedules are
discrete. Open: can we define a kernel or acquisition function over schedules
that respects budget B, phase/distance structure, and uncertainty over G(S)
without becoming an engineering-heavy detour? This direction is attractive for
a Zanella-facing principled story, but it should follow set-function
diagnostics that clarify whether the search landscape is smooth enough for BO
to be useful.

Status: Gate 5 verdict is `BO_unclear`. A candidate GP over schedules can be
stated formally, but empirical support remains insufficient: distance/|ΔG|
rank signal is borderline and many high-quality anchors still have sampled
one-swap improvements. BO stays future work unless a later diagnostic shows a
stronger kernel-smoothness story.

### OQ-T3 — Statistical stability of Diagnostic Framework C at K=30 (with pair / pool sampling)

Diagnostics U_B^{MC,N}, R_B, I_B, P_B^{level}, C_B^{MC,N} all carry BCa CIs
over seeds. But P_B^{level} and ζ_{B,C} also depend on pair / schedule
sampling; bootstrapping only over seeds while pair / schedule samples are
sparse understates uncertainty. Open:

Status: mostly closed for the current thesis scope. Gate 3b gives a positive
B=2 result and a negative B=3/4 result for the tested deployable Q family; Gate
5 gives targeted local and DR diagnostics but not a theorem-grade positive.
Further nested-bootstrap work is only useful if a new surrogate family is
proposed.

### OQ-T4 — Online-state sufficiency for Theorem D

Theorem D is useful only if some compact z_t admits small ‖V − V̂‖_∞.
Protocol C (bucketed (signal_quartile, phase) state) found this fails. Gate 3b
also suggests that cheap offline pairwise composition is insufficient at
B=3/4. Open: does a value-function view V_t(b,z_t) recover predictive value
using budget b, phase, continuous signals, and trajectory summaries?
Appendix-grade unless set-function diagnostics justify moving from offline
search to online value approximation.

Status: Gate 6 tested held-out prediction of single-step correction value
using current Protocol A state features. Time+state ridge did not improve over
time-only ridge (MSE 0.016975 vs 0.016893; mean seed-level MSE reduction
−0.000082, 95 % CI [−0.000267,+0.000111]). This closes the simple online
trigger path for the current feature set.

### OQ-T5 — Secondary backbone external validity

Which (model, corrector) pair is feasible **and** likely to lie in a
different regime than ProSeCo-OWT? Candidates: ReMDM-conf, MDLM with a
non-Gibbs partial-resample corrector, LLaDA-SFT (only after the Tier 3
protocol issue is fixed). Required: enough headroom (U_B > 0) to make
the diagnostic comparison meaningful.

Status: open only if Prof. Zanella requires external validity. The next
experiment should be a small headroom/metric/corrector-degeneracy gate, not a
full scheduling run.

### OQ-T6 — PRISM feasibility (decision rule)

Without pretrained weights, can a usable quality head be trained in 1–2
weeks? Decision rule: if no, cite as related work and use QM_t (existing
PRISM-style mass) as a candidate signal. Do not pivot the thesis around
PRISM. A non-separable PRISM use is not pursued in this thesis but is not
ruled out by the Empirical Ranker-Class Limitation (`candidate_theorems.md` §1.5).

### OQ-T7 — Minimum experiment set for August writing freeze

Minimum set is now complete for the ProSeCo-OWT case study: Phase 0, K=30,
Gate 3a/3b, Gate 4/5, and Gate 6 state-predictability audit. The next step is
writing and supervisor approval of scope, not another scheduler.

### OQ-L1 — Literature freshness check before final thesis claim

Before final thesis claims, verify novelty positioning against current
masked-diffusion LM literature: ProSeCo, PRISM, entropy-adaptive
Gibbs / EAGS, Denoising Entropy, KLASS, DEMASK, and recent
trajectory-search / test-time-scaling work. Do not claim the area is
untouched; claim a specific gap in **fixed-budget trajectory-level
corrector timing** with a regime-diagnostic framework.

Status: deferred until July; flagged here to avoid forgetting.

---

## Carry-over LaTeX-prose questions (writing phase)

| OQ | Item | Status |
|---|---|---|
| OQ-W1 | ch6 LaTeX prose for Theorem A combining step | Initial full draft written; polish after Phase 0 results |
| OQ-W2 | ch6 LaTeX prose for Diagnostics A′, A″ | Initial full draft written; diagnostics only, not theorem variants |
| OQ-W3 | ch6 Empirical Ranker-Class Limitation (formal time-only part + scoped empirical part) | Initial full draft written; verify wording after Phase 0 re-confirmation |
| OQ-W4 | ch7 single-backbone caveat scoping (regime III vs IV language) | Defer until Phase 1 outcome |

---

## Resolved questions (summary only)

| Question | Resolution |
|---|---|
| Approximate additivity realistic? | η_B measured (σ_ξ at 0.174/0.240/0.309 for B=2/3/4); A′ is now a diagnostic scale, not a refinement |
| Entropy as proxy? | Spearman ρ(ψ,Δ) ≈ 0.10–0.15; ε_R is a diagnostic only; tested separable rankers do not recover headroom |
| Practical oracle estimate? | MC-oracle (best-of-100) used as practical upper bound at B ∈ {2,3,4}; **not** the exhaustive oracle |
| ProSeCo novelty? | ProSeCo provides no Δ_t / proxy-regret / scheduling theory — confirmed |
| L∞ ε vs ε_R? | ε_R is **not** a theorem constant. The safe selected-schedule statement is the finite-pool form of Theorem A (= Theorem B′ with Q := A). R_B = ρ(A,G) and (1−|ρ|)·σ_Δ are reported as A″ diagnostics only. |
| √B vs B² bound? | Superseded. A′ is now an additivity-scale diagnostic only; no √B regret theorem is active. |
| Choice of F? | F = − GPT-2 NLL on 512-token window; treated as relative within-run metric |
| Budget unit? | B = corrector-placement budget; B_NFE = c_corr · B (c_corr = 2 for ProSeCo) |
| Budget sensitivity? | B ∈ {2,3,4,8,16}; ranker saturation at B=8 |
| Adaptive (Protocol C) shrinkage? | ε̃/ε ∈ [0.983, 0.986] — no shrinkage; honest negative |
| Held-out state predictability? | Gate 6 negative: time+state ridge MSE 0.016975 vs time-only 0.016893; no support for simple online trigger |
| Theorem B exact-Q form? | Proved (`candidate_theorems.md` §2.1) |
| Theorem B estimated-Q̂ constant? | 2 α_B (not 4 α_B); proof in §2.2 |
| Theorem B′ high-prob form? | Proved over fixed, no-leakage candidate pool C_B (§2.3); κ_B against MC pool, never the exhaustive oracle |
| Theorem D constant? | 2Tδ; honest about not having c B δ in general |
| Proposition C → Diagnostic Framework C | Renamed; it is a definition + protocol, not a proven proposition |
| Q_t vs Q(S) collision | Resolved: signal is QM_t; Q(S) is the pairwise surrogate; historical files use `Q_t` |
| Refinement A′ status? | **Demoted to additivity-scale diagnostic** (was: variance-form regret refinement); does not control selected-schedule regret without finite-pool conversion |
| Refinement A″ status? | **Demoted to rankability diagnostic** (was: rank-form regret refinement); ε_R is not used as a theorem constant |
| Negative-Result Corollary status? | **Reframed as Empirical Ranker-Class Limitation**; formal part for time-only / seed-averaged separable ψ; empirical part on tested rankers; does not rule out feature-conditioned separable rankers, non-separable PRISM, pairwise / online / search policies |
| Phase 2b decision gate? | Run if **either** (G2a-pop) population P_B^pop > R_B^pop **or** (G2a-feat) ξ_{i,t,t'} feature-predictable on held-out seeds (R² ≥ pre-specified threshold) |
| Phase 1 uncertainty? | Nested bootstrap over (seeds, pair-pool) for P_B^{level}, ζ_{B,C}; do not classify regime unless CI of P_B − R_B and ζ_{B,C} − η_{B,C} stable |
| MC-oracle vs exhaustive oracle? | U_B^{MC,N} reports best-of-N pool oracle; U_B^* (exhaustive) is unobservable and never reported |

Full provenance: `research/proof_worklog.md`, `research/proof_ledger.md`.
