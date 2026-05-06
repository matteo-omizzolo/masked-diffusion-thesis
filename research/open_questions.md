> TECHNICAL WORKLOG.
> Current thesis status starts at `START_HERE.md`.
> Current theory-first plan: `docs/06_theory_first_research_plan.md`.
> Theorem stack: `research/candidate_theorems.md` §0–§7.

# Open Questions — Current

**Updated:** 2026-05-06 (post theory-correction pass).
The theorem stack is formalized; remaining questions concern empirical
estimation of bound constants, statistical stability of regime diagnostics,
and the literature-freshness check before the final thesis claim.

---

## Active open questions — theory-first programme

### OQ-T1 — Theorem B at which level of analysis?

Theorem B can be instantiated at three levels (`candidate_theorems.md` §2.4):
Level 1 diagnostic (uses true counterfactual G({t,t'})), Level 2 population
(optimises Q̄ for one global schedule), Level 3 feature-conditioned (Q̂_i
from observable features, no true G_i at inference). Open: at which level
does Theorem B's hypothesis hold for ProSeCo-OWT? The thesis can claim a
deployable scheduler **only at Level 3**.

Tests: Phase 1 settles Levels 1–2; Phase 2b settles Level 3.

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

Open: a clean estimation protocol that reports each constant separately with
paired BCa CIs and a no-leakage pool construction.

### OQ-T3 — Statistical stability of Diagnostic Framework C at K=30 (with pair / pool sampling)

Diagnostics U_B^{MC,N}, R_B, I_B, P_B^{level}, C_B^{MC,N} all carry BCa CIs
over seeds. But P_B^{level} and ζ_{B,C} also depend on pair / schedule
sampling; bootstrapping only over seeds while pair / schedule samples are
sparse understates uncertainty. Open:

- At K=30 paired seeds and the planned pair-pool size, does the **nested
  bootstrap** (seeds × sampled pairs/schedules) CI for P_B − R_B exclude 0?
- If borderline, K=60 may be needed; alternatively grow the pair pool.
- Do not classify Regime III vs IV unless the CI for P_B − R_B (or
  ζ_{B,C} − η_{B,C}) is stable under both seed-bootstrap and pair-pool
  resampling.

### OQ-T4 — Online-state sufficiency for Theorem D

Theorem D is useful only if some compact z_t admits small ‖V − V̂‖_∞.
Protocol C (bucketed (signal_quartile, phase) state) found this fails.
Open: does adding (b_t, u_t, continuous H_t/M_t^{-1}) recover predictive
value? Appendix-grade unless Phase 4 demonstrates it.

### OQ-T5 — Secondary backbone for Diagnostic Framework C external validity

Which (model, corrector) pair is feasible **and** likely to lie in a
different regime than ProSeCo-OWT? Candidates: ReMDM-conf, MDLM with a
non-Gibbs partial-resample corrector, LLaDA-SFT (only after the Tier 3
protocol issue is fixed). Required: enough headroom (U_B > 0) to make
the diagnostic comparison meaningful.

### OQ-T6 — PRISM feasibility (decision rule)

Without pretrained weights, can a usable quality head be trained in 1–2
weeks? Decision rule: if no, cite as related work and use QM_t (existing
PRISM-style mass) as a candidate signal. Do not pivot the thesis around
PRISM. A non-separable PRISM use is not pursued in this thesis but is not
ruled out by the Empirical Ranker-Class Limitation (`candidate_theorems.md` §1.5).

### OQ-T7 — Minimum experiment set for August writing freeze

Working candidate: Phase 0 + Phase 1 sparse pairwise + Phase 2a population
pairwise. Phase 2b feature-conditioned only if Phase 2a beats rankers.
If Phase 2a fails, the thesis claim becomes "tested separable rankers do not
recover MC-oracle headroom; pairwise also fails on this regime; Diagnostic
Framework C explains why; CD-G provides an existence proof." Do not enter
Phase 4 (Theorem D / online controller) unless Phase 1/2 succeed and time
remains.

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
| OQ-W1 | ch6 LaTeX prose for Theorem A combining step | TODO; defer until after Phase 0 results |
| OQ-W2 | ch6 LaTeX prose for Diagnostics A′, A″ | TODO; diagnostics only, not theorem variants |
| OQ-W3 | ch6 LaTeX environment for Empirical Ranker-Class Limitation (formal time-only part + scoped empirical part) | TODO |
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
