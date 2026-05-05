> TECHNICAL WORKLOG.
> Current thesis status starts at `START_HERE.md`.
> Current theory-first plan: `docs/06_theory_first_research_plan.md`.
> Theorem stack: `research/candidate_theorems.md` §0–§7.

# Open Questions — Current

**Updated:** 2026-05 (post theory-first formalization).
The theorem stack has been formalized; remaining questions concern *empirical
estimation* of the bound constants and *regime classification*. Older writing-phase
LaTeX-prose questions are tracked in §6 below as a thin status table.

---

## Active open questions — theory-first programme

### OQ-T1 — Theorem B with estimated Q̂

The estimated-Q̂ form has constant 2α_B (derived in `candidate_theorems.md` §2.2).
Open: under **leave-one-seed-out** estimation of (Δ̂, ξ̂), is the empirical α_B small
enough that 2α_B does not swamp the (η_B − ζ_B) improvement?

Tests: cross-validated estimate of α_B on Phase 1 sparse pair data; compare to
measured ζ_B and η_B.

---

### OQ-T2 — Estimating ζ_B, α_B, ω_B without leakage

The three constants have orthogonal sources of error:
- ζ_B: pairwise approximation error (model bias).
- α_B: surrogate estimation error (finite training pool).
- ω_B: optimization gap of the scheduler.

We need an estimation protocol that reports each separately with paired-bootstrap CIs
and that avoids using test G to fit Q̂. Phase 2 train/test split is the obvious vehicle;
ω_B is reported by the optimizer.

---

### OQ-T3 — Statistical stability of regime diagnostics at K=30

For Proposition C the diagnostics U_B, R_B, I_B, P_B, C_B all have BCa CIs over seeds.
Open: at K=30 paired seeds, do the CIs separate adjacent regimes (e.g. III vs IV)
on ProSeCo-OWT? If not, K=60 may be needed for Phase 1.

---

### OQ-T4 — Sufficiency of compact online state z_t

For Theorem D to be useful, the compact state z_t must carry enough value-function
information that ‖V − V̂‖_∞ is small. Protocol C found this fails for the bucketed
(signal_quartile, phase) state. Open: does adding (b_t, u_t, continuous H_t/M_t^{-1})
recover predictive value? This is appendix-grade unless Phase 4 demonstrates it.

---

### OQ-T5 — Secondary backbone choice for Proposition C external validity

Which (model, corrector) pair is feasible *and* expected to lie in a different regime
than ProSeCo-OWT? Candidates: ReMDM-conf, MDLM-conf with partial resample, LLaDA-SFT
(only after the Tier 3 protocol issue is fixed). Required: enough headroom (U_B > 0)
to make the diagnostic comparison meaningful.

---

### OQ-T6 — PRISM feasibility

Without pretrained weights, can a usable quality head be trained in 1–2 weeks?
Decision rule: if no, cite as related work and use Q_t (existing PRISM-style mass)
as a signal candidate; do not pivot the thesis around PRISM.

---

### OQ-T7 — Minimum experiment set for August writing freeze

What is the smallest experiment set that defends the central thesis claim?
Working candidate: Phase 0 + Phase 1 sparse pairwise + Phase 2 held-out pairwise
scheduler on ProSeCo-OWT only. If Phase 2 succeeds, the thesis claim is "interaction-aware
scheduling on a single backbone, with regime-diagnostic protocol for transfer."
If Phase 2 fails (regime IV), the thesis claim is "rankers fail; pairwise also fails;
diagnostic framework explains why; CD-G provides existence proof."

---

## Carry-over technical questions

### OQ-W1 — ch6 LaTeX prose for Theorem A combining step

The combining-step argument is in `research/proof_worklog.md` Entry 6 and
`candidate_theorems.md` §1.2. LaTeX prose in `thesis/chapters/ch6_contribution.tex`
remains TODO.

### OQ-W2 — ch6 LaTeX prose for Refinements A′, A″ and Negative-Result Corollary

Formal statements locked. LaTeX prose TODO. Defer until Phase 0/1 results are in
so the empirical anchors can be cited.

### OQ-W3 — External validity caveat scoping

Phrase the single-backbone caveat in ch7 *after* Phase 1 outcome is known; the
language differs between regime III (interaction-driven, generality unknown) and
regime IV (chaotic, generality presumably negative).

---

## Resolved questions (summary only)

| Question | Resolution |
|---|---|
| Approximate additivity realistic? | η_B measured (σ_ξ at 0.174/0.240/0.309 for B=2/3/4); A′ refinement |
| Entropy as proxy? | Spearman ρ(ψ,Δ) ≈ 0.10–0.15; ε_R = 0.07→0.39 across B; rankers fail |
| True G(S_B*)? | MC oracle (best-of-100) used as practical upper bound at B ∈ {2,3,4} |
| ProSeCo novelty? | ProSeCo provides no Δ_t / proxy-regret / scheduling theory — confirmed |
| L∞ ε vs ε_R? | ε_R adopted as Refinement A″ (operative form) |
| √B vs B² bound? | A′ (√B form) is tighter; adopted |
| Choice of F? | F = − GPT-2 NLL on 512-token window |
| Budget sensitivity? | B ∈ {2,3,4,8,16}; ranker saturation at B=8 |
| Adaptive (Protocol C) shrinkage? | ε̃/ε ∈ [0.983, 0.986] — no shrinkage; honest negative |
| Theorem B exact-Q form? | Proved (`candidate_theorems.md` §2.1) |
| Theorem B estimated-Q̂ constant? | 2α_B (not 4α_B); proof in §2.2 |
| Theorem D constant? | 2Tδ; honest about not having cBδ in general |

Full provenance: `research/proof_worklog.md` Entries 5–8;
`research/proof_ledger.md`.
