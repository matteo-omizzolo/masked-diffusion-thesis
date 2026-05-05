> TECHNICAL WORKLOG.
> Current thesis status starts at `START_HERE.md`.
> Current theory-first plan: `docs/06_theory_first_research_plan.md`.

# Open Questions — Current

**Updated:** May 2026 (post-Phase-3b, post-Protocol-C).
Most pre-Phase-2b questions are resolved. Only genuinely unresolved technical
points remain below.

---

## Active open questions (theory-first phase)

### OQ-TF1 — Theorem B: pairwise surrogate regret

Formalize the bound on regret of pairwise surrogate scheduling ψ₂(t,t') vs. the joint
oracle G({t,t'}). What are the key assumptions? Does it reduce to Theorem A when interactions
are negligible?

**Status:** Not yet formalized. Planned in `docs/06_theory_first_research_plan.md` §2.

---

### OQ-TF2 — Proposition C: separable ranker failure construction

Find an explicit construction (noise level, interaction pattern) where any separable
per-step ranker achieves constant additive regret independent of B. Relate to the
measured σ_ξ and Phase 2b failure.

**Status:** Not yet written. Empirical evidence in `results/phase2b_proseco_owt/`.

---

### OQ-TF3 — Theorem D: online budgeted controller abstraction

Define the budgeted online decision problem. Characterize the value function. Bound the
gap to the offline oracle as a function of state space richness and budget B.

**Status:** Not yet formalized. Protocol C is an honest empirical negative; Theorem D
is the formal companion. Planned in `docs/06_theory_first_research_plan.md` §4.

---

### OQ-TF4 — Regime diagnostics

Define measurable quantities that classify a trajectory as:
- "additivity-regime": interactions negligible, ranker failure is calibration-only;
- "interaction-regime": pairwise ξ_{t,t'} structured, pairwise scheduler justified;
- "online-regime": information arrives too late for open-loop scheduling.

**Status:** Not yet defined. Phase 0 audit precedes this.

---

### OQ-TF5 — Phase 0 reproducibility gate

Can we reproduce the Phase 2b and Phase 3a results locally (K=3 smoke, then K=30)
before launching new HPC experiments?

**Status:** Phase 0 checklist in `docs/05_next_steps.md`. Not yet run.

---

## Active open questions (writing phase)

### OQ-1 — Theorem A proof prose (LaTeX write-up)

The combining-step argument (Lemma A1 + Lemma A2 + two applications of
approximate additivity) is sketched in `research/proof_worklog.md` Entry 6
but not yet a clean LaTeX narrative in `thesis/chapters/ch6_contribution.tex`.

Two steps need careful notational treatment:
1. Route the exchange argument through the additive surrogate A(S) := ∑ Δ_t,
   not G directly.
2. Apply two instances of assumption (2) to connect A(S_B*) and A(Ŝ_B) back to
   G(S_B*) and G(Ŝ_B).

**Status:** Skeleton exists; section bodies marked TODO.

---

### OQ-2 — Refinement A′ formal derivation

The variance-form η_B ≤ σ_ξ · √B/√2 is empirically motivated (measured on 9000 MC
rows) and stated as a theorem in `research/candidate_theorems.md`, but the formal
proof under the mixing/cancellation hypothesis on (ξ_{t,t'}) needs a clean
order-statistics write-up in ch6.

**Status:** Formal statement locked; proof derivation exists; LaTeX prose TODO.

---

### OQ-3 — Refinement A″ formal derivation

The rank-based ε_R := (1 − |ρ_B|) · σ_Δ is empirically anchored but the
formal derivation under the Gaussian-A hypothesis (that A(Ŝ_B) − A(S_A*) has
a known order-statistics distribution) needs a clean proof in ch6.

**Status:** Formal statement locked; proof derivation exists; LaTeX prose TODO.

---

### OQ-4 — Negative-Result Corollary formal statement

The corollary (any separable per-step ranker is bounded by the mean_delta_oracle
envelope, which enters the NULL band at B = 8 on OWT) needs a formal corollary
environment in ch6 with a proof citing the Phase 2b paired CIs.

**Status:** Empirically anchored; formal statement in `research/candidate_theorems.md`;
LaTeX corollary environment TODO.

---

### OQ-5 — External validity (single-backbone caveat)

All primary results are on ProSeCo-OWT. The LLaDA-SFT bounded probe was
inconclusive (T3, K=8). Whether CD-G/BS-AG recovery rates transfer to a second
backbone is unknown.

**Status:** Not addressed in main thesis; mentioned as a limitation.
Not authorized for new experiments without supervisor approval.

---

## Resolved questions (summary only)

| Question | Resolution |
|---|---|
| Approximate additivity realistic? | σ_ξ measured (0.174/0.240/0.309 at B=2/3/4); η_B via Refinement A′ |
| Entropy as proxy? | Spearman ρ ≈ 0.10–0.15; all three signals similar; ε_R measured |
| True G(S_B*)? | MC oracle (best-of-100) used as practical upper bound at B∈{2,3,4} |
| Corrector definition? | ProSeCo annealed refinement; 2 NFEs per loop |
| ProSeCo novelty? | Confirmed: ProSeCo does not provide proxy-regret or Δ_t measurements |
| ε_R as calibration? | Adopted as Refinement A″ |
| √B vs B² bound? | A′ (√B form) is tighter; adopted |
| TCR ≠ Δ_t? | Both measured in Phase 1; Δ_t used exclusively |
| Choice of F? | F = −GPT-2 NLL on 512-token window |
| Budget sensitivity? | Phase 2b covers B∈{2,3,4,8,16}; saturation at B=8 |
| Δ_open, ε shrinkage (adaptive)? | Protocol C: Δ_open > 0 confirmed; ε̃/ε ≈ 0.983–0.986 (no meaningful shrinkage) |

Full resolution details: `research/proof_worklog.md` Entries 5–8.
Old 416-line version archived at `docs/archive/repo_cleanup_20260505/old_research_worklogs/open_questions_pre_cleanup.md`.
