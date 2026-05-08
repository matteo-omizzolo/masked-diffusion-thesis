> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


# Phase 3 Alternative Plan — Combinatorial Baselines + Theory Finalisation

*Created: 2026-04-20.*
*Replaces the proposed PRISM pivot — see `PHASE3_DIRECTION_AUDIT.md` for the rejection rationale.*

> **Phase 3a result (2026-04-20).** **POSITIVE.** Job 479941 (K=30 paired,
> 4 shards × 4 A100s, wall-clock 7h59m) ran both **CD-G** (coordinate descent
> with true-G feedback, ≤ 65 G-calls per cell) and **BS-AG** (beam search
> width 8, cheap-A ranking on extension candidates with true-G rollouts on
> top 8) at B ∈ {2, 3, 4, 8}. Both methods PASS at every budget with paired
> BCa CIs strictly above 0:
>
> | Method | B = 2 | B = 3 | B = 4 | B = 8 |
> |---|---|---|---|---|
> | CD-G Δ | +0.356 PASS | +0.327 PASS | +0.380 PASS | +0.322 PASS |
> | BS-AG Δ | +0.289 PASS | +0.252 PASS | +0.220 PASS | +0.153 PASS |
> | CD/oracle | 78.9 % | 74.1 % | 84.3 % | n/a |
> | BS/oracle | 64.1 % | 57.1 % | 48.8 % | n/a |
>
> Both methods clear the **"≥ 50 % of MC-oracle headroom"** decision rule at
> B ∈ {2, 3, 4} (CD-G at all three; BS-AG at B ∈ {2, 3} and just under at
> B = 4). Both PASS at B = 8 even though the MC oracle was not anchored
> there — i.e. the search class exceeds the ranker envelope at the budget
> where `mean_delta_oracle` itself is NULL. Thesis story per the decision
> table: *"scheduling has learnable combinatorial structure → scheduling is
> search, not signals; chapter on combinatorial corrector scheduling lands."*
>
> Caveats (all flagged in `PHASE3A_COMBINATORIAL_RESULTS.md`):
> CD-G's true-G feedback is not deployable (existence/structural result);
> BS-AG is closer to practical (cheap-A ranking + O(B) true-G rollouts) but
> still requires a true-G evaluator; ProSeCo-OWT only — no external validity;
> no theoretical guarantee on closure ratio.
>
> **Phase 3b consequence:** The Negative-Result Corollary as stated in the
> §"Phase 3b" block below ("no greedy single-step ranking policy can satisfy
> 𝔼[G(π_signal)] − 𝔼[G(uniform)] > 2σ_F for B ≥ B_*") needs its scope
> rescoped explicitly to the **greedy/separable-ranker class**. Phase 3a
> shows non-greedy search procedures *exceed* this bound on the same
> ProSeCo-OWT system, so the corollary cannot be stated as a universal
> bound on "informed scheduling". The empirical anchor (top-K MC internal
> Jaccard ≈ random) still motivates the corollary; the corollary still kills
> the entire learned-per-step-ranker programme (PRISM and similar). See
> `THEORY_STATUS.md` (Honesty Ledger 2026-04-20 entries) and
> `PHASE3A_COMBINATORIAL_RESULTS.md` §6 for the rescoped statement.

---

## Why this dominates PRISM right now

Phase 2b shows three independent signatures that the corrector-scheduling gain is **combinatorial, not signal-driven** at the single-step granularity:

1. `mean_delta_oracle` (uses ground-truth Δ_t) saturates by B=8.
2. Top-10 MC schedules overlap `mean_delta_oracle`'s picks at ~1.5× random baseline.
3. Top-10 MC schedules are no more internally consistent than bottom-10 MC schedules.

A learned per-token quality signal (PRISM) addresses none of these. It's a 2–4 week, multi-week-of-HPC investment that targets a hypothesis the data already falsified. The two cheap experiments below directly test the *actual* dominant hypothesis ("scheduling is combinatorial") and produce a thesis-grade contribution either way.

---

## Phase 3a — Combinatorial scheduling baselines *(estimate: 1–2 days)*

### Question
> Is the +0.45 MC-oracle headroom recoverable by **any** non-greedy scheduling procedure that does not require ground-truth labels?

### Procedure
Two baselines, both starting from a uniform schedule and using paired-G evaluation on ProSeCo-OWT:

1. **Coordinate descent (CD)** — at each iteration, sample a (in_position, out_position) swap from {schedule_steps} × {non_schedule_steps}, re-evaluate G, accept if improved. Stop when no swap improves over a window of N=16 attempts. Surrogate option: use the *cheap* additive A(S) for swap acceptance, then re-evaluate the final schedule with true G — much faster.

2. **Beam search with rollouts (BS)** — maintain a beam of W=8 partial schedules of size k. Expand each by every legal next position; rank by cheap-A; rollout-evaluate top W; advance to k+1. Repeat until k=B.

Both run on **30 seeds × B ∈ {2, 3, 4, 8}** to mirror Phase 2b's paired protocol.

### Outputs
- `results/phase3a/coord_descent_paired.json` — CD vs uniform paired CIs per (B, seed)
- `results/phase3a/beam_search_paired.json` — BS vs uniform paired CIs per (B, seed)
- `results/phase3a/oracle_gap_closure.json` — Δ_CD / Δ_oracle and Δ_BS / Δ_oracle ratios per B
- `figures/phase3a/oracle_gap_closure.{png,pdf}`

### Decision rules
| Outcome | Reading | Thesis story |
|---|---|---|
| CD or BS recovers ≥50 % of MC-oracle headroom across B | scheduling has *learnable* combinatorial structure | "scheduling is search, not signals" — pivot to a sequel paper or a chapter on combinatorial corrector scheduling |
| CD/BS recover only marginal gains (within 2σ_F) | the gain is *irreducibly unstructured* at single-step granularity | clean negative-result corollary; Theorem A is empirically vacuous on this system; thesis chapter writes itself |
| CD wins but BS doesn't (or vice versa) | the topology of useful schedules has structure that one search method exploits | document the structural difference; useful for the methods chapter |

### HPC requirement
- CD with cheap-A surrogate: ~30 minutes per (seed, B) on 1 GPU; full 30×4 = 6h on 1 A100, runnable as a single short job. With 4 GPUs sharded by seed, ~1.5h.
- BS rollout-eval: ~2× CD cost, fits in same job.

### Pre-requisites
- The infrastructure to evaluate G(S) for an arbitrary corrector schedule already exists (used by `mean_delta_oracle` and the MC oracle in Phase 2b).
- The cheap surrogate A(S) (additive Δ̂ over schedule steps) is also already computed.
- A new `scripts/run_phase3a_combinatorial.py` orchestrator + an `hpc/phase3a_combinatorial.sbatch` are the only new artefacts.

---

## Phase 3b — Theory finalisation *(estimate: 2–3 days)*

### Goal
Convert the post–Phase-2b candidate refinements into formal statements with empirical anchors. Phase 2b empirically validates two of them; Phase 3a will validate or close out the third.

### Statements to formalise

**Refinement A′ (variance-form additivity slack).**
Already in `THEORY_STATUS.md` as a candidate. Tighten with Phase 2b σ_ξ measurement (residual = G − A per (seed, schedule); compute σ_ξ across MC rows). Prove: `𝔼|G(S) − A(S)| ≤ σ_ξ · √B / √2`. This replaces Proposition C's `B(B−1)/2` looseness with a √B-tighter bound.

**Refinement A″ (rank-based ε_R).**
Phase 2b shows pooled Spearman ρ(A, G) decays from 0.66 (B=2) to 0.39 (B=8). Define `ε_R(B) := (1 − |ρ_B|) · σ_Δ`. Prove: `G(S_B*) − G(Ŝ_B) ≤ 2B · ε_R(B) + 2η_B`. This is *the* calibration measure that survives the data, replacing the L∞-norm ε used in Theorem A.

**Negative-Result Corollary (new).**
Under the empirical condition observed in Phase 2b (top-K MC schedules show internal Jaccard ≈ random baseline), prove:
> No greedy single-step ranking policy `π_signal` can satisfy `𝔼[G(π_signal)] − 𝔼[G(uniform)] > 2σ_F` for B ≥ B_*, where B_* is the smallest budget at which top-K MC internal Jaccard falls within (random_baseline ± 0.02).

Empirically B_* ≈ 3 on ProSeCo-OWT.

### Outputs
- `research/candidate_theorems.md` — promote A′, A″ from "candidate" to "stated and proven (informal)"; add Negative-Result Corollary.
- `docs/thesis/theory/THEORY_STATUS.md` — update Honesty Ledger; add Phase 2b empirical anchor entries.
- `research/proof_worklog.md` — formal proofs (informal first, formal pass after Phase 3a closes).
- Optional: `thesis/chapters/ch6_theory.tex` skeleton with the refined theorems.

### Decision rules
- If Refinement A″ proves cleanly and matches Phase 2b ρ-decay numerically → ch6 writes itself; primary theorem of thesis.
- If Negative-Result Corollary proves under realistic conditions → thesis has a stress-test result on the entire informed-corrector literature.

---

## How Phase 3a and Phase 3b interact

Run **in parallel**. Phase 3a (HPC) and Phase 3b (notebook) do not share resources.

If Phase 3a recovers the gap → Negative-Result Corollary is *not* unconditional; it bounds *greedy* policies, not search. The thesis becomes "search beats signals" with the corollary as a structural reason for greedy failure.

If Phase 3a does not recover the gap → Negative-Result Corollary applies to a broader policy class; the thesis becomes "irreducible combinatorial gain — even search can't fully close it within budget".

Either way, both Phase 3a and Phase 3b produce thesis content.

---

## What success looks like by end of week

1. `results/phase3a/oracle_gap_closure.json` produced; verdict written.
2. `THEORY_STATUS.md` shows Refinement A′ and A″ as "stated, informally proven, empirically anchored".
3. Negative-Result Corollary stated with Phase 2b empirical conditions.
4. `docs/archive/chronicles/RESULTS_STATUS.md §13` summarises the Phase 3 result (archived historical chronicle).
5. `CANONICAL_RESEARCH_DIRECTION.md` updated with the new "Phase 3 verdict" block.
6. A two-page Phase 3 PDF status report at `docs/pdf/status/phase3_status_report_2026-04-XX.pdf`.

This puts the thesis in defensible final-chapter posture within ~one week, instead of betting 2–4 weeks on a PRISM pivot whose premise is already falsified.

---

## What this plan deliberately does NOT do

- It does not propose porting to MDLM. That is a Tier-2 external-validity check, only justified if Phase 3a surfaces a positive.
- It does not propose Phase 2c MAUVE F-swap as a mainline workstream. The Phase 2b verdict is robust given Cohen's d up to ±2 in many cells; F-swap can be revisited if Phase 3a pivots toward defending the small-B win.
- It does not pursue PRISM, learned critic, or any other model-branch work, because the Phase 2b smoking guns falsify the premise that motivates them.

---

## Provenance

- Audit: `PHASE3_DIRECTION_AUDIT.md`
- Phase 2b results: `results/phase2b/` and `results/phase2b_proseco_owt/`
- Theory: `THEORY_STATUS.md`, `research/candidate_theorems.md`
- Canonical scope: `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`, `CLAUDE.md`
