# Phase 3a — Combinatorial Scheduling Baselines (Results)

*Created: 2026-04-20. Job 479941 (gnode02, ProSeCo-OWT). Wall-clock 7h59m.*
*Status: **COMPLETE — primary positive result.***

---

## TL;DR

On ProSeCo-OWT, a fixed corrector budget `B` is best treated as a **combinatorial
schedule-optimisation problem**. Two non-greedy search procedures — CD-G
(coordinate descent with true-G feedback) and BS-AG (beam search with cheap-A
ranking and true-G rollouts) — beat the paired uniform baseline at every
budget tested (`B ∈ {2, 3, 4, 8}`) and recover **49 – 84 %** of the Phase 2b
MC-oracle headroom. Phase 2b's negative result therefore lands as a statement
about the **greedy-ranker policy class**, not about the existence of recoverable
scheduling structure.

---

## 1. What is being tested

Phase 2b established two facts:

- The MC oracle (best-of-100 random schedules per seed) sits at +0.45 paired
  G over uniform at `B ∈ {2, 3, 4}`. The +0.36 gap between mean-profile
  oracle (`mean_delta_oracle`) and MC oracle is *per-instance* (`RESULTS_STATUS.md`
  §12.2).
- Greedy single-step rankers — including the cheating ground-truth
  `mean_delta_oracle` — only beat uniform at small `B`, and saturate by `B = 8`
  (`PHASE3_DIRECTION_AUDIT.md` smoking gun #1).

Phase 3a asks the complementary question: does *any* non-greedy search procedure
recover the +0.45 headroom? If yes, the right policy abstraction for fixed-budget
corrector scheduling is **combinatorial search**, not single-step ranking.

---

## 2. Methods

Both methods score every candidate schedule with the same paired-G estimator
used in Phase 2b (`F = −GPT-2 NLL` on a 512-token reference window;
`evaluate_schedule` in `src/mdm_playground/scheduling/evaluate.py`).

### 2.1 CD-G — Coordinate Descent with true-G feedback

- Initial schedule: uniform stride-T/B step set.
- Each iteration: sample one (in_position, out_position) swap from
  (schedule, non-schedule) using the seed-derived RNG, evaluate the new
  schedule's `G`, accept iff `G_new > G_current`.
- Stop on a window of `cd_window = 16` consecutive non-improving attempts or
  on `cd_max_attempts = 64`.
- Per-cell budget: ≤ 65 G-calls (1 init + 64 swaps).

CD-G uses **the true pipeline-evaluated G for every accept/reject decision** —
it is therefore a *structural / existence* result, not a deployable inference-time
scheduler. The fair reading is: "if you could spend `O(B · cd_attempts)` G-calls,
this is the gain you can structurally extract."

### 2.2 BS-AG — Beam Search with cheap-A ranking + true-G rollouts

- Round 1: rank every singleton by the cheap additive surrogate `A({t}) = Δ_t`
  (using Phase 1 `protocol_a` Δ_t traces — *no* extra G-call). Take the top
  `bs_beam_width = 8` and rollout-evaluate them with true G.
- Rounds 2 … B: for every (current beam, new position) extension, rank by
  cheap A; rollout-evaluate the top 8 with true G; keep the 8 with best
  rollout-G as the next beam.
- Final return: best-G schedule across the final beam.
- Per-cell budget: `8 × B` true-G rollouts.

BS-AG is closer to a *practical* method than CD-G: it uses cheap-A ranking to
prune the candidate set and only spends G-calls on the surviving 8 beam
extensions per round. It still uses true G to score those rollouts, so it is
not free of pipeline evaluation, but its G-call count is `O(B)` rather than
`O(cd_attempts)`.

### 2.3 Paired protocol

Mirrors Phase 2b exactly:

- `K = 30` paired seeds, sharded 4-way (one shard per A100 GPU).
- `B ∈ {2, 3, 4, 8}` per seed × method.
- Each (seed, B, method) row computes `Δ̂ = G_method − G_uniform` against the
  same seed's uniform baseline replayed from Phase 2b's `policy_raw.json`.
- Paired BCa bootstrap 95 % CI per (B, method); decision rule
  PASS / NULL / FAIL identical to Phase 2b (ANALYSIS_SPEC §3).
- Oracle-gap closure ratio per B is `Δ_method / Δ_oracle`, where `Δ_oracle` is
  the Phase 2b MC-oracle paired headroom from `mc_oracle.json`.

---

## 3. Main result

### 3.1 Paired Δ vs uniform (file-backed)

Numbers below come directly from
`results/phase3a_proseco_owt/{cd_paired,bs_paired,oracle_gap_closure}.json`.

| Method | B = 2 | B = 3 | B = 4 | B = 8 |
|---|---|---|---|---|
| **CD-G Δ** (paired BCa CI) | **+0.356** [+0.284, +0.432] **PASS** | **+0.327** [+0.240, +0.421] **PASS** | **+0.380** [+0.285, +0.479] **PASS** | **+0.322** [+0.215, +0.427] **PASS** |
| **BS-AG Δ** (paired BCa CI) | **+0.289** [+0.209, +0.374] **PASS** | **+0.252** [+0.158, +0.352] **PASS** | **+0.220** [+0.113, +0.331] **PASS** | **+0.153** [+0.035, +0.263] **PASS** |
| **Δ_oracle** (Phase 2b MC) | +0.451 | +0.441 | +0.450 | n/a (MC oracle not run at B=8) |
| **CD-G / oracle ratio** | 78.9 % | 74.1 % | 84.3 % | n/a |
| **BS-AG / oracle ratio** | 64.1 % | 57.1 % | 48.8 % | n/a |

Both methods PASS at every budget tested with paired CI strictly above 0.

### 3.2 Reading the oracle-gap closure

At `B ∈ {2, 3, 4}`, CD-G recovers 74 – 84 % of the +0.45 MC-oracle paired
headroom; BS-AG recovers 49 – 64 %. CD-G's superiority is consistent with its
unbounded G-call budget vs BS-AG's `O(B)` budget — the gap between them is the
price BS-AG pays for using cheap-A ranking instead of true-G ranking on the
expansion candidates.

At `B = 8`, no MC-oracle anchor exists (Phase 2b ran the oracle only at
`B ∈ {2, 3, 4}`), so the closure ratio is undefined. The paired Δ vs uniform
is nonetheless **strongly positive** for both methods at `B = 8` (CD-G +0.32,
BS-AG +0.15, both PASS). This is the most informative single number for the
thesis story: at the budget where `mean_delta_oracle` falls into the NULL
band (Phase 2b smoking gun #1), CD-G still beats uniform by ≥ 3 σ. The
single-step *ranker* ceiling does not bound the *search* class.

### 3.3 Variance interpretation

Phase 2b's variance decomposition at B = 4 (62 % within-seed, 38 %
between-seed; `PHASE3_DIRECTION_AUDIT.md` §A1.3) showed that schedule choice
matters. Phase 3a confirms this is exploitable: search procedures that *can*
attend to schedule structure capture the within-seed variance that ranking
procedures miss.

---

## 4. Why this changes the Phase 3 story

The Phase 2b verdict ("the gain is combinatorial, not signal-driven at the
single-step granularity") was correct but its consequence was slightly
overreached in the original Negative-Result Corollary draft: that draft spoke
generically of "no greedy single-step ranking policy" without separating the
*ranker policy class* from the *search policy class*.

Phase 3a refines the verdict in two binding ways:

1. **The headroom is real and largely recoverable** by procedures that do not
   rely on a learned per-token quality signal. The +0.45 oracle headroom is not
   a sampling artifact and is not "irreducibly unstructured": it has structure
   that combinatorial search can reach.

2. **The wrong solution class is greedy ranking**, not "informed scheduling
   in general." A separable per-step score `ψ(s_t)` followed by top-B selection
   collapses the combinatorial search space into a sort, and the data shows
   that sort throws away most of the gain (Phase 2b) — but the search itself,
   when allowed to operate on schedules instead of scores, does work.

The thesis story therefore tightens from "scheduling is hard" to:
> **Fixed-budget corrector allocation is a combinatorial trajectory-control
> problem; cheap greedy rankers are the wrong solution class for it. Search
> procedures over schedules — even ones that use the cheap-A surrogate to
> prune candidates — recover most of the oracle headroom.**

The PRISM rejection (`PHASE3_DIRECTION_AUDIT.md`) still stands. PRISM is a
learned per-token quality signal, and the cell that bounds it
(`mean_delta_oracle` at `B = 8`) sits in the NULL band. The refined reason
to reject is: the recoverable structure does not factor through any per-step
ranker.

---

## 5. Honest caveats

### 5.1 CD-G's true-G feedback is not deployable

CD-G calls the full pipeline G(S) on every swap acceptance test. On
ProSeCo-OWT this is roughly the cost of generating one full sample. A
schedule of size B reached by ≤ 65 swaps therefore costs ≈ 65 × generation
cost — orders of magnitude more than running the corrector itself. CD-G is
an **upper-envelope existence result**: it tells the thesis that the
combinatorial gain is reachable in principle, not that an inference-time
scheduler can pay this cost. Any deployability claim about CD-G is
overclaiming.

### 5.2 BS-AG is closer to practical, but still not free

BS-AG's G-call count is `O(B · beam_width) = O(8 B)` — a small constant
multiple of B. This is the realistic candidate for a deployable scheduler if
one is willing to pay a one-shot search cost per generation. But BS-AG still
needs a true-G evaluator, so it requires either (i) a tractable proxy for G
that has not been validated here, or (ii) a meta-search wrapper around a
draft-and-rescore protocol. The B = 8 result (+0.15 PASS) is encouraging
but tighter than CD-G; future work (out of thesis scope) would have to test
whether further beam-width or rollout improvements close that gap.

### 5.3 No external validity yet

Phase 3a was run on ProSeCo-OWT only. Whether the search-vs-ranker dichotomy
holds on a different (backbone, corrector, F) triple — e.g. MDLM + ReMDM-conf,
or any non-NLL F — is open. The Phase 2b verdict already held on
ProSeCo-OWT only, so this is not a new caveat, but it does cap the
generality of the Phase 3a positive claim.

### 5.4 No theoretical guarantee on closure ratio

The 49 – 84 % closure ratios are empirical descriptors. There is no
theorem yet that says CD-G or BS-AG should achieve any particular fraction
of MC-oracle headroom. The Negative-Result Corollary scoped to the
ranking class (Phase 3b) does not imply or bound the search-class
positive — this is a separate analysis that the thesis does not have to
ship to be defensible.

---

## 6. What remains open (Phase 3b agenda)

The theory contract that lands in the thesis is now:

1. **Refinement A″ (rank-based ε_R) — formal proof.** Empirically anchored
   by Phase 2b ρ-decay (0.66 → 0.39 across `B ∈ {2..16}`); needs the
   order-statistics derivation.
2. **Refinement A′ (variance-form additivity slack) — formal proof.** σ_ξ
   measured directly from Phase 2b MC residuals on 9 000 (G − A) pairs;
   needs the mixing/cancellation hypothesis written out.
3. **Negative-Result Corollary — restate scoped to greedy/separable
   ranking class.** Old broad framing ("no greedy ranking beats uniform by
   > 2 σ_F") is empirically tight for the ranker class but is *not*
   contradicted by Phase 3a's search-class positive. The corollary should
   read: "any policy of the form `Ŝ_B = top-B(ψ)` for separable per-step
   `ψ` is bounded above by the Phase 2b `mean_delta_oracle` envelope, which
   on ProSeCo-OWT enters the NULL band by B = 8." Phase 3a shows the
   envelope can be exceeded by procedures that do not factor through
   separable scores, so the corollary's reach is the ranker class only.

Out of scope for the Phase 3 finishing track:

- A formal lower bound on what CD-G or BS-AG can recover.
- Any deployable-schedule construction (would require either a tractable
  G surrogate that matches G at search resolution, or a meta-search
  wrapper).
- External validity replication on another backbone.

---

## 7. Provenance

| Artefact | Path |
|---|---|
| Raw CD per-(seed, B) rows | `results/phase3a_proseco_owt/cd_raw.json` |
| Raw BS per-(seed, B) rows | `results/phase3a_proseco_owt/bs_raw.json` |
| Per-shard raw (4) | `results/phase3a_proseco_owt/{cd,bs}_raw.shard{0..3}-of-4.json` |
| CD paired CIs per B | `results/phase3a_proseco_owt/cd_paired.json` |
| BS paired CIs per B | `results/phase3a_proseco_owt/bs_paired.json` |
| Oracle-gap closure | `results/phase3a_proseco_owt/oracle_gap_closure.json` |
| Per-seed cd / bs JSONs | `results/phase3a_proseco_owt/per_seed/{cd,bs}_rows_seed{seed}.json` |
| Manifest (per shard) | `results/phase3a_proseco_owt/manifest.json` |
| Bar chart (PNG / PDF) | `figures/phase3a/oracle_gap_closure.{png,pdf}` |
| Orchestrator | `scripts/run_phase3a_combinatorial.py` |
| Analyzer | `scripts/analyze_phase3a.py` |
| Sbatch | `hpc/phase3a_combinatorial.sbatch` |
| Job log (stdout) | `out/phase3a_combinatorial_479941.out` |
| Job log (per-shard) | `out/phase3a_combinatorial_479941_shard{0..3}.log` |

Run config: T = 64, K = 30, B_values = 2,3,4,8, corrector_steps = 1,
cd_max_attempts = 64, cd_window = 16, bs_beam_width = 8, shard_count = 4,
checkpoint = `~/mdm/checkpoints/proseco_owt`. Wall-clock 28 681 s
(per-shard mean 53 min/seed across 30 seeds total).

---

## 8. Cross-references

- Phase 2b verdict: `RESULTS_STATUS.md` §12
- Direction audit (PRISM rejection): `PHASE3_DIRECTION_AUDIT.md`
- Phase 3a/3b plan (pre-result): `PHASE3_ALTERNATIVE_PLAN.md`
- Theory ledger (post-Phase-3a entry): `docs/thesis/theory/THEORY_STATUS.md`
- Canonical research direction (Phase-3a-aware framing):
  `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
- Navigation: `docs/thesis/CURRENT_INDEX.md`
