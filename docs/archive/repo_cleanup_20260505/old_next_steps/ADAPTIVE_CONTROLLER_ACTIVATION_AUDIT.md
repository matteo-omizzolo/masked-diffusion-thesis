> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


> **STATUS:** AUDIT (independent re-audit; supersedes the labeling — not the
> evidence — of `ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md` and
> `POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md` only on the activation question)
> **LAST VERIFIED:** 2026-04-25
> **SCOPE:** Skeptical re-evaluation of whether adaptive-budgeted controllers
> should remain Future Work, be activated as theory only, or as theory + a
> bounded pilot. Re-audits the standing recommendation in light of the actual
> repo evidence (OWT Phase 1 + Phase 2b + Phase 3a closed; bounded LLaDA-SFT
> closed). Decisive verdict in §D. No reopening of any HPC contract.

---

# Adaptive Controller Activation Audit (Independent Re-audit)

This document is an independent skeptical audit. Prior decisions are read as
hypotheses, not adopted by default. The standing classification is:

> "Adaptive-budgeted controllers are Future Work; Protocol C may be run on
> LLaDA-SFT artefacts; theory remains a conditional sketch."
> — `POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`, 2026-04-22.

The audit re-examines that label against current evidence and identifies one
defect in the standing plan that materially changes the recommended action.

---

## A. What is currently established?

Each item is classified as **supported / plausible-unproven / speculative /
contradicted / not-assessable**, with the load-bearing artefact cited.

### A.1 OWT open-loop mainline

| Item | Status | Anchor |
|---|---|---|
| Theorem A bound (open-loop ranker, L∞ form) | **Supported as a theorem; empirically vacuous on OWT at every B ∈ {4, 8, 16}** | `THEORY_STATUS.md` Honesty Ledger |
| Refinement A′ (variance-form η_B via σ_ξ) | **Supported empirically (σ_ξ measured on 9 000 G−A pairs); proof pending Phase 3b** | `results/phase2b/theorem_a_constants.json` (σ_ξ = 0.174 / 0.240 / 0.309 at B = 2, 3, 4) |
| Refinement A″ (rank-based ε_R via ρ(A,G)) | **Supported empirically (ρ pooled = 0.60 / 0.54 / 0.46 at B = 2, 3, 4); proof pending Phase 3b** | same file |
| Negative-Result Corollary, **rescoped to ranker class** | **Supported empirically; formal statement pending Phase 3b** | `PHASE3A_COMBINATORIAL_RESULTS.md` + `THEORY_STATUS.md` |
| Phase 3a search-class positive (CD-G 74–84 %, BS-AG 49–64 %; both PASS at B = 8) | **Supported empirically on OWT K = 30** | `results/phase3a_proseco_owt/oracle_gap_closure.json` |
| Phase 2b MC-oracle-minus-uniform paired = +0.45 at B ∈ {2, 3, 4}, CIs exclude zero | **Supported** | `results/phase2b/mc_oracle.json` |
| Phase 2b MC top-K vs oracle Jaccard ≈ 1.20 / 1.18 / 1.30× random at B = 2 / 3 / 4 | **Supported** | `results/phase2b/combinatorial_diagnostics.json` |
| Phase 2b within-seed share of MC variance: 0.69 / 0.64 / 0.62 at B = 2 / 3 / 4 | **Supported** | same file |

### A.2 LLaDA-SFT bounded probe (T3)

| Item | Status | Anchor |
|---|---|---|
| Uniform-not-beaten (ranker observation) corroborates at T3 | **Supported under bounded resolution** | `CROSS_BACKBONE_REPLICATION_RESULTS.md` §10 |
| Positive MC-oracle headroom does **not** transfer at tested (T = 64, B ∈ {2, 4}, GPT-2 ref) | **Supported under bounded resolution; CI [0, 0] at B = 4; CI [−4.07, −1.07] at B = 2** | `results/cross_backbone/proseco_llada_sft_bounded/phase2b/mc_oracle.json` |
| Phase 3a NOT authorized on LLaDA-SFT | **Decision; reopening precondition pre-registered** | `POST_CROSS_BACKBONE_DECISION.md` §6 |

### A.3 Adaptive controller direction

| Item | Status | Anchor |
|---|---|---|
| Theorem A-ad (F1, FH-CMDP) reduces to Theorem A in degenerate state | **Plausible-unproven** as currently written; the reduction needs care because the threshold-λ policy is not the open-loop top-B policy on individual trajectories — both belong to a broader policy class but differ even when ψ̃ ignores z beyond s_t | `ADAPTIVE_BUDGETED_CONTROLLERS.md` §2.1 |
| Theorem A-ad has all four error terms estimable from data | **Supported in principle; never measured** | same file §4 |
| Δ_open > 0 on OWT | **Supported empirically** at B ∈ {2, 3, 4} via Phase 2b MC oracle headroom | `results/phase2b/mc_oracle.json` |
| Δ_open > 0 on LLaDA-SFT | **Contradicted under bounded protocol** (CI [0, 0] at B = 4) | `POST_CROSS_BACKBONE_DECISION.md` |
| State conditioning ε̃ < ε on any backbone | **Not assessable** — no measurement exists in the repo | none |
| Phase 1 Protocol A on OWT contains per-step Δ_t and per-step signals (entropy, inverse margin, quality mass, unmasked fraction, n_revisable, n_masked) for 50 seeds × 64 steps = 3 200 paired observations | **Supported** | `results/phase1_proseco_owt_full/protocol_a/trajectory_*.json` (verified: 50 files, T = 64, full per-step signal + Δ_t structure) |

### A.4 Defect in the standing recommendation

The standing `POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md` scopes Protocol C to
**LLaDA-SFT** artefacts. But on LLaDA-SFT the bounded probe established
Δ_open ≈ 0 at the tested budgets (CI [0, 0] at B = 4). Protocol C's reported
quantity `Δ_close(π̂_λ,1) / Δ_open` is therefore **mathematically
uninterpretable** (≈ 0 / 0). The decision rule "> 0.5 → preliminary positive,
< 0.5 → honest negative" cannot fire on a 0/0 ratio.

The standing recommendation is correct in shape (run Protocol C, no GPU,
reuse artefacts, pre-register decision rule) but **wrong in target dataset**.
The right target is **OWT**, where Δ_open is measured at +0.45 paired and
the Phase 1 Protocol A trajectory data is on disk.

This is the single defect that motivates an audit-time deviation from the
2026-04-22 recommendation.

---

## B. Is adaptive control actually justified now?

### B.1 What evidence supports moving from open-loop to adaptive?

Three load-bearing observations:

1. **Within-seed variance dominates between-seed variance at small B.** On
   OWT Phase 2b, within-seed share is 0.69 / 0.64 / 0.62 at B ∈ {2, 3, 4}
   (`results/phase2b/combinatorial_diagnostics.json`). The optimal schedule
   is largely a function of the *seed-specific realisation*, not the
   ensemble average. This is a necessary (not sufficient) condition for
   adaptivity to be useful.

2. **Top-K MC ∩ oracle Jaccard barely exceeds random.** Ratios 1.20 / 1.18 /
   1.30× random at B ∈ {2, 3, 4} mean: the schedules that produce the best
   G are largely *not* the schedules an ensemble-mean ranker would pick.
   Some of this is noise; some of it is true seed-conditional structure.
   No current artefact disentangles those.

3. **Search class (CD-G with true G feedback) recovers ≤ 84 % of the MC
   oracle.** The unrecovered ≥ 16 % is an upper bound on what any *better*
   policy — adaptive or otherwise — could add over CD-G, conditional on the
   MC oracle being a meaningful ceiling.

### B.2 What evidence is still missing?

Five explicit gaps, all addressable from existing data:

1. **State-conditional ε̃.** No measurement exists of
   RMS(Δ_t − ψ̃(z_t)) where z_t includes any feature beyond s_t. The
   standing claim "state conditioning shrinks ε" is conjectural until
   measured.

2. **η̃_B under a non-random schedule distribution.** Phase 2b's σ_ξ was
   measured under random MC schedules. If a threshold-λ policy concentrates
   on a low-σ region of schedule space, η̃_B may be smaller (good); if it
   concentrates on a high-σ region (e.g. correlated bursts), η̃_B may be
   larger (bad). No measurement exists.

3. **Closure ratio for a deterministic state-conditional ranker.** Phase 2b's
   `entropy_top_B_per_trajectory` ranks each seed's signal vector and picks
   top-B; Phase 2b shows it FAILS to beat uniform (paired CI excludes 0 in
   the wrong direction). But Phase 2b does **not** test top-B by
   `bucket-mean Δ̄(z)`: a per-seed function that uses the bucket-mean as
   the proxy rather than the raw signal. The latter is a strictly different
   ranker because it folds in trajectory phase information.

4. **Strict-generalisation reduction Theorem A-ad → Theorem A.** As written
   in `ADAPTIVE_BUDGETED_CONTROLLERS.md` §2.1, the reduction claim is
   slightly imprecise: top-B-by-ψ and threshold-by-ψ̃ differ even when
   ψ̃(z) reduces to ψ(s_t), because the latter spends Bin(T, p) on each
   seed and the former spends exactly B. The reduction goes through if
   stated for the *abstract* policy class (top-B is one member, threshold-λ
   another), not for the specific policies. Current write-up needs
   tightening.

5. **No formal proof exists for Theorem A-ad** (currently a sketch under
   stated assumptions). The Lagrangian-duality + swap argument is
   straightforward but has not been written.

### B.3 Is adaptive control a natural next move, or still too weakly motivated?

**Natural** in the sense that A-ad strictly contains A as a special case
(when stated for the abstract policy class) — every open-loop result is a
special case. **Operationally weak** in the sense that:

- Phase 2b's `entropy_top_B_per_trajectory` already tests one member of the
  state-conditional ranker class and FAILS. This is moderate evidence
  against state-blind-by-anything-but-s_t adaptivity.
- The remaining ≥ 16 % adaptive headroom over CD-G is small relative to
  the 49 % BS-AG already recovers, *and* Phase 3a explicitly does not bound
  it formally.

The move is justified as **theory + bounded pilot in an Appendix**, not as
a thesis-core direction.

### B.4 Would activating it strengthen or distract from the thesis?

Activating as **theory-only Appendix F + bounded OWT pilot** strengthens the
thesis if and only if:

- The formal A-ad statement adds a clean conditional theorem to the
  Future-Work section without overclaiming;
- The bounded pilot delivers an interpretable Δ_close / Δ_open ratio (which
  requires a non-zero Δ_open — which requires running on OWT, not
  LLaDA-SFT);
- The decision rule is pre-registered and produces a binary verdict
  (preliminary-positive / honest-negative).

Activating as **a major thesis branch** distracts: it would re-open the
empirical infrastructure question (state-conditional Δ_t estimation
requires either fresh GPU runs or aggressive offline reanalysis), and
would compete with main-thesis writing budget at the T-6-month point of
an MSc thesis.

### B.5 Is there a bounded adaptive-controller step that is scientifically
meaningful now?

Yes: **Protocol C on OWT** (CPU-only, ≤ 1 day of work, reuses
`results/phase1_proseco_owt_full/protocol_a/` + `results/phase2b_proseco_owt/`
+ `results/phase2b/`). Specification in §C.4 below.

---

## C. Strategic options and their tradeoffs

The five options to evaluate. Each is summarised against scientific value,
novelty, tractability, thesis fit, risk, and the evidence likely to be
gained.

### Option 1 — Keep adaptive control as Future Work only

| Dimension | Assessment |
|---|---|
| Scientific value | Low (no new evidence; the theory remains a sketch) |
| Novelty | Zero (current state) |
| Tractability | Trivial |
| Thesis fit | Defensible (one-paragraph future-work mention in ch5/ch6) |
| Risk | Misses an estimable conditional theorem with all four error terms measurable from existing data |
| Evidence gained | None |

### Option 2 — Activate theory only

Promote Theorem A-ad from candidate to formally stated + proved (as a
conditional theorem on stated assumptions), positioned in Appendix F. No
empirical commitment.

| Dimension | Assessment |
|---|---|
| Scientific value | Moderate (a clean conditional theorem with explicit proof, contributing to a thesis-quality theoretical scaffold) |
| Novelty | Moderate (formal F1 statement with the strict-generalisation reduction tightened) |
| Tractability | High (≤ 1 day of write-up; the math is essentially Lemma A2 + Lagrangian duality) |
| Thesis fit | High (Appendix F or short ch6 §) |
| Risk | Theorem with unmeasured constants — exactly what the direction audit warned against |
| Evidence gained | None empirical; proof of A-ad as a closed-form bound |

### Option 3 — Activate bounded pilot now, on OWT (not LLaDA-SFT)

Theory promotion (as Option 2) + bounded Protocol C on OWT artefacts only,
no GPU. Pre-registered decision rule produces a binary verdict on whether
state conditioning materially shrinks ε.

| Dimension | Assessment |
|---|---|
| Scientific value | High (one new measurement: ε̃ vs ε on OWT bucketed state, plus an additive-surrogate Δ_close estimate with explicit η̃_B uncertainty) |
| Novelty | Moderate (state-conditional calibration on bucketed z, no learned controller) |
| Tractability | High (≤ 1 day of CPU-only work; existing artefacts cover all inputs) |
| Thesis fit | High (Appendix F with conditional theorem + preliminary measurement, no main-body claim) |
| Risk | Low — the test is binary and pre-registered; even a negative outcome is informative (strengthens the ranker-class corollary) |
| Evidence gained | (a) ε̃ / ε ratio per signal × B; (b) Δ_close_A / Δ_open ratio with σ_ξ uncertainty band; (c) Hamming distance between threshold schedule and best MC schedule per seed |

### Option 4 — Activate as a major thesis branch

Make adaptive control part of the thesis main body. Run state-conditional
Δ_t estimation on multiple backbones, compare F1 / F3 algorithms, expand
state space.

| Dimension | Assessment |
|---|---|
| Scientific value | Potentially high in 12 months; not in 2 |
| Novelty | High (new theorem class, new algorithmic study) |
| Tractability | Low at thesis scale (function approximator, sample complexity, multi-backbone effort) |
| Thesis fit | Poor — displaces ch3 writing and ch6 proof finalisation |
| Risk | High — premature framework escalation, theorem loss, scope creep into token-selection / learned control |
| Evidence gained | A second-class study; not the right shape for an MSc thesis |

### Option 5 — Bounded hybrid (theory + lightweight pilot, on OWT)

Identical to Option 3. Listed separately to make explicit that the
"hybrid" framing in the user's prompt is the same as Option 3.

### Comparison summary

| Option | Theory | Pilot | Backbone | GPU | Verdict shape |
|---|---|---|---|---|---|
| 1 — Future Work only | sketch | none | — | none | unchanged |
| 2 — Theory only | formal A-ad in Appendix F | none | — | none | A-ad has unmeasured constants |
| 3 — Theory + OWT pilot | formal A-ad in Appendix F | Protocol C | OWT (Phase 1 + 2b) | none | binary: preliminary-positive vs honest-negative |
| 4 — Major branch | full A-ad + algorithmic study | multi-backbone, learned controllers | OWT + LLaDA-SFT + others | substantial | thesis pivot |
| 5 — Bounded hybrid | (= Option 3) | (= Option 3) | (= Option 3) | (= Option 3) | (= Option 3) |

---

## D. Decisive verdict

**ACTIVATE THEORY + BOUNDED OWT PILOT (Option 3).**

The standing recommendation in `POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`
is shape-correct (Protocol C, no new GPU, decision-rule-pre-registered) and
target-incorrect (LLaDA-SFT artefacts have Δ_open ≈ 0, making the test
uninterpretable). Re-targeted to OWT — where Δ_open is measured at +0.45
paired and Phase 1 Protocol A delivers 50 seeds × 64 steps with full
per-step Δ_t and per-step signals — Protocol C becomes a tractable,
informative, decision-binary calibration test.

The verdict has two layers:

1. **Theory.** Promote Theorem A-ad from candidate (sketch) to **formally
   stated and proved as a conditional theorem** in
   `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`, with the
   strict-generalisation reduction stated for the abstract policy class.
   Positioned for **Appendix F**, not main body.

2. **Empirical.** Run **Protocol C on OWT** (CPU only, ≤ 1 day, reuses
   existing artefacts). Pre-register decision rule. Produces a binary
   verdict on whether the threshold-λ policy on bucketed state z = (s_t,
   phase(t)) recovers a measurable fraction of the MC-oracle headroom.

What this verdict explicitly **does not commit the thesis to**:

- A Particle-Gibbs / cSMC implementation (F3 stays analytical-only).
- A learned controller of any form.
- A state abstraction richer than (s_t, phase(t)).
- A cross-backbone Protocol C replication.
- A move of A-ad into the main body.
- A reopening of LLaDA-SFT Phase 3a.
- Any GPU work whatsoever.

What this verdict **rejects** from prior framings:

- "Run Protocol C on LLaDA-SFT" — wrong target dataset (Δ_open ≈ 0).
- "Adaptive is Future Work, no formal theorem in the thesis" — leaves an
  estimable conditional theorem on the table.
- "Adaptive is a thesis pivot" — Phase 2b ranker negative is moderate
  evidence against rich adaptivity, and the remaining ≥ 16 % CD-G gap is
  too narrow to justify a full pivot.

### D.1 Justification chain

E1 (within-seed variance 62 % at B = 4) + E2 (top-K Jaccard 1.2× random) +
E3 (Δ_open = +0.45 measured on OWT) + E4 (Phase 1 OWT trajectory data on
disk, no GPU needed) → bounded pilot is justified.

A2 (signals weak unconditionally) + A3 (Phase 2b's
`entropy_top_B_per_trajectory` already FAILS) + A6 (within-seed variance
not partitionable into adaptive-removable vs irreducible) → pilot must be
**bounded** (CPU only, additive-surrogate G estimator with σ_ξ bound), not
a major thesis investment.

The remaining ≥ 16 % CD-G gap is genuinely interesting but its size is
exactly what a bounded adaptive pilot can probe without committing the
thesis.

### D.2 Pre-registered decision rule for the pilot

After Protocol C runs:

| Outcome | Action |
|---|---|
| ε̃ / ε ≤ 0.7 AND Δ_close_A / Δ_open ≥ 0.5 (after σ_ξ uncertainty) | **Preliminary positive** — Appendix F includes A-ad as formal theorem + Protocol C as preliminary evidence |
| ε̃ / ε > 0.9 OR Δ_close_A / Δ_open < 0.3 | **Honest negative** — Appendix F includes A-ad as existence-only theorem + Protocol C as a refinement of the ranker-class corollary (state-conditional rankers also bounded by `mean_delta_oracle` envelope on OWT) |
| Otherwise (middle case) | **Inconclusive at K = 50 OWT** — Appendix F includes A-ad as formal theorem + Protocol C as bounded null result with explicit Tier-3 caveat |

Decision rule fires on results of Protocol C analysis output JSON, not on
post-hoc inspection.

### D.3 Hard scope guards (non-negotiable)

The activation does not authorise any of:

- New GPU/HPC runs of any kind on any backbone.
- Function approximators (a learned ψ̃).
- State expansion beyond (s_t, phase(t)).
- Cross-backbone replication of Protocol C.
- Particle-Gibbs / cSMC implementation.
- Promotion of A-ad to main body in any branch of the verdict.
- Reopening of LLaDA-SFT Phase 3a.
- Any change to the OWT Phase 2b / Phase 3a mainline framing.
- Any displacement of ch3 / Zanella meeting / Phase 3b theory finalisation.

Violation of any of these returns the audit to "future work only."

---

## E. What this audit does NOT change

- Theorem A (open-loop) remains the main-body theorem.
- Refinements A′ and A″ remain the load-bearing refinements pending Phase 3b
  formal proofs.
- Negative-Result Corollary remains scoped to the ranker class.
- Phase 3a's CD-G + BS-AG search-class positive remains the load-bearing
  empirical positive on OWT.
- LLaDA-SFT bounded probe remains closed; Phase 3a on that backbone remains
  not authorised.
- ch3 / ch5 / ch6 / ch7 main-body framing remains unchanged.
- Zanella meeting writeup remains valid; the meeting brief's "Future Work:
  Adaptive State-Conditional Controllers" framing now has a measured
  pilot result attached, but the main-body is unchanged.

The activation is **strictly additive** to the appendix and the theory
canon; it does not perturb the mainline.

---

## F. Sequencing after this audit

In order:

1. **Theory promotion** — tighten `ADAPTIVE_BUDGETED_CONTROLLERS.md` to
   formal A-ad statement with explicit proof under the abstract policy
   class, and an honest reduction to Theorem A. (See Phase 2 of this
   activation.)

2. **Experiment plan** — write
   `ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md` specifying Protocol C on OWT
   exactly. Pre-register decision rule.

3. **Implementation** — implement Protocol C as a CPU-only Python module
   (`src/mdm_playground/analysis/protocol_c.py`) + script
   (`scripts/run_protocol_c_owt.py`) + tests
   (`tests/test_protocol_c.py`).

4. **Run** — execute Protocol C on OWT, write JSON outputs to
   `results/protocol_c_owt/`.

5. **Decide** — apply pre-registered decision rule; write
   `POST_ADAPTIVE_CONTROLLER_DECISION.md` with the binary verdict.

6. **Doc updates** — narrow updates to `THEORY_STATUS.md` (mark Theorem
   A-ad as formally stated + conditionally proved with one measured
   calibration constant), `CANONICAL_RESEARCH_DIRECTION.md` (one-line
   addition under "Out of scope as of Phase 3 audit" referring to Appendix
   F), `CURRENT_INDEX.md` (add Appendix F entry under §9 if the verdict is
   preliminary-positive), and ch5 / ch6 / ch7 LaTeX (only if preliminary-
   positive; otherwise leave LaTeX unchanged and let A-ad live in
   `ADAPTIVE_BUDGETED_CONTROLLERS.md`).

The whole activation fits inside ≤ 2 days of work and produces a single,
well-defined empirical artefact.

---

## G. Honesty ledger

| Claim | Tag |
|---|---|
| Theorem A-ad reduces to Theorem A under abstract-policy-class statement | `[Plausible-unproven; reduction needs careful statement]` |
| ε̃ < ε on OWT bucketed state | `[Conjecture; pilot will measure]` |
| Δ_open on OWT ≥ +0.45 paired at B ∈ {2, 3, 4} | `[Empirically anchored]` |
| Δ_open on LLaDA-SFT ≈ 0 at tested setup | `[Empirically anchored, T3]` |
| Protocol C on LLaDA-SFT artefacts is uninterpretable (0 / 0) | `[Inferred from prior empirics, not from a Protocol C run on LLaDA-SFT]` |
| State-conditional ranker class is bounded by the same envelope as signal-only ranker class | `[Conjecture for the (s_t, phase(t)) bucketing; pilot will probe]` |
| The threshold-λ policy on bucketed state recovers a measurable fraction of MC-oracle headroom on OWT | `[Conjecture; pilot will measure]` |
| Activating Theorem A-ad as Appendix F preserves Theorem A as the main-body theorem | `[Plausible by construction; verified by §E above]` |

---

## H. Links

- Standing recommendation (re-targeted by this audit): `POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`
- Phase-1 audit (still load-bearing for the broader "do not pivot" verdict): `ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md`
- Theory doc (target of Phase 2 of activation): `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`
- Open-loop canonical theory: `docs/thesis/theory/THEORY_STATUS.md`
- Phase 3a empirical anchor: `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`
- Phase 2b empirical anchor: `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` (LLaDA-SFT side; OWT side in `THEORY_STATUS.md` Honesty Ledger)
- Existing per-step OWT data (Protocol A): `results/phase1_proseco_owt_full/protocol_a/trajectory_*.json`
- Existing OWT MC oracle: `results/phase2b/mc_oracle.json`
- Existing OWT theorem A constants: `results/phase2b/theorem_a_constants.json`

---

*End of independent activation audit. Verdict: Option 3 (theory + bounded
OWT pilot). Sequencing in §F. The audit does not authorise any GPU, any
state-space expansion, any policy-learning, or any main-body change.*
