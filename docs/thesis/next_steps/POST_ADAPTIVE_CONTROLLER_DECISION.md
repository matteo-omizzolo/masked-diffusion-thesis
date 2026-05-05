> **STATUS:** DECISION (closes the bounded OWT Protocol C pilot — terminal node
> of the activation audit launched 2026-04-25)
> **LAST VERIFIED:** 2026-04-26
> **SCOPE:** Final go/no-go on whether Theorem A-ad and Protocol C move from
> Future Work to Appendix-active or remain Future Work, conditional on the
> Protocol C pilot result recorded in
> `results/protocol_c_owt/protocol_c_summary.json`. Reads downstream from
> `ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md`,
> `ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md`, and
> `ADAPTIVE_BUDGETED_CONTROLLERS.md` §4.2.

---

# Post-Adaptive-Controller Decision — Bounded OWT Pilot

## TL;DR

**Theorem A-ad is promoted from "candidate sketch" to "formally proved
conditional theorem" in Appendix F of the thesis.** Protocol C is
**closed with an honest-negative verdict** on OWT: bucketed-state
conditioning on z = (s_t, phase(t)) shrinks calibration error by
**< 1.7 %** across all three signals (ε̃ / ε ∈ {0.983, 0.985, 0.986}),
and the resulting state-conditional ranker policy's additive-surrogate
close ratio Δ_close_A / Δ_open is **dominated by the Refinement-A′
σ_ξ · √B / √2 uncertainty band at every B ∈ {2, 3, 4}**. The
state-conditional ranker class is empirically bounded by the same
σ_ξ-driven uncertainty as the signal-only ranker class — i.e., the
ranker-class Negative-Result Corollary extends, on this dataset, to the
bucketed-state ranker class. Theorem A-ad lives in **Appendix F as
formally-stated existence-only**, with Protocol C as a clearly-labeled
preliminary null. **Phase 3a-style search procedures (CD-G, BS-AG)
remain the only known route to non-trivial closure on OWT, and they are
out of the adaptive-controller scope.** No GPU/HPC was used. No further
adaptive-controller work is authorised by this decision.

## 1. The pilot, executed exactly as pre-registered

Per `ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md`:

- **Inputs.** 50 OWT Phase 1 Protocol A trajectories (T = 64; per-step
  Δ_t and per-step signals), 30 OWT Phase 2b mc_rows seeds (100 MC
  rows × B ∈ {2, 3, 4} = 9 000 schedules), Phase 2b
  `theorem_a_constants.json` (σ_ξ pooled per B), Phase 2b
  `mc_oracle.json` (best-of-100 G per seed × B).
- **Computation.** All steps in the experiment plan §3, including the
  abstract score-policy class members σ_λ (threshold) and σ_topB
  (top-B-bucketed) for ψ̃ defined as the bucket-mean of Δ_t over the 50
  Phase 1 trajectories with z = (signal_quartile, phase) (4 × 3 = 12
  buckets per signal).
- **Δ_open.** Computed as `mean_over_seeds(best-of-100 MC G) −
  mean_over_seeds(closest_MC_to_uniform G)` per B = 2, 3, 4 →
  +0.371, +0.386, +0.412. This is a tiny per-seed-paired underestimate
  of the +0.45 paired figure in
  `mc_oracle.json` because it uses an MC-sampled uniform proxy rather
  than the exact paired uniform G; the verdict is robust to this
  understatement (the close ratio decision rule fires on the eps_ratio
  leg, not on Δ_open's exact value).
- **Decision rule.** Pre-registered three-way classifier on
  (eps_ratio, delta_close_ratio_threshold_after_uncertainty);
  outcome_class field of the summary JSON drives the verdict.

Code: `src/mdm_playground/analysis/protocol_c.py`,
`scripts/run_protocol_c_owt.py`. Tests:
`tests/test_protocol_c.py` (24 unit tests, all PASS).
Output: `results/protocol_c_owt/protocol_c_summary.json`.

## 2. Result — the four numerical answers

### Q-adapt-2 — does state conditioning shrink ε?

**No, not on this dataset.** Pooled (50 seeds × T = 64 = 3 200 paired
observations per signal):

| Signal | ε (linear fit) | ε̃ (bucket-mean) | ε̃ / ε |
|---|---|---|---|
| entropy | 0.1335 | 0.1313 | **0.9832** |
| inverse_margin | 0.1335 | 0.1314 | **0.9846** |
| quality_mass_proxy | 0.1335 | 0.1315 | **0.9856** |

Bucketing on (signal_quartile × phase) with 4 × 3 = 12 buckets shrinks
the calibration RMS by < 1.7 % at the best signal. The pre-registered
positive threshold was ε̃ / ε ≤ 0.7 (≥ 30 % shrinkage); the observed
shrinkage is **17–20 × short of that threshold**. Q-adapt-2 receives a
clean negative answer for the (s_t, phase(t)) state.

### Q-adapt-1 — Δ_open > 0?

**Yes, empirically** at B ∈ {2, 3, 4}:

| B | Δ_open (best MC − uniform proxy) | σ_ξ · √B / √2 | uncertainty / Δ_open |
|---|---|---|---|
| 2 | 0.371 | 0.174 | 0.469 |
| 3 | 0.386 | 0.294 | 0.760 |
| 4 | 0.412 | 0.437 | **1.061** |

By B = 4 the σ_ξ uncertainty band is **larger than Δ_open itself** —
i.e., any additive-surrogate-derived close estimate at B ≥ 4 is
indistinguishable from zero under Refinement A′'s honest accounting.
This is the same structural barrier the open-loop story already
documents in the rescoped Negative-Result Corollary; Protocol C
inherits it.

### Q-adapt-3 — what is the η̃_B slack under the threshold policy?

The σ_ξ measured under random MC schedules (Phase 2b) is the upper
bound used here. Protocol C does **not** measure η̃_B under the
threshold policy directly because doing so would require running the
ProSeCo corrector on the threshold schedules (GPU work, out of
audit-authorised scope). The Refinement-A′ band σ_ξ · √B / √2 is
therefore a **conservative** estimate.

The Hamming distance between the threshold schedule and the best MC
schedule is **3.93 / 5.87 / 7.47** at B = 2 / 3 / 4 (max possible
Hamming is 2B = 4 / 6 / 8); fraction of seeds with exact match is
**0.000 across all signals × all B**. The threshold-λ policy lives in a
disjoint region of schedule space from the high-G MC schedules. This
provides indirect evidence that η̃_B for the threshold policy is **not
smaller** than the random-MC σ_ξ — possibly larger.

### Q-adapt-close — Δ_close / Δ_open after uncertainty

Both σ_λ (threshold) and σ_topB (top-B-bucketed) collapse to the same
schedule per seed × per B (with cap-based tuning λ chosen so the cap
is binding for all 50 seeds), so their close ratios are **identical**:

| Signal | B | Δ_close_A | raw ratio | after σ_ξ-band ratio |
|---|---|---|---|---|
| entropy | 2 | 0.179 | **0.484** | **+0.015** |
| entropy | 3 | 0.219 | 0.566 | −0.194 |
| entropy | 4 | 0.307 | 0.744 | −0.316 |
| inverse_margin | 2 | 0.176 | 0.475 | +0.007 |
| inverse_margin | 3 | 0.210 | 0.543 | −0.217 |
| inverse_margin | 4 | 0.303 | 0.734 | −0.326 |
| quality_mass_proxy | 2 | 0.171 | 0.461 | −0.007 |
| quality_mass_proxy | 3 | 0.211 | 0.546 | −0.214 |
| quality_mass_proxy | 4 | 0.319 | 0.774 | −0.286 |

The **best after-uncertainty ratio is +0.015** (entropy at B = 2) —
two orders of magnitude below the 0.5 preliminary-positive threshold,
and below the 0.3 honest-negative threshold. At B ∈ {3, 4} every
after-uncertainty ratio is **negative**: the σ_ξ uncertainty band
exceeds the additive-surrogate close estimate. This is the σ_ξ
domination that the rescoped Negative-Result Corollary already
predicts for the ranker class.

### Verdict

```json
{
  "best_signal": "entropy",
  "best_B": 2,
  "eps_ratio_at_best": 0.983,
  "delta_close_ratio_at_best_after_uncertainty": 0.015,
  "outcome_class": "honest_negative"
}
```

The decision rule fires on the **eps_ratio > 0.9** leg (state
conditioning does not shrink ε meaningfully on the bucketed state),
which is independently sufficient for honest-negative without needing
the close-ratio leg. The close-ratio leg also independently fires
(< 0.3) at B ≥ 2 with after-uncertainty accounting.

## 3. What this verdict implies for the thesis

### 3.1 Theorem A-ad — promote to Appendix F as formal but existence-only

The conditional theorem (Theorem A-ad, abstract policy class form) is
proved in `ADAPTIVE_BUDGETED_CONTROLLERS.md` §2.1 under stated
assumptions. The bound shape

    𝔼[G(σ_{π*_B})] − 𝔼[G(σ_{π̂_ψ̃})] ≤ 2 B ε̃ + 2 η̃_B + 𝒪(√(B / N_cal))

is now **a formally stated proved-under-assumptions theorem** in the
thesis appendix. Protocol C's pilot data shows that on OWT with
bucketed state z = (s_t, phase(t)):

- ε̃ = 0.131, B = 2 → 2 B ε̃ = 0.524.
- η̃_B (upper bounded by σ_ξ · √B / √2) ≥ 0.174 at B = 2.
- 2 B ε̃ + 2 η̃_B ≥ 0.872 at B = 2.
- Δ_close (best-case at B = 2) = 0.179.

The bound is **inert by ≈ 4.9 ×** at B = 2 and worse at B ≥ 3 — the
same kind of inertia the open-loop Theorem A exhibits, with the same
diagnostic conclusion (the L∞ form does not land on this triple). The
Refinements A′ / A″ ideas can be ported to A-ad in future work
(rank-form ε̃_R, variance-form η̃_B), but doing so is **out of scope
for this thesis**.

### 3.2 Negative-Result Corollary — extends to bucketed state

Phase 3a's rescoped Negative-Result Corollary bounds any policy
`Ŝ_B = top-B(ψ)` for separable per-step ψ on OWT. Protocol C
empirically extends that envelope: any policy
`σ_{ψ̃}` for ψ̃ a bucket-mean on (s_t, phase(t)) is also bounded by
the σ_ξ-driven uncertainty. This is a **mild generalisation** of the
corollary to coarse state-conditional rankers; it does **not**
generalise to "informed scheduling in general" (Phase 3a's CD-G and
BS-AG search procedures explicitly exceed this envelope).

The wording in `THEORY_STATUS.md` is updated narrowly to reflect this:
"the corollary characterises the ranker class only" → "the corollary
characterises the ranker class — including state-conditional rankers
on the bucketed state z = (s_t, phase(t)) tested by Protocol C — but
not the search class."

### 3.3 What is closed by this decision

- **Theorem A-ad as a formal Appendix-F theorem.** Done. The theorem
  is stated and proved as a conditional bound in
  `ADAPTIVE_BUDGETED_CONTROLLERS.md` §2.1.
- **Protocol C as a one-shot calibration test on OWT.** Done. Output
  in `results/protocol_c_owt/protocol_c_summary.json`. No re-run
  authorised.
- **The "is adaptive control thesis-active?" question.** Closed:
  thesis-active for the theorem (Appendix F) only; not thesis-active
  for the experiment (the pilot is a documented null).

### 3.4 What this decision does NOT do

- Does **not** authorise running Protocol C on a different state
  abstraction (richer z, function approximator, etc.).
- Does **not** authorise running Protocol C on a different backbone.
- Does **not** authorise any GPU/HPC work for the adaptive direction.
- Does **not** displace the OWT Phase 2b / Phase 3a mainline.
- Does **not** reopen LLaDA-SFT Phase 3a.
- Does **not** modify Theorem A's main-body status.
- Does **not** introduce A-ad as a main-body theorem.

## 4. Why the negative is informative, not a failure

The verdict is **scientifically valuable**, not a setback, for three
reasons:

1. **It tightens the ranker-class corollary.** Before Protocol C, the
   corollary was scoped to "separable per-step ψ"; after Protocol C,
   we have explicit empirical evidence that the corollary's envelope
   also bounds **bucketed-state rankers** on OWT. Future readers of
   the thesis can no longer dismiss the corollary as "only bounds
   raw-signal rankers."

2. **It demonstrates Theorem A-ad's bound is a useful diagnostic.**
   The bound's inertness on this dataset is the same shape as
   Theorem A's inertness on the same dataset; both tell a consistent
   story about why ranker-class methods cannot recover the
   MC-oracle headroom.

3. **It documents an honest negative under a pre-registered decision
   rule.** Pre-registration prevents the kind of post-hoc adaptive
   pivot that the activation audit warned against (`§4.5`,
   "opportunity cost on the running Phase 2b job"). The result is
   defensible at the meeting and at the defence.

A null Protocol C is **strictly preferable** to running an
unstructured GPU-heavy alternative; the activation audit's verdict
("Option 3, bounded OWT pilot") was the right call and the bounded
pilot gave the right kind of answer.

## 5. Updates to do (narrow, conservative)

In order:

1. **`THEORY_STATUS.md`** — Honesty Ledger: update the
   "Adaptive-budgeted controllers (Theorem A-ad)" row to reflect
   formally-proved-conditional status + Protocol C honest-negative
   pilot result. The Negative-Result Corollary row receives a one-line
   addendum: "extension to bucketed-state ranker class on OWT
   (Protocol C, 2026-04-26)".

2. **`ADAPTIVE_BUDGETED_CONTROLLERS.md`** — §5 Honesty ledger row for
   Theorem A-ad becomes `[Proved, conditional]` with the constants
   `ε̃, η̃_B` measured by Protocol C and the bound diagnosed inert on
   OWT at every B ∈ {2, 3, 4}. §4.2 Protocol C section receives the
   pilot result as a closing paragraph.

3. **`CANONICAL_RESEARCH_DIRECTION.md`** — no change to mainline.
   Optional one-line addendum to the existing Future-Work list noting
   the bounded pilot's null outcome.

4. **`CURRENT_INDEX.md`** — add the three new docs under §2 (canonical
   thesis documents):
   - `ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md`
   - `ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md`
   - `POST_ADAPTIVE_CONTROLLER_DECISION.md` (this file)
   And under §5 (active scripts):
   - `scripts/run_protocol_c_owt.py`
   And under §7 (active analysis code):
   - `src/mdm_playground/analysis/protocol_c.py`
   And under §8 (results files):
   - `results/protocol_c_owt/`

5. **`thesis/chapters/ch5_informed_correctors.tex`,
   `ch6_contribution.tex`, `ch7_experiments.tex`** — no LaTeX change.
   The pilot verdict is a null; the thesis text already cites
   "Adaptive State-Conditional Controllers" as Future Work, with
   pointer to `ADAPTIVE_BUDGETED_CONTROLLERS.md`. The pointer can stay;
   no new claims to add to the body.

## 6. Single re-opening precondition

The bounded-pilot null does **not** close the broader adaptive
direction permanently. A future precondition for re-opening:

- A new measurement (offline reanalysis or external paper) that shows
  ε̃ shrinks by ≥ 30 % (ε̃ / ε ≤ 0.7) on a richer state abstraction
  on OWT or another backbone.

Absent such a measurement, the adaptive direction stays at
"Appendix F formal theorem + Protocol C honest negative" and is **not**
on the thesis critical path.

## 7. Cost summary

- Audit: 1 day.
- Theory tightening: 1 day.
- Implementation: ~ 1 day (module + script + tests).
- Pilot run: 60 seconds wall time, CPU only.
- Total: ≤ 3 days end-to-end.
- GPU hours used: **0**.

## 8. Honesty ledger

| Claim | Tag |
|---|---|
| ε̃ / ε ≈ 0.983 across all three signals on OWT bucketed state z = (s_t, phase(t)) | `[Empirically anchored 2026-04-26 — n=3200 paired observations]` |
| State-conditional ranker class on bucketed (s_t, phase(t)) does not recover Δ_open after σ_ξ uncertainty subtraction at any B ∈ {2, 3, 4} on OWT | `[Empirically anchored 2026-04-26]` |
| Hamming distance between threshold schedule and best MC schedule ≈ 2B | `[Empirically anchored 2026-04-26]` |
| Theorem A-ad as formal conditional theorem | `[Proved under (1)–(4); abstract-policy-class reduction to Theorem A explicit]` |
| Theorem A-ad bound non-vacuous on OWT | `[Refuted on this dataset; bound is inert by ≈ 4.9 × at B = 2]` |
| Negative-Result Corollary extends to bucketed-state ranker class | `[Empirically supported on OWT 2026-04-26; not generalised to other states or backbones]` |
| The bounded LLaDA-SFT probe — had it been run for Protocol C — would have produced 0/0 at B = 4 | `[Inferred from POST_CROSS_BACKBONE_DECISION.md, not separately verified]` |
| Adaptive-controller direction is closed at thesis level | `[Decision; reopening precondition stated in §6]` |

## 9. Links

- Activation audit (this study's start): `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md`
- Experiment plan: `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md`
- Theory: `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`
- Pilot output: `results/protocol_c_owt/protocol_c_summary.json`
- Module: `src/mdm_playground/analysis/protocol_c.py`
- Tests: `tests/test_protocol_c.py`
- Open-loop canonical theory: `docs/thesis/theory/THEORY_STATUS.md`
- OWT mainline: `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
- Phase 2b empirical anchor: `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`
- LLaDA-SFT probe (closed): `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md`
- Prior recommendation (re-targeted by activation audit): `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`

---

*End of bounded OWT Protocol C pilot. Theorem A-ad lands in Appendix F
as formal conditional theorem; Protocol C closes with an honest
negative on bucketed state. No new GPU/HPC. No mainline displacement.
Re-opening precondition stated in §6.*
