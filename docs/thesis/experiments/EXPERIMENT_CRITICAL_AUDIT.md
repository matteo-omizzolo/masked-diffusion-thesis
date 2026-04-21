# Experiment Critical Audit — Phase 1 ProSeCo-OWT

**Author:** Claude Code (rigorous co-advisor review)
**Date:** 2026-04-19
**Status:** Historical audit of `results/phase1_proseco_owt_full/` (N=50, T=64) and
predecessor pilot `results/phase1_proseco_owt/` (N=20, T=64). Kept for provenance;
the current thesis mainline is Phase 2b / Phase 3a.

**Headline verdict:** The canonical claim *"entropy_bot_B beats uniform by +29% at B=8
(thesis-grade evidence)"* **is not supported** by the data. It confuses the **additive
surrogate** A(S) = Σ ⟨Δ_t⟩ with the **true joint gain** G(S) = F(y_T^S) − F(y_T^base).
When the pipeline actually runs entropy_bot_B end-to-end, its G(S) at B=8 is **0.077** —
roughly **10× worse** than uniform's **0.758**. The +29% number is a comparison between
two *different* quantities measured on two *different* code paths, and one of them (A(S))
has a documented slack η₉₅ = 0.68 at B=8 that exceeds the claimed surplus of ≈0.20.

This audit also documents: single-seed policy evaluation (no standard errors on G),
pilot↔full-run numerical flip at the same B despite seed-aligned runs, Theorem A bound
vacuous at all B, and several implementation ambiguities that make the current numbers
unsafe to cite in the thesis.

---

## 1. Central finding — A(S) vs G(S) confusion

### 1.1 What the canonical doc claims

From `docs/thesis/CURRENT_INDEX.md`:

> Phase 1 — COMPLETE. entropy_bot_B +29% over uniform at B=8.
> Entropy-minimum scheduling principle confirmed at thesis-grade (N=50).

From `docs/thesis/experiments/RESULTS_STATUS.md` (full-run policy table, B=8):

| Policy          | G      | vs uniform |
|-----------------|--------|------------|
| uniform         | 0.688  | —          |
| entropy_bot_B   | 0.887  | +29%       |
| middle          | 1.118  | +62%       |
| oracle          | 2.281  | +231%      |

### 1.2 Where these numbers actually come from

These numbers are read from `results/phase1_proseco_owt_full/inverted_policy_analysis.json`
(keyed under `policy_G`). They are written by
`archive/legacy_framework/scripts/analyze_inverted_policies.py`.
The relevant line is:

```python
# archive/legacy_framework/scripts/analyze_inverted_policies.py:112
G_list.append(sum(pt[t]["delta"] for t in steps))
```

i.e. for a policy that selects `steps = sel(per_t, B)`, the "G" reported is

> A(S) := Σ_{t ∈ S} ⟨Δ_t⟩

where ⟨Δ_t⟩ is the **Protocol A** per-step one-loop marginal gain measured on the
*uncorrected* base trajectory y_t^base. This is the **additive surrogate**, not the
true joint gain of running predictor + chosen corrector loops end-to-end.

### 1.3 What the pipeline-evaluated G(S) actually is

The summary file `results/phase1_proseco_owt_full/summary.json` stores a separate
`policy_comparison` block (keyed `B4`, `B8`, `B16`) that *is* produced by
`src/mdm_playground/scheduling/evaluate.py::evaluate_schedule`, i.e. by actually running
the ProSeCo-OWT predictor+corrector pipeline with the selected step set S and measuring
F(y_T^S) − F(y_T^base). These are the real G(S) numbers.

At **B=8** the true G(S) values are:

| Policy            | G(S) true | vs uniform |
|-------------------|-----------|------------|
| uniform           | **0.75847** | —        |
| middle            | 0.65179   | −14%      |
| oracle (mean-field top-B of ⟨Δ_t⟩) | 0.63706 | −16%      |
| front / entropy_top_B / margin_top_B / quality_top_B / entropy_burn_in_gated | 0.06559 | −91% |
| bottom_B (= entropy_bot_B signal) | **0.07664** | **−90%** |
| back              | 0.08759   | −88%      |

Every single policy loses to uniform at B=8 on the true G. The "champion" entropy_bot_B
loses by a factor of **≈10×**, not wins by +29%.

At B=16 the pattern is similar: only the mean-field oracle beats uniform
(1.007 vs 0.936), entropy_bot_B lands at 0.466 (−50%), middle at 0.725 (−23%).

At B=4 *no* policy beats uniform (uniform 0.544, oracle 0.504, middle 0.374,
bottom_B 0.072).

### 1.4 Why this is not a rounding error

Three independent reasons the A(S) column in the canonical table is misleading:

1. **Units mismatch.** Δ_t is measured against y_t^base, so Σ_t Δ_t is the sum of
   gains as if each corrector ran on the untouched base trajectory. Under Protocol B
   the loops compose: a loop at t=23 changes the trajectory that the next loop at
   t=31 sees. Writing "G" on an additive sum pretends this composition is free.

2. **Additivity slack η exceeds the claim.** The same summary reports
   η_95(B=8) = **0.680** (eta_95 under `eta_by_B["8"]`). The additive surrogate and
   the true G differ by up to ≈0.68 at the 95th percentile across M=30 sampled
   schedules. The "entropy_bot_B beats uniform by ≈0.20" claim is entirely inside
   that slack, i.e. it is within noise *by the document's own assumption-B
   measurement*. We should have flagged this at the time η was computed.

3. **Sign-flipping empirical check.** The pipeline-evaluated `bottom_B` is exactly
   the entropy_bot_B policy (the signal used is `signal_traces["bottom_B"] =
   mean_entropy` at `archive/legacy_framework/scripts/run_phase1_proseco_owt.py:285`, and `allocation.py::bottom_B`
   returns the B indices of smallest signal). Its true G at B=8 is 0.077.
   The additive surrogate says +0.199 above uniform; the pipeline says −0.682
   below uniform. Sign flipped, magnitude bigger, same policy.

### 1.5 Why Theorem A did not catch this

Theorem A (`CANONICAL_RESEARCH_DIRECTION.md`) bounds

> G(S_B*) − G(Ŝ_B) ≤ 2Bε + 2η_B

with ε the proxy calibration RMS and η_B the additivity slack at budget B.

The full-run `theorem_A_bound_check` records:

| B  | 2Bε+2η₉₅ | G_oracle (additive) | bound_useful |
|----|----------|---------------------|--------------|
| 4  | 1.895    | 0.613               | false        |
| 8  | 3.497    | 1.198               | false        |
| 16 | 6.986    | 2.272               | false        |

At every B, the bound is 3× bigger than the thing it is bounding. The theorem is
currently **vacuous** on this system. A proxy-based allocator is allowed to be up
to 3× worse than the oracle at B=8, which is exactly what happened on the true G.
The theorem therefore did *not* fail in a mathematical sense — it correctly predicts
that no guarantee was available — but nothing in the experimental report treats
`bound_useful=false` as a red flag, and the `RESULTS_STATUS.md` table is presented
as a win anyway.

### 1.6 Summary of §1

The thesis does not currently have *any* experimental evidence that a signal-adaptive
corrector schedule beats uniform on the true joint gain G(S). It has evidence that,
**under the additivity surrogate**, signal-minimum selection out-scores uniform at
B=8 — but the measured additivity slack at B=8 is larger than the claimed surplus,
and the one end-to-end pipeline evaluation we did shows entropy_bot_B losing by 10×.

This audit was the reason `CURRENT_INDEX.md` and `RESULTS_STATUS.md` were
reframed around the Phase 2b paired-evaluation / Phase 3a scheduling mainline.
The underlying warning remains: A(S) vs G(S) must not be conflated, and any
future comparison must be paired and seed-aligned.

---

## 2. Secondary finding — single-seed policy comparison, no confidence intervals

`archive/legacy_framework/scripts/run_phase1_proseco_owt.py::run_policy_comparison` evaluates every (policy, B)
combination against a **single** sample pair:

```python
# archive/legacy_framework/scripts/run_phase1_proseco_owt.py:289
seed = seed_base + 88888
```

Every policy at every B is evaluated once with seed 88930. There is no standard error
on the G values in `summary.json.policy_comparison`; they are point estimates from one
trajectory each. The numbers that look like ranking signal (uniform 0.758, middle 0.652,
oracle 0.637, bottom_B 0.077 at B=8) could easily be dominated by that single seed's
quality functional variance.

Evidence that this is not paranoia:

### 2.1 Pilot vs full-run numerical flip

Same script, same seed structure, two different N:

| Policy    | Pilot N=20 G (B=8) | Full N=50 G (B=8) | Flip? |
|-----------|--------------------|--------------------|-------|
| uniform   | 0.403              | 0.758              | ×1.88 |
| middle    | 0.415 ✓ beats      | 0.652 ✗           | Yes   |
| bottom_B  | 0.436 ✓ beats      | 0.077 ✗           | Yes (10×) |
| oracle    | 0.840 ✓ beats      | 0.637 ✗           | Yes   |
| front     | −0.076             | 0.066              | –     |
| back      | 0.043              | 0.088              | –     |

In the N=20 pilot the conclusion was "middle, bottom_B, oracle all beat uniform". In
the N=50 full run the conclusion is "nothing beats uniform". The policy_comparison
block uses one seed per (policy, B) in each run; N is the number of Protocol-A trajectories
that feeds the Δ calibration and η/γ numbers but does **not** multiply the number of
policy_comparison evaluations. So what we are seeing in the above table is pure
single-seed noise changing the story entirely.

### 2.2 Consequence

No conclusion about policy ranking at B ∈ {4, 8, 16} on this dataset can be drawn
from the current policy_comparison block. We do not know whether uniform truly
dominates (§1.3) or whether that is a second single-seed artifact. The only way to
answer this is to evaluate each (policy, B) on ≥ 20–30 independent base trajectories
and report G_mean ± SE, then do a paired t-test against uniform.

---

## 3. Third finding — mean signal profile is near-monotone, so "signal-adaptive"
  collapses to "positional"

`mean_delta_profile` in `inverted_policy_analysis.json` rises near-monotonically from
~−0.03 at t=0 to a plateau ~0.14–0.16 in t ∈ [20, 45], then decays to ~0.06 at t=63.
The mean entropy profile mirrors this (high → low as t grows).

Under the current top-B / bottom-B policies we select B positions on the **trajectory
mean** of each signal (line 285: `signal_traces = { ... "entropy_top_B": mean_entropy,
...}`). This means:

- `entropy_top_B` ≡ `front` (high-entropy = early steps)
- `entropy_bot_B` ≡ `back` selected from the tail where entropy is lowest
- `middle` is hand-picked from t ∈ [T/4, 3T/4]

All three "signal-adaptive" policies are in fact *positional* policies under different
names. The informative experiment — *do per-trajectory signals identify the best steps
within a single trajectory?* — is masked by the use of cross-trajectory means.

The Protocol A Spearman correlation between per-trajectory per-step signal and
per-step Δ is:

| Signal             | ⟨ρ⟩     | σ      |
|--------------------|---------|--------|
| entropy            | −0.191  | 0.252  |
| inverse_margin     | −0.200  | 0.234  |
| quality_mass_proxy | −0.185  | 0.246  |
| unmasked_fraction  | +0.060  | 0.318  |

So, per-trajectory, low entropy correlates with *high* Δ (ρ ≈ −0.19). That is a real
but weak signal. The **policy** evaluation, however, does not exploit this — it uses
trajectory-averaged profiles. Any claim about signal-adaptivity in the current data
is therefore a claim about a population profile, not a per-trajectory decision rule.

This is also why "entropy_bot_B ≈ back" in the pipeline G: both policies select the
same late steps when signals are averaged.

---

## 4. Fourth finding — the "oracle" is not an oracle

The label "oracle" in `policy_comparison` and `inverted_policy_analysis.json` refers
to **top-B selection on the trajectory mean of ⟨Δ_t⟩**, not per-trajectory top-B, and
certainly not true per-trajectory joint-optimum schedules. This is explicit in the
additive-surrogate construction.

Consequences:

1. When we say "oracle beats uniform by 231% at B=8 under A(S)" we mean: if you could
   see the population-mean Δ profile and allocate B corrector loops to its peaks, the
   **additive** surrogate is 2.3× larger than uniform's. That is a population-level
   upper bound on A, not a per-sample planning oracle.

2. Under the true G in `summary.json.policy_comparison`, this "oracle" *loses* to
   uniform at B=4 and B=8 and only ~7% beats uniform at B=16. So even the most
   optimistic additive-based recommendation is pipeline-losing at small B.

3. A true "per-trajectory joint oracle" would require enumerating or sampling
   schedules S of size B and picking argmax_S G(S) per trajectory. That is
   tractable for B=4 (combinatorial) and samplable (e.g. via Monte Carlo over random
   schedules) for B=8. The current pipeline does *not* evaluate this, so we have no
   empirical ceiling for how much corrector scheduling *could* gain.

Without an honest oracle, we cannot tell whether "uniform ≥ all proxy-based policies"
is (a) a deep fact about ProSeCo-OWT (low gain-from-scheduling headroom), or (b) a
Protocol-A-is-the-wrong-surrogate artifact, or (c) both.

---

## 5. Fifth finding — quality functional F is a weak signal

F = −NLL of GPT-2 on the first 512 tokens of the decoded sequence. This is used both
to compute Δ_t at Protocol A and to compute G at policy evaluation.

Issues:

1. GPT-2 NLL is not calibrated as a semantic quality metric. A schedule that
   marginally lowers NLL on 50 samples may reflect fluency artifacts (tokenization
   boundaries, repetition penalties interacting with left-context masks), not the
   "correction quality" we want to claim.

2. The canonical doc does not report F's own scale / baseline variance. If sample-to-sample
   F-variance is ≈ 0.8, then a full-run G of 0.76 at N=50 has SE ≈ 0.8/√50 ≈ 0.11.
   The claimed "entropy_bot_B +0.20 over uniform" is ≈ 1.8σ on the additive
   surrogate, inside a slack of η₉₅=0.68 on G, and indistinguishable from zero once
   η is accounted for.

3. A more informative F would be (a) MAUVE against a held-out OWT reference, or
   (b) a held-out LLM judge score, or (c) a task metric if we restrict to a
   conditional setting. Until F's behavior is characterized, the policy-ranking
   results are only as trustworthy as its variance.

---

## 6. Sixth finding — cross-backbone / cross-corrector generality

All current evidence is on a single backbone+corrector pair: ProSeCo-OWT checkpoint
(scheduling/backends/proseco_owt.py) with the ProSeCo "annealed-refinement" corrector
at the model's default temperature/alpha schedule.

The thesis claims a *signal-adaptive corrector scheduling principle*. Even if the
entropy-minimum claim were statistically sound on ProSeCo-OWT (it is not, per §1), it
would still need to hold across at least:

- A second backbone (MDLM checkpoint on OWT), to rule out this being a ProSeCo-specific
  artifact.
- A second corrector kernel (a simple resample-then-remask kernel, or MDLM-conf), to
  rule out the ProSeCo corrector's own NFE schedule dominating the signal.
- A second quality functional (MAUVE with owt-reference, or a held-out judge).

These are explicitly listed as "next-phase" items in
`docs/thesis/experiments/HPC_NEXT_RUN_PLAN.md` but not as audit items against the
current "complete" claim.

---

## 7. Seventh finding — the "inverted direction" framing is provisional

The analysis in `PROSECO_ANALYSIS.md` describes the Spearman correlations as
"inverted direction (NEG)" and argues that this is a substantive finding: "low entropy
predicts high Δ". Two caveats that the current doc underplays:

1. The absolute correlation is small (⟨ρ⟩ ≈ −0.19, σ ≈ 0.25). Per-trajectory the sign
   of ρ flips fairly often (we have no reported per-trajectory sign counts — we should
   compute the fraction of the 50 trajectories with ρ < 0).

2. The "inversion" may be an artifact of the ProSeCo corrector's own internal
   behavior — at low-entropy positions R_t may be near-final and the single corrector
   loop is working in a near-committed distribution where small local noise shows
   the largest Δ_t, not because the step is *beneficial* but because
   F sometimes improves under resampling near converged context. We need to check
   whether NLL improvements at low-entropy steps persist after the subsequent
   predictor runs, or whether they are "erased" before T. The Protocol B G=0.08
   for entropy_bot_B is exactly the kind of evidence that suggests *erased*.

---

## 8. Corrections to the canonical docs

The following specific claims in `docs/thesis/CURRENT_INDEX.md` and
`docs/thesis/experiments/RESULTS_STATUS.md` need to be rewritten:

### 8.1 `CURRENT_INDEX.md`

- Strike: "Phase 1 — COMPLETE. entropy_bot_B +29% over uniform at B=8. Entropy-minimum
  scheduling principle confirmed at thesis-grade (N=50)."
- Replace with: "Phase 1 — INSUFFICIENT. Additive-surrogate signal shows entropy_bot_B
  > uniform, but pipeline-evaluated G(S) at N=50, single seed, shows entropy_bot_B ≈
  0.08 vs uniform ≈ 0.76 at B=8. η₉₅(B=8) = 0.68 exceeds the additive surplus of
  0.20. Theorem A bound is vacuous at all B. Next phase must measure G(S) with
  paired standard errors across ≥ 30 base seeds per policy before any scheduling
  principle is asserted."

### 8.2 `RESULTS_STATUS.md`

- The policy table under "Phase 1 full run (N=50)" is currently printed from
  `inverted_policy_analysis.json::policy_G` (i.e. A(S)). Either:
  - relabel columns to `A(S)` with a prominent note that this is the additive
    surrogate, **or**
  - replace with the true G(S) column from `summary.json::policy_comparison`,
    but with "single seed; no SE" as a footnote and the caveat that no policy beats
    uniform at B=8.

### 8.3 `PROSECO_ANALYSIS.md`

- The C6 line ("passes with inverted signal direction") needs to be downgraded to
  "Spearman ⟨ρ⟩ ≈ −0.19 ± 0.25 — weak and sign-noisy; inversion direction is a
  hypothesis, not a confirmed property".
- The "top Δ_t steps" list (t=23, t=31, …) is computed on the mean profile. The
  per-trajectory variance in where Δ_t peaks should be reported (we do not currently
  have this; add to analysis spec).

### 8.4 `CANONICAL_EXPERIMENT_OVERVIEW.md` / Theorem A assumption ledger

- Add an explicit red flag: "Current full-run evidence does not satisfy Assumption B
  (approximate additivity) at a useful level: η₉₅(B=8) > mean |Δ_t| × B × 0.75. The
  theorem is therefore vacuous at B=8 on ProSeCo-OWT and cannot currently be used
  to claim a performance bound."

---

## 9. What this audit does **not** rule out

To be rigorous, these are still live possibilities that the above audit does not kill:

1. **Signal-adaptive scheduling still works, but the signal must be per-trajectory
   not mean-profile.** Our current evidence uses trajectory-averaged signals, which
   collapses every "signal_top_B" to a positional policy. A per-trajectory bottom-B
   over entropy on each of 30+ base samples might still outperform uniform. We have
   not tested this.

2. **The additive surrogate is wrong at B=8 but useful at smaller budgets.** At B=2
   or B=3, η_B is smaller; the additive prediction may track true G. Worth testing.

3. **The quality functional is the problem.** GPT-2 NLL is not calibrated for
   semantic correction; maybe with MAUVE/judge F, the signal-adaptive effect would
   come back.

4. **Theorem A's vacuous bound is a real fact about this system** (not a proof
   failure). If so, the thesis has two legitimate outputs: (a) the negative result
   that on ProSeCo-OWT uniform is within corrector-scheduling error of optimal, and
   (b) a characterization of when the bound is non-vacuous (smaller η_B regime →
   smaller T or smaller corrector kernel). This would be a smaller, but honest,
   thesis contribution.

All four are worth investigating; §10 describes how.

---

## 10. Recommendations for the next phase (expanded in `NEXT_PHASE_EXPERIMENT_PLAN.md`)

1. **Must-have:** re-evaluate every (policy, B) on ≥ 30 independent base seeds.
   Report G_mean ± SE, paired t-test vs uniform. Do this before touching any new
   backbone.
2. **Must-have:** compute true per-trajectory oracle via random-schedule Monte Carlo
   (sample 200 schedules of size B per trajectory, take argmax G). This gives an
   honest headroom estimate.
3. **Must-have:** compute `A(S) vs G(S)` **paired** across the 30× sampled schedules
   used for η; report Spearman rank correlation between A- and G-rankings at each B.
   If ρ_A,G < 0.5 at B=8 we have quantitative proof that the additive surrogate
   cannot be used as a thesis-level proxy.
4. **Should-have:** per-trajectory signal-adaptive policies (not mean-profile).
5. **Should-have:** swap F to MAUVE vs owt-reference, repeat the critical rows.
6. **Should-have:** add MDLM backbone + simple resample corrector as a second
   (backbone, corrector) pair.
7. **Defer:** anything cross-domain or cross-model-family until the above are clean
   on ProSeCo-OWT.

---

## 11. Appendix — evidence files audited

- `results/phase1_proseco_owt_full/summary.json` (N=50, T=64)
- `results/phase1_proseco_owt_full/inverted_policy_analysis.json` (post-hoc A(S))
- `results/phase1_proseco_owt/summary.json` (pilot N=20)
- `archive/legacy_framework/scripts/analyze_inverted_policies.py` (line 112 —
  additive surrogate definition)
- `archive/legacy_framework/scripts/run_phase1_proseco_owt.py` (lines 285, 289 — signal traces + single seed)
- `src/mdm_playground/scheduling/evaluate.py` (`evaluate_schedule` — true G(S))
- `src/mdm_playground/scheduling/allocation.py` (`bottom_B` — signal-minimum selection)

The A(S) vs G(S) distinction is the single most important takeaway. Every downstream
recommendation in this document (single-seed SE, oracle Monte Carlo, paired A↔G
rank, per-trajectory signals, F swap, backbone swap) exists because once §1 is true,
we no longer know which direction the real signal is pointing.
