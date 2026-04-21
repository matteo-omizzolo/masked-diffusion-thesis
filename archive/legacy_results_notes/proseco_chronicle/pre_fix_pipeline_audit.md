# Pre-Fix Pipeline Audit

**Date:** April 2026  
**Auditor:** Claude Code  
**Purpose:** Document the exact structure of the Phase 1 experiment pipeline before any patches are applied, identify all structural inconsistencies, and record the precise code locations to modify.

---

## 1. Protocol A — Definition in Code Terms

### Where Δ_t is computed

`scripts/run_phase1_pilot.py`, `run_protocol_a()`, lines 77–109.

```python
y_base   = gen.run_base(seed=seed)                          # no-correction trajectory
y_branch = gen.run_branch(t_corrected=t, seed=seed)         # same seed → CRN
gain = estimate_single_step_gain(y_base, y_branch, F="neg_nll")
delta = gain["delta"]   # = y_branch["neg_nll"] - y_base["neg_nll"]
```

`estimate_single_step_gain` is in `src/mdm_playground/scheduling/gain.py`.
It computes `delta = F(y_branch) - F(y_base)` where F = "neg_nll" looks up
`y["neg_nll"]` — the GPT-2 reference-model negative NLL of the final tokens.

**Assessment:** Δ_t computation is correct. CRN is implemented (same seed for
base and branch). `neg_nll` is the right quality functional.

### Where TCR_t is computed

Same call, `gain.py` lines 68–69:
```python
n_changed = int((tokens_base != tokens_branch).sum())
tcr = n_changed / D
```
Hamming distance between final token sequences, normalized by D.

**Assessment:** Correct. TCR_t ≠ Δ_t distinction is maintained.

### Where entropy / margin / quality_mass are computed

`backends/mdlm.py`, `_extract_signals()`, called inside `_run_loop()` at line 293.

```python
signals = self._extract_signals(x, t)   # AFTER predictor step, BEFORE corrector
per_step_signals.append({"t": step_i, **signals})
```

`run_protocol_a` then reads signals from `y_base["per_step_signals"][t]` (line 91 of the pilot script). These are the signals from the **base** trajectory at step t.

**Timing:** Signals are extracted on the post-predictor, pre-corrector state at step t. This is correct — the signal at step t should describe the state at which the scheduling decision is made.

### Which positions are used for the signals — THE BUG

`_extract_signals()` in `backends/mdlm.py`, line 338:
```python
revisable = (x[0] != self.mask_id)   # unmasked positions
```

`signals.py` docstring (line 9–10):
> "Signals are computed only over **revisable** positions  
> (i.e., **unmasked** positions that a corrector could resample)"

**This is the core semantic error.** The MDLM corrector (`_apply_corrector`)
resamples **masked** positions (`x == mask_id`), not unmasked ones. The
"revisable" concept in the code means "unmasked", but the corrector can only
act on "masked" tokens. These are opposite sets.

Because MDLM commits highest-confidence tokens first, the unmasked (committed)
positions have near-zero entropy by construction. Computing entropy over them
gives ≈ 0 everywhere. This explains why all signals = 0 in the real MDLM run.

**What "revisable" should mean** for this experiment: the set of positions the
corrector can change at step t = the **masked** positions. Their entropy is the
uncertainty the corrector could reduce. Their inverse margin is the model's
confidence gap at still-undecided positions.

### Baseline trajectory definition

`y_base = gen.run_base(seed=seed)` — full predictor run, no correction, fixed seed.
Branch trajectories use the same seed with `torch.manual_seed(seed)` at the start
of `_run_loop`. The corrector fires at exactly step t_corrected.

**Assessment:** Correct. CRN is maintained. f_base is constant within a trajectory.

---

## 2. Protocol B — Definition in Code Terms

### Where G(S) is computed

`src/mdm_playground/scheduling/evaluate.py`, `evaluate_schedule()`:
```python
y_base  = generator.run_base(seed=seed)
y_sched = generator.run_with_schedule(allocation=allocation, seed=seed)
gain_info = estimate_single_step_gain(y_base, y_sched, F=F)
G = gain_info["delta"]  # = y_sched["neg_nll"] - y_base["neg_nll"]
```

**Assessment:** Correct. G(S) is the joint quality gain from the full scheduled trajectory.

### Where A(S) = sum(Δ_t) is computed

`scripts/run_phase1_pilot.py`, `run_protocol_b()`, line 82:
```python
A = sum(delta_trace.get(t, 0.0) for t in allocation)
```
where `delta_trace` = mean Δ_t across Protocol A trajectories (line 130).

**Assessment:** Correct. A(S) is the additive surrogate using per-step means.

### Pairwise interaction estimation

`run_phase1_pilot.py`, lines 156–180:
- For each pair (t, t'): compute `y_base`, `y_t`, `y_tp`, `y_pair` with the **same seed**
- `ξ = G({t,t'}) - Δ_t - Δ_{t'}`

**Assessment:** Correct. CRN maintained across all four evaluations.

---

## 3. Structural Inconsistencies

### 3.1 Wrong eligible-position set for signals [CRITICAL BUG]

**Location:** `backends/mdlm.py` line 338; `signals.py` line 96.

**Expected:** `revisable = (x == mask_id)` — masked positions the corrector can change.  
**Actual:** `revisable = (x != mask_id)` — unmasked (committed) positions.

The committed positions have near-zero entropy (MDLM only commits when confident).
Computing entropy over them gives ≈ 0. All three signals (entropy, inverse margin,
quality_mass_proxy) are ≈ 0 for all steps in all real MDLM trajectories.

**Effect:** ε_rms ≈ 0.992 (maximum possible), Spearman = NaN. No calibration
information whatsoever. Protocol A's central output (signal–gain correlation)
is entirely invalidated.

**Note on `unmasked_fraction`:** After fixing the signal positions to masked,
`unmasked_fraction` must still be computed as `(x != mask_id).sum() / D`
(i.e., the fraction of non-masked tokens). The fix only changes which positions
entropy/margin are computed over — not the unmasked-fraction scalar.

### 3.2 Corrector resamples all masked positions [CRITICAL DESIGN BUG]

**Location:** `backends/mdlm.py` `_apply_corrector()`, line 323.

**Expected behavior:** A "corrector" for MDLM should resample a small, targeted
subset of masked positions — specifically those where the model is most uncertain —
so that it (a) leverages available context, (b) does not overwrite positions the
predictor would have confidently placed, and (c) does not disrupt the predictor's
incremental denoising schedule.

**Actual behavior:** Resamples ALL masked positions at once. At step t=0, this
means resampling ~97% of the 1024-token sequence from a model with 1.4% context.
The resulting sequence is near-random with no coherence. Subsequent predictor
steps find an anomalous state (most tokens placed but with low confidence) and
cannot recover → Δ_t << 0 at all early/mid steps.

**Why this produces all-negative Δ_t:** MDLM's predictor is trained to denoise
from a fully-masked initial state, placing tokens one at a time in confidence
order. Starting from a state where 97% of tokens are already placed (but with
low confidence from a single aggressive corrector sweep) is out-of-distribution
for the predictor. The generated text has poor coherence → higher NLL under GPT-2.

**Effect:** All 1280 (t, trajectory) data points have Δ_t ≤ 0. The "oracle"
schedule G(S_B*) ≈ 0 (correct nowhere or only at the last 1–2 steps). Theorem A
is vacuous not because ε is large (signal is broken) but also because G itself
is ≤ 0 — there is no positive gain to award.

### 3.3 Theorem A check uses entropy signal (which is 0) for G(Ŝ_B) estimate

**Location:** `scripts/run_phase1_pilot.py`, `compute_summary()`, lines 272–276.

```python
mean_entropy = entropy.mean(axis=0)
top_B_idx = np.argsort(mean_entropy)[::-1][:B]
G_top_B_estimate = float(mean_delta[top_B_idx].sum())
```

Because mean_entropy ≈ 0 everywhere, `argsort` returns an arbitrary ordering.
`G_top_B_estimate` uses mean_delta at those randomly-ordered indices, giving
a near-zero or slightly-negative value. This is not the estimated gain of any
meaningful schedule.

**Effect:** The theorem A check is trivially vacuous due to both bugs.

### 3.4 Surrogate and real backend share the same "revisable = unmasked" convention

The `signals.py` public API also has `revisable = tokens != mask_id` (line 96).
The SurrogateGenerator, however, generates correct signals analytically without
using `compute_signals`. So the surrogate test results are not affected by
this bug — the surrogate signal values are correct by construction.

The discrepancy means: surrogate mode produces valid (synthetic but structured)
signals, while real MDLM mode produces zero signals. This is why the
`phase1_interpretation.md` (written from a surrogate run) described sensible
non-zero Spearman correlations, while the real MDLM result showed NaN.

### 3.5 No burn-in safeguard in the corrector

The corrector fires at every step t ∈ {0, …, T−1} regardless of how little
context is available. The experiment design (`docs/experiments/entropy_proxy_experiment.md`)
does not require a burn-in skip, but Proposition B predicts that early steps
are low-gain. The combined effect of (a) all-masked resample and (b) no safeguard
means the corrector is maximally harmful at exactly the steps where context is
minimal (t ≈ 0, u_t ≈ 0).

---

## 4. Logging Omissions

All required fields are logged for Protocol A and B. The only omission is that
`per_step_signals` in the real run contains near-zero values, making downstream
calibration analyses (e.g., `analyze_phase1.py`) produce artifacts rather than
errors. No critical logging fields are missing.

---

## Summary of Required Fixes

| Priority | File | Change |
|----------|------|--------|
| CRITICAL | `backends/mdlm.py` `_extract_signals()` line 338 | Change `!= mask_id` to `== mask_id`; fix `unmasked_fraction` computation; update docstring |
| CRITICAL | `backends/mdlm.py` `_apply_corrector()` | Replace full resample with confidence-guided partial resample + burn-in safeguard |
| MODERATE | `signals.py` docstring | Clarify that "revisable" = eligible update set, which for MDLM correctors = masked positions |
| MODERATE | `scripts/run_phase1_pilot.py` `compute_summary()` | Pass corrector_mode parameter through so Theorem A check uses the correct signal |
| LOW | Add tests | Test that signals are non-zero when masked positions exist; test corrector respects fraction |

---

## Structural elements that are CORRECT (do not change)

- CRN implementation (same seed for base and branch)
- Δ_t computation (`gain.py`) — correct formula and separation from TCR
- Protocol B joint-gain computation (`evaluate.py`) — correct
- Pairwise ξ computation — correct, CRN maintained
- η_B computation — correct formula, valid for any corrector
- Theorem A bound formula `2Bε + 2η` — correct; the problem is the inputs (ε, G)
- T_low identification logic — correct; just needs non-degenerate mean_delta values
- Summary JSON structure — correct and complete
