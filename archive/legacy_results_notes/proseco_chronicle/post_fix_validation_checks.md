# Post-Fix Validation Checks

**Date:** April 2026  
**Purpose:** Document what was verified after applying Bug #1 and Bug #2 fixes to `backends/mdlm.py`.

---

## Fixes Applied

### Fix 1: Signal computation positions (`_extract_signals`)

**File:** `src/mdm_playground/scheduling/backends/mdlm.py`, `_extract_signals()`

**Change:**
```python
# Before (wrong):
revisable = (x[0] != self.mask_id)   # unmasked positions — near-zero entropy by construction
n_rev = int(revisable.sum())
...
return {"unmasked_fraction": float(n_rev / D), ...}

# After (correct):
eligible = (x[0] == self.mask_id)    # masked positions = corrector's action set
n_eligible = int(eligible.sum())
n_unmasked = D - n_eligible           # separate unmasked_fraction computation
...
return {"unmasked_fraction": float(n_unmasked / D), ...}
```

**Why:** MDLM commits tokens in confidence order; committed positions have p_argmax ≈ 1,
so their entropy ≈ 0 by construction. The signal must measure uncertainty at positions
the corrector can still change, which are the **masked** positions.

`unmasked_fraction` continues to measure predictor progress (committed fraction) —
it was always semantically correct, just computed as a side-effect of the wrong `revisable`
variable.

### Fix 2: Corrector design (`_apply_corrector`)

**File:** `src/mdm_playground/scheduling/backends/mdlm.py`, `_apply_corrector()`

**Change:** Replaced full masked-position resample with confidence-guided partial resample.

New behaviour:
1. **Burn-in safeguard:** return x unchanged if `n_unmasked / D < min_unmasked_fraction` (default 0.05)
2. **Eligible set:** only masked positions (`x[0] == mask_id`)
3. **Selection:** bottom `correction_fraction` (default 0.20 = 20%) by model confidence
4. **Resample:** only the selected positions; all others unchanged

Old behaviour preserved as `legacy_resample_all=True` flag for diagnostic comparison.

### Fix 3: `signals.py` docstring

**File:** `src/mdm_playground/scheduling/signals.py`

Updated module docstring and inline comment to clarify:
- "Revisable" means the corrector's action set, which for MDLM = **masked** positions
- The default (`tokens != mask_id`) is the wrong convention for MDLM
- Callers must pass `revisable_mask = (tokens == mask_id)` explicitly for MDLM use
- `unmasked_fraction` is always computed as `(tokens != mask_id).sum() / D`, independent
  of the revisable definition

---

## Validation Tests

**File:** `tests/test_scheduling_pipeline.py`  
**Result:** 27/27 passing

### TestSignalSanity (10 tests)

| Test | What it verifies |
|------|-----------------|
| `test_signals_nonzero_when_masked_positions_exist` | entropy > 0 with uniform logits over masked positions |
| `test_signals_zero_when_no_masked_positions` | entropy = 0 when no masked positions |
| `test_n_revisable_equals_masked_count` | n_revisable counts masked tokens correctly |
| `test_unmasked_fraction_is_committed_fraction` | unmasked_fraction = (L - n_masked) / L |
| `test_unmasked_fraction_equals_one_when_all_committed` | unmasked_fraction = 1.0 at end of generation |
| `test_entropy_decreases_with_more_peaked_logits` | peaked > uniform in confidence → lower entropy |
| `test_inverse_margin_bounds` | inverse_margin ∈ [0, 1] |
| `test_quality_mass_proxy_bounds` | quality_mass_proxy ∈ [0, 1] |
| `test_peaked_logits_give_low_inverse_margin` | near-deterministic logits → inverse_margin < 0.1 |
| `test_signals_dict_has_all_required_keys` | all 5 required keys present |

### TestCorrectorSanity (7 tests)

| Test | What it verifies |
|------|-----------------|
| `test_corrector_only_modifies_selected_masked_positions` | unmasked positions unchanged |
| `test_corrector_modifies_approximately_correct_fraction` | ~correction_fraction × n_masked positions changed |
| `test_burn_in_skip_when_context_too_sparse` | returns x unchanged below min_unmasked_fraction |
| `test_burn_in_fires_above_threshold` | fires and changes tokens when above threshold |
| `test_legacy_resample_all_modifies_all_masked` | legacy flag enables old (full resample) behaviour |
| `test_corrector_noop_when_no_masked_positions` | safe no-op when nothing to resample |
| `test_corrector_output_has_correct_shape` | shape (1, L) preserved |

### TestProtocolSanity (6 tests)

| Test | What it verifies |
|------|-----------------|
| `test_delta_is_branch_minus_base` | Δ_t = F(branch) − F(base) numerically |
| `test_delta_negative_when_branch_worse` | sign correct for degraded branch |
| `test_tcr_is_separate_from_delta` | same quality but different tokens → delta=0, tcr>0 |
| `test_tcr_zero_when_tokens_identical` | identical token sequences → TCR = 0 |
| `test_tcr_one_when_all_tokens_differ` | all tokens differ → TCR = 1 |
| `test_gain_dict_has_required_keys` | delta, tcr, n_changed all present |

### TestComputeSignals (4 tests)

| Test | What it verifies |
|------|-----------------|
| `test_mdlm_masked_revisable_gives_nonzero_entropy` | revisable_mask=masked → non-zero entropy |
| `test_default_unmasked_revisable_gives_zero_entropy_for_committed` | default (unmasked) → near-zero entropy with peaked committed logits |
| `test_unmasked_fraction_independent_of_revisable_mask` | unmasked_fraction same regardless of revisable definition |
| `test_n_revisable_matches_mask` | n_revisable counts revisable_mask correctly |

---

## Fast Surrogate Sanity Run

**Command:** `python scripts/run_phase1_pilot.py --surrogate --T 32 --N 5 --M 5 --P 20 --B_values 4,8`

**Result:** PASS — see `docs/experiments/results/fast_sanity_run.md` for full record.

Key numbers:
- ε (entropy, RMS) = **0.017** (was 0.992 with broken pipeline)
- Spearman(H, Δ) = **−0.19** (was NaN)
- Elapsed: 8.2 s (CPU)

---

## What Remains Before Results Are Citable

1. **HPC rerun** with real MDLM and corrected code (push + sbatch)
2. **Inspect corrected Δ_t profile:** verify some positive Δ_t values appear
3. **Inspect ε:** verify ε < 0.5 so Theorem A has meaningful non-trivial regime
4. **Update** `experiment_result_analysis.md` and `post_experiment_conclusion.md` with corrected results
5. **Update** `open_questions.md`, `proof_worklog.md` with corrected empirical record
