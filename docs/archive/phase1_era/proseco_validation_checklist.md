> **STATUS:** ARCHIVED
> **ARCHIVED ON:** 2026-04-22
> **SUPERSEDED BY:** `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md`
> **REASON:** Pre-launch checklist; executed. Provenance only.

---

# ProSeCo Experiment Validation Checklist

**Date:** 2026-04-17  
**Status:** Pipeline validated on surrogate; real-backend checks pending GPU run.

This checklist must be verified before accepting any ProSeCo results as meaningful.

---

## Structural checks (surrogate run — complete)

### ✓ Check 1: Surrogate pipeline runs end-to-end

**Command:**
```bash
python scripts/run_phase1_proseco.py \
    --surrogate --T 16 --N 5 --M 3 --P 10 --B_values 2,4 --seed 42 \
    --out_dir results/phase1_proseco_surrogate_sanity
```

**Result:** Passed (8.1s, CPU). All outputs generated correctly.

| Quantity | Result |
|---|---|
| ε (entropy, RMS) | 0.017 (non-degenerate) |
| Steps with Δ_t > 0 | 16/16 (surrogate by design) |
| Theorem A | Non-vacuous at B=2,4 |
| Policy comparison | Ran cleanly, 20 records |
| Pairwise γ (95th) | 0.009 (small relative to G) |

---

## Structural checks (real ProSeCo backend — pending GPU)

These checks require running on HPC with the MDLM checkpoint.

### Check 2: Signals are non-trivial

**What to verify:**
- `H_t`, `M_t^{-1}`, `Q_t` are NOT all-zero across steps
- Signals vary meaningfully across t = 0, …, T-1
- `unmasked_fraction` increases monotonically (it should, by MDLM design)

**Failure mode:** if signals are all zero, either:
1. The signal is computed over the wrong positions (bug in `revisable` mask), OR
2. The backbone produces near-uniform logits (unlikely for a well-trained model)

**Diagnostic:** print `per_step_signals` from a single `run_base` call. Check that:
- At t < 5: signals ≈ 0 (R_t ≈ ∅, no unmasked tokens)
- At t = 32: signals non-zero (u_t ≈ 0.5)
- At t = 63: signals reflect near-complete sequence

### Check 3: Corrector is genuinely correcting (not just re-predicting)

**What to verify:**
- `TCR_t > 0` for some t (tokens change after correction)
- `TCR_t` is NOT always 0 (corrector is doing something)
- `TCR_t` is NOT always 1 (corrector is not destroying everything)

**Expected behavior:**
- At early steps (u_t small, R_t small): TCR_t ≈ 0 (few unmasked positions to correct)
- At mid steps (u_t ≈ 0.3–0.7): TCR_t moderate (some committed tokens revised)
- At late steps (u_t ≈ 1.0): TCR_t larger (many committed tokens available to revise)

**Failure mode:** TCR_t = 0 everywhere → corrector output identical to input → bug in
the corrector application (possibly `unmasked` mask always empty, or argmax produces
same tokens as predictor).

### Check 4: Some Δ_t > 0 exists

This is the critical structural prerequisite for the thesis.

**What to verify:**
- At least some steps t have `Δ_t > 0` (corrector beneficial at those steps)
- `n_positive_delta_steps > 0` in `summary.json`

**Expected behavior:**
- Early steps (R_t small): Δ_t ≈ 0 (no revisions → no gain)
- Mid/late steps (R_t large): Δ_t ≷ 0 (corrector may improve or slightly hurt)

**If all Δ_t ≤ 0:** the corrector is never beneficial → either:
1. The annealed refinement is not effective on this backbone (backbone not corrector-trained)
2. GPT-2 NLL is insensitive to the kind of revision the corrector makes
3. `corrector_steps` is too few (try `corrector_steps=4`)
4. F is dominated by noise (try more trajectories N=50)

**Action if all Δ_t ≤ 0:**
- Increase corrector_steps to 4 or 8
- Try sampling instead of argmax (`corrector_xs = sample(log_p_corr)`)
- Check that corrector_x genuinely differs from x (TCR_t check above)
- Consider downloading proseco-owt checkpoint (co-trained corrector)
- Document as "corrector not beneficial for MDLM backbone" (informative negative result)

### Check 5: TCR_t and Δ_t logged separately

**Verified by design:** `run_phase1_proseco.py` records both `delta` and `tcr` separately
for each (t, i). `evaluate_schedule` and `estimate_single_step_gain` also keep them separate.

No formula anywhere conflates TCR_t with Δ_t. ✓

### Check 6: Schedule-level run composes multiple corrections

**What to verify:**
- `run_with_schedule({10: 1, 30: 1, 50: 1}, seed=s)` applies corrector at steps 10, 30, 50
- The final tokens differ from `run_base` (G ≠ 0 in most schedules)
- `G(S) ≠ 0` for random non-trivial schedules

**Diagnostic:** print `y_sched["schedule_steps"]` and compare tokens to base.

### Check 7: Protocol A/B outputs sufficient for theorem estimation

After a pilot run (N=20, M=15, P=120), verify:

| Quantity | Requirement | Check |
|---|---|---|
| ε (entropy) | Non-degenerate, not ≈ 1 | `calibration.entropy.eps_rms < 1.0` |
| Spearman(H, Δ) | Not NaN | `calibration.entropy.spearman_mean` defined |
| η_B | Finite for all B | `eta_by_B.B4.eta_95` defined |
| γ | Finite | `gamma.gamma_95` defined |
| n_positive_delta | > 0 | `n_positive_delta_steps > 0` |

---

## Decision tree after real run

```
Run complete → check results/phase1_proseco/summary.json

n_positive_delta_steps > 0?
├─ YES → Continue to full analysis
│         ε non-degenerate?
│         ├─ YES → Theorem A potentially non-vacuous → continue
│         └─ NO  → Signals don't predict gains → revisit signal design
└─ NO  → Corrector never beneficial
          TCR_t > 0?
          ├─ NO  → Corrector not applying changes → check code (revisable mask)
          └─ YES → Corrector changes tokens but hurts quality
                   Increase corrector_steps or download proseco-owt
```

---

## Additional lightweight tests

Run from repo root with `remdm311` env:

```bash
# Test 1: Backend import
python -c "
import sys; sys.path.insert(0, 'src')
# ProSeCo backend imports (without loading checkpoint)
from mdm_playground.scheduling.backends import proseco
print('Import OK')
"

# Test 2: Signal extraction logic (no GPU needed)
python -c "
import torch, sys; sys.path.insert(0, 'src')
from mdm_playground.scheduling.backends.proseco import ProSeCoGenerator
import numpy as np
# Simulate p_x0 and x tensors
B, L, V = 1, 16, 50258
mask_id = 50257
p_x0 = torch.softmax(torch.randn(B, L, V), dim=-1)
x = torch.full((B, L), mask_id, dtype=torch.long)
x[0, :8] = torch.randint(0, V-1, (8,))  # 8 unmasked tokens

# Create minimal generator shell (no checkpoint)
class FakeGen:
    mask_id = mask_id
    device = 'cpu'
    def _extract_signals(self, x, p_x0):
        return ProSeCoGenerator._extract_signals(self, x, p_x0)

gen = FakeGen()
sigs = gen._extract_signals(x, p_x0)
print('Signals:', sigs)
assert sigs['unmasked_fraction'] == 0.5, f'Expected 0.5, got {sigs[\"unmasked_fraction\"]}'
assert sigs['n_revisable'] == 8
assert sigs['entropy'] > 0
print('Signal extraction test PASSED')
"

# Test 3: Surrogate end-to-end
python scripts/run_phase1_proseco.py \
    --surrogate --T 8 --N 2 --M 2 --P 5 --B_values 2 --seed 0 \
    --out_dir /tmp/proseco_test
```
