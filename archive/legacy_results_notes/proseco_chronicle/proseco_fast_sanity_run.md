# ProSeCo Fast Sanity Run

**Date:** 2026-04-17  
**Status:** SURROGATE PASSED — real backend pending GPU (HPC submission required)

---

## What was run

### Surrogate sanity (local, CPU)

**Command:**
```bash
/home/3316152/.conda/envs/remdm311/bin/python scripts/run_phase1_proseco.py \
    --surrogate --T 16 --N 5 --M 3 --P 10 --B_values 2,4 --seed 42 \
    --out_dir results/phase1_proseco_surrogate_sanity
```

**Environment:** Login node (no GPU). Python 3.11 (remdm311 env). PyTorch 2.10.0+cu128.  
**Elapsed:** 8.1 seconds  
**Output:** `results/phase1_proseco_surrogate_sanity/`

### Signal extraction unit test (no checkpoint)

```python
# 8/16 positions unmasked, random logits
sigs = gen._extract_signals(x, p_x0)
# Expected: unmasked_fraction=0.5, n_revisable=8, entropy>0
# Result: {'entropy': 10.33, 'inverse_margin': 1.00, 'quality_mass_proxy': 1.00,
#           'unmasked_fraction': 0.5, 'n_revisable': 8, 'n_masked': 8}
# PASSED
```

---

## Surrogate sanity results

| Metric | Value | Status |
|---|---|---|
| ε (entropy, RMS) | 0.017 | ✓ Non-degenerate |
| ε (margin, RMS) | 0.018 | ✓ Non-degenerate |
| Spearman(H, Δ) | −0.232 ± 0.099 | ✓ Non-NaN (note: negative in surrogate — expected) |
| Steps with Δ_t > 0 | 16/16 | ✓ (surrogate always positive by design) |
| γ (95th pct) | 0.009 | ✓ Small |
| Theorem A (B=2) | Non-vacuous | ✓ 2Bε+2η=0.091 < G_oracle=0.102 |
| Theorem A (B=4) | Non-vacuous | ✓ 2Bε+2η=0.181 < G_oracle=0.190 |
| Policy comparison | 20 records | ✓ All policies computed |
| Wall time | 8.1s | ✓ Fast |

**Negative Spearman in surrogate:** The surrogate generates entropy signals independently
of delta gains (synthetic), so negative Spearman is expected and uninformative. The real
run will show the true correlation.

---

## What the surrogate run validates

| Component | Validated? |
|---|---|
| Protocol A loop (run_base + T × run_branch) | ✓ |
| Signal extraction from per_step_signals | ✓ |
| estimate_single_step_gain (Δ_t, TCR_t) | ✓ |
| Protocol B loop (evaluate_schedule) | ✓ |
| Pairwise gamma estimation | ✓ |
| Summary computation (ε, η_B, γ, T_low) | ✓ |
| Policy comparison (10 policies × 2 budgets) | ✓ |
| Output JSON structure | ✓ |
| Import and instantiation of ProSeCoGenerator | ✓ (class imported; surrogate used) |
| Real checkpoint loading | ✗ — requires GPU on HPC |
| ProSeCo corrector mechanism | ✗ — requires GPU on HPC |
| Non-zero Δ_t from real corrector | ✗ — requires GPU on HPC |

---

## Real backend sanity run (pending)

To run the smallest real ProSeCo experiment:

```bash
# On HPC compute node (submit via sbatch or interactive):
python scripts/run_phase1_proseco.py \
    --T 16 --N 5 --M 3 --P 10 \
    --B_values 2,4 \
    --seed 42 \
    --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --corrector_steps 2 \
    --out_dir results/phase1_proseco_real_sanity
```

Estimated runtime on A100: ~5–10 minutes.

**Minimum checks for real sanity:**

1. Backend loads without error (checkpoint loading, tokenizer)
2. `per_step_signals` is non-zero at t > 5 (unmasked_fraction > 0)
3. TCR_t > 0 for some t (corrector changes tokens)
4. At least 1 step with Δ_t > 0 (corrector is beneficial somewhere)
5. summary.json is generated without error

If all 5 pass: proceed to full pilot (N=20, T=64, M=15, P=120).

---

## Why the local environment cannot run the real sanity

The login node (`slogin.hpc.unibocconi.it`) has no GPU. The MDLM model runs on CPU
but at ~500× slower speed (estimated ~30 hours for a single trajectory at T=64).
The surrogate mode is the maximum viable test on the login node.

**To run real sanity:** submit `hpc/phase1_proseco.sbatch` with the sanity parameters
(uncomment the sanity block and comment the pilot block).

---

## What remains unknown until real run

1. Whether any Δ_t > 0 with the ProSeCo corrector on MDLM backbone
2. Whether entropy over unmasked positions correlates with Δ_t
3. The actual magnitude of η_B and γ with a real corrector
4. Whether signal-guided scheduling beats uniform on this backend
5. Whether the corrector is effective enough to make Theorem A non-vacuous

These are answered by the pilot run (N=20, T=64, submitted via `sbatch hpc/phase1_proseco.sbatch`).
