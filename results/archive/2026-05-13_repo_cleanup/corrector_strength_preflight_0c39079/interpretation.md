# Corrector-Strength Preflight Interpretation

SHA: unknown | T=64 | debug=False | surrogate=False

## Gate verdict: **PREFLIGHT PASS**

| Gate | Pass | Notes |
|------|------|-------|
| G1_canonical_reproduction | ✅ | max_abs_error=0.0; n_checked=30 |
| G2_standard_nontrivial | ✅ | max_corrector_n_changed_strength_1=64 |
| G3_strength_variants_differ | ✅ | mean_corrector_n_changed_s0=4.066666666666666; mean_corrector_n_changed_s1=32.56666666666667 |
| G4_no_correction_noop | ✅ | max_abs_delta_no_correction=0.0 |
| G5_crn_consistent | ✅ |  |

## Per-strength summary

| Strength | mean Δ_t | mean corr_nch | mean G_pair | mean ξ | P(ξ>0) |
|----------|---------|---------------|-------------|--------|--------|
| no_correction | 0.0000 | 0.0 | 0.0000 | 0.0000 | 0.000 |
| strength_0 | 0.0426 | 4.1 | 0.0721 | -0.0250 | 0.000 |
| strength_1 | 0.1638 | 32.6 | 0.2292 | -0.0909 | 0.222 |
| strength_2 | 0.2282 | 46.0 | 0.3246 | -0.1304 | 0.222 |

## Next decision

All preflight gates passed. Proceed to main Phase B experiment (K=30).
Strength levels confirmed: no_correction (sanity), weak (k=0), standard (k=1), strong (k=2).
