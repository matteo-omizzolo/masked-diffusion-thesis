# ReMDM Evaluation Results

## T=128 steps (100 samples, seed=42, OWT reference)

| strategy    | gen_ppl | entropy | MAUVE  |
|-------------|---------|---------|--------|
| mdlm        |  60.914 |  5.507  | 0.170  |
| remdm-conf  |  57.579 |  5.499  | 0.440  |
| remdm-loop  |  59.632 |  5.538  | 0.396  |

## T=1000 steps (100 samples, seed=42, OWT reference)

| strategy    | gen_ppl | entropy | MAUVE  |
|-------------|---------|---------|--------|
| mdlm        |  52.269 |  5.446  | 0.590  |
| remdm-conf  |  37.321 |  5.357  | 0.325  |
| remdm-loop  |  30.296 |  5.390  | 0.684  |

## Key findings

### gen_ppl improvement T=128 → T=1000
- mdlm:       60.9 → 52.3  (−14%)
- remdm-conf: 57.6 → 37.3  (−35%)
- remdm-loop: 59.6 → 30.3  (−49%)  ← largest gain

### MAUVE ranking inversion
At T=128:  remdm-conf (0.440) > remdm-loop (0.396) > mdlm (0.170)
At T=1000: remdm-loop (0.684) > mdlm (0.590) > remdm-conf (0.325)

remdm-conf drops from best to worst MAUVE as steps increase — possible diversity
collapse or mode-seeking behaviour under high step counts.
remdm-loop benefits most from additional steps, reaching best gen_ppl AND MAUVE.

### Paper alignment (ReMDM arxiv 2503.00307)
Paper (Table 1, T=1000): reports remdm-conf gen_ppl ~34–38 range — our 37.3 ✓
MAUVE comparison: paper primary metric is remdm-conf vs mdlm.
Our result shows the opposite at T=1000 (mdlm MAUVE > remdm-conf), which is a
notable thesis finding worth investigating further.
