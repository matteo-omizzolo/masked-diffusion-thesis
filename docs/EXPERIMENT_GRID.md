# Experiment Grid - Thesis Comparison

## Overview

Systematic comparison of MDLM baseline vs ReMDM variants for thesis research.

**Fixed parameters** (all experiments):
- Dataset: openwebtext-streaming
- Model: kuleshov-group/mdlm-no_flashattn-fp32-owt (DiT small)
- Sequence length: 1024 tokens
- Nucleus sampling: p=0.9
- Precision: fp32
- Seed: 42 (project), 1 (sampling)
- Batch size: 1

## Experiment Grid

### Mini Validation (Quick Sanity Check)

**Purpose**: Verify all strategies work before scaling up  
**Parameters**: 64 steps, 2 batches per strategy  
**Runtime**: ~1 min per experiment  
**Total samples**: 8 sequences (2 per strategy × 4 strategies)

| Config | Strategy | Status |
|--------|----------|--------|
| `mdlm_hpc_owt_mini.yaml` | mdlm (baseline) | ✅ Tested |
| `remdm_hpc_owt_mini_rescale.yaml` | remdm-rescale | ✅ Tested |
| `remdm_hpc_owt_mini_cap.yaml` | remdm-cap | ✅ Tested |
| `remdm_hpc_owt_mini_loop.yaml` | remdm-loop | ✅ Tested |

**Run command**:
```bash
bash scripts/run_experiment_grid.sh mini
```

### Production (Thesis Results)

**Purpose**: Generate meaningful comparison for thesis  
**Parameters**: 256 steps, 10 batches per strategy  
**Runtime**: ~15-20 min per experiment  
**Total samples**: 40 sequences (10 per strategy × 4 strategies)

| Config | Strategy | Status |
|--------|----------|--------|
| `mdlm_hpc_owt_prod.yaml` | mdlm (baseline) | ⏳ Pending |
| `remdm_hpc_owt_prod_rescale.yaml` | remdm-rescale | ⏳ Pending |
| `remdm_hpc_owt_prod_cap.yaml` | remdm-cap | ⏳ Pending |
| `remdm_hpc_owt_prod_loop.yaml` | remdm-loop | ⏳ Pending |

**Run command**:
```bash
bash scripts/run_experiment_grid.sh prod
```

## Metrics to Compare

Each experiment outputs:
1. **Perplexity** (gen_ppl): Lower is better
2. **Entropy**: Measure of diversity
3. **MAUVE**: Similarity to reference distribution (0-1, higher is better)
4. **Generated text quality**: Manual inspection

## Analysis Plan

1. **Baseline establishment**: What is MDLM's performance?
2. **ReMDM improvements**: Do any variants beat baseline?
3. **Strategy comparison**: Which ReMDM variant performs best?
4. **Trade-offs**: Perplexity vs diversity vs quality

## Results Summary

### Mini Validation Results

| Strategy | Steps | Batches | Runtime | PPL | MAUVE | Status |
|----------|-------|---------|---------|-----|-------|--------|
| mdlm | 64 | 2 | 46s | TBD | TBD | ✅ |
| remdm-rescale | 64 | 2 | 48s | TBD | TBD | ✅ |
| remdm-cap | 64 | 2 | 46s | TBD | TBD | ✅ |
| remdm-loop | 64 | 2 | 72s | TBD | TBD | ✅ |

### Production Results

| Strategy | Steps | Batches | Runtime | PPL | MAUVE | Status |
|----------|-------|---------|---------|-----|-------|--------|
| mdlm | 256 | 10 | - | - | - | ⏳ |
| remdm-rescale | 256 | 10 | - | - | - | ⏳ |
| remdm-cap | 256 | 10 | - | - | - | ⏳ |
| remdm-loop | 256 | 10 | - | - | - | ⏳ |

**Update this table after running production experiments.**

## Notes

- Mini experiments validated: all 4 strategies working (Jan 25, 2026)
- Production experiments: run after validating mini grid
- remdm-conf excluded due to dtype bug (see STRATEGY_TEST_RESULTS.md)
- All experiments use streaming dataset to avoid 40GB download

## Next Steps

1. ✅ Create experiment configs (done)
2. ✅ Test mini grid (done)
3. ⏳ Run production grid
4. ⏳ Collect metrics from summary.json files
5. ⏳ Analyze text quality
6. ⏳ Create comparison plots
7. ⏳ Write thesis results section
