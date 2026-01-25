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

**Purpose**: Generate thesis comparison of MDLM baseline vs 3 ReMDM variants

**Parameters**: 256 steps, 10 batches per strategy  
**Runtime**: ~15-20 min per experiment (60-80 min total)  
**Total samples**: 40 sequences (10 per strategy × 4 strategies)

| Config | Strategy | Purpose |
|--------|----------|----------|
| `mdlm_hpc_owt_prod.yaml` | mdlm (baseline) | Standard MDLM sampling |
| `remdm_hpc_owt_prod_rescale.yaml` | remdm-rescale | Logit rescaling (most stable) |
| `remdm_hpc_owt_prod_cap.yaml` | remdm-cap | Probability capping |
| `remdm_hpc_owt_prod_loop.yaml` | remdm-loop | Iterative refinement |

**Run command**:
```bash
bash scripts/run_experiment_grid.sh

# Monitor (4 jobs × ~15-20 min = 60-80 min total)
watch -n 10 'squeue -u $USER'
```

## Metrics to Compare

Each experiment outputs:
1. **Perplexity** (gen_ppl): Lower is better
2. **Entropy**: Measure of diversity
3. **MAUVE**: Similarity to reference distribution (0-1, higher is better)
4. **Generated text quality**: Manual inspection

## Analysis Plan

### Running Experiments

```bash
# Submit all 4 experiments
bash scripts/run_experiment_grid.sh

# Monitor progress
watch -n 10 'squeue -u $USER'
```

### Evaluation Workflow

1. **Wait for completion**: `squeue -u $USER` returns empty
2. **Evaluate metrics**:
```bash
# Print comparison table
python scripts/evaluate_text.py results/

# Export CSV for thesis
python scripts/evaluate_text.py results/ --output thesis_metrics.csv
```

3. **Inspect text quality**:
```bash
cat results/*/external_remdm/generated_sequences.json | python3 -m json.tool | less
```

### Evaluation Metrics

**Automatic metrics** (computed by `evaluate_text.py`):
- **Perplexity** (gen_ppl): Lower is better (fluency)
- **MAUVE**: 0-1, higher is better (similarity to reference)
- **Distinct-1/2**: Vocabulary diversity (unique unigrams/bigrams)
- **Entropy**: Sample diversity measure
- **Length**: Average tokens/chars per sample

**Manual evaluation**:
- Coherence: Does text make sense?
- Quality: Grammar, factuality, relevance
- Creativity: Novel vs repetitive content

## Results Summary

| Strategy | Steps | Batches | PPL | MAUVE | Distinct-1 | Distinct-2 | Status |
|----------|-------|---------|-----|-------|------------|------------|--------|
| mdlm | 256 | 10 | - | - | - | - | ⏳ |
| remdm-rescale | 256 | 10 | - | - | - | - | ⏳ |
| remdm-cap | 256 | 10 | - | - | - | - | ⏳ |
| remdm-loop | 256 | 10 | - | - | - | - | ⏳ |

**Update this table after running experiments.**

## Notes

- **remdm-conf excluded**: Dtype bug in upstream (see STRATEGY_TEST_RESULTS.md)
- **Only production configs**: Smoke/mini configs removed (validated, not needed for thesis)
- All experiments use streaming OpenWebText (no 40GB download)
- Evaluation: `python scripts/evaluate_text.py results/` for metrics table

## Next Steps

1. ✅ Environment setup complete
2. ✅ Configs validated (mini runs successful)
3. ✅ Repository cleaned
4. ⏳ **Run production**: `bash scripts/run_experiment_grid.sh`
5. ⏳ **Evaluate**: `python scripts/evaluate_text.py results/ -o thesis_metrics.csv`
6. ⏳ Analyze results and write thesis section
