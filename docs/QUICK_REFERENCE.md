# Quick Reference: Running Experiments

## Current State ✅

**Repository locked and committed** (2 commits pushed)
- All configs tested and working
- MDLM baseline validated (job 9584)
- 3 ReMDM variants validated (jobs 9581-9583)
- Documentation complete

## Before Scaling Up

✅ **Never run setup in parallel jobs** - race conditions with git/pip
✅ **Environment is ready** - all deps installed in conda env `masked-diffusion`
✅ **Jobs only run experiments** - no git submodule update in sbatch anymore

## Experiment Commands

### Quick Validation (recommended first)
```bash
# Test all 4 strategies in mini mode (~4 min total)
bash scripts/run_experiment_grid.sh mini

# Monitor
watch -n 5 'squeue -u $USER'

# Check results
ls -lt results/ | head
```

### Production Run (for thesis)
```bash
# Run full comparison grid (4 jobs × ~15-20 min = 60-80 min total)
bash scripts/run_experiment_grid.sh prod

# Monitor
watch -n 10 'squeue -u $USER'
```

### Individual Jobs
```bash
# Run one specific experiment
sbatch slurm/remdm_smoke.sbatch configs/mdlm_hpc_owt_prod.yaml
sbatch slurm/remdm_smoke.sbatch configs/remdm_hpc_owt_prod_rescale.yaml
```

## Expected Results

### Mini (64 steps, 2 batches)
- Runtime: ~45-75s per job
- Output: 2 sequences per strategy
- Purpose: Sanity check

### Production (256 steps, 10 batches)
- Runtime: ~15-20 min per job
- Output: 10 sequences per strategy
- Purpose: Thesis comparison

## Result Files

Each job creates `results/<timestamp>_remdm/`:
```
summary.json              # Metrics: PPL, entropy, MAUVE
samples.pt                # Raw tokens
external_remdm/
  ├── generated_sequences.json  # Decoded text
  ├── config_tree.txt           # Full config
  └── main.log                  # Upstream logs
```

## Analysis Workflow

1. Run experiments: `bash scripts/run_experiment_grid.sh prod`
2. Wait for completion: `squeue -u $USER` (empty = done)
3. List results: `ls -lt results/ | head -10`
4. **Evaluate all runs**:
```bash
# Print metrics table
python scripts/evaluate_text.py results/

# Export to CSV for thesis
python scripts/evaluate_text.py results/ --output thesis_metrics.csv

# Quiet mode (CSV only)
python scripts/evaluate_text.py results/ -o metrics.csv --quiet
```

5. Compare strategies: Open CSV in Excel/pandas for analysis
6. Inspect text quality:
```bash
cat results/*/external_remdm/generated_sequences.json | python3 -m json.tool | less
```

### Metrics Computed

- **Perplexity** (gen_ppl): Lower = better fluency
- **MAUVE**: 0-1, higher = closer to reference distribution
- **Distinct-1/2**: Vocabulary diversity (unique n-grams)
- **Entropy**: Sample diversity
- **Length**: Average tokens/chars per sample

## Troubleshooting

**Job fails immediately**: Check `err/remdm_smoke_<jobid>.err`
**Out of memory**: Reduce `num_sample_batches` in config
**Wrong strategy in results**: Check that sbatch received correct config path

## Next Steps

1. ⏳ Run mini grid to revalidate all 4 strategies
2. ⏳ Run production grid for thesis results
3. ⏳ Extract metrics from summary.json files
4. ⏳ Create comparison table/plots
5. ⏳ Analyze text quality differences
6. ⏳ Write thesis section

## Documentation

- [README.md](../README.md) - Overview and quick start
- [docs/HPC_SETUP.md](HPC_SETUP.md) - Detailed setup guide
- [docs/EXPERIMENT_GRID.md](EXPERIMENT_GRID.md) - Experiment plan and tracking
- [docs/STRATEGY_TEST_RESULTS.md](STRATEGY_TEST_RESULTS.md) - Mini validation results
