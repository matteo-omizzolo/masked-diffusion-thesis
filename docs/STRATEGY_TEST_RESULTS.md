# ReMDM Strategy Test Results

## Summary

Tested all 4 ReMDM strategies with mini production config (64 steps, streaming OpenWebText):

| Strategy | Status | Notes |
|----------|--------|-------|
| **rescale** | ✅ SUCCESS | Logit rescaling - most stable |
| **conf** | ❌ FAILED | Confidence-based - dtype bug confirmed |
| **cap** | ✅ SUCCESS | Probability capping |
| **loop** | ✅ SUCCESS | Iterative refinement |

## Test Details

**Config**: 64 sampling steps, openwebtext-streaming dataset, 2 batches, fp32 precision  
**Hardware**: RTX 2080 Ti (11.5GB), 8 CPUs, 32GB RAM  
**Runtime**: ~45-75 seconds per strategy

## Failed Strategy: remdm-conf

The confidence-based strategy (`remdm-conf`) fails with dtype mismatch:

```
RuntimeError: Index put requires the source and destination dtypes match, 
got BFloat16 for the destination and Float for the source.
```

**Root Cause Analysis**:
- Line 863: `confident_score = -torch.ones_like(x).to(torch.bfloat16) * torch.inf`
- Line 751: `conf_values = -p_x0[batch_indices, feature_indices, xs]` (inherits Float32 from p_x0)
- Line 752: `conf[unmask_mask] = conf_values[unmask_mask]` → dtype mismatch

**Fix Options**:
1. **Current policy (recommended)**: Exclude from experiments, use working strategies
2. **Runtime monkeypatch**: Patch in wrapper without touching upstream (see remdm_dtype_patch.py)
3. **Manual upstream edit**: Modify `external/remdm/diffusion.py` line 752 (not recommended)

**For thesis**: Use the 3 working strategies (rescale/cap/loop). No need to fix upstream bug.

## Successful Results

### rescale (Job 9582)
- Result dir: `results/20260125_143805_remdm/`
- Runtime: 48 seconds
- Status: COMPLETED (0:0)

### cap (Job 9583)  
- Result dir: `results/20260125_143805_remdm/` (shared timestamp)
- Runtime: 46 seconds
- Status: COMPLETED (0:0)

### loop (Job 9581)
- Result dir: `results/20260125_143439_remdm/`
- Runtime: 72 seconds  
- Status: COMPLETED (0:0)

## Generated Outputs

Each successful run produces:
- `summary.json` - Run metadata with strategy/steps info
- `samples.pt` - Raw PyTorch tensor samples
- `external_remdm/generated_sequences.json` - Decoded text sequences
- `external_remdm/config_tree.txt` - Full Hydra config tree
- `external_remdm/main.log` - Upstream ReMDM logs

## Quick Reference

### Test single strategy
```bash
sbatch slurm/remdm_smoke.sbatch configs/remdm_hpc_owt_mini_rescale.yaml
```

### Test all working strategies (parallel)
```bash
for strategy in rescale cap loop; do
    sbatch slurm/remdm_smoke.sbatch configs/remdm_hpc_owt_mini_${strategy}.yaml
done
```

### Monitor jobs
```bash
squeue -u $USER
watch -n 5 'squeue -u $USER'
```

### Check results
```bash
ls -lt results/ | head
cat results/<timestamp>_remdm/summary.json | python3 -m json.tool
cat results/<timestamp>_remdm/external_remdm/generated_sequences.json | python3 -m json.tool | head -50
```

## Recommendations

1. **For thesis experiments**: Use `remdm-rescale` as baseline (most stable)
2. **For comparisons**: Test rescale vs cap vs loop (all working)
3. **Avoid**: `remdm-conf` until upstream dtype bug is fixed
4. **Next steps**: 
   - Run full 256-step experiments with `configs/remdm_hpc_owt.yaml`
   - Compare text quality across strategies
   - Benchmark inference speed differences

## Notes

- Parallel job submission may hit git submodule lock (removed from sbatch script)
- NFS temp file cleanup warnings are non-critical (OSError: Device or resource busy)
- All tests used fp32 precision to maximize compatibility
- Streaming dataset prevents large downloads (OpenWebText would be ~40GB)

---
**Last updated**: 2026-01-25  
**Environment**: HPC cluster (dsba partition, RTX 2080 Ti, CUDA 12.4.0)
**See also**: [EXPERIMENT_GRID.md](EXPERIMENT_GRID.md) for full experiment workflow
