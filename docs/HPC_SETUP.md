# HPC Setup Guide

Comprehensive setup for masked diffusion models on the HPC cluster.

## Prerequisites

```bash
# Clone with submodules
git clone --recursive <repo-url>
cd masked-diffusion-thesis

# Create conda environment
conda create -n masked-diffusion python=3.9 -y
conda activate masked-diffusion
```

## One-Time Setup

**On GPU node** (required for compiling CUDA extensions):

```bash
# Submit setup job
sbatch slurm/setup_env_once.sbatch

# Monitor progress
squeue -u $USER
tail -f out/setup_env_once_*.out
```

**Duration**: ~45 minutes  
**What it installs**: PyTorch 2.2.2, Lightning, flash-attn 2.5.6, causal-conv1d 1.4.0, mamba-ssm 1.2.0, all dependencies

**Environment snapshot** saved to: `logs/env_snapshots/env_snapshot_*.txt`

## Verification

```bash
# Activate environment
conda activate masked-diffusion

# Run preflight check
python scripts/preflight_check.py

# Should pass 7/7 checks:
# ✓ Core deps, ML frameworks, Compiled extensions
# ✓ GPU & CUDA, Repo structure, Upstream imports, Env vars
```

## Running Jobs

### Smoke Test (1 minute)
```bash
sbatch slurm/remdm_smoke.sbatch

# Check output
cat out/remdm_smoke_<JOBID>.out
ls results/<timestamp>_remdm/
```

### Custom Job

1. **Create config** (copy from `configs/remdm_hpc_smoke.yaml`)
2. **Create sbatch script** or use `python scripts/run_remdm.py --config <your-config>.yaml`
3. **Submit**: `sbatch <your-script>.sbatch`

## Configuration Reference

### Key Parameters

```yaml
remdm:
  data: wikitext2              # Dataset: wikitext2, text8, openwebtext-streaming
  steps: 16                    # Sampling steps (16=fast, 256=production)
  num_sample_batches: 1        # Number of batches to generate
  strategy: remdm-rescale      # remdm-rescale, remdm-cap, remdm-loop
  backbone: hf_dit             # hf_dit (DiT) or hf_dimamba (DiMamba)
  precision: 32                # 32 or bf16 (use 32 for stability)
  env_patch: true              # Enable HPC environment patches
```

### Dataset Options

- **wikitext2** / **text8**: Small, fast, ideal for smoke tests
- **openwebtext-split**: Full dataset (large, requires caching)
- **openwebtext-streaming**: Streaming mode (memory-efficient)

### Sampling Strategies

- **remdm-rescale**: Rescaling-based remasking (recommended)
- **remdm-cap**: Confidence capping
- **remdm-loop**: Loop-based with eta/t_on/t_off parameters
- **remdm-conf**: Confidence-based (has dtype issues, requires patch)

## Output Structure

```
results/<timestamp>_remdm/
├── summary.json                # Metrics and metadata
├── samples.pt                  # Generated tokens (PyTorch)
├── meta.json                   # Run configuration
└── external_remdm/
    ├── generated_sequences.json  # Text samples + MAUVE/entropy
    └── config_tree.txt           # Full Hydra config
```

## Troubleshooting

### Setup Issues

**Problem**: Flash-attn compilation fails  
**Solution**: Check CUDA version, ensure running on GPU node

**Problem**: Conda not found  
**Solution**: Update `CONDA_BASE` in `setup_hpc_env.sh` to match your cluster

### Runtime Issues

**Problem**: Out of memory (OOM)  
**Solutions**:
- Use streaming datasets: `data: openwebtext-streaming`
- Reduce batch size: `eval_batch_size: 1`
- Use smaller dataset for testing: `data: wikitext2`

**Problem**: Dataset download timeout/error  
**Solutions**:
```bash
# Clean corrupted cache
rm -rf ~/.cache/huggingface/datasets/

# Use smaller dataset
data: wikitext2  # instead of openwebtext-split
```

**Problem**: FileNotFoundError in results path  
**Solution**: Fixed in v1.0, uses absolute paths. Reinstall: `pip install -e . --no-deps`

### Job Issues

**Check job status**:
```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

**View logs**:
```bash
tail -50 out/<jobname>_<JOBID>.out
tail -50 err/<jobname>_<JOBID>.err
```

**Disk space**:
```bash
df -h $HOME
du -sh ~/.cache/huggingface/
```

## Environment Details

### Cluster Specs
- **Partition**: dsba
- **GPU**: NVIDIA RTX 2080 Ti (compute 7.5, 11.5 GB)
- **CUDA**: 12.4.0 (PyTorch uses 12.1)
- **GCC**: 12

### Python Packages (pinned)
- **PyTorch**: 2.2.2+cu121
- **Lightning**: 2.2.1
- **Transformers**: 4.38.2
- **Hydra**: 1.3.2
- **flash-attn**: 2.5.6
- **causal-conv1d**: 1.4.0 (downgraded for compatibility)
- **mamba-ssm**: 1.2.0 (downgraded for compatibility)

### Environment Variables
```bash
export HF_HOME=$HOME/.cache/huggingface
export TMPDIR=$HOME/.tmp  # Avoids cross-device link errors
export CUDA_HOME=/opt/share/libs/nvidia/cuda-12.4.0
export TORCH_CUDA_ARCH_LIST="7.5"  # RTX 2080 Ti only
```

## Advanced: Manual Setup

If you need to run setup manually (not recommended):

```bash
# Activate conda
conda activate masked-diffusion

# Load modules
module load gcc-12.2.1/gcc-12 nvidia/cuda-12.4.0

# Run setup script
bash scripts/setup_hpc_env.sh
```

## Maintenance

### Updating Dependencies

Edit `pyproject.toml`, then:
```bash
pip install -e . --no-deps
```

### Cleaning Cache

```bash
# Remove old datasets (keeps models)
rm -rf ~/.cache/huggingface/datasets/

# Check space
du -sh ~/.cache/huggingface/*
```

### Re-installing Compiled Extensions

```bash
pip uninstall flash-attn causal-conv1d mamba-ssm -y
bash scripts/setup_hpc_env.sh  # Re-run relevant sections
```

## References

- **ReMDM Paper**: [arXiv:2503.00307](https://arxiv.org/abs/2503.00307)
- **ReMDM GitHub**: [external/remdm/README.md](../external/remdm/README.md)
- **Flash Attention**: https://github.com/Dao-AILab/flash-attention
- **Mamba**: https://github.com/state-spaces/mamba
