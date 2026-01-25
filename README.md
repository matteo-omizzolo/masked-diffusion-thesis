# Masked Diffusion Models - HPC Thesis Repository

Reproducible HPC environment for masked discrete diffusion models research, integrating upstream ReMDM and PRISM codebases.

## Quick Start

### 1. One-Time Setup (on GPU node)
```bash
# Submit setup job (installs all dependencies including flash-attn, mamba-ssm)
sbatch slurm/setup_env_once.sbatch

# Monitor: squeue -u $USER
# Takes ~45 minutes, installs to conda env: masked-diffusion
```

### 2. Run Smoke Test
```bash
# Quick validation (~1 min, uses tiny wikitext2 dataset)
sbatch slurm/remdm_smoke.sbatch

# Check results in out/remdm_smoke_<JOBID>.out
# Generated samples saved to results/<timestamp>_remdm/
```

### 3. Run Production Experiments
```bash
# Use production config with streaming OpenWebText
python scripts/run_remdm.py --config configs/remdm_hpc_owt.yaml
```

## Repository Structure

```
├── configs/                    # Experiment configurations
│   ├── remdm_hpc_smoke.yaml   # Smoke test (wikitext2, 16 steps)
│   └── remdm_hpc_owt.yaml     # Production (openwebtext-streaming)
├── external/                   # Upstream submodules (DO NOT MODIFY)
│   ├── remdm/                 # ReMDM codebase
│   └── PRISM/                 # PRISM codebase
├── scripts/
│   ├── setup_hpc_env.sh       # Main installation script
│   ├── preflight_check.py     # Pre-run validation
│   └── run_remdm.py           # ReMDM wrapper
├── slurm/
│   ├── setup_env_once.sbatch  # One-time environment setup
│   └── remdm_smoke.sbatch     # Example job script
└── docs/                       # Documentation
```

## HPC Cluster Configuration

- **Partition**: dsba, **GPU**: RTX 2080 Ti (11.5GB), **CUDA**: 12.4.0, **GCC**: 12
- **Python**: 3.9, **Conda env**: masked-diffusion
- **PyTorch**: 2.2.2+cu121, **Lightning**: 2.2.1, **flash-attn**: 2.5.6

## Configuration Examples

**Smoke Test** (configs/remdm_hpc_smoke.yaml):
```yaml
remdm:
  data: wikitext2           # Tiny dataset for fast validation
  steps: 16                 # Minimal sampling steps
  num_sample_batches: 1     # Single batch
  strategy: remdm-rescale   # Stable strategy
```

**Production** (configs/remdm_hpc_owt.yaml):
```yaml
remdm:
  data: openwebtext-streaming  # Large dataset, streaming mode
  steps: 256                   # Full sampling
  num_sample_batches: 100      # Production batch count
```

## Output Files

Each run creates `results/<timestamp>_remdm/`:
- **summary.json** - Metrics (perplexity, entropy, MAUVE)
- **samples.pt** - Generated token sequences
- **external_remdm/generated_sequences.json** - Full text samples

## Troubleshooting

**Out of Memory**: Use streaming datasets, reduce batch size  
**Dataset Timeout**: Clean cache `rm -rf ~/.cache/huggingface/datasets/`  
**Module Errors**: Reinstall `pip install -e . --no-deps`

See [docs/HPC_SETUP.md](docs/HPC_SETUP.md) for detailed documentation.

## Citation

```bibtex
@article{wang2025remasking,
  title={Remasking Discrete Diffusion Models with Inference-Time Scaling},
  author={Wang, Guanghan and Schiff, Yair and Sahoo, Subham and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2503.00307},
  year={2025}
}
```
