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

### 3. Run Experiments

**Quick validation (64 steps, ~1 min each)**:
```bash
# Test baseline MDLM
sbatch slurm/remdm_smoke.sbatch configs/mdlm_hpc_owt_mini.yaml

# Or run full mini grid (baseline + 3 ReMDM variants)
bash scripts/run_experiment_grid.sh mini
```

**Production experiments (256 steps, ~15-20 min each)**:
```bash
# Run complete comparison grid
bash scripts/run_experiment_grid.sh prod

# This submits 4 jobs:
#   - MDLM baseline (standard sampling)
#   - ReMDM rescale (logit rescaling)
#   - ReMDM cap (probability capping)
#   - ReMDM loop (iterative refinement)
```

## Repository Structure

```
├── configs/                    # Experiment configurations
│   ├── remdm_hpc_smoke.yaml   # Smoke test (wikitext2, 16 steps)
│   ├── mdlm_hpc_owt_mini.yaml # MDLM baseline (64 steps)
│   ├── mdlm_hpc_owt_prod.yaml # MDLM baseline (256 steps)
│   ├── remdm_hpc_owt_mini_*.yaml  # ReMDM variants (64 steps)
│   └── remdm_hpc_owt_prod_*.yaml  # ReMDM variants (256 steps)
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
    ├── HPC_SETUP.md           # Detailed setup guide
    └── STRATEGY_TEST_RESULTS.md  # Strategy comparison & results
```

## HPC Cluster Configuration

- **Partition**: dsba, **GPU**: RTX 2080 Ti (11.5GB), **CUDA**: 12.4.0, **GCC**: 12
- **Python**: 3.9, **Conda env**: masked-diffusion
- **PyTorch**: 2.2.2+cu121, **Lightning**: 2.2.1, **flash-attn**: 2.5.6

## Configuration Examples

### Experiment Grid (for thesis)

**Mini validation** (64 steps, 2 batches, ~1 min each):
- `mdlm_hpc_owt_mini.yaml` - MDLM baseline
- `remdm_hpc_owt_mini_rescale.yaml` - ReMDM rescale
- `remdm_hpc_owt_mini_cap.yaml` - ReMDM cap
- `remdm_hpc_owt_mini_loop.yaml` - ReMDM loop

**Production** (256 steps, 10 batches, ~15-20 min each):
- `mdlm_hpc_owt_prod.yaml` - MDLM baseline
- `remdm_hpc_owt_prod_rescale.yaml` - ReMDM rescale
- `remdm_hpc_owt_prod_cap.yaml` - ReMDM cap
- `remdm_hpc_owt_prod_loop.yaml` - ReMDM loop

### Config Structure

**Smoke Test** (configs/remdm_hpc_smoke.yaml):
```yaml
remdm:
  data: wikitext2           # Tiny dataset for fast validation
  steps: 16                 # Minimal sampling steps
  num_sample_batches: 1     # Single batch
  strategy: remdm-rescale   # Stable strategy
```

## Experiment Results

After running experiments, evaluate results:

```bash
# Generate metrics table
python scripts/evaluate_text.py results/

# Export to CSV
python scripts/evaluate_text.py results/ --output thesis_metrics.csv
```

**Metrics computed**:
- Perplexity (lower = better fluency)
- MAUVE (higher = closer to reference)
- Distinct-1/2 (vocabulary diversity)
- Entropy (sample diversity)
- Text length statistics

Each run generates:
- **summary.json** - Upstream metrics (perplexity, entropy, MAUVE)
- **samples.pt** - Generated token sequences
- **external_remdm/generated_sequences.json** - Decoded text samples

View individual run:
```bash
ls -lt results/ | head
cat results/<timestamp>_*/summary.json | python3 -m json.tool
```

## Troubleshooting

**Out of Memory**: Use streaming datasets, reduce batch size  
**Dataset Timeout**: Clean cache `rm -rf ~/.cache/huggingface/datasets/`  
**Module Errors**: Reinstall `pip install -e . --no-deps`  
**remdm-conf dtype bug**: See [docs/STRATEGY_TEST_RESULTS.md](docs/STRATEGY_TEST_RESULTS.md#failed-strategy-remdm-conf) for fix options

See [docs/HPC_SETUP.md](docs/HPC_SETUP.md) and [docs/EXPERIMENT_GRID.md](docs/EXPERIMENT_GRID.md) for detailed documentation.

## Citation

```bibtex
@article{wang2025remasking,
  title={Remasking Discrete Diffusion Models with Inference-Time Scaling},
  author={Wang, Guanghan and Schiff, Yair and Sahoo, Subham and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2503.00307},
  year={2025}
}
```
