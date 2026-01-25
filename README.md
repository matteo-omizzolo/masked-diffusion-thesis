# Masked Diffusion Thesis - HPC Setup

MSc thesis repository for masked diffusion models (ReMDM, PRISM) with reproducible HPC experiments.

## Quick Start

```bash
# 1. Create conda environment (once)
conda create -n masked-diffusion python=3.9 -y
conda activate masked-diffusion

# 2. Run setup on GPU node (once, ~30-45 min)
sbatch slurm/setup_env_once.sbatch

# 3. Verify installation
python scripts/preflight_check.py

# 4. Run experiments
sbatch slurm/remdm_smoke.sbatch
```

## Documentation

- **[HPC_SETUP.md](docs/HPC_SETUP.md)** - Complete setup guide with troubleshooting
- **[REMDM_INTEGRATION.md](docs/REMDM_INTEGRATION.md)** - ReMDM integration details

## Key Files

### Setup & Validation
- `scripts/setup_hpc_env.sh` - Main installer (installs all dependencies + CUDA extensions)
- `scripts/preflight_check.py` - Environment validator
- `scripts/debug_env.py` - Diagnostic tool

### Slurm Jobs
- `slurm/setup_env_once.sbatch` - One-time setup job
- `slurm/remdm_smoke.sbatch` - Smoke test job

### Experiment Scripts
- `scripts/run_remdm.py` - ReMDM experiment runner
- `configs/*.yaml` - Experiment configurations

## Repository Structure

```
├── configs/           # Experiment configurations
├── docs/              # Documentation
├── external/          # Git submodules (ReMDM, PRISM) - DO NOT MODIFY
├── logs/              # Experiment and setup logs
├── results/           # Experiment results
├── scripts/           # Setup and experiment scripts
├── slurm/             # Slurm batch job scripts
└── src/               # Main package code
    └── masked_diffusion_thesis/
        ├── integrations/  # Wrappers for upstream repos
        ├── models/        # Model implementations
        └── utils/         # Utilities
```

## Requirements

- Python 3.9
- CUDA 12.1/12.4
- GCC 12
- Conda environment
- GPU: RTX 2080 Ti (or compatible)

## Notes

- Upstream repos (external/remdm, external/PRISM) are git submodules - never modify them
- All dependencies including compiled CUDA extensions are installed by `scripts/setup_hpc_env.sh`
- Environment snapshots saved in `logs/env_snapshots/` for reproducibility
- Use `scripts/debug_env.py` if you encounter issues

## Troubleshooting

If setup fails:
1. Check logs in `logs/slurm/`
2. Run diagnostic: `python scripts/debug_env.py`
3. See troubleshooting guide: [docs/HPC_SETUP.md](docs/HPC_SETUP.md)

## Author

Matteo Mizzolo - MSc Thesis 2026
