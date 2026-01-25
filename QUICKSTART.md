# Quick Start Guide

Get running in 3 steps:

## 1. Setup (once, ~45 min)

```bash
sbatch slurm/setup_env_once.sbatch
# Installs all dependencies on GPU node
```

## 2. Smoke Test (~1 min)

```bash
sbatch slurm/remdm_smoke.sbatch
# Validates installation with tiny dataset
```

## 3. Run Experiments

```bash
# Edit config
cp configs/remdm_hpc_smoke.yaml configs/my_exp.yaml
vim configs/my_exp.yaml

# Run
python scripts/run_remdm.py --config configs/my_exp.yaml
```

## Check Results

```bash
# Monitor job
squeue -u $USER

# View output
tail -f out/remdm_smoke_*.out

# See results
ls results/
cat results/<timestamp>_remdm/summary.json
```

## Need Help?

- Full guide: [docs/HPC_SETUP.md](docs/HPC_SETUP.md)
- Main README: [README.md](README.md)
