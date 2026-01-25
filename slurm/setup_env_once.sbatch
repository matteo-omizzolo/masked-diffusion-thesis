#!/bin/bash
################################################################################
# Slurm Job: One-time HPC Environment Setup
#
# Purpose: Build and install ALL dependencies (including compiled CUDA extensions)
#          on a GPU node. This only needs to be run once (or after major updates).
#
# Usage: sbatch slurm/setup_env_once.sbatch
#
# Output: Check logs/slurm/setup_env_once_<jobid>.log
################################################################################

#SBATCH --job-name=setup_env_once
#SBATCH --account=3316152
#SBATCH --partition=dsba               # Using dsba partition (has RTX 2080 Ti)
#SBATCH --gres=gpu:nv-2080:1           # Request 1 RTX 2080 Ti GPU
#SBATCH --cpus-per-task=4              # 4 CPUs for parallel compilation
#SBATCH --mem=16G                      # 16GB RAM should be enough
#SBATCH --time=02:00:00                # 2 hours max (flash-attn takes ~20 min)
#SBATCH --output=logs/slurm/setup_env_once_%j.log
#SBATCH --error=logs/slurm/setup_env_once_%j.err

################################################################################
# Configuration
################################################################################

# TODO: Update conda environment name if different
CONDA_ENV_NAME="masked-diffusion"

# Repo root - use absolute path or SLURM_SUBMIT_DIR
REPO_ROOT="${SLURM_SUBMIT_DIR:-$HOME/masked-diffusion-thesis}"

################################################################################
# Setup
################################################################################

echo "========================================"
echo "Slurm Job: One-time HPC Environment Setup"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Repository: $REPO_ROOT"
echo ""

# Create log directories
mkdir -p "$REPO_ROOT/logs/slurm"
mkdir -p "$REPO_ROOT/logs/env_snapshots"

################################################################################
# Run setup script
################################################################################

cd "$REPO_ROOT"

echo "Running setup script: scripts/setup_hpc_env.sh"
echo ""

# Run the main setup script
bash scripts/setup_hpc_env.sh

SETUP_EXIT_CODE=$?

################################################################################
# Finalize
################################################################################

if [ $SETUP_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Setup completed successfully!"
    echo "========================================"
    echo "Finished at: $(date)"
    echo ""
    echo "Next steps:"
    echo "  1. Review this log: logs/slurm/setup_env_once_${SLURM_JOB_ID}.log"
    echo "  2. Review environment snapshot: logs/env_snapshots/latest.txt"
    echo "  3. Run preflight check:"
    echo "     srun --partition=gpu --gres=gpu:1 --pty python scripts/preflight_check.py"
    echo "  4. Submit test job:"
    echo "     sbatch slurm/remdm_smoke.sbatch"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Setup FAILED with exit code $SETUP_EXIT_CODE"
    echo "========================================"
    echo "Finished at: $(date)"
    echo ""
    echo "Check logs for errors:"
    echo "  - This log: logs/slurm/setup_env_once_${SLURM_JOB_ID}.log"
    echo "  - Flash-attn: logs/flash_attn_install.log"
    echo "  - Causal-conv1d: logs/causal_conv1d_install.log"
    echo "  - Mamba-ssm: logs/mamba_ssm_install.log"
    echo ""
    exit $SETUP_EXIT_CODE
fi
