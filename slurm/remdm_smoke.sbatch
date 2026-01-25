#!/bin/bash
################################################################################
# Slurm Job: ReMDM Smoke Test
#
# Purpose: Quick test run of ReMDM to verify environment is working
#
# Usage: sbatch slurm/remdm_smoke.sbatch
################################################################################

#SBATCH --job-name=remdm_smoke
#SBATCH --account=3316152
#SBATCH --partition=dsba
#SBATCH --nodes=1
#SBATCH --gres=gpu:nv-2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

set -euo pipefail

################################################################################
# Setup
################################################################################

echo "========================================"
echo "Slurm Job: ReMDM Smoke Test"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate masked-diffusion

cd ~/masked-diffusion-thesis

# Set threading environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Set HPC environment variables (matching setup script)
export HF_HOME="$HOME/.cache/huggingface"
export TMPDIR="$HOME/.tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TOKENIZERS_PARALLELISM="false"

# Load modules (matching setup script)
# TODO: Update module names if different on your cluster
module purge
module load gcc-12.2.1/gcc-12 nvidia/cuda-12.4.0

################################################################################
# Preflight Check
################################################################################

echo ""
echo "Running preflight check..."
echo ""

if python scripts/preflight_check.py; then
    echo ""
    echo "✓ Preflight check passed"
    echo ""
else
    echo ""
    echo "✗ Preflight check FAILED"
    echo "Fix environment issues before running jobs."
    echo "See setup instructions: docs/HPC_SETUP.md"
    echo ""
    exit 1
fi

################################################################################
# Run ReMDM
################################################################################

# Accept config file as argument, default to smoke test
CONFIG="${1:-configs/remdm_hpc_smoke.yaml}"

echo ""
echo "Running ReMDM with config: $CONFIG"
echo ""

python -u scripts/run_remdm.py --config "$CONFIG"

################################################################################
# Done
################################################################################

echo ""
echo "========================================"
echo "Job completed successfully!"
echo "========================================"
echo "Finished at: $(date)"
echo ""

