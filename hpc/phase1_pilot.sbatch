#!/bin/bash
#SBATCH --job-name=phase1_pilot
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# Phase 1 Pilot: entropy-proxy experiment (Protocol A + B)
# Empirically estimates ε (calibration) and η_B (additivity) for Theorem A.
#
# Submit from your laptop:
#   bash hpc/submit.sh phase1-pilot
#
# Or directly on the cluster:
#   sbatch hpc/phase1_pilot.sbatch

set -euo pipefail

echo "=========================================="
echo "Phase 1 Pilot: Entropy-Proxy Experiment"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
cd "$REPO"

# Activate conda env
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p out err results/phase1_pilot figures/phase1_pilot

# Make sure scheduling package is installed
pip install -e . -q --break-system-packages
pip install scipy matplotlib -q --break-system-packages

# ---- Protocol A + B (real MDLM mode) ----
# Pilot: N=20 trajectories, T=64 steps, M=15 schedules, P=120 pairs
# Adjust N/M/P for the full run (N=50, T=128, M=30, P=300)

srun python -u scripts/run_phase1_pilot.py \
    --T 64 \
    --N 20 \
    --M 15 \
    --P 120 \
    --B_values 4,8,16 \
    --seed 42 \
    --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --out_dir results/phase1_pilot

# ---- Generate figures ----
python scripts/analyze_phase1.py \
    --results_dir results/phase1_pilot \
    --out_dir figures/phase1_pilot

echo "=========================================="
echo "Done: $(date)"
echo "Results: $REPO/results/phase1_pilot/"
echo "Figures: $REPO/figures/phase1_pilot/"
echo "=========================================="
