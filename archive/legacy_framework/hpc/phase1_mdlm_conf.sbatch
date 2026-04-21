#!/bin/bash
#SBATCH --job-name=phase1_mdlm_conf
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# Phase 1 MDLM-conf Experiment: Protocol A + B
#
# Corrector: resample top_k=20 least-confident MASKED positions per step.
# Fixes both bugs from job 478600:
#   Bug #1: signals now over masked positions (not unmasked)
#   Bug #2: corrector now resamples K=20 positions (not all ~990)
#
# Expected: non-zero Δ_t at mid-to-late steps; informative entropy signal.
#
# Preflight (run locally before submitting):
#   python scripts/debug_mdlm_conf_load.py \
#       --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
#       --device cpu --T 4 --top_k 5
#
# Submit:
#   bash hpc/push.sh && sbatch hpc/phase1_mdlm_conf.sbatch
#
# Runtime estimate: ~3-5 hours for PILOT (N=20, T=64, M=15, P=120)

set -euo pipefail

echo "=========================================="
echo "Phase 1 MDLM-conf Experiment"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p out err results/phase1_mdlm_conf figures/phase1_mdlm_conf

pip install -e . -q --break-system-packages
pip install scipy matplotlib -q --break-system-packages

# -----------------------------------------------------------------------
# FAST SANITY RUN (N=5, T=16, M=3, P=10) — uncomment to validate first
# -----------------------------------------------------------------------
# srun python -u scripts/run_phase1_mdlm_conf.py \
#     --T 16 --N 5 --M 3 --P 10 \
#     --B_values 2,4 \
#     --seed 42 \
#     --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
#     --top_k 20 \
#     --out_dir results/phase1_mdlm_conf_sanity

# -----------------------------------------------------------------------
# PILOT RUN (N=20, T=64, M=15, P=120)
# -----------------------------------------------------------------------
srun python -u scripts/run_phase1_mdlm_conf.py \
    --T 64 \
    --N 20 \
    --M 15 \
    --P 120 \
    --B_values 4,8,16 \
    --seed 42 \
    --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --top_k 20 \
    --out_dir results/phase1_mdlm_conf 2>&1

# -----------------------------------------------------------------------
# FULL RUN (N=50, T=64, M=30, P=300) — increase --time to 12:00:00
# -----------------------------------------------------------------------
# srun python -u scripts/run_phase1_mdlm_conf.py \
#     --T 64 \
#     --N 50 \
#     --M 30 \
#     --P 300 \
#     --B_values 4,8,16 \
#     --seed 42 \
#     --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
#     --top_k 20 \
#     --out_dir results/phase1_mdlm_conf_full

echo "=========================================="
echo "Done: $(date)"
echo "Results: $REPO/results/phase1_mdlm_conf/"
echo "=========================================="
