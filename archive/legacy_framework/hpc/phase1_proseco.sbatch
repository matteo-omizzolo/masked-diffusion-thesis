#!/bin/bash
#SBATCH --job-name=phase1_proseco
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

# Phase 1 ProSeCo Experiment: Protocol A + B
# Main empirical platform for the thesis:
#   "Signal-adaptive corrector scheduling for masked diffusion LMs"
#
# Uses ProSeCo-style annealed-refinement corrector on MDLM checkpoint.
# Corrector: corrector_steps=2 inner NFEs per loop, action_set=unmasked positions.
#
# Submit:
#   bash hpc/push.sh && sbatch hpc/phase1_proseco.sbatch
#
# Runtime estimate: ~3-5 hours for PILOT (N=20, T=64, M=15, P=120)
# Runtime estimate: ~8 hours for FULL (N=50, T=64, M=30, P=300)

set -euo pipefail

echo "=========================================="
echo "Phase 1 ProSeCo Experiment"
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

mkdir -p out err results/phase1_proseco figures/phase1_proseco

# Install updated package
pip install -e . -q --break-system-packages
pip install scipy matplotlib -q --break-system-packages

# -----------------------------------------------------------------------
# FAST SANITY RUN (N=5, T=16, M=3, P=10)
# Uncomment to validate backend before the real run.
# -----------------------------------------------------------------------
# srun python -u scripts/run_phase1_proseco.py \
#     --T 16 --N 5 --M 3 --P 10 \
#     --B_values 2,4 \
#     --seed 42 \
#     --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
#     --corrector_steps 2 \
#     --out_dir results/phase1_proseco_sanity

# -----------------------------------------------------------------------
# PILOT RUN (N=20, T=64, M=15, P=120)
# -----------------------------------------------------------------------
srun python -u scripts/run_phase1_proseco.py \
    --T 64 \
    --N 20 \
    --M 15 \
    --P 120 \
    --B_values 4,8,16 \
    --seed 42 \
    --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --corrector_steps 2 \
    --out_dir results/phase1_proseco

# -----------------------------------------------------------------------
# FULL RUN (N=50, T=64, M=30, P=300)
# To run full: comment out PILOT, uncomment FULL, increase --time to 12:00:00
# -----------------------------------------------------------------------
# srun python -u scripts/run_phase1_proseco.py \
#     --T 64 \
#     --N 50 \
#     --M 30 \
#     --P 300 \
#     --B_values 4,8,16 \
#     --seed 42 \
#     --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
#     --corrector_steps 2 \
#     --out_dir results/phase1_proseco_full

echo "=========================================="
echo "Done: $(date)"
echo "Results: $REPO/results/phase1_proseco/"
echo "=========================================="
