#!/bin/bash
#SBATCH --job-name=phase1_proseco_owt
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

# Phase 1 ProSeCo-OWT Experiment: Protocol A + B
#
# Backend: ProSeCo-OWT (kuleshov-group/proseco-owt) — co-trained backbone.
# Corrector: annealed-refinement on UNMASKED positions, corrector_steps=1.
#
# This is the first run where Δ_t > 0 is expected at mid-to-late trajectory
# because the backbone was co-trained to respond to the corrector perturbation.
#
# Preflight (run on HPC login node before submitting):
#   Step 1 — stage checkpoint (once):
#     module load miniconda3 && source $(conda info --base)/etc/profile.d/conda.sh
#     conda activate remdm311
#     python scripts/stage_proseco_owt.py \
#         --dest ~/mdm/checkpoints/proseco_owt
#
#   Step 2 — CPU validation:
#     python scripts/debug_proseco_owt_load.py \
#         --checkpoint ~/mdm/checkpoints/proseco_owt \
#         --device cpu --T 4
#
#   Step 3 — submit:
#     bash hpc/push.sh && sbatch hpc/phase1_proseco_owt.sbatch
#
# Runtime estimate: ~4-6 hours for PILOT (N=20, T=64, M=15, P=120)

set -euo pipefail

echo "=========================================="
echo "Phase 1 ProSeCo-OWT Experiment"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
CHECKPOINT=~/mdm/checkpoints/proseco_owt
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

mkdir -p out err results/phase1_proseco_owt figures/phase1_proseco_owt

pip install -e . -q --break-system-packages
pip install scipy matplotlib -q --break-system-packages
pip install einops -q --break-system-packages

# Validate checkpoint is staged
python -c "
from pathlib import Path
p = Path('$CHECKPOINT')
assert p.exists(), f'Checkpoint dir not found: {p}'
assert (p / 'config.json').exists(), 'config.json missing — run stage_proseco_owt.py first'
assert (p / 'modeling_proseco.py').exists(), 'modeling_proseco.py missing'
print(f'Checkpoint OK: {p}')
" 2>&1

# -----------------------------------------------------------------------
# SANITY RUN (N=5, T=16, M=3, P=10) — uncomment to validate first
# -----------------------------------------------------------------------
# srun python -u scripts/run_phase1_proseco_owt.py \
#     --T 16 --N 5 --M 3 --P 10 \
#     --B_values 2,4 \
#     --seed 42 \
#     --checkpoint $CHECKPOINT \
#     --corrector_steps 1 \
#     --out_dir results/phase1_proseco_owt_sanity 2>&1

# -----------------------------------------------------------------------
# PILOT RUN (N=20, T=64, M=15, P=120)
# -----------------------------------------------------------------------
srun python -u scripts/run_phase1_proseco_owt.py \
    --T 64 \
    --N 20 \
    --M 15 \
    --P 120 \
    --B_values 4,8,16 \
    --seed 42 \
    --checkpoint $CHECKPOINT \
    --corrector_steps 1 \
    --out_dir results/phase1_proseco_owt 2>&1

# -----------------------------------------------------------------------
# FULL RUN (N=50, T=64, M=30, P=300) — increase --time to 12:00:00
# -----------------------------------------------------------------------
# srun python -u scripts/run_phase1_proseco_owt.py \
#     --T 64 \
#     --N 50 \
#     --M 30 \
#     --P 300 \
#     --B_values 4,8,16 \
#     --seed 42 \
#     --checkpoint $CHECKPOINT \
#     --corrector_steps 1 \
#     --out_dir results/phase1_proseco_owt_full 2>&1

echo "=========================================="
echo "Done: $(date)"
echo "Results: $REPO/results/phase1_proseco_owt/"
echo "=========================================="
