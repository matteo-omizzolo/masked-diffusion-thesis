#!/bin/bash
#SBATCH --job-name=phase1_proseco_owt_full
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# Phase 1 ProSeCo-OWT FULL RUN (N=50, T=64, M=30, P=300)
#
# Prerequisite: pilot (job 479382) passed 5-6/6 §7.2 criteria.
# entropy_bot_B beats uniform at all B; C6 passes with inverted signal direction.
# This is the primary thesis evidence base.

set -euo pipefail

echo "=========================================="
echo "Phase 1 ProSeCo-OWT FULL RUN"
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

mkdir -p out err results/phase1_proseco_owt_full figures/phase1_proseco_owt_full

pip install -e . -q --break-system-packages
pip install scipy matplotlib safetensors -q --break-system-packages
pip install einops -q --break-system-packages

python -c "
from pathlib import Path
p = Path('$CHECKPOINT')
assert p.exists(), f'Checkpoint dir not found: {p}'
assert (p / 'config.json').exists(), 'config.json missing'
assert (p / 'modeling_proseco.py').exists(), 'modeling_proseco.py missing'
print(f'Checkpoint OK: {p}')
" 2>&1

srun python -u scripts/run_phase1_proseco_owt.py \
    --T 64 \
    --N 50 \
    --M 30 \
    --P 300 \
    --B_values 4,8,16 \
    --seed 42 \
    --checkpoint $CHECKPOINT \
    --corrector_steps 1 \
    --out_dir results/phase1_proseco_owt_full 2>&1

echo "=========================================="
echo "Done: $(date)"
echo "Results: $REPO/results/phase1_proseco_owt_full/"
echo "=========================================="
