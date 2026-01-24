#!/bin/bash
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

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate masked-diffusion

cd ~/masked-diffusion-thesis

# Keep submodule in sync (optional but OK)
git submodule update --init --recursive

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun --cpu-bind=cores --hint=nomultithread python -u scripts/run_remdm.py \
  --config configs/remdm_hpc_smoke.yaml

