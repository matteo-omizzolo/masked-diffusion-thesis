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

# 1) Activate conda environment
# IMPORTANT: use the environment you will install deps into (not your Mac venv).
# Example name below; change it to your actual env.
source activate masked-diffusion

# 2) Move into your thesis repo (change path to where you cloned it)
cd ~/masked-diffusion-thesis

# 3) Ensure submodules are present (external/remdm)
git submodule update --init --recursive

# 4) CPU threading settings (helps avoid oversubscription)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 5) Run a SMALL ReMDM job via your wrapper (prints logs live because of -u)
# Use srun so stdout is streamed (useful with tail -f).
srun --cpu-bind=cores --hint=nomultithread python -u scripts/run_remdm.py \
  --config configs/remdm_hpc_smoke.yaml

