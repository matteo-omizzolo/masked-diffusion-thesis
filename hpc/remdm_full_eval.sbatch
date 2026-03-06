#!/bin/bash
#SBATCH --job-name=remdm_eval
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
# stud QOS: MaxSubmitJobsPerUser=2, so max 2 array tasks at once.
# Tasks 0-1: mdlm + remdm-conf (core thesis comparison).
# Task 2 (remdm-loop): submit separately via `bash hpc/submit.sh eval-loop`.
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --out=out/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err
#SBATCH --mail-type=END,ARRAY_TASKS_DONE
#SBATCH --mail-user=3316152@studbocconi.it

set -euo pipefail

STRATEGIES=("mdlm" "remdm-conf" "remdm-loop")
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Strategy:  $STRATEGY"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

# 1) Repo location on HPC
REPO=~/mdm/masked-diffusion-thesis
cd "$REPO"

# 2) Activate environment
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 3) CPU threading (avoid oversubscription)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4) Create output dirs
OUT_DIR=results/full_eval/$STRATEGY
mkdir -p out err "$OUT_DIR"

# 5) Run full eval: 100 batches, 128 steps, MAUVE enabled
srun python -u -m mdm_playground.cli.run \
    --method remdm \
    --model_id /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --remdm_strategy "$STRATEGY" \
    --steps 128 \
    --remdm_num_batches 100 \
    --out_dir "$OUT_DIR" \
    --seed 42

echo "=========================================="
echo "Done: $(date)"
echo "Outputs at: $OUT_DIR"
echo "=========================================="
