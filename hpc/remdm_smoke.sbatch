#!/bin/bash
#SBATCH --job-name=remdm_smoke
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

set -euo pipefail

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

# 1) Repo location on HPC
REPO=~/mdm/masked-diffusion-thesis
cd "$REPO"

# 2) Activate environment
# If using conda (recommended on HPC):
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate remdm
# Fallback: venv created by hpc/setup_env.sh
elif [ -d "$REPO/.venv" ]; then
    source "$REPO/.venv/bin/activate"
else
    echo "ERROR: no conda env 'remdm' and no .venv found. Run hpc/setup_env.sh first." >&2
    exit 1
fi

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 3) Ensure submodules are present
git submodule update --init --recursive

# 4) CPU threading (avoid oversubscription)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 5) Create output dirs
mkdir -p out err results/remdm_smoke

# 6) Run smoke test: 2 batches of ancestral MDLM sampling (baseline, no remasking)
#    This verifies the checkpoint loads and the pipeline runs end-to-end.
#    Expected output: perplexity ~26-28 BPB on OpenWebText.
srun python -u -m mdm_playground.cli.run \
    --method remdm \
    --model_id /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --strategy remdm_conf \
    --steps 64 \
    --remdm_num_batches 2 \
    --out_dir results/remdm_smoke \
    --seed 42

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
