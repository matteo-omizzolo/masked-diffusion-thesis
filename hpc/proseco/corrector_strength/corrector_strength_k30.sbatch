#!/bin/bash
#SBATCH --job-name=cs_k30
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/3316152/mdm/masked-diffusion-thesis/logs/corrector_strength_k30_%j.out
#SBATCH --error=/home/3316152/mdm/masked-diffusion-thesis/logs/corrector_strength_k30_%j.err

set -euo pipefail

REPO="$HOME/mdm/masked-diffusion-thesis"
CHECKPOINT="$HOME/mdm/checkpoints/proseco_owt"

# SHA resolution: env var → git (if .git exists) → .hpc_git_sha file
SHA="${CS_K30_GIT_SHA:-}"
if [ -z "$SHA" ]; then
    SHA=$(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || true)
fi
if [ -z "$SHA" ] && [ -f "$REPO/.hpc_git_sha" ]; then
    SHA=$(cat "$REPO/.hpc_git_sha")
fi
if [ -z "$SHA" ]; then
    echo "[cs_k30] ERROR: no git SHA available. Submit with --export=ALL,CS_K30_GIT_SHA=<sha> or create .hpc_git_sha" >&2
    exit 1
fi

OUT_DIR="$REPO/results/corrector_strength_k30_${SHA}"

echo "[cs_k30] Job: $SLURM_JOB_ID | SHA: $SHA"
echo "[cs_k30] Node: $SLURM_NODELIST | GPU: $CUDA_VISIBLE_DEVICES"
echo "[cs_k30] Output: $OUT_DIR"

mkdir -p logs

# Activate conda environment
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

# Verify Python version (must be >=3.10)
python3 -c "import sys; assert sys.version_info >= (3, 10), f'Need Python>=3.10, got {sys.version}'"
echo "[cs_k30] Python OK: $(python3 --version)"

# Sanity: GPU visible
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}')"

# Main K=30 run: all 30 seeds (42-71), T=64, all 4 strength levels, n_per_stratum=3
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71"
python3 "$REPO/scripts/proseco/corrector_strength/run_corrector_strength_preflight.py" \
    --checkpoint "$CHECKPOINT" \
    --device cuda \
    --out-dir "$OUT_DIR" \
    --seeds $SEEDS \
    --strengths no_correction strength_0 strength_1 strength_2

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[cs_k30] Gate PASS — running analysis..."
    python3 "$REPO/scripts/proseco/corrector_strength/analyze_corrector_strength_preflight.py" "$OUT_DIR"
    echo "[cs_k30] Analysis complete. Results: $OUT_DIR"
else
    echo "[cs_k30] Gate FAIL or partial run (exit code $EXIT_CODE). Check $OUT_DIR/summary.json"
fi

echo "[cs_k30] Done. Exit: $EXIT_CODE"
exit $EXIT_CODE
