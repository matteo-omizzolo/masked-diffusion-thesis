#!/bin/bash
#SBATCH --job-name=cs_preflight
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/corrector_strength_preflight_%j.out
#SBATCH --error=logs/corrector_strength_preflight_%j.err

set -euo pipefail

REPO="$HOME/mdm/masked-diffusion-thesis"
CHECKPOINT="$HOME/mdm/checkpoints/proseco_owt"

# SHA resolution: env var → git (if .git exists) → .hpc_git_sha file
SHA="${CS_PREFLIGHT_GIT_SHA:-}"
if [ -z "$SHA" ]; then
    SHA=$(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || true)
fi
if [ -z "$SHA" ] && [ -f "$REPO/.hpc_git_sha" ]; then
    SHA=$(cat "$REPO/.hpc_git_sha")
fi
if [ -z "$SHA" ]; then
    echo "[cs_preflight] ERROR: no git SHA available. Submit with --export=ALL,CS_PREFLIGHT_GIT_SHA=<sha> or create .hpc_git_sha" >&2
    exit 1
fi

OUT_DIR="$REPO/results/corrector_strength_preflight_${SHA}"

echo "[cs_preflight] Job: $SLURM_JOB_ID | SHA: $SHA"
echo "[cs_preflight] Node: $SLURM_NODELIST | GPU: $CUDA_VISIBLE_DEVICES"
echo "[cs_preflight] Output: $OUT_DIR"

mkdir -p logs

# Activate conda environment
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

# Verify Python version (must be >=3.10)
python3 -c "import sys; assert sys.version_info >= (3, 10), f'Need Python>=3.10, got {sys.version}'"
echo "[cs_preflight] Python OK: $(python3 --version)"

# Sanity: GPU visible
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}')"

# Run preflight (seeds 42, 43; T=64; 3 pairs per stratum; all 4 strength levels)
python3 "$REPO/scripts/run_corrector_strength_preflight.py" \
    --checkpoint "$CHECKPOINT" \
    --device cuda \
    --out-dir "$OUT_DIR" \
    --seeds 42 43 \
    --strengths no_correction strength_0 strength_1 strength_2

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[cs_preflight] Gate PASS — running analysis..."
    python3 "$REPO/scripts/analyze_corrector_strength_preflight.py" "$OUT_DIR"
    echo "[cs_preflight] Analysis complete. Results: $OUT_DIR"
else
    echo "[cs_preflight] Gate FAIL (exit code $EXIT_CODE). Check $OUT_DIR/summary.json"
fi

echo "[cs_preflight] Done. Exit: $EXIT_CODE"
exit $EXIT_CODE
