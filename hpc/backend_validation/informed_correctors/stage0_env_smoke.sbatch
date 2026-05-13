#!/bin/bash
#SBATCH --job-name=ic_text8_s0
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3316152/mdm/masked-diffusion-thesis/logs/ic_text8_stage0_%j.out
#SBATCH --error=/home/3316152/mdm/masked-diffusion-thesis/logs/ic_text8_stage0_%j.err

set -euo pipefail

REPO="${REPO:-$HOME/mdm/masked-diffusion-thesis}"
IC_REPO="${IC_REPO:-$REPO/external_repos/informed-correctors}"
TEXT8_DATA_DIR="${TEXT8_DATA_DIR:-$HOME/mdm/data/text8_md4}"

cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-remdm311}"

SHA="$(git rev-parse --short HEAD 2>/dev/null || cat .hpc_git_sha 2>/dev/null || echo nogit)"
OUT_DIR="$REPO/results/backend_validation/informed_correctors/stage0_env_smoke_${SHA}"
mkdir -p "$OUT_DIR" logs

echo "[stage0] repo=$REPO"
echo "[stage0] informed-correctors=$IC_REPO"
echo "[stage0] text8_data_dir=$TEXT8_DATA_DIR"
echo "[stage0] output=$OUT_DIR"

python3 scripts/backend_validation/informed_correctors/check_text8_training_feasibility.py \
  --external-repo "$IC_REPO" \
  --data-dir "$TEXT8_DATA_DIR" \
  --check-imports \
  --check-jax \
  --check-data \
  --write-report "$OUT_DIR/feasibility.json"

echo "[stage0] complete"

