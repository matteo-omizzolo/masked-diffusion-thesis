#!/bin/bash
#SBATCH --job-name=ic_text8_s1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3316152/mdm/masked-diffusion-thesis/logs/ic_text8_stage1_%j.out
#SBATCH --error=/home/3316152/mdm/masked-diffusion-thesis/logs/ic_text8_stage1_%j.err

set -euo pipefail

# Tiny training-loop smoke only. Do not use this as a useful checkpoint.

REPO="${REPO:-$HOME/mdm/masked-diffusion-thesis}"
IC_REPO="${IC_REPO:-$REPO/external_repos/informed-correctors}"
TEXT8_DATA_DIR="${TEXT8_DATA_DIR:-$HOME/mdm/data/text8_md4}"
TEXT8_WORD_VOCAB_PKL="${TEXT8_WORD_VOCAB_PKL:-$TEXT8_DATA_DIR/text8_word_vocab.pkl}"
IC_STAGE1_STEPS="${IC_STAGE1_STEPS:-50}"
IC_STAGE1_BATCH_SIZE="${IC_STAGE1_BATCH_SIZE:-8}"

cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-remdm311}"

SHA="$(git rev-parse --short HEAD 2>/dev/null || cat .hpc_git_sha 2>/dev/null || echo nogit)"
OUT_DIR="$REPO/results/backend_validation/informed_correctors/stage1_tiny_train_${SHA}"
RUN_DIR="$OUT_DIR/upstream_copy"
mkdir -p "$OUT_DIR" "$RUN_DIR" logs

test -d "$IC_REPO/text8" || { echo "missing informed-correctors text8 tree: $IC_REPO/text8" >&2; exit 2; }
test -f "$TEXT8_DATA_DIR/text8/text8.zip" || {
  echo "missing Text8 zip: $TEXT8_DATA_DIR/text8/text8.zip" >&2
  echo "Stage Text8 manually before Stage 1; this job does not download data." >&2
  exit 3
}

if [ ! -f "$TEXT8_WORD_VOCAB_PKL" ]; then
  echo "[stage1] creating minimal word-vocab pickle at $TEXT8_WORD_VOCAB_PKL"
  mkdir -p "$(dirname "$TEXT8_WORD_VOCAB_PKL")"
  TEXT8_WORD_VOCAB_PKL="$TEXT8_WORD_VOCAB_PKL" python3 - <<'PY'
import os
import pickle
from pathlib import Path
path = Path(os.environ["TEXT8_WORD_VOCAB_PKL"])
vocab = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "it"}
with path.open("wb") as f:
    pickle.dump(vocab, f)
PY
fi

rsync -a --delete "$IC_REPO/text8/" "$RUN_DIR/text8/"

RUN_DIR="$RUN_DIR" TEXT8_DATA_DIR="$TEXT8_DATA_DIR" python3 - <<'PY'
import os
from pathlib import Path
run_dir = Path(os.environ["RUN_DIR"])
text8_data_dir = os.environ["TEXT8_DATA_DIR"]
path = run_dir / "text8" / "md4" / "input_pipeline.py"
text = path.read_text()
old = 'DATA_DIR = "/root/md4/data_dir"'
new = f'DATA_DIR = {text8_data_dir!r}'
if old not in text:
    raise SystemExit(f"[stage1] expected DATA_DIR line not found in {path}")
path.write_text(text.replace(old, new))
PY

export PYTHONPATH="$RUN_DIR/text8:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TEXT8_WORD_VOCAB_PKL
export IC_STAGE1_STEPS
export IC_STAGE1_BATCH_SIZE

python3 "$RUN_DIR/text8/md4/main.py" \
  --config="$REPO/scripts/backend_validation/informed_correctors/hollow_text8_stage1_config.py" \
  --workdir="$OUT_DIR/workdir" \
  --sharded=false

echo "[stage1] complete: $OUT_DIR"

