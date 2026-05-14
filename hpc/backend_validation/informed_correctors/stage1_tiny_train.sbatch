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
# Default to the dedicated CUDA13 JAX env. The old remdm311 env is reserved
# for PyTorch/ProSeCo work and is known to fail Stage 1 under the
# cuda12-on-cuda13-MIG combination (CLAUDE.md issue #14). Override via
# `CONDA_ENV=...` only if explicitly testing another env.
conda activate "${CONDA_ENV:-ic_text8_jax13}"

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

# Patch 1: redirect input_pipeline DATA_DIR away from /root/md4/data_dir.
path = run_dir / "text8" / "md4" / "input_pipeline.py"
text = path.read_text()
old = 'DATA_DIR = "/root/md4/data_dir"'
new = f'DATA_DIR = {text8_data_dir!r}'
if old not in text:
    raise SystemExit(f"[stage1] expected DATA_DIR line not found in {path}")
path.write_text(text.replace(old, new))

# Patch 2: make `import distrax` in utils.py defensive. distrax 0.1.8 (the
# only published version) is incompatible with jax[cuda12]'s newest JAX
# (missing jax.interpreters.xla.pytype_aval_mappings). distrax is only
# used by DiscretizedLogisticMixture (ImageNet code path) which Text8
# HollowMD4 training never reaches. Both the import AND the class
# definition that inherits from distrax.Distribution must be guarded,
# because the class is evaluated at module load time.
utils_path = run_dir / "text8" / "md4" / "utils.py"
utils_text = utils_path.read_text()

old_import = "import distrax"
new_import = (
    "try:\n"
    "    import distrax\n"
    "except Exception as _distrax_exc:  # informed-correctors Text8 patch\n"
    "    distrax = None\n"
)
if old_import not in utils_text:
    raise SystemExit(
        f"[stage1] expected 'import distrax' line not found in {utils_path}"
    )
utils_text = utils_text.replace(old_import, new_import, 1)

# Find the DiscretizedLogisticMixture class block (between `class
# DiscretizedLogisticMixture(distrax.Distribution):` and the next
# `\nclass ` or `\ndef ` at column 0) and wrap it in `if distrax is not
# None:` so the body parses but only runs when distrax is healthy.
import re
match = re.search(
    r"^class DiscretizedLogisticMixture\(distrax\.Distribution\):.*?(?=^class |^def |\Z)",
    utils_text,
    flags=re.MULTILINE | re.DOTALL,
)
if not match:
    raise SystemExit(
        f"[stage1] DiscretizedLogisticMixture class not found in {utils_path}"
    )
class_body = match.group(0)
indented = "    " + class_body.rstrip("\n").replace("\n", "\n    ") + "\n"
guarded = (
    "if distrax is not None:  # informed-correctors Text8 patch\n"
    + indented
    + "else:\n"
    + "    DiscretizedLogisticMixture = None  # type: ignore[assignment]\n\n"
)
utils_text = utils_text[: match.start()] + guarded + utils_text[match.end() :]
utils_path.write_text(utils_text)
print("[stage1] patched DATA_DIR, distrax import, and DiscretizedLogisticMixture class")
PY

export PYTHONPATH="$RUN_DIR/text8:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TEXT8_WORD_VOCAB_PKL
export IC_STAGE1_STEPS
export IC_STAGE1_BATCH_SIZE

# JAX / CUDA workaround for A100 MIG mode under CUDA-13 driver: disable
# allocator preallocation so JAX cooperates with MIG-partitioned memory.
# See CLAUDE.md known issue #14 and
# docs/09_informed_correctors_training_contingency.md §Stage 1 current blocker.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

python3 "$RUN_DIR/text8/md4/main.py" \
  --config="$REPO/scripts/backend_validation/informed_correctors/hollow_text8_stage1_config.py" \
  --workdir="$OUT_DIR/workdir" \
  --sharded=false

echo "[stage1] complete: $OUT_DIR"

