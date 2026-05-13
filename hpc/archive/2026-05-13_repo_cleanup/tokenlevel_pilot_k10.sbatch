#!/bin/bash
# ---------------------------------------------------------------------------
# Bounded token-level feature extraction pilot — K=10 seeds, T=64.
#
# Purpose: extract pre-correction token-level features along the canonical
# Phase 1 base trajectory on the same A100/CUDA hardware that produced
# results/phase1_proseco_owt_full/protocol_a/ and
# results/phase1_interaction_diag_nogit/xi_raw.json, so that the new
# features are RNG-aligned with the canonical xi targets.
#
# This is a PILOT. The pre-registered expansion gate is in
# scripts/analyze_tokenlevel_state_predictability.py (P6 vs P1 on xi,
# with seed-bootstrap CI excluding 0 on the positive side and >=60 % of
# seeds improved). If the gate does not pass, do NOT expand to K=30.
#
# Submit with:
#   sbatch hpc/tokenlevel_pilot_k10.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=tokenlevel_pilot_k10
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/3316152/mdm/masked-diffusion-thesis/logs/tokenlevel_pilot_k10_%j.out
#SBATCH --error=/home/3316152/mdm/masked-diffusion-thesis/logs/tokenlevel_pilot_k10_%j.err

set -euo pipefail

# --- environment ------------------------------------------------------------
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

# --- paths -----------------------------------------------------------------
cd "/home/3316152/mdm/masked-diffusion-thesis"
export PROSECO_OWT_CHECKPOINT="$HOME/mdm/checkpoints/proseco_owt"
GIT_SHA="$(git rev-parse --short HEAD)"
OUT_DIR="results/tokenlevel_features_proseco_pilot_${GIT_SHA}"

# --- preflight -------------------------------------------------------------
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required for RNG alignment'"
python -c "import mdm_playground; print('mdm_playground OK')"
test -d "$PROSECO_OWT_CHECKPOINT" || { echo "missing checkpoint"; exit 2; }

# --- extraction (K=10 seeds matching the first 10 Gate 3a seeds) -----------
python scripts/extract_tokenlevel_features_proseco.py \
    --checkpoint "$PROSECO_OWT_CHECKPOINT" \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --T 64 \
    --device cuda \
    --out-dir "$OUT_DIR"

# --- alignment + analysis --------------------------------------------------
python scripts/analyze_tokenlevel_state_predictability.py \
    --features-dir "$OUT_DIR" \
    --xi-file results/phase1_interaction_diag_nogit/xi_raw.json \
    --protocol-a-dir results/phase1_proseco_owt_full/protocol_a \
    --out-dir "results/tokenlevel_state_predictability_pilot_${GIT_SHA}" \
    --n-folds 5 \
    --alignment-tol 1e-3

# --- sanity dump for the run record ----------------------------------------
echo "--- aggregate verdict ---"
python -c "
import json, pathlib
p = pathlib.Path('results/tokenlevel_state_predictability_pilot_${GIT_SHA}/aggregate_stats.json')
d = json.load(open(p))
print('alignment_status:', json.dumps(d['meta']['alignment_status'], indent=2))
print('verdict:', json.dumps(d['verdict'], indent=2))
"
echo "DONE: pilot complete."
