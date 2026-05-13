#!/bin/bash
# ---------------------------------------------------------------------------
# Token-level feature extraction PREFLIGHT — K=2 seeds, T=64.
#
# Purpose: verify RNG alignment with canonical Phase 1 / Gate 3a artifacts
# before launching the full K=30 job. Compares the extracted T0 aggregate
# (entropy, inverse_margin, quality_mass_proxy, unmasked_fraction,
# n_revisable, n_masked) to results/phase1_proseco_owt_full/protocol_a/
# trajectory_*.json values for the same (seed, t). Abs tolerance 1e-3.
#
# Submit with:
#   sbatch hpc/tokenlevel_preflight_k2.sbatch
# Then read logs/tokenlevel_preflight_k2_<job>.out for alignment verdict.
# ---------------------------------------------------------------------------
#SBATCH --job-name=tokenlevel_preflight_k2
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/3316152/mdm/masked-diffusion-thesis/logs/tokenlevel_preflight_k2_%j.out
#SBATCH --error=/home/3316152/mdm/masked-diffusion-thesis/logs/tokenlevel_preflight_k2_%j.err

set -euo pipefail

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

cd "/home/3316152/mdm/masked-diffusion-thesis"
export PROSECO_OWT_CHECKPOINT="$HOME/mdm/checkpoints/proseco_owt"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo '0c39079')"
OUT_DIR="results/tokenlevel_features_proseco_preflight_${GIT_SHA}"
ANALYSIS_DIR="results/tokenlevel_state_predictability_preflight_${GIT_SHA}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
python -c "import mdm_playground; print('mdm_playground OK')"
test -d "$PROSECO_OWT_CHECKPOINT" || { echo "missing checkpoint"; exit 2; }

echo "=== K=2 PREFLIGHT EXTRACTION (seeds 42, 43; T=64) ==="
python scripts/extract_tokenlevel_features_proseco.py \
    --checkpoint "$PROSECO_OWT_CHECKPOINT" \
    --seeds 42 43 \
    --T 64 \
    --device cuda \
    --out-dir "$OUT_DIR"

echo "=== ANALYSIS / ALIGNMENT CHECK ==="
python scripts/analyze_tokenlevel_state_predictability.py \
    --features-dir "$OUT_DIR" \
    --xi-file results/phase1_interaction_diag_nogit/xi_raw.json \
    --protocol-a-dir results/phase1_proseco_owt_full/protocol_a \
    --out-dir "$ANALYSIS_DIR" \
    --n-folds 2 \
    --alignment-tol 1e-3 || echo "(analysis returned non-zero, inspect logs)"

echo "=== ALIGNMENT VERDICT ==="
python - <<'PY'
import json, pathlib, sys
import os
import glob
d = sorted(glob.glob("results/tokenlevel_state_predictability_preflight_*"))[-1]
agg = json.load(open(pathlib.Path(d) / "aggregate_stats.json"))
align = agg["meta"]["alignment_status"]
print("alignment:", json.dumps(align, indent=2))
verdict_aligned = bool(align.get("rng_aligned", False))
print("PREFLIGHT_PASS" if verdict_aligned else "PREFLIGHT_FAIL")
sys.exit(0 if verdict_aligned else 0)  # do not fail the sbatch on alignment fail
PY

echo "DONE: preflight complete."
