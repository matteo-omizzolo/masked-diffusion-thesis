#!/bin/bash
# ---------------------------------------------------------------------------
# Token-level feature extraction + analysis — FULL K=30, T=64.
#
# Runs the bounded extraction pilot at the full Gate 3a seed set (seeds
# 42-71) so that the analysis against canonical xi_raw.json has 30
# bootstrap units, matching every other K=30 result in this thesis.
#
# PRECONDITION: only submit after the archived token-level preflight has
# reported PREFLIGHT_PASS in its log (alignment.rng_aligned == true). The
# original preflight sbatch and output are archived under the 2026-05-13
# repo-cleanup archive; see docs/archive/2026-05-13_repo_cleanup/MANIFEST.md.
#
# Submit with:
#   sbatch hpc/proseco/tokenlevel/tokenlevel_k30.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=tokenlevel_k30
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/3316152/mdm/masked-diffusion-thesis/logs/tokenlevel_k30_%j.out
#SBATCH --error=/home/3316152/mdm/masked-diffusion-thesis/logs/tokenlevel_k30_%j.err

set -euo pipefail

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

cd "/home/3316152/mdm/masked-diffusion-thesis"
export PROSECO_OWT_CHECKPOINT="$HOME/mdm/checkpoints/proseco_owt"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo '0c39079')"
OUT_DIR="results/tokenlevel_features_proseco_k30_${GIT_SHA}"
ANALYSIS_DIR="results/tokenlevel_state_predictability_k30_${GIT_SHA}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
python -c "import mdm_playground; print('mdm_playground OK')"
test -d "$PROSECO_OWT_CHECKPOINT" || { echo "missing checkpoint"; exit 2; }

# Preflight sanity: refuse to run if preflight result is absent or failed.
shopt -s nullglob
PF_CANDIDATES=(results/tokenlevel_state_predictability_preflight_* results/archive/2026-05-13_repo_cleanup/tokenlevel_state_predictability_preflight_*)
shopt -u nullglob
if [ "${#PF_CANDIDATES[@]}" -eq 0 ]; then
    echo "ERROR: no preflight directory found. Restore or rerun the archived preflight before resubmission." >&2
    exit 3
fi
PF_LATEST="$(ls -dt "${PF_CANDIDATES[@]}" | head -1)"
python - <<PY
import json, sys
agg = json.load(open("$PF_LATEST/aggregate_stats.json"))
ok = bool(agg["meta"]["alignment_status"].get("rng_aligned"))
print("preflight rng_aligned:", ok, "from", "$PF_LATEST")
sys.exit(0 if ok else 4)
PY

echo "=== K=30 EXTRACTION (seeds 42..71; T=64) ==="
python scripts/proseco/state_predictability/extract_tokenlevel_features_proseco.py \
    --checkpoint "$PROSECO_OWT_CHECKPOINT" \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
            52 53 54 55 56 57 58 59 60 61 \
            62 63 64 65 66 67 68 69 70 71 \
    --T 64 \
    --device cuda \
    --out-dir "$OUT_DIR"

echo "=== K=30 ANALYSIS ==="
python scripts/proseco/state_predictability/analyze_tokenlevel_state_predictability.py \
    --features-dir "$OUT_DIR" \
    --xi-file results/phase1_interaction_diag_nogit/xi_raw.json \
    --protocol-a-dir results/phase1_proseco_owt_full/protocol_a \
    --out-dir "$ANALYSIS_DIR" \
    --n-folds 5 \
    --alignment-tol 1e-3

echo "=== VERDICT ==="
python - <<PY
import json
agg = json.load(open("$ANALYSIS_DIR/aggregate_stats.json"))
print("alignment_status:", json.dumps(agg["meta"]["alignment_status"], indent=2))
print("verdict:", json.dumps(agg["verdict"], indent=2))
PY

echo "DONE: K=30 token-level run complete."
