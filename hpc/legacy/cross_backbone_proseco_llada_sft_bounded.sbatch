#!/bin/bash
#SBATCH --job-name=cross_backbone_llada_bounded
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# LEGACY / PROVENANCE — DO NOT SUBMIT FOR CURRENT STATUS.
# Closed LLaDA-SFT bounded probe. This script references historical script paths
# and may overwrite old cross-backbone outputs if re-run.
#
# Bounded cross-backbone replication on ProSeCo-LLaDA-SFT:
#   Stage 0: preflight load check
#   Stage 1: Protocol A (K=8)
#   Stage 2: Phase 2b bounded (B={2,4}, mc_P=24)
#   Stage 3: Phase 3a bounded (B={2,4}, reduced CD/BS budgets)

set -euo pipefail

echo "=========================================="
echo "Cross-backbone bounded replication (LLaDA-SFT)"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
CHECKPOINT=~/mdm/checkpoints/proseco_llada_sft
RUN_ROOT=$REPO/results/cross_backbone/proseco_llada_sft_bounded
PROTO_A_DIR=$RUN_ROOT/protocol_a
PHASE2B_RAW=$RUN_ROOT/phase2b_raw
PHASE2B_AGGR=$RUN_ROOT/phase2b
PHASE3A_RAW=$RUN_ROOT/phase3a_raw
PHASE3A_AGGR=$RUN_ROOT/phase3a
FIG_ROOT=$REPO/figures/cross_backbone/proseco_llada_sft_bounded

cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p out err "$RUN_ROOT" "$FIG_ROOT"/phase2b "$FIG_ROOT"/phase3a

pip install -e . -q --break-system-packages
pip install "transformers==4.38.2" huggingface_hub safetensors einops scipy matplotlib -q --break-system-packages

python -c "
from pathlib import Path
p = Path('$CHECKPOINT')
assert p.exists(), f'Checkpoint dir not found: {p}'
required = ['config.json', 'configuration_llada.py', 'modeling_llada.py', 'model.safetensors.index.json']
missing = [x for x in required if not (p / x).exists()]
assert not missing, f'Missing checkpoint files: {missing}'
print(f'Checkpoint OK: {p}')
" 2>&1

echo "Running LLADA preflight..."
python -u scripts/debug_proseco_llada_sft_load.py \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --T 4 \
  --corrector_steps 1 2>&1 | tee "out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_preflight.log"

echo "Running Protocol A (bounded)..."
python -u scripts/run_protocol_a_proseco_snapshot.py \
  --checkpoint "$CHECKPOINT" \
  --backend proseco_llada_sft \
  --T 64 \
  --K 8 \
  --seed_start 142 \
  --corrector_steps 1 \
  --out_dir "$PROTO_A_DIR" 2>&1 | tee "out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_protocol_a.log"

echo "Running Phase 2b bounded..."
python -u scripts/proseco/reproduction/run_phase2b_proseco_owt.py \
  --checkpoint "$CHECKPOINT" \
  --backend proseco_llada_sft \
  --protocol_a_dir "$PROTO_A_DIR" \
  --out_dir "$PHASE2B_RAW" \
  --T 64 \
  --K 8 \
  --B_values 2,4 \
  --mc_B_values 2,4 \
  --mc_P 24 \
  --seed_start 142 \
  --corrector_steps 1 \
  --shard_idx 0 \
  --shard_count 1 \
  --resume 2>&1 | tee "out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_phase2b.log"

python -c "
import json
from pathlib import Path
out = Path('$PHASE2B_RAW')
policy_all, mc_all = [], []
for f in sorted(out.glob('policy_raw.shard*-of-*.json')):
    policy_all.extend(json.loads(f.read_text()))
for f in sorted(out.glob('mc_raw.shard*-of-*.json')):
    mc_all.extend(json.loads(f.read_text()))
(out / 'policy_raw.json').write_text(json.dumps(policy_all, indent=2))
(out / 'mc_raw.json').write_text(json.dumps(mc_all, indent=2))
print(f'Aggregated phase2b: {len(policy_all)} policy rows, {len(mc_all)} MC rows')
" 2>&1

python -u scripts/proseco/reproduction/analyze_phase2b.py \
  --results_dir "$PHASE2B_RAW" \
  --out_dir "$PHASE2B_AGGR" \
  --figures_dir "$FIG_ROOT/phase2b" 2>&1 | tee "out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_phase2b_analysis.log"

echo "Running Phase 3a bounded..."
python -u scripts/proseco/reproduction/run_phase3a_combinatorial.py \
  --checkpoint "$CHECKPOINT" \
  --backend proseco_llada_sft \
  --protocol_a_dir "$PROTO_A_DIR" \
  --out_dir "$PHASE3A_RAW" \
  --T 64 \
  --K 8 \
  --B_values 2,4 \
  --seed_start 142 \
  --corrector_steps 1 \
  --cd_max_attempts 24 \
  --cd_window 8 \
  --bs_beam_width 4 \
  --shard_idx 0 \
  --shard_count 1 \
  --resume 2>&1 | tee "out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_phase3a.log"

python -c "
import json
from pathlib import Path
out = Path('$PHASE3A_RAW')
cd_all, bs_all = [], []
for f in sorted(out.glob('cd_raw.shard*-of-*.json')):
    cd_all.extend(json.loads(f.read_text()))
for f in sorted(out.glob('bs_raw.shard*-of-*.json')):
    bs_all.extend(json.loads(f.read_text()))
(out / 'cd_raw.json').write_text(json.dumps(cd_all, indent=2))
(out / 'bs_raw.json').write_text(json.dumps(bs_all, indent=2))
print(f'Aggregated phase3a: {len(cd_all)} CD rows, {len(bs_all)} BS rows')
" 2>&1

python -u scripts/proseco/reproduction/analyze_phase3a.py \
  --results_dir "$PHASE3A_RAW" \
  --phase2b_dir "$PHASE2B_RAW" \
  --phase2b_aggr "$PHASE2B_AGGR" \
  --out_dir "$PHASE3A_AGGR" \
  --figures_dir "$FIG_ROOT/phase3a" 2>&1 | tee "out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_phase3a_analysis.log"

echo "=========================================="
echo "Done: $(date)"
echo "Run root: $RUN_ROOT"
echo "Figures:  $FIG_ROOT"
echo "=========================================="
