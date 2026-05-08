#!/bin/bash
#SBATCH --job-name=xb_llada_p2b
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:3g.40gb:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=23:50:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# LEGACY / PROVENANCE — DO NOT SUBMIT FOR CURRENT STATUS.
# Closed LLaDA-SFT resume job. Preserved only to explain historical outputs.
#
# Phase 2b ONLY, LLaDA-SFT backbone, 4 parallel shards inside one job.
#
# QOS constraints (sacctmgr show qos stud):
#   MaxJobsPerUser=1  → cannot run 4 sibling jobs in parallel
#   MaxWall=24h       → fits 1 seed / shard / 12-15h comfortably
#   MaxTRES=gpu/4     → max 4 GPU slices per job
#
# Design:
#   - Shard_count = 8 (1 seed per shard, per-seed time ~11-15h on this backbone)
#   - This sbatch runs shards SHARD_START..SHARD_START+3 in parallel on 4 MIG slices
#   - Submit twice: once with SHARD_START=0 (seeds 142-145), once with SHARD_START=4 (seeds 146-149)
#   - Second submission waits for first to finish (MaxJobsPerUser=1)
#
# Scope:
#   - DOES NOT run preflight (Protocol A already validated the backbone).
#   - DOES NOT run Protocol A (complete on disk).
#   - DOES NOT run Phase 3a (blocked on POST_CROSS_BACKBONE_DECISION.md).
#   - DOES NOT run analyze_phase2b (runs separately after all 8 shards complete).

set -euo pipefail

: "${SHARD_START:?SHARD_START must be set (0 or 4)}"
SHARD_COUNT=8
N_PARALLEL=4

case "$SHARD_START" in
  0|4) ;;
  *) echo "ERROR: SHARD_START must be 0 or 4, got $SHARD_START" >&2; exit 1 ;;
esac

echo "=========================================="
echo "Phase 2b (LLaDA-SFT) — shards ${SHARD_START}..$((SHARD_START+3)) of ${SHARD_COUNT}"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
CHECKPOINT=~/mdm/checkpoints/proseco_llada_sft
RUN_ROOT=$REPO/results/cross_backbone/proseco_llada_sft_bounded
PROTO_A_DIR=$RUN_ROOT/protocol_a
PHASE2B_RAW=$RUN_ROOT/phase2b_raw

cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p out err

# ----------------------------------------------------------------------
# SAFETY: Refuse to run if Protocol A is not complete.
# ----------------------------------------------------------------------
python -c "
from pathlib import Path
p = Path('$PROTO_A_DIR')
need = [p / f'trajectory_{i}.json' for i in range(8)] + [p / 'run_config.json']
missing = [str(x) for x in need if not x.exists()]
assert not missing, f'Protocol A incomplete: missing {missing}'
print(f'Protocol A OK: {len(need)} files present in {p}')
"

# ----------------------------------------------------------------------
# Launch N_PARALLEL shards in parallel, one per MIG slice.
# ----------------------------------------------------------------------
declare -a PIDS=()
for OFFSET in $(seq 0 $((N_PARALLEL - 1))); do
  SHARD_IDX=$((SHARD_START + OFFSET))
  LOG="out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard${SHARD_IDX}of${SHARD_COUNT}.log"
  echo "Launching shard ${SHARD_IDX}/${SHARD_COUNT} on CUDA_VISIBLE_DEVICES=${OFFSET} → ${LOG}"

  (
    export CUDA_VISIBLE_DEVICES=$OFFSET
    python -u scripts/run_phase2b_proseco_owt.py \
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
      --shard_idx "$SHARD_IDX" \
      --shard_count "$SHARD_COUNT" \
      --resume
  ) > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo "Launched ${#PIDS[@]} shards, pids: ${PIDS[*]}"
echo "Waiting for all shards to finish..."

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    echo "Shard pid=$PID FAILED"
    FAIL=$((FAIL + 1))
  fi
done

echo "=========================================="
echo "All shards for SHARD_START=${SHARD_START} finished: $(date)"
echo "Failures: ${FAIL} / ${N_PARALLEL}"
echo "Per-seed outputs: $PHASE2B_RAW/per_seed/"
ls -la "$PHASE2B_RAW/per_seed/" || true
echo "=========================================="

exit $FAIL
