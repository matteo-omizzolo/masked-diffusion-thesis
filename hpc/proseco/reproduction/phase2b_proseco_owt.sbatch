#!/bin/bash
#SBATCH --job-name=phase2b_proseco_owt
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# Phase 2b ProSeCo-OWT: Paired K-seed policy comparison + MC oracle
#
# Consumes Phase 1 protocol_a trajectories (seeds 42..91 available from the
# full run; we use K=30, so seeds 42..71) to avoid re-running the expensive
# Protocol A per-trajectory Δ_t and signal extraction.
#
# Design
#   K=30 paired seeds × 10 policies × |B_values|=5 = 1500 schedule evals
#   + K=30 × |mc_B_values|=3 × mc_P=100 = 9000 MC schedule evals
#   → 10,500 schedule evaluations total, ~2625 per 4-GPU shard.
#
# Splits K=30 seeds across 4 A100s via --shard_idx {0,1,2,3} --shard_count 4.
# Each shard pins CUDA_VISIBLE_DEVICES to one GPU and runs as a background
# process. `wait` blocks until all four complete before aggregation.
#
# Resume:
#   If any shard fails mid-run, re-submit with the same args. --resume causes
#   each shard to skip seeds whose per_seed/{policy,mc}_rows_seed{s}.json
#   already exist on disk.
#
# Downstream:
#   scripts/proseco/reproduction/analyze_phase2b.py aggregates shard-level policy_raw.shard*.json
#   and mc_raw.shard*.json into policy_comparison_paired.json, mc_oracle.json,
#   and rank_A_vs_G_phase2b.json per ANALYSIS_SPEC §3b.

set -euo pipefail

echo "=========================================="
echo "Phase 2b ProSeCo-OWT — Paired K-seed + MC oracle"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
CHECKPOINT=~/mdm/checkpoints/proseco_owt
PROTOCOL_A_DIR=$REPO/results/phase1_proseco_owt_full/protocol_a
# K=30 critical replication — write to a new folder (git cf89e00) so the
# existing canonical results/phase2b_proseco_owt/ are never at risk of
# being overwritten.  Compare this folder against canonical after the run.
OUT_DIR=$REPO/results/phase2b_k30_rep_cf89e00
cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "GPU count: $(nvidia-smi -L | wc -l)"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p out err "$OUT_DIR" "$OUT_DIR/per_seed" figures/phase2b_proseco_owt

pip install -e . -q --break-system-packages
pip install scipy matplotlib safetensors einops -q --break-system-packages

# Validate prerequisites
python -c "
from pathlib import Path
ckpt = Path('$CHECKPOINT')
pa = Path('$PROTOCOL_A_DIR')
assert ckpt.exists() and (ckpt / 'config.json').exists(), f'Checkpoint missing: {ckpt}'
traj = sorted(pa.glob('trajectory_*.json'))
assert len(traj) >= 30, f'Phase 1 protocol_a has only {len(traj)} trajectories; need >= 30'
print(f'Checkpoint OK: {ckpt}')
print(f'Protocol A OK: {len(traj)} trajectories in {pa}')
" 2>&1

# Common CLI args for all shards
COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --protocol_a_dir "$PROTOCOL_A_DIR"
    --out_dir "$OUT_DIR"
    --T 64
    --K 30
    --B_values 2,3,4,8,16
    --mc_B_values 2,3,4
    --mc_P 100
    --seed_start 42
    --corrector_steps 1
    --shard_count 4
    # --resume intentionally omitted: this is a fresh replication run.
    # Do not add --resume unless re-submitting a partially-failed job.
)

echo "=========================================="
echo "Launching 4 shards in parallel (one per GPU)"
echo "=========================================="

# Launch one shard per GPU as a background process.
# CUDA_VISIBLE_DEVICES pins each Python process to a single A100.
# PIDs are captured so we can propagate individual shard failures.
SHARD_PIDS=()
for SHARD in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES=$SHARD
        LOG="out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard${SHARD}.log"
        echo "[shard $SHARD] GPU=$CUDA_VISIBLE_DEVICES log=$LOG"
        python -u scripts/proseco/reproduction/run_phase2b_proseco_owt.py \
            "${COMMON_ARGS[@]}" \
            --shard_idx $SHARD \
            > "$LOG" 2>&1
        EXIT_CODE=$?
        echo "[shard $SHARD] EXIT=$EXIT_CODE"
        exit $EXIT_CODE
    ) &
    SHARD_PIDS+=($!)
done

# Block until all four shards finish; propagate any failure.
SHARD_FAILED=0
for i in "${!SHARD_PIDS[@]}"; do
    wait "${SHARD_PIDS[$i]}" || { echo "[shard $i] FAILED (PID ${SHARD_PIDS[$i]})"; SHARD_FAILED=1; }
done
if [ "$SHARD_FAILED" -ne 0 ]; then
    echo "One or more shards failed — aborting aggregation. Check shard logs in out/." >&2
    exit 1
fi

echo "=========================================="
echo "All shards done — running aggregation"
echo "=========================================="

# Concatenate shard-level outputs into flat policy_raw.json / mc_raw.json
# so downstream analyze_phase2b.py has a single file to consume.
python -c "
import json
from pathlib import Path
out = Path('$OUT_DIR')
policy_all, mc_all = [], []
for f in sorted(out.glob('policy_raw.shard*-of-*.json')):
    policy_all.extend(json.loads(f.read_text()))
for f in sorted(out.glob('mc_raw.shard*-of-*.json')):
    mc_all.extend(json.loads(f.read_text()))
(out / 'policy_raw.json').write_text(json.dumps(policy_all, indent=2))
(out / 'mc_raw.json').write_text(json.dumps(mc_all, indent=2))
print(f'Aggregated: {len(policy_all)} policy rows, {len(mc_all)} MC rows')
"

echo "=========================================="
echo "Done: $(date)"
echo "Results (replication): $OUT_DIR"
echo "Canonical:             $REPO/results/phase2b_proseco_owt"
echo "Next:    python scripts/proseco/reproduction/analyze_phase2b.py --results_dir $OUT_DIR"
echo "Compare: diff replication vs canonical per_seed/ to verify K=30 gate."
echo "=========================================="
