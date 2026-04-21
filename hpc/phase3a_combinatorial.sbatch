#!/bin/bash
#SBATCH --job-name=phase3a_combinatorial
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# Phase 3a — Combinatorial scheduling baselines (CD-G + BS-AG) on ProSeCo-OWT.
#
# Tests whether non-greedy search procedures recover the +0.45 MC-oracle
# headroom over uniform observed in Phase 2b without ground-truth Δ_t.
#
# Design (mirrors Phase 2b sharding)
#   K=30 paired seeds × |B_values|=4 cells, two methods per cell (CD-G + BS-AG).
#   CD-G G-calls per cell ≤ cd_max_attempts+1 (=65); BS-AG G-calls per cell
#   = beam_width · B (=8B). Total G-calls per shard ≈ 8 seeds × (65×4 + 8(2+3+4+8))
#                                                 = 8 × (260 + 136) = 3168.
#
# Resume:
#   --resume causes each shard to skip seeds whose per_seed/{cd,bs}_rows_seed{s}.json
#   already exist on disk.
#
# Downstream:
#   scripts/analyze_phase3a.py merges shard outputs into cd_raw.json / bs_raw.json
#   and computes oracle_gap_closure.json with Δ_CD/Δ_oracle and Δ_BS/Δ_oracle
#   ratios per B (against the Phase 2b uniform paired baseline + MC oracle bound).

set -euo pipefail

echo "=========================================="
echo "Phase 3a ProSeCo-OWT — CD-G + BS-AG baselines"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
CHECKPOINT=~/mdm/checkpoints/proseco_owt
PROTOCOL_A_DIR=$REPO/results/phase1_proseco_owt_full/protocol_a
OUT_DIR=$REPO/results/phase3a_proseco_owt
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

mkdir -p out err "$OUT_DIR" figures/phase3a

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

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --protocol_a_dir "$PROTOCOL_A_DIR"
    --out_dir "$OUT_DIR"
    --T 64
    --K 30
    --B_values 2,3,4,8
    --seed_start 42
    --corrector_steps 1
    --cd_max_attempts 64
    --cd_window 16
    --bs_beam_width 8
    --shard_count 4
    --resume
)

echo "=========================================="
echo "Launching 4 shards in parallel (one per GPU)"
echo "=========================================="

for SHARD in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES=$SHARD
        LOG="out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard${SHARD}.log"
        echo "[shard $SHARD] GPU=$CUDA_VISIBLE_DEVICES log=$LOG"
        python -u scripts/run_phase3a_combinatorial.py \
            "${COMMON_ARGS[@]}" \
            --shard_idx $SHARD \
            > "$LOG" 2>&1
        echo "[shard $SHARD] EXIT=$?"
    ) &
done

wait

echo "=========================================="
echo "All shards done — concatenating shard outputs"
echo "=========================================="

# Concatenate shard-level outputs into flat cd_raw.json / bs_raw.json so
# downstream analyze_phase3a.py has a single file per method to consume.
python -c "
import json
from pathlib import Path
out = Path('$OUT_DIR')
cd_all, bs_all = [], []
for f in sorted(out.glob('cd_raw.shard*-of-*.json')):
    cd_all.extend(json.loads(f.read_text()))
for f in sorted(out.glob('bs_raw.shard*-of-*.json')):
    bs_all.extend(json.loads(f.read_text()))
(out / 'cd_raw.json').write_text(json.dumps(cd_all, indent=2))
(out / 'bs_raw.json').write_text(json.dumps(bs_all, indent=2))
print(f'Aggregated: {len(cd_all)} CD rows, {len(bs_all)} BS rows')
"

echo "=========================================="
echo "Done: $(date)"
echo "Results: $OUT_DIR"
echo "Next:    python scripts/analyze_phase3a.py --results_dir $OUT_DIR \\"
echo "                --phase2b_dir $REPO/results/phase2b_proseco_owt"
echo "=========================================="
