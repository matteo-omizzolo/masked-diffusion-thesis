#!/bin/bash
#SBATCH --job-name=phase1_interaction_diag
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# Phase 1 — Sparse pairwise interaction diagnostics (Gate 3a).
#
# Measures xi_{t,t'} = G({t,t'}) - Delta_t - Delta_{t'} for a pre-registered
# sparse stratified set of corrector-placement pairs using paired CRN.
#
# Delta_t loaded from Phase 1 Protocol A (no re-evaluation of singletons).
# Only G({t,t'}) is newly evaluated per pair.
#
# Compute estimate: K=30 seeds x ~72 pairs x 1 G-call = 2160 G-calls total.
# 4 shards x ~540 G-calls each, at ~13 s/G-call => ~2h/shard.
# Time limit: 6h with buffer.
#
# NOTE: No pip install in this script. The remdm311 conda environment must
# have all required packages pre-installed. See HPC issue #12 in CLAUDE.md:
# compute nodes have no outbound internet access, so pip install at job start
# fails. Verify with: conda activate remdm311 && python -c "import mdm_playground"
#
# Resume: --resume causes each shard to skip seeds with existing
# per_seed/xi_rows_seed{s}.json.
#
# Gate: Gate 3a (sparse pair diagnostics).
# This gate does NOT validate Theorem B. Gate 3b (schedule-level validation)
# is the required next step before authorising a pairwise scheduler.
#
# Output dir: results/phase1_interaction_diag_${GIT_SHA}
# where GIT_SHA is computed at submission time from the current HEAD.
# This ensures the output is keyed to the exact code version that was submitted.

set -euo pipefail

echo "=========================================="
echo "Phase 1 interaction diagnostics — Gate 3a"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
CHECKPOINT=~/mdm/checkpoints/proseco_owt
PROTOCOL_A_DIR=$REPO/results/phase1_proseco_owt_full/protocol_a

# Compute output dir from current git SHA at submission time.
GIT_SHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo "nogit")
OUT_DIR=$REPO/results/phase1_interaction_diag_${GIT_SHA}

echo "Git SHA:   $GIT_SHA"
echo "Out dir:   $OUT_DIR"

cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:    $(which python)"
echo "PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "GPU count: $(nvidia-smi -L | wc -l)"

# Validate prerequisites before allocating GPU time.
python -c "
from pathlib import Path
ckpt = Path('$CHECKPOINT')
pa = Path('$PROTOCOL_A_DIR')
assert ckpt.exists() and (ckpt / 'config.json').exists(), f'Checkpoint missing: {ckpt}'
traj = sorted(pa.glob('trajectory_*.json'))
assert len(traj) >= 30, f'Protocol A has only {len(traj)} trajectories; need >= 30'
print(f'Checkpoint OK: {ckpt}')
print(f'Protocol A OK: {len(traj)} trajectories in {pa}')
" 2>&1

# Soft check for analysis dependencies (runner does not need matplotlib).
python -c "
try:
    import scipy, matplotlib
    print(f'scipy: {scipy.__version__}  matplotlib: {matplotlib.__version__}  (OK for analyzer)')
except ImportError as e:
    print(f'WARNING: {e} — analysis plots will be skipped; runner is not affected')
" 2>&1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p out err "$OUT_DIR/per_seed" "figures/phase1_interaction_diag"

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --protocol_a_dir "$PROTOCOL_A_DIR"
    --out_dir "$OUT_DIR"
    --T 64
    --K 30
    --seed_start 42
    --corrector_steps 1
    --pair_rng_seed 42
    --shard_count 4
    --resume
)

echo "=========================================="
echo "Pair list preview (printed by shard 0 only)"
echo "=========================================="

# Preview pair count (validation — not optional).
python -c "
import sys
sys.path.insert(0, '$REPO/src')
sys.path.insert(0, '$REPO/scripts')
from run_phase1_interaction_diagnostics import make_pair_list
pairs = make_pair_list(T=64, rng_seed=42)
phase_counts = {}
for p in pairs:
    k = f\"{p['phase_t']}-{p['phase_tp']}\"
    phase_counts[k] = phase_counts.get(k, 0) + 1
print(f'Total pairs: {len(pairs)}')
for k, v in sorted(phase_counts.items()):
    print(f'  {k}: {v}')
d_short = sum(1 for p in pairs if p['distance'] <= 5)
d_long  = sum(1 for p in pairs if p['distance'] >= 40)
print(f'  short (d<=5): {d_short}  long (d>=40): {d_long}')
" 2>&1

echo "=========================================="
echo "Launching 4 shards in parallel (one per GPU)"
echo "=========================================="

declare -a SHARD_PIDS=()
for SHARD in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES=$SHARD
        LOG="out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard${SHARD}.log"
        echo "[shard $SHARD] GPU=$CUDA_VISIBLE_DEVICES log=$LOG"
        python -u scripts/run_phase1_interaction_diagnostics.py \
            "${COMMON_ARGS[@]}" \
            --shard_idx $SHARD \
            > "$LOG" 2>&1
        EXIT_CODE=$?
        echo "[shard $SHARD] EXIT=${EXIT_CODE}"
        exit ${EXIT_CODE}
    ) &
    SHARD_PIDS+=($!)
done

# Wait for all shards and collect exit codes.
FAILED_SHARDS=0
for i in "${!SHARD_PIDS[@]}"; do
    PID=${SHARD_PIDS[$i]}
    if ! wait "$PID"; then
        echo "ERROR: shard $i (PID $PID) failed"
        FAILED_SHARDS=$((FAILED_SHARDS + 1))
    fi
done

if [ "$FAILED_SHARDS" -gt 0 ]; then
    echo "ERROR: $FAILED_SHARDS shard(s) failed. Aborting merge step."
    exit 1
fi

echo "=========================================="
echo "All shards done — concatenating shard outputs"
echo "=========================================="

python -c "
import json, time
from pathlib import Path
import numpy as np

out = Path('$OUT_DIR')

# Assert exactly 4 shard files exist.
shard_files = sorted(out.glob('xi_raw.shard*-of-4.json'))
assert len(shard_files) == 4, f'Expected 4 shard files, got {len(shard_files)}: {shard_files}'

# Merge and validate row counts.
all_rows = []
for f in shard_files:
    shard_rows = json.loads(f.read_text())
    all_rows.extend(shard_rows)

pair_list = json.loads((out / 'pair_list.json').read_text())
n_pairs = len(pair_list)
seeds = sorted(set(r['seed'] for r in all_rows))
n_seeds = len(seeds)
expected_rows = n_seeds * n_pairs
if len(all_rows) != expected_rows:
    print(f'WARNING: expected {expected_rows} rows ({n_seeds} seeds x {n_pairs} pairs), '
          f'got {len(all_rows)}. Some seeds may be missing.')

(out / 'xi_raw.json').write_text(json.dumps(all_rows, indent=2))

# Write merged manifest.
shard_manifests = sorted(out.glob('manifest.shard*.json'))
merged_manifest = {
    'gate': 'Gate3a_sparse_pair_diagnostics',
    'merged_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'n_shards': len(shard_manifests),
    'n_rows': len(all_rows),
    'n_seeds': n_seeds,
    'n_pairs': n_pairs,
    'shards': [json.loads(f.read_text()) for f in shard_manifests],
}
(out / 'manifest.json').write_text(json.dumps(merged_manifest, indent=2))

xi_vals = np.array([r['xi'] for r in all_rows])
print(f'Merged: {len(all_rows)} rows | {n_seeds} seeds | {n_pairs} pairs')
print(f'xi: mean={float(np.mean(xi_vals)):.4f} std={float(np.std(xi_vals)):.4f} '
      f'P(>0)={float(np.mean(xi_vals > 0)):.3f}')
print(f'Wrote xi_raw.json and manifest.json to {out}')
" 2>&1

echo "=========================================="
echo "Done: $(date)"
echo "Results: $OUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Pull results from HPC:"
echo "     rsync -avz 3316152@slogin.hpc.unibocconi.it:~/mdm/masked-diffusion-thesis/$OUT_DIR/ ./$OUT_DIR/"
echo "  2. Run analyzer locally:"
echo "     python scripts/analyze_phase1_interactions.py \\"
echo "         --out_dir $OUT_DIR \\"
echo "         --figures_dir figures/phase1_interaction_diag \\"
echo "         --phase3a_dir results/phase3a_proseco_owt"
echo "  3. Inspect figures/ and interpretation.md."
echo "  4. Gate 3b decision: proceed only if Gate 3a shows structured non-zero xi."
echo "=========================================="
