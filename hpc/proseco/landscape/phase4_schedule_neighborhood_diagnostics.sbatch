#!/bin/bash
#SBATCH --job-name=phase4_schedule_neighborhood
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# Phase 4 — targeted schedule-neighborhood diagnostics on ProSeCo-OWT.
#
# This job is a K=10 pilot over B in {2,3,4}. It samples one-swap
# neighborhoods around CD-G, BS-AG, top-MC, random-MC, uniform, and middle
# anchors; it also evaluates exact diminishing-return triples around the
# eligible anchor contexts. It does not implement a new scheduler.
#
# The output directory is SHA-tagged:
#   results/phase4_schedule_neighborhood_<gitsha>/
#
# Resume behavior:
#   --resume skips completed per_seed/diagnostics_rows_seed*.json files.

set -euo pipefail

echo "=========================================="
echo "Phase 4 ProSeCo-OWT — schedule-neighborhood diagnostics"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
CHECKPOINT=~/mdm/checkpoints/proseco_owt
PROTOCOL_A_DIR=$REPO/results/phase1_proseco_owt_full/protocol_a
PHASE2B_DIR=$REPO/results/phase2b_proseco_owt
PHASE3A_DIR=$REPO/results/phase3a_proseco_owt
cd "$REPO"

GIT_SHA="${PHASE4_GIT_SHA:-}"
if [ -z "$GIT_SHA" ]; then
    GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || true)
fi
if [ -z "$GIT_SHA" ] && [ -f .hpc_git_sha ]; then
    GIT_SHA=$(cat .hpc_git_sha)
fi
if [ -z "$GIT_SHA" ]; then
    echo "ERROR: no git SHA available. Submit with --export=ALL,PHASE4_GIT_SHA=<sha> or create .hpc_git_sha." >&2
    exit 1
fi
OUT_DIR=$REPO/results/phase4_schedule_neighborhood_${GIT_SHA}

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:     $(which python)"
echo "PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:       $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "GPU count:  $(nvidia-smi -L | wc -l)"
echo "Git SHA:    $GIT_SHA"
echo "Out dir:    $OUT_DIR"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p out err "$OUT_DIR" "$OUT_DIR/logs"

# Avoid compute-node package installation. The environment is pre-provisioned;
# expose the rsync'd source tree directly without build isolation or internet.
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

python -c "
import mdm_playground
from pathlib import Path
ckpt = Path('$CHECKPOINT')
pa = Path('$PROTOCOL_A_DIR')
p2 = Path('$PHASE2B_DIR')
p3 = Path('$PHASE3A_DIR')
assert ckpt.exists() and (ckpt / 'config.json').exists(), f'Checkpoint missing: {ckpt}'
assert len(list(pa.glob('trajectory_*.json'))) >= 10, f'Need >=10 Protocol A trajectories in {pa}'
assert (p2 / 'mc_raw.json').exists(), f'Missing Phase 2b MC rows: {p2}'
assert (p2 / 'policy_raw.json').exists(), f'Missing Phase 2b policy rows: {p2}'
assert (p3 / 'cd_raw.json').exists(), f'Missing Phase 3a CD rows: {p3}'
assert (p3 / 'bs_raw.json').exists(), f'Missing Phase 3a BS rows: {p3}'
print('Prerequisites OK; mdm_playground import OK')
"

python -m py_compile scripts/proseco/landscape/run_schedule_neighborhood_diagnostics.py
python -m py_compile scripts/proseco/landscape/analyze_schedule_neighborhood_diagnostics.py

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --protocol_a_dir "$PROTOCOL_A_DIR"
    --phase2b_dir "$PHASE2B_DIR"
    --phase3a_dir "$PHASE3A_DIR"
    --out_dir "$OUT_DIR"
    --T 64
    --K 10
    --B_values 2,3,4
    --seed_start 42
    --corrector_steps 1
    --neighborhood_max_neighbors 32
    --dr_triples_per_context 3
    --dr_anchor_types cd_g,bs_ag,top_mc,random_mc,middle
    --shard_count 4
    --resume
)

echo "=========================================="
echo "Launching 4 shards in parallel (one per allocated GPU)"
echo "=========================================="

for SHARD in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES=$SHARD
        LOG="$OUT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard${SHARD}.log"
        echo "[shard $SHARD] GPU=$CUDA_VISIBLE_DEVICES log=$LOG"
        python -u scripts/proseco/landscape/run_schedule_neighborhood_diagnostics.py \
            "${COMMON_ARGS[@]}" \
            --shard_idx "$SHARD" \
            > "$LOG" 2>&1
        echo "[shard $SHARD] EXIT=$?"
    ) &
done

wait

echo "=========================================="
echo "All shards done — merging outputs"
echo "=========================================="

python -c "
import json
from pathlib import Path
out = Path('$OUT_DIR')
neighborhood, triples = [], []
for f in sorted(out.glob('neighborhood_raw.shard*-of-*.json')):
    neighborhood.extend(json.loads(f.read_text()))
for f in sorted(out.glob('triples_raw.shard*-of-*.json')):
    triples.extend(json.loads(f.read_text()))
(out / 'neighborhood_raw.json').write_text(json.dumps(neighborhood, indent=2, sort_keys=True))
(out / 'triples_raw.json').write_text(json.dumps(triples, indent=2, sort_keys=True))
configs = [json.loads(f.read_text()) for f in sorted(out.glob('config.shard*-of-*.json'))]
manifests = [json.loads(f.read_text()) for f in sorted(out.glob('manifest.shard*-of-*.json'))]
if configs:
    cfg = configs[0]
    cfg['merged_shards'] = len(configs)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2, sort_keys=True))
if manifests:
    manifest = {
        'phase': 'phase4_schedule_neighborhood_diagnostics',
        'git_sha': '$GIT_SHA',
        'hpc_job_id': '$SLURM_JOB_ID',
        'node_list': '$SLURM_NODELIST',
        'merged_shards': len(manifests),
        'n_neighborhood_rows': len(neighborhood),
        'n_triple_rows': len(triples),
        'shards': manifests,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
logs = []
for f in sorted(out.glob('command_log.shard*-of-*.txt')):
    logs.append(f'===== {f.name} =====\\n' + f.read_text())
(out / 'command_log.txt').write_text('\\n'.join(logs))
print(f'Merged {len(neighborhood)} neighborhood rows and {len(triples)} triple rows into {out}')
"

echo "=========================================="
echo "Done:    $(date)"
echo "Results: $OUT_DIR"
echo "Next:    fetch results and run scripts/proseco/landscape/analyze_schedule_neighborhood_diagnostics.py locally"
echo "=========================================="
