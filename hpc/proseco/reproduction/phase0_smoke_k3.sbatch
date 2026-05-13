#!/bin/bash
#SBATCH --job-name=phase0_smoke_k3
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Phase 0 K=3 smoke — minimal sanity run before K=30 replication.
# K=3, T=64, B_values=2,4, mc_B_values=2,4, mc_P=8 (lightweight MC).
# Gate: 11 passed (PROSECO_OWT_CHECKPOINT set) confirmed on HPC 2026-05-06.
# Do NOT use output for thesis conclusions; gate only. K=30 full run next.
# PROVENANCE RECORD: SHA d4edc92 is hardcoded intentionally — this script
# documents the specific smoke run (job 489457, gnode01, 2026-05-06).
# For a fresh run at a new commit, copy and update SHA, ROOT, and timestamp.
# nvidia-smi and cp alias commands use || true intentionally (non-critical).

set -euo pipefail

REPO="$HOME/mdm/masked-diffusion-thesis"
CKPT="$HOME/mdm/checkpoints/proseco_owt"
SHA="d4edc92"
ROOT="$REPO/results/phase0_smoke_${SHA}"
PROTO_A="$REPO/results/phase1_proseco_owt_full/protocol_a"

cd "$REPO"
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

export PROSECO_OWT_CHECKPOINT="$CKPT"
export CUDA_VISIBLE_DEVICES=0

echo "=== Phase 0 K=3 Smoke ==="
echo "  SHA:      $SHA"
echo "  ROOT:     $ROOT"
echo "  CKPT:     $CKPT"
echo "  PROTO_A:  $PROTO_A"
echo "  hostname: $(hostname)"
echo "  date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi unavailable"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count())"
echo ""

mkdir -p "$ROOT"

# --- Preflight status record ---
cat > "$ROOT/preflight_status.txt" <<EOF
preflight_command: PROSECO_OWT_CHECKPOINT=$CKPT pytest tests/proseco/reproduction/test_phase0_preflight.py -q
preflight_result:  11 passed, 0 skipped, 0 failed
preflight_date:    2026-05-06
preflight_node:    slnode01 (login)
preflight_device:  cpu (T=4)
backend:           ProSeCo-OWT real (model.safetensors + GPT-2 scorer)
EOF

# --- Smoke config JSON ---
python3 -c "
import json, os, socket, datetime
cfg = {
    'git_sha': '$SHA',
    'branch': 'theory-first-corrector-timing',
    'K': 3, 'T': 64,
    'B_values': [2, 4], 'mc_B_values': [2, 4], 'mc_P': 8,
    'seed_start': 42,
    'protocol_a_dir': '$PROTO_A',
    'checkpoint': '$CKPT',
    'smoke_root': '$ROOT',
    'hostname': socket.gethostname(),
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'preflight_status': '11 passed, 0 skipped on HPC slnode01 cpu 2026-05-06',
}
with open('$ROOT/smoke_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('smoke_config.json written.')
"

# --- Phase 2b: policy comparison + MC oracle ---
echo "--- Phase 2b (K=3, T=64, B_values=2,4, mc_P=8) ---"
PHASE2B_CMD="python scripts/proseco/reproduction/run_phase2b_proseco_owt.py \
  --checkpoint $CKPT \
  --protocol_a_dir $PROTO_A \
  --out_dir $ROOT/phase2b_raw \
  --K 3 --T 64 --B_values 2,4 --mc_B_values 2,4 --mc_P 8 \
  --seed_start 42 --shard_idx 0 --shard_count 1"
echo "CMD: $PHASE2B_CMD"
eval "$PHASE2B_CMD"

cp "$ROOT/phase2b_raw/policy_raw.shard0-of-1.json" \
   "$ROOT/phase2b_raw/policy_raw.json" 2>/dev/null || true
cp "$ROOT/phase2b_raw/mc_raw.shard0-of-1.json" \
   "$ROOT/phase2b_raw/mc_raw.json" 2>/dev/null || true

# --- Phase 3a: combinatorial (CD-G + BS-AG) ---
echo "--- Phase 3a (K=3, T=64, B_values=2,4) ---"
PHASE3A_CMD="python scripts/proseco/reproduction/run_phase3a_combinatorial.py \
  --checkpoint $CKPT \
  --protocol_a_dir $PROTO_A \
  --out_dir $ROOT/phase3a_raw \
  --K 3 --T 64 --B_values 2,4 \
  --seed_start 42 --shard_idx 0 --shard_count 1"
echo "CMD: $PHASE3A_CMD"
eval "$PHASE3A_CMD"

cp "$ROOT/phase3a_raw/cd_raw.shard0-of-1.json" \
   "$ROOT/phase3a_raw/cd_raw.json" 2>/dev/null || true
cp "$ROOT/phase3a_raw/bs_raw.shard0-of-1.json" \
   "$ROOT/phase3a_raw/bs_raw.json" 2>/dev/null || true

# --- Aggregate analysis ---
echo "--- Analyze phase2b ---"
python scripts/proseco/reproduction/analyze_phase2b.py \
  --results_dir "$ROOT/phase2b_raw" \
  --out_dir "$ROOT/phase2b" \
  --figures_dir "$ROOT/figures/phase2b"

echo "--- Analyze phase3a ---"
python scripts/proseco/reproduction/analyze_phase3a.py \
  --results_dir "$ROOT/phase3a_raw" \
  --phase2b_dir "$ROOT/phase2b_raw" \
  --phase2b_aggr "$ROOT/phase2b" \
  --out_dir "$ROOT/phase3a" \
  --figures_dir "$ROOT/figures/phase3a"

# --- Command log ---
{
  echo "# Phase 0 K=3 Smoke Commands"
  echo "# SHA: $SHA  branch: theory-first-corrector-timing"
  echo "# Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  Node: $(hostname)"
  echo ""
  echo "$PHASE2B_CMD"
  echo ""
  echo "$PHASE3A_CMD"
} > "$ROOT/smoke_commands.log"

echo ""
echo "=== Smoke complete. Output: $ROOT ==="
find "$ROOT" -maxdepth 3 -type f | sort
