#!/bin/bash
# Pull logs and results from Bocconi HPC cluster
# Usage: bash hpc/pull.sh

set -euo pipefail

REMOTE_USER="3316152"
REMOTE_HOST="slogin.hpc.unibocconi.it"
REMOTE_DIR="~/mdm/masked-diffusion-thesis"

echo "=========================================="
echo "Pulling results from Bocconi HPC"
echo "=========================================="
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}"
echo ""

# Get the repo root (parent of hpc/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Local repo: $REPO_ROOT"
echo ""

# Create local directories if they don't exist
mkdir -p logs
mkdir -p results

# Helper: rsync a remote dir only if it exists (avoids --ignore-missing-args which is GNU-only)
sync_if_exists() {
    local remote_path="$1"
    local local_path="$2"
    shift 2
    # Check if remote dir exists before rsyncing (avoids GNU --ignore-missing-args)
    if ssh "${REMOTE_USER}@${REMOTE_HOST}" "[ -d ${remote_path} ]" 2>/dev/null; then
        rsync -av "$@" "${REMOTE_USER}@${REMOTE_HOST}:${remote_path}/" "${local_path}/"
    else
        echo "  (skipped — ${remote_path} does not exist on remote)"
    fi
}

# Pull logs
echo "Syncing logs..."
sync_if_exists "${REMOTE_DIR}/logs" ./logs
echo ""

# Pull out/ and err/ (SLURM job output files)
echo "Syncing out/ and err/..."
sync_if_exists "${REMOTE_DIR}/out" ./out
sync_if_exists "${REMOTE_DIR}/err" ./err
echo ""

# Pull results
echo "Syncing results..."
sync_if_exists "${REMOTE_DIR}/results" ./results
echo ""

# Pull outputs (if exists)
echo "Syncing outputs (if any)..."
sync_if_exists "${REMOTE_DIR}/outputs" ./outputs
echo ""

# Pull checkpoints (if exists - but exclude large .pt/.ckpt files by default)
echo "Syncing checkpoint metadata (excluding large .pt/.ckpt files)..."
sync_if_exists "${REMOTE_DIR}/checkpoints" ./checkpoints \
    --exclude='*.pt' \
    --exclude='*.ckpt' \
    --exclude='*.pth' \
    --exclude='*.npz' \
    --exclude='*.safetensors'

echo ""
echo "=========================================="
echo "✅ Pull complete!"
echo "=========================================="
echo ""
echo "Check your results:"
echo "  ls -lh out/ err/ results/"
echo ""
echo "View latest SLURM output:"
echo "  tail -100 out/remdm_smoke_*.out"
echo ""
echo "Note: Large checkpoint files (*.pt, *.ckpt) are excluded by default."
echo "To download them manually:"
echo "  scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/checkpoints/model.ckpt ./checkpoints/"
