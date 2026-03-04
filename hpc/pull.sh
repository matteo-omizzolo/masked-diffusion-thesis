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

# Pull logs
echo "Syncing logs..."
rsync -av --ignore-missing-args \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/logs/" \
    ./logs/ \
    || echo "No logs directory on remote (this is OK if job hasn't started yet)"

echo ""

# Pull results
echo "Syncing results..."
rsync -av --ignore-missing-args \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/results/" \
    ./results/ \
    || echo "No results directory on remote (this is OK if job hasn't finished yet)"

echo ""

# Pull outputs (if exists)
echo "Syncing outputs (if any)..."
rsync -av --ignore-missing-args \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/outputs/" \
    ./outputs/ \
    || echo "No outputs directory on remote"

echo ""

# Pull checkpoints (if exists - but exclude large .pt/.ckpt files by default)
echo "Syncing checkpoint metadata (excluding large .pt/.ckpt files)..."
rsync -av --ignore-missing-args \
    --exclude='*.pt' \
    --exclude='*.ckpt' \
    --exclude='*.pth' \
    --exclude='*.npz' \
    --exclude='*.safetensors' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/checkpoints/" \
    ./checkpoints/ \
    || echo "No checkpoints directory on remote"

echo ""
echo "=========================================="
echo "✅ Pull complete!"
echo "=========================================="
echo ""
echo "Check your results:"
echo "  ls -lh logs/"
echo "  ls -lh results/"
echo ""
echo "View latest log:"
echo "  tail -100 logs/remdm-*.out | tail -50"
echo ""
echo "Note: Large checkpoint files (*.pt, *.ckpt) are excluded by default."
echo "To download them manually:"
echo "  scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/checkpoints/model.ckpt ./checkpoints/"
