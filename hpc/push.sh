#!/bin/bash
# Push code to Bocconi HPC cluster
# Usage: bash hpc/push.sh

set -euo pipefail

REMOTE_USER="3316152"
REMOTE_HOST="slogin.hpc.unibocconi.it"
REMOTE_DIR="~/mdm/masked-diffusion-thesis"
REPO_NAME="masked-diffusion-thesis"

echo "=========================================="
echo "Pushing code to Bocconi HPC"
echo "=========================================="
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}"
echo "Target: ${REMOTE_DIR}"
echo ""

# Get the repo root (parent of hpc/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Local repo: $REPO_ROOT"
echo ""

# Create remote directory if it doesn't exist
echo "Creating remote directory..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

echo ""
echo "Syncing files..."
echo "Excluding: .git, .venv, __pycache__, logs, results, checkpoints, *.pt, *.ckpt"
echo ""

# Sync with rsync
rsync -av --delete \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='.vscode' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='wandb' \
    --exclude='logs' \
    --exclude='results' \
    --exclude='checkpoints' \
    --exclude='out' \
    --exclude='err' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    --exclude='*.npz' \
    --exclude='*.safetensors' \
    --exclude='.ipynb_checkpoints' \
    ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
echo "=========================================="
echo "✅ Push complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Submit job: bash hpc/submit.sh"
echo "  2. Monitor: ssh ${REMOTE_USER}@${REMOTE_HOST} 'squeue -u ${REMOTE_USER}'"
echo "  3. Fetch results: bash hpc/pull.sh"
