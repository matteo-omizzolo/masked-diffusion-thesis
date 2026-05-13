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

# Record commit SHA so HPC sbatches can stamp output folders with provenance.
# .git/ is excluded from rsync (saves bandwidth and avoids submodule issues),
# so this fallback file is the only SHA source on the cluster. If the worktree
# has uncommitted tracked changes, append "-dirty" so output folder names make
# it obvious that the run used unpushed code.
if git rev-parse --short HEAD >/dev/null 2>&1; then
    SHA="$(git rev-parse --short HEAD)"
    if ! git diff --quiet HEAD -- 2>/dev/null; then
        SHA="${SHA}-dirty"
    fi
    printf '%s\n' "$SHA" > .hpc_git_sha
    echo "Wrote .hpc_git_sha: $SHA"
else
    echo "warning: not a git repo or no HEAD; .hpc_git_sha not written" >&2
fi

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
echo "  1. ProSeCo-OWT is closed as a case study; no K=30 replication is currently open."
echo "     Before any sbatch, run a Codex pre-submit review (see hpc/README.md)."
echo "     Current backend-validation entry point: hpc/backend_validation/informed_correctors/stage0_env_smoke.sbatch"
echo "  2. Monitor: ssh ${REMOTE_USER}@${REMOTE_HOST} 'squeue -u ${REMOTE_USER}'"
echo "  3. Fetch results: bash hpc/pull.sh"
