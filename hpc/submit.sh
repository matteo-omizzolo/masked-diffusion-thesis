#!/bin/bash
# Submit job to Bocconi HPC cluster
# Usage: bash hpc/submit.sh

set -euo pipefail

REMOTE_USER="3316152"
REMOTE_HOST="slogin.hpc.unibocconi.it"
REMOTE_DIR="~/mdm/masked-diffusion-thesis"

echo "=========================================="
echo "Submitting job to Bocconi HPC"
echo "=========================================="
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}"
echo ""

# SSH to cluster, cd to repo, and submit job
JOB_OUTPUT=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" << 'ENDSSH'
cd ~/mdm/masked-diffusion-thesis
echo "Current directory: $(pwd)"
echo ""
echo "Submitting job..."
sbatch hpc/job.sh
ENDSSH
)

echo "$JOB_OUTPUT"
echo ""

# Extract job ID from output (format: "Submitted batch job 12345")
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")

if [ -n "$JOB_ID" ]; then
    echo "=========================================="
    echo "✅ Job submitted successfully!"
    echo "=========================================="
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor job:"
    echo "  squeue -u ${REMOTE_USER}"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "View logs (once running):"
    echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ~/mdm/masked-diffusion-thesis/logs/remdm-${JOB_ID}.out'"
    echo ""
    echo "Cancel job:"
    echo "  scancel ${JOB_ID}"
else
    echo "⚠️  Warning: Could not extract job ID"
    echo "Check output above for errors"
fi
