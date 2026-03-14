#!/bin/bash
# Submit job to Bocconi HPC cluster
# Usage: bash hpc/submit.sh [smoke|eval|eval-loop|t1000|t1000-loop]   (default: smoke)
#
# Targets:
#   smoke      — 2-batch smoke test (remdm-conf, no MAUVE)
#   eval       — full eval array: tasks 0-1 (mdlm + remdm-conf, 100 batches, 128 steps, MAUVE)
#   eval-loop  — single task 2 (remdm-loop); submit after eval jobs complete
#   t1000      — T=1000 eval array: tasks 0-1 (mdlm + remdm-conf, paper-comparable gen_ppl)
#   t1000-loop — single task 2 (remdm-loop) at T=1000; submit after t1000 completes

set -euo pipefail

TARGET="${1:-smoke}"
REMOTE_USER="3316152"
REMOTE_HOST="slogin.hpc.unibocconi.it"

EXTRA_SBATCH_OPTS=""
case "$TARGET" in
  smoke)     SBATCH_FILE="hpc/remdm_smoke.sbatch" ;;
  eval)       SBATCH_FILE="hpc/remdm_full_eval.sbatch" ;;
  eval-loop)  SBATCH_FILE="hpc/remdm_full_eval.sbatch"
              EXTRA_SBATCH_OPTS="--array=2-2" ;;  # NOTE: sbatch opts must come before script
  t1000)      SBATCH_FILE="hpc/remdm_t1000_eval.sbatch" ;;
  t1000-loop) SBATCH_FILE="hpc/remdm_t1000_eval.sbatch"
              EXTRA_SBATCH_OPTS="--array=2-2" ;;
  t1000p)     SBATCH_FILE="hpc/remdm_t1000_parallel.sbatch" ;;  # all 3 strategies on 3 GPUs, 1 job
  sweep)      SBATCH_FILE="hpc/remdm_sweep.sbatch" ;;          # T=256 + T=512 step sweep
  *)
    echo "ERROR: unknown target '$TARGET'. Use 'smoke', 'eval', 'eval-loop', 't1000', 't1000-loop', 't1000p', or 'sweep'."
    exit 1
    ;;
esac

echo "=========================================="
echo "Submitting job to Bocconi HPC"
echo "=========================================="
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}"
echo "Target: ${TARGET} (${SBATCH_FILE})"
echo ""

# SSH to cluster, cd to repo, and submit job
JOB_OUTPUT=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "
cd ~/mdm/masked-diffusion-thesis
echo 'Current directory: \$(pwd)'
echo ''
echo 'Submitting job...'
sbatch ${EXTRA_SBATCH_OPTS} ${SBATCH_FILE}
")

echo "$JOB_OUTPUT"
echo ""

# Extract job ID from output (format: "Submitted batch job 12345")
JOB_ID=$(echo "$JOB_OUTPUT" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*$' || echo "")

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
    if [ "$TARGET" = "eval" ]; then
        echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ~/mdm/masked-diffusion-thesis/out/remdm_eval_${JOB_ID}_0.out'"
        echo "  (array tasks: ${JOB_ID}_0 = mdlm, ${JOB_ID}_1 = remdm-conf)"
        echo "  When done: bash hpc/submit.sh eval-loop   (submits remdm-loop as task 2)"
    elif [ "$TARGET" = "eval-loop" ]; then
        echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ~/mdm/masked-diffusion-thesis/out/remdm_eval_${JOB_ID}_2.out'"
    elif [ "$TARGET" = "t1000" ]; then
        echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ~/mdm/masked-diffusion-thesis/out/remdm_t1000_${JOB_ID}_0.out'"
        echo "  (array tasks: ${JOB_ID}_0 = mdlm, ${JOB_ID}_1 = remdm-conf)"
        echo "  When done: bash hpc/submit.sh t1000-loop   (submits remdm-loop as task 2)"
    elif [ "$TARGET" = "t1000-loop" ]; then
        echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ~/mdm/masked-diffusion-thesis/out/remdm_t1000_${JOB_ID}_2.out'"
    else
        echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ~/mdm/masked-diffusion-thesis/out/remdm_smoke_${JOB_ID}.out'"
    fi
    echo ""
    echo "Cancel job:"
    echo "  scancel ${JOB_ID}"
else
    echo "⚠️  Warning: Could not extract job ID"
    echo "Check output above for errors"
fi
