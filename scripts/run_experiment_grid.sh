#!/bin/bash
################################################################################
# Experiment Grid Runner
#
# Purpose: Submit all 4 production experiments for thesis comparison
#
# Usage:
#   bash scripts/run_experiment_grid.sh
################################################################################

set -euo pipefail

echo "=== Running Production Experiment Grid (256 steps) ==="
echo ""

# Baseline: MDLM
echo "Submitting: MDLM baseline (prod, 256 steps, 10 batches)"
JOB1=$(sbatch slurm/remdm_smoke.sbatch configs/mdlm_hpc_owt_prod.yaml | awk '{print $4}')
echo "  Job ID: $JOB1"

# ReMDM variants
echo "Submitting: ReMDM rescale (prod, 256 steps, 10 batches)"
JOB2=$(sbatch slurm/remdm_smoke.sbatch configs/remdm_hpc_owt_prod_rescale.yaml | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting: ReMDM cap (prod, 256 steps, 10 batches)"
JOB3=$(sbatch slurm/remdm_smoke.sbatch configs/remdm_hpc_owt_prod_cap.yaml | awk '{print $4}')
echo "  Job ID: $JOB3"

echo "Submitting: ReMDM loop (prod, 256 steps, 10 batches)"
JOB4=$(sbatch slurm/remdm_smoke.sbatch configs/remdm_hpc_owt_prod_loop.yaml | awk '{print $4}')
echo "  Job ID: $JOB4"

echo ""
echo "Submitted 4 production experiments: $JOB1, $JOB2, $JOB3, $JOB4"
echo "Expected runtime: ~15-20 min each (60-80 min total)"
echo ""
echo "Monitor with: watch -n 10 'squeue -u \$USER'"
echo ""
echo "After completion, evaluate with:"
echo "  python scripts/evaluate_text.py results/ --output thesis_metrics.csv"

