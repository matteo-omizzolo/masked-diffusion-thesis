#!/bin/bash
################################################################################
# Experiment Grid Runner
#
# Purpose: Submit all experiments for thesis comparison
#
# Quick validation (64 steps, ~1 min each):
#   bash scripts/run_experiment_grid.sh mini
#
# Production experiments (256 steps, ~15-20 min each):
#   bash scripts/run_experiment_grid.sh prod
################################################################################

set -euo pipefail

MODE="${1:-mini}"

if [[ "$MODE" == "mini" ]]; then
    echo "=== Running Mini Experiment Grid (64 steps) ==="
    echo ""
    
    # Baseline: MDLM
    echo "Submitting: MDLM baseline (mini)"
    JOB1=$(sbatch slurm/remdm_smoke.sbatch configs/mdlm_hpc_owt_mini.yaml | awk '{print $4}')
    echo "  Job ID: $JOB1"
    
    # ReMDM variants
    echo "Submitting: ReMDM rescale (mini)"
    JOB2=$(sbatch slurm/remdm_smoke.sbatch configs/remdm_hpc_owt_mini_rescale.yaml | awk '{print $4}')
    echo "  Job ID: $JOB2"
    
    echo "Submitting: ReMDM cap (mini)"
    JOB3=$(sbatch slurm/remdm_smoke.sbatch configs/remdm_hpc_owt_mini_cap.yaml | awk '{print $4}')
    echo "  Job ID: $JOB3"
    
    echo "Submitting: ReMDM loop (mini)"
    JOB4=$(sbatch slurm/remdm_smoke.sbatch configs/remdm_hpc_owt_mini_loop.yaml | awk '{print $4}')
    echo "  Job ID: $JOB4"
    
    echo ""
    echo "Submitted 4 mini experiments: $JOB1, $JOB2, $JOB3, $JOB4"
    echo "Monitor with: squeue -u \$USER"
    
elif [[ "$MODE" == "prod" ]]; then
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
    echo "Expected runtime: ~15-20 min each"
    echo "Monitor with: watch -n 10 'squeue -u \$USER'"
    
else
    echo "Usage: $0 [mini|prod]"
    echo ""
    echo "  mini - Quick validation (64 steps, 2 batches)"
    echo "  prod - Production experiments (256 steps, 10 batches)"
    exit 1
fi

echo ""
echo "Check results later with:"
echo "  ls -lt results/ | head"
echo "  cat results/<timestamp>_*/summary.json | python3 -m json.tool"
