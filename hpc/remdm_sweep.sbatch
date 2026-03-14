#!/bin/bash
#SBATCH --job-name=remdm_sweep
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# Step-sweep eval: runs T=256 then T=512, all 3 strategies in parallel (3× A100).
# Purpose: characterise MAUVE inversion between T=128 and T=1000.
# Estimated runtime: ~40min (T=256) + ~80min (T=512) + MAUVE (~40min) ≈ 2.5h

set -euo pipefail

REPO=~/mdm/masked-diffusion-thesis
cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPUs:    $(python -c 'import torch; print(torch.cuda.device_count())')"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

BATCHES=100
SEED=42

run_step() {
    local STEPS=$1
    local OUT_BASE="results/sweep/T${STEPS}"

    echo ""
    echo "=========================================="
    echo "STEP COUNT: T=$STEPS   Start: $(date)"
    echo "=========================================="

    for STRATEGY in mdlm remdm-conf remdm-loop; do
        mkdir -p "$OUT_BASE/$STRATEGY"
    done

    # Launch all 3 strategies in parallel on GPUs 0/1/2
    CUDA_VISIBLE_DEVICES=0 python -u -m mdm_playground.cli.run \
        --method remdm \
        --model_id /home/3316152/mdm/checkpoints/mdlm.ckpt \
        --remdm_strategy mdlm \
        --steps $STEPS \
        --remdm_num_batches $BATCHES \
        --out_dir "$OUT_BASE/mdlm" \
        --seed $SEED \
        > >(while IFS= read -r line; do echo "[T${STEPS}/mdlm]      $line"; done) 2>&1 &
    PID_MDLM=$!

    CUDA_VISIBLE_DEVICES=1 python -u -m mdm_playground.cli.run \
        --method remdm \
        --model_id /home/3316152/mdm/checkpoints/mdlm.ckpt \
        --remdm_strategy remdm-conf \
        --steps $STEPS \
        --remdm_num_batches $BATCHES \
        --out_dir "$OUT_BASE/remdm-conf" \
        --seed $SEED \
        > >(while IFS= read -r line; do echo "[T${STEPS}/remdm-conf] $line"; done) 2>&1 &
    PID_CONF=$!

    CUDA_VISIBLE_DEVICES=2 python -u -m mdm_playground.cli.run \
        --method remdm \
        --model_id /home/3316152/mdm/checkpoints/mdlm.ckpt \
        --remdm_strategy remdm-loop \
        --steps $STEPS \
        --remdm_num_batches $BATCHES \
        --out_dir "$OUT_BASE/remdm-loop" \
        --seed $SEED \
        > >(while IFS= read -r line; do echo "[T${STEPS}/remdm-loop] $line"; done) 2>&1 &
    PID_LOOP=$!

    echo "PIDs: mdlm=$PID_MDLM  remdm-conf=$PID_CONF  remdm-loop=$PID_LOOP"

    STATUS=0
    wait $PID_MDLM      || { echo "ERROR: T=$STEPS mdlm failed (exit $?)";      STATUS=1; }
    wait $PID_CONF      || { echo "ERROR: T=$STEPS remdm-conf failed (exit $?)"; STATUS=1; }
    wait $PID_LOOP      || { echo "ERROR: T=$STEPS remdm-loop failed (exit $?)"; STATUS=1; }

    echo "T=$STEPS done: $(date)  status=$STATUS"

    if [ $STATUS -eq 0 ]; then
        python3 scripts/aggregate_results.py --results_dir "$OUT_BASE"
    fi

    return $STATUS
}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start:  $(date)"
echo "Plan:   T=256 → T=512 (sequential), all strategies parallel"
echo "=========================================="

FINAL_STATUS=0
run_step 256 || FINAL_STATUS=1
run_step 512 || FINAL_STATUS=1

echo ""
echo "=========================================="
echo "Sweep complete: $(date)"
echo "Exit status: $FINAL_STATUS"
echo "=========================================="

exit $FINAL_STATUS
