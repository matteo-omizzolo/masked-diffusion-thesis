#!/bin/bash
#SBATCH --job-name=remdm_t1000p
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

set -euo pipefail

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $SLURM_JOB_GPUS"
echo "Start:     $(date)"
echo "Mode:      3 strategies in parallel (3× A100)"
echo "=========================================="

REPO=~/mdm/masked-diffusion-thesis
cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPUs available: $(python -c 'import torch; print(torch.cuda.device_count())')"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

STEPS=1000
BATCHES=100
SEED=42

for STRATEGY in mdlm remdm-conf remdm-loop; do
    mkdir -p "results/t1000_eval/$STRATEGY"
done

echo ""
echo "Launching 3 strategies in parallel..."

# mdlm → GPU 0
CUDA_VISIBLE_DEVICES=0 python -u -m mdm_playground.cli.run \
    --method remdm \
    --model_id /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --remdm_strategy mdlm \
    --steps $STEPS \
    --remdm_num_batches $BATCHES \
    --out_dir results/t1000_eval/mdlm \
    --seed $SEED \
    > >(while IFS= read -r line; do echo "[mdlm]      $line"; done) \
    2>&1 &
PID_MDLM=$!

# remdm-conf → GPU 1
CUDA_VISIBLE_DEVICES=1 python -u -m mdm_playground.cli.run \
    --method remdm \
    --model_id /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --remdm_strategy remdm-conf \
    --steps $STEPS \
    --remdm_num_batches $BATCHES \
    --out_dir results/t1000_eval/remdm-conf \
    --seed $SEED \
    > >(while IFS= read -r line; do echo "[remdm-conf] $line"; done) \
    2>&1 &
PID_CONF=$!

# remdm-loop → GPU 2
CUDA_VISIBLE_DEVICES=2 python -u -m mdm_playground.cli.run \
    --method remdm \
    --model_id /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --remdm_strategy remdm-loop \
    --steps $STEPS \
    --remdm_num_batches $BATCHES \
    --out_dir results/t1000_eval/remdm-loop \
    --seed $SEED \
    > >(while IFS= read -r line; do echo "[remdm-loop] $line"; done) \
    2>&1 &
PID_LOOP=$!

echo "PIDs: mdlm=$PID_MDLM  remdm-conf=$PID_CONF  remdm-loop=$PID_LOOP"
echo ""

# Wait for all three and collect exit codes
STATUS=0
wait $PID_MDLM      || { echo "ERROR: mdlm failed (exit $?)";      STATUS=1; }
wait $PID_CONF      || { echo "ERROR: remdm-conf failed (exit $?)"; STATUS=1; }
wait $PID_LOOP      || { echo "ERROR: remdm-loop failed (exit $?)"; STATUS=1; }

echo ""
echo "=========================================="
echo "All strategies finished: $(date)"
echo "Exit status: $STATUS"
echo "=========================================="

if [ $STATUS -eq 0 ]; then
    echo ""
    echo "Results:"
    python3 scripts/aggregate_results.py --results_dir results/t1000_eval
fi

exit $STATUS
