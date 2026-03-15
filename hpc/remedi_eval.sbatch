#!/bin/bash
#SBATCH --job-name=remedi_eval
#SBATCH --account=3316152
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3316152@studbocconi.it

# RemeDi-RL evaluation: runs T=32 (paper default), T=128, T=1000 sequentially.
# Uses scripts/remedi_eval.py — standalone, does NOT go through the ReMDM pipeline.

set -euo pipefail

REPO=~/mdm/masked-diffusion-thesis
cd "$REPO"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate remdm311

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python:   $(which python)"
echo "Start:    $(date)"
echo "=========================================="

OWT_REF="/home/3316152/mdm/data/owt_reference_1000.json"
NUM_SAMPLES=100
SEED=42
CKPT="maple-research-lab/RemeDi-RL"

# Verify OWT reference exists
if [ ! -f "$OWT_REF" ]; then
    echo "ERROR: OWT reference not found at $OWT_REF"
    echo "Pre-stage with: python3 scripts/stage_owt_reference.py"
    exit 1
fi

run_remedi() {
    local STEPS=$1
    local OUT_DIR="results/remedi/T${STEPS}"
    echo ""
    echo "--- RemeDi T=$STEPS  Start: $(date) ---"
    mkdir -p "$OUT_DIR"
    python -u scripts/remedi_eval.py \
        --ckpt_path "$CKPT" \
        --steps "$STEPS" \
        --num_samples $NUM_SAMPLES \
        --out_dir "$OUT_DIR" \
        --owt_ref "$OWT_REF" \
        --seed $SEED
    echo "--- T=$STEPS done: $(date) ---"
}

# T=32 is the paper's default (32 steps per 32-token block)
run_remedi 32
# T=128 for direct comparison with our ReMDM T=128 results
run_remedi 128
# T=1000 for high-compute comparison
run_remedi 1000

echo ""
echo "=========================================="
echo "All RemeDi evals done: $(date)"
echo "=========================================="

# Aggregate results
python3 -c "
import json, os
base = os.path.expanduser('$REPO')
rows = []
for T in [32, 128, 1000]:
    p = f'{base}/results/remedi/T{T}/summary.json'
    if os.path.exists(p):
        d = json.load(open(p))
        rows.append({'T': T, 'gen_ppl': d['gen_ppl'],
                     'entropy': d['entropy'], 'MAUVE': d['MAUVE']})
        print(f'T={T:4d}  gen_ppl={d[\"gen_ppl\"]:.3f}  entropy={d[\"entropy\"]:.3f}  MAUVE={d[\"MAUVE\"]:.3f}')
out = f'{base}/results/remedi/comparison.json'
json.dump(rows, open(out, 'w'), indent=2)
print(f'Saved {out}')
"
