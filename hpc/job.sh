#!/bin/bash
#SBATCH --job-name=remdm              # Job name
#SBATCH --partition=stud              # Bocconi HPC student partition (mandatory)
#SBATCH --qos=stud                    # Quality of service (mandatory)
#SBATCH --account=3316152             # Your account ID (mandatory)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=02:00:00               # Time limit (hh:mm:ss)
#SBATCH --output=logs/%x-%j.out       # Standard output log
#SBATCH --error=logs/%x-%j.err        # Standard error log

# Fail fast: exit on any error, undefined variable, or pipe failure
set -euo pipefail

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Python environment
# This repo uses venv at .venv/
if [ -d ".venv" ]; then
    echo "Activating venv at .venv/"
    source .venv/bin/activate
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
elif [ -d "venv" ]; then
    echo "Activating venv at venv/"
    source venv/bin/activate
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
elif command -v conda &> /dev/null; then
    echo "Attempting to activate conda environment 'myenv'..."
    eval "$(conda shell.bash hook)"
    conda activate myenv || echo "Warning: conda env 'myenv' not found, using base"
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
else
    echo "No virtual environment found, using system Python"
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
fi

echo ""
echo "=========================================="
echo "Running ReMDM sampling..."
echo "=========================================="
echo ""

# Run the main script
# Adjust the config path and arguments as needed for your experiment
python scripts/run_remdm.py --config configs/remdm.yaml

# Print completion info
echo ""
echo "=========================================="
echo "Job completed successfully!"
echo "End Time: $(date)"
echo "=========================================="
