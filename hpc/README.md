# Bocconi HPC Workflow

Scripts for running experiments on Bocconi HPC cluster without VS Code Remote-SSH.

## Quick Start

```bash
# 1. Push code to cluster
bash hpc/push.sh

# 2. Submit job
bash hpc/submit.sh

# 3. Monitor job (optional)
ssh 3316152@slogin.hpc.unibocconi.it "squeue -u 3316152"

# 4. Pull results back
bash hpc/pull.sh
```

## Detailed Workflow

### 1. Push Code (`hpc/push.sh`)

Syncs local code to the cluster using rsync:

```bash
bash hpc/push.sh
```

**What it does:**
- Creates remote directory: `~/mdm/masked-diffusion-thesis/`
- Syncs all code files
- Excludes: `.git`, `.venv`, `__pycache__`, `logs/`, `results/`, `*.pt`, `*.ckpt`
- Uses `--delete` to remove old files on remote

**Location on cluster:** `~/mdm/masked-diffusion-thesis/`

### 2. Submit Job (`hpc/submit.sh`)

Submits a Slurm batch job:

```bash
bash hpc/submit.sh
```

**What it does:**
- SSHs to cluster
- Changes to repo directory
- Runs `sbatch hpc/job.sh`
- Returns job ID

**Example output:**
```
✅ Job submitted successfully!
Job ID: 12345

Monitor job:
  squeue -u 3316152
  squeue -j 12345

View logs (once running):
  ssh 3316152@slogin.hpc.unibocconi.it 'tail -f ~/mdm/masked-diffusion-thesis/logs/remdm-12345.out'
```

### 3. Monitor Job

**Check job status:**
```bash
ssh 3316152@slogin.hpc.unibocconi.it "squeue -u 3316152"
```

**View live logs:**
```bash
# Replace 12345 with your job ID
ssh 3316152@slogin.hpc.unibocconi.it "tail -f ~/mdm/masked-diffusion-thesis/logs/remdm-12345.out"
```

**Check job details:**
```bash
ssh 3316152@slogin.hpc.unibocconi.it "scontrol show job 12345"
```

**Cancel job:**
```bash
ssh 3316152@slogin.hpc.unibocconi.it "scancel 12345"
```

### 4. Pull Results (`hpc/pull.sh`)

Fetches logs and results back to local machine:

```bash
bash hpc/pull.sh
```

**What it downloads:**
- `logs/` - Slurm output logs
- `results/` - Experiment results
- `outputs/` - Additional outputs
- `checkpoints/` - Metadata only (excludes large `.pt`/`.ckpt` files)

**To download large checkpoint files manually:**
```bash
scp 3316152@slogin.hpc.unibocconi.it:~/mdm/masked-diffusion-thesis/checkpoints/model.ckpt ./checkpoints/
```

## Job Configuration

The job script (`hpc/job.sh`) includes:

**Mandatory Bocconi HPC settings:**
```bash
#SBATCH --partition=stud      # Student partition
#SBATCH --qos=stud            # Quality of service
#SBATCH --account=3316152     # Your account ID
```

**Resource allocation:**
```bash
#SBATCH --gres=gpu:1          # 1 GPU
#SBATCH --cpus-per-task=4     # 4 CPU cores
#SBATCH --mem=32G             # 32 GB RAM
#SBATCH --time=02:00:00       # 2 hour time limit
```

**Customize for your experiment:**

Edit `hpc/job.sh` to change:
- `--time=HH:MM:SS` - Time limit
- `--mem=XG` - Memory
- `--gres=gpu:N` - Number of GPUs
- Config file: `python scripts/run_remdm.py --config configs/YOUR_CONFIG.yaml`

## Environment Setup

The job script automatically detects and activates your Python environment:

1. **venv** (`.venv/` or `venv/`) - Activates with `source .venv/bin/activate`
2. **conda** - Activates `myenv` environment
3. **System Python** - Falls back if no environment found

**To use a specific conda environment on HPC:**
```bash
# On cluster (first time setup):
ssh 3316152@slogin.hpc.unibocconi.it

# Create environment
conda create -n myenv python=3.10
conda activate myenv
cd ~/mdm/masked-diffusion-thesis
pip install -r requirements.txt

# Or copy your local venv (after push):
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common Workflows

### Quick iteration
```bash
# Edit code locally
vim configs/remdm.yaml

# Push and submit
bash hpc/push.sh && bash hpc/submit.sh

# Check if running
ssh 3316152@slogin.hpc.unibocconi.it "squeue -u 3316152"

# Pull results when done
bash hpc/pull.sh
```

### Debug failing job
```bash
# Pull latest logs
bash hpc/pull.sh

# Check error log
cat logs/remdm-*.err

# Fix issue, push, resubmit
bash hpc/push.sh
bash hpc/submit.sh
```

### Multiple experiments
```bash
# Edit job.sh to use different config
vim hpc/job.sh
# Change: --config configs/experiment1.yaml

bash hpc/push.sh && bash hpc/submit.sh

# Repeat for experiment2
vim hpc/job.sh
# Change: --config configs/experiment2.yaml

bash hpc/push.sh && bash hpc/submit.sh
```

## Troubleshooting

### "Permission denied" when running scripts
```bash
chmod +x hpc/*.sh
```

### "Cannot find module X"
Environment not activated or dependencies not installed on cluster:
```bash
ssh 3316152@slogin.hpc.unibocconi.it
cd ~/mdm/masked-diffusion-thesis
source .venv/bin/activate  # or conda activate myenv
pip install -r requirements.txt
```

### "No such file or directory: logs/remdm-*.out"
Job hasn't started yet or logs directory not created. The job script creates it automatically, but you can create it manually:
```bash
ssh 3316152@slogin.hpc.unibocconi.it "mkdir -p ~/mdm/masked-diffusion-thesis/logs"
```

### Job stuck in queue
Check partition availability and your account limits:
```bash
ssh 3316152@slogin.hpc.unibocconi.it "sinfo -p stud"
ssh 3316152@slogin.hpc.unibocconi.it "sacct -u 3316152"
```

### Out of memory / Time limit exceeded
Increase resources in `hpc/job.sh`:
```bash
#SBATCH --mem=64G              # More memory
#SBATCH --time=04:00:00        # More time
```

## File Structure

```
hpc/
├── job.sh          # Slurm batch script (submit with sbatch)
├── push.sh         # Push code to cluster
├── submit.sh       # Submit job remotely
├── pull.sh         # Fetch results back
└── README.md       # This file
```

## Notes

- **No Remote-SSH:** All connections are short-lived SSH commands
- **macOS compatible:** Scripts use bash and standard Unix tools
- **Excludes large files:** Checkpoints and model files are not synced automatically
- **Clean workspace:** Use `--delete` flag to keep remote in sync with local

## References

- [Bocconi HPC Documentation](https://hpc.unibocconi.it/)
- Slurm commands: `squeue`, `sbatch`, `scancel`, `scontrol`, `sinfo`, `sacct`
