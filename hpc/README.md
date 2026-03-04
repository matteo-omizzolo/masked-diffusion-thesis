# Bocconi HPC Workflow

Scripts for running experiments on the Bocconi HPC cluster.

---

## Quick Start

```bash
# 1. Push code to cluster
bash hpc/push.sh

# 2. Upload checkpoint (first time only — done separately, not via push.sh)
ssh 3316152@slogin.hpc.unibocconi.it "mkdir -p ~/mdm/checkpoints"
scp checkpoints/mdlm.ckpt 3316152@slogin.hpc.unibocconi.it:~/mdm/checkpoints/mdlm.ckpt

# 3. Set up environment (first time only)
ssh 3316152@slogin.hpc.unibocconi.it "cd ~/mdm/masked-diffusion-thesis && bash hpc/setup_env.sh"

# 4. Submit job
bash hpc/submit.sh

# 5. Monitor
ssh 3316152@slogin.hpc.unibocconi.it "squeue -u 3316152"

# 6. Pull results back
bash hpc/pull.sh
```

---

## Cluster Constraints (Bocconi stud partition)

| Resource | Limit |
|---|---|
| Max GPUs per job | **4** |
| Partition | `stud` |
| QOS | `stud` |
| Account | `3316152` |

The current job script (`hpc/remdm_smoke.sbatch`) requests 1 GPU — well within limits.

---

## File Structure

```
hpc/
├── remdm_smoke.sbatch   # Slurm job script for the ReMDM smoke test
├── push.sh              # rsync code to cluster (excludes checkpoints/results)
├── submit.sh            # SSH + sbatch hpc/remdm_smoke.sbatch
├── pull.sh              # Fetch out/, err/, results/ back to local
├── setup_env.sh         # Create conda env and install dependencies on HPC
└── README.md            # This file

checkpoints/             # Local checkpoint storage (gitignored, not synced by push.sh)
└── mdlm.ckpt            # MDLM checkpoint (~2.5 GB) — upload manually via scp
```

**Checkpoint location on HPC:** `~/mdm/checkpoints/mdlm.ckpt` (outside the repo)

---

## Detailed Workflow

### Push code (`hpc/push.sh`)

```bash
bash hpc/push.sh
```

Rsyncs the repo to `~/mdm/masked-diffusion-thesis/` on the cluster. Automatically excludes:
`.git`, `.venv`, `__pycache__`, `logs/`, `results/`, `checkpoints/`, `*.pt`, `*.ckpt`.

> Checkpoints are never synced by push.sh — upload them once manually via scp (see Quick Start).

---

### Submit job (`hpc/submit.sh`)

```bash
bash hpc/submit.sh
```

SSHes to the cluster and runs `sbatch hpc/remdm_smoke.sbatch`. Prints the job ID on success.

**Job outputs go to:**
- `out/remdm_smoke_<job_id>.out` — stdout
- `err/remdm_smoke_<job_id>.err` — stderr

---

### Monitor

```bash
# Job queue
ssh 3316152@slogin.hpc.unibocconi.it "squeue -u 3316152"

# Live stdout
ssh 3316152@slogin.hpc.unibocconi.it "tail -f ~/mdm/masked-diffusion-thesis/out/remdm_smoke_<job_id>.out"

# Job details
ssh 3316152@slogin.hpc.unibocconi.it "scontrol show job <job_id>"

# Cancel
ssh 3316152@slogin.hpc.unibocconi.it "scancel <job_id>"

# Past jobs
ssh 3316152@slogin.hpc.unibocconi.it "sacct -u 3316152 --format=JobID,JobName,State,Elapsed,ExitCode"
```

---

### Pull results (`hpc/pull.sh`)

```bash
bash hpc/pull.sh
```

Downloads `out/`, `err/`, and `results/` from the cluster to local. Large binary files (`.pt`, `.ckpt`) are excluded.

---

## Customising the Job Script

Edit `hpc/remdm_smoke.sbatch` to change resources or the command:

```bash
#SBATCH --gres=gpu:1          # GPUs (max 4 on stud partition)
#SBATCH --cpus-per-task=8     # CPU cores
#SBATCH --mem=32G             # RAM
#SBATCH --time=02:00:00       # Wall time (HH:MM:SS)
```

The `srun` command at the bottom is what actually runs — edit it to change strategy, steps, config, etc.

---

## Common Workflows

### Iterate on an experiment

```bash
# 1. Edit config or code locally
# 2. Push + submit in one line
bash hpc/push.sh && bash hpc/submit.sh
# 3. Pull when done
bash hpc/pull.sh
```

### Debug a failing job

```bash
bash hpc/pull.sh
cat err/remdm_smoke_<job_id>.err
# fix issue, then:
bash hpc/push.sh && bash hpc/submit.sh
```

---

## Troubleshooting

**"Permission denied" on scripts**
```bash
chmod +x hpc/*.sh
```

**Job stuck in queue**
```bash
ssh 3316152@slogin.hpc.unibocconi.it "sinfo -p stud"   # check partition availability
```

**Module not found / import errors**
```bash
ssh 3316152@slogin.hpc.unibocconi.it
cd ~/mdm/masked-diffusion-thesis
conda activate remdm
pip install -e .
```

**Out of memory or time limit**

Increase `--mem` or `--time` in `hpc/remdm_smoke.sbatch`. Remember: max 4 GPUs.

---

## References

- Slurm commands: `squeue`, `sbatch`, `scancel`, `scontrol`, `sinfo`, `sacct`
