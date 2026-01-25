# HPC Setup Guide

This document provides step-by-step instructions for setting up your environment on the HPC cluster to run masked diffusion experiments.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [One-Time Setup](#one-time-setup)
4. [Verification](#verification)
5. [Running Experiments](#running-experiments)
6. [Troubleshooting](#troubleshooting)
7. [Technical Details](#technical-details)

---

## Overview

The setup process installs **all** required dependencies, including compiled CUDA extensions (flash-attn, causal-conv1d, mamba-ssm), in a reproducible way. This is done **once** on a GPU node, after which you can run experiments without chasing missing dependencies.

**Key Features:**
- One-command setup script
- Handles problematic filesystem issues (cross-device link errors)
- Pins package versions for reproducibility
- Creates environment snapshots
- Preflight checks before each run

---

## Prerequisites

### 1. Clone Repository with Submodules

```bash
cd ~
git clone --recursive https://github.com/your-org/masked-diffusion-thesis.git
cd masked-diffusion-thesis

# If already cloned without submodules:
git submodule update --init --recursive
```

### 2. Create Conda Environment

Create a conda environment with Python 3.9:

```bash
conda create -n masked-diffusion python=3.9 -y
conda activate masked-diffusion
```

**Note:** The repo supports Python 3.10+ locally, but the HPC cluster has Python 3.9, so we use 3.9 for compatibility.

### 3. Verify HPC Modules

Check that required modules are available:

```bash
module avail gcc
module avail cuda
```

You should see:
- `gcc-12.2.1/gcc-12` (or similar GCC 12 version)
- `nvidia/cuda-12.4.0` (or similar CUDA 12 version)

**TODO:** If your cluster uses different module names, update these in:
- [`scripts/setup_hpc_env.sh`](../scripts/setup_hpc_env.sh) (lines 22-23)
- [`slurm/remdm_smoke.sbatch`](../slurm/remdm_smoke.sbatch) (line 58)

---

## One-Time Setup

### Option A: Interactive GPU Node (Recommended for First Time)

This allows you to monitor the installation in real-time.

```bash
# Request an interactive GPU node
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash

# Once on GPU node, run setup
cd ~/masked-diffusion-thesis
bash scripts/setup_hpc_env.sh
```

**Expected Duration:** 30-45 minutes
- flash-attn: ~15-20 minutes
- causal-conv1d: ~5-10 minutes
- mamba-ssm: ~5 minutes
- Other packages: ~10 minutes

### Option B: Batch Job (Hands-off)

Submit the setup as a batch job:

```bash
cd ~/masked-diffusion-thesis
sbatch slurm/setup_env_once.sbatch
```

Monitor progress:

```bash
tail -f logs/slurm/setup_env_once_<jobid>.log
```

---

## Verification

### 1. Review Environment Snapshot

After setup completes, review the environment snapshot:

```bash
cat logs/env_snapshots/latest.txt
```

This file contains:
- Python, GCC, NVCC versions
- PyTorch and CUDA info
- Compiled extension versions
- Complete pip freeze output
- Complete conda list output

### 2. Run Preflight Check

Run the preflight checker to verify all dependencies:

```bash
# On interactive GPU node
srun --partition=gpu --gres=gpu:1 --pty python scripts/preflight_check.py

# Output should show all green checkmarks (✓)
```

**Check Categories:**
- ✓ Core dependencies (PyYAML, NumPy, PyTorch, etc.)
- ✓ ML frameworks (Lightning, Hydra, Transformers, etc.)
- ✓ Compiled CUDA extensions (flash-attn, causal-conv1d, mamba-ssm)
- ✓ GPU & CUDA availability
- ✓ Repository structure
- ✓ Upstream ReMDM imports
- ✓ Environment variables

If any checks fail, see [Troubleshooting](#troubleshooting).

### 3. Test Dry Run

Test the ReMDM wrapper in dry-run mode (no GPU needed):

```bash
python scripts/run_remdm.py --config configs/remdm.yaml
```

This should print the command that would be executed without actually running it.

---

## Running Experiments

### Smoke Test

Submit a quick smoke test to verify everything works end-to-end:

```bash
sbatch slurm/remdm_smoke.sbatch
```

This job:
1. Loads modules and activates conda environment
2. Runs preflight check (fails fast if environment is broken)
3. Runs a minimal ReMDM experiment

Monitor:

```bash
tail -f out/remdm_smoke_<jobid>.out
```

### Full Experiments

Once the smoke test passes, you can run full experiments:

```bash
# Edit config as needed
vim configs/remdm_hpc.yaml

# Submit job
sbatch slurm/remdm_production.sbatch  # TODO: Create this
```

---

## Troubleshooting

### Flash-attn Installation Fails

**Symptoms:**
- "Errno 18 Invalid cross-device link"
- Compilation errors during `pip install flash-attn`

**Solutions:**

1. **Check TMPDIR:** Ensure `TMPDIR=$HOME/.tmp` is set
   ```bash
   echo $TMPDIR  # Should be /home/<user>/.tmp
   ls -ld $TMPDIR  # Should exist
   ```

2. **Check CUDA paths:**
   ```bash
   echo $CUDA_HOME  # Should be /opt/share/libs/nvidia/cuda-12.4.0
   which nvcc       # Should be in CUDA_HOME/bin
   nvcc --version   # Should show CUDA 12.4
   ```

3. **Check compiler:**
   ```bash
   which gcc        # Should be GCC 12
   gcc --version    # Should show 12.x
   ```

4. **Check build environment:**
   ```bash
   echo $TORCH_CUDA_ARCH_LIST  # Should be "7.5" for RTX 2080 Ti
   ```

5. **Manually retry:**
   ```bash
   pip install flash-attn==2.5.6 --no-build-isolation --no-cache-dir --verbose
   ```

6. **Check logs:**
   ```bash
   cat logs/flash_attn_install.log
   ```

### Causal-conv1d Installation Fails

**Symptoms:**
- Build errors during `pip install causal-conv1d`

**Solutions:**

1. **Check if GCC/NVCC are in path:**
   ```bash
   module list  # Should show gcc-12 and cuda-12.4
   ```

2. **Manually retry from GitHub:**
   ```bash
   pip install git+https://github.com/Dao-AILab/causal-conv1d.git \
     --no-build-isolation --no-cache-dir
   ```

3. **Check logs:**
   ```bash
   cat logs/causal_conv1d_install.log
   ```

**Note:** If causal-conv1d fails, you can still use the DiT backbone (which only requires flash-attn). DiMamba will not work.

### Preflight Check Fails

**Symptom:** `python scripts/preflight_check.py` shows red ✗ marks

**Solutions:**

1. **Identify which check failed:**
   - Core dependencies: Re-run setup script
   - Compiled extensions: See above
   - GPU not available: Make sure you're on a GPU node
   - Upstream ReMDM imports: Check that submodules are initialized

2. **Re-run specific installations:**
   ```bash
   # Re-activate environment
   conda activate masked-diffusion
   module load gcc-12.2.1/gcc-12 nvidia/cuda-12.4.0
   
   # Re-install problematic package
   pip install <package> --no-cache-dir --force-reinstall
   ```

3. **Check Python version:**
   ```bash
   python --version  # Should be 3.9.x
   ```

### ReMDM Subprocess Fails

**Symptoms:**
- Job fails with "ImportError: cannot import name 'flash_attn'"
- Job fails with "ModuleNotFoundError: No module named 'causal_conv1d'"

**Solutions:**

1. **Check that environment is activated in job script:**
   ```bash
   # Should be in your .sbatch file:
   conda activate masked-diffusion
   ```

2. **Check that modules are loaded:**
   ```bash
   # Should be in your .sbatch file:
   module load gcc-12.2.1/gcc-12 nvidia/cuda-12.4.0
   ```

3. **Enable env_patch in config:**
   ```yaml
   # In your config file (e.g., configs/remdm_hpc_smoke.yaml)
   wrapper:
     env_patch: true  # Ensures runtime env vars are set
   ```

4. **Check backbone configuration:**
   - For DiT: Only requires flash-attn ✓
   - For DiMamba: Requires flash-attn + causal-conv1d + mamba-ssm

   ```yaml
   # configs/remdm_hpc_smoke.yaml
   remdm:
     backbone: "dit"  # Use "dit" if mamba deps are not installed
     require_optional_backbones: false  # Set to true only if using DiMamba
   ```

### Packaging Version Conflict

**Symptoms:**
- "ERROR: packaging 25.x is not compatible with lightning"

**Solution:**
```bash
pip uninstall packaging -y
pip install packaging==24.2
```

This is handled automatically by the setup script, but may occur if you install packages manually.

---

## Technical Details

### Why TMPDIR and No Cache?

**Problem:** Some HPC filesystems use different mount points for `/tmp` (local SSD) and home directories (network storage). When pip tries to move files from `/tmp` to `~/.local`, it fails with "Errno 18 Invalid cross-device link".

**Solution:** Set `TMPDIR=$HOME/.tmp` so all temp operations happen in the same filesystem. Also disable pip cache with `PIP_NO_CACHE_DIR=1` and `--no-cache-dir`.

### Why TORCH_CUDA_ARCH_LIST?

**Background:** CUDA code is compiled for specific GPU architectures (compute capabilities). Without specifying this, PyTorch extensions will try to compile for all architectures (3.5 through 9.0), which:
1. Takes much longer (hours instead of minutes)
2. May fail on some architectures
3. Produces bloated binaries

**Solution:** Set `TORCH_CUDA_ARCH_LIST="7.5"` to only compile for RTX 2080 Ti (compute capability 7.5).

**Other GPU compute capabilities:**
- RTX 3090: 8.6
- A100: 8.0
- V100: 7.0
- T4: 7.5

**TODO:** If your cluster has different GPUs, update this in:
- [`scripts/setup_hpc_env.sh`](../scripts/setup_hpc_env.sh) (line 31)

### Why Pin Packaging?

**Background:** PyTorch Lightning 2.2.1 requires `packaging<25`, but newer versions of pip install `packaging>=25` by default.

**Solution:** Explicitly pin `packaging==24.2` to avoid conflicts.

### Why No Build Isolation?

**Background:** By default, pip creates an isolated environment for building each package, which can:
1. Miss environment variables (CUDA_HOME, CC, etc.)
2. Not see PyTorch installed in the main environment
3. Fail to find CUDA libraries

**Solution:** Use `--no-build-isolation` for compiled extensions so they inherit our carefully configured build environment.

### Environment Patching

The `env_patch` option in `ReMDMRunConfig` sets runtime environment variables for the ReMDM subprocess:

- **HF_HOME:** HuggingFace cache location
- **TMPDIR/TEMP/TMP:** Filesystem-safe temp directory
- **TOKENIZERS_PARALLELISM:** Disable to avoid warnings in multi-process context

This ensures the upstream ReMDM process has the same environment setup as the main script.

### Submodule Integration Strategy

**Problem:** Upstream repos (external/remdm, external/PRISM) may import optional dependencies eagerly, causing ImportError if not installed.

**Constraint:** We cannot modify submodule code.

**Solutions:**

1. **Install all optional dependencies** (our approach for HPC)
   - Pro: Simple, everything works
   - Con: Longer initial setup time

2. **Lazy imports in submodule** (would require patching)
   - Not viable due to constraint

3. **Subprocess isolation**
   - Our wrapper runs ReMDM as a subprocess with its own environment
   - PYTHONPATH includes external/remdm
   - Environment variables are patched if needed

4. **Backbone selection**
   - Use DiT backbone (only needs flash-attn) if Mamba deps fail
   - Set `require_optional_backbones: false` in config

---

## Summary Workflow

```
1. Create conda env (once)
   conda create -n masked-diffusion python=3.9
   
2. Run setup (once)
   sbatch slurm/setup_env_once.sbatch
   
3. Verify (once)
   python scripts/preflight_check.py
   
4. Run experiments (many times)
   sbatch slurm/remdm_smoke.sbatch
   sbatch slurm/remdm_production.sbatch
```

---

## Additional Resources

- [ReMDM upstream repo](https://github.com/kuleshov-group/ReMDM)
- [PRISM upstream repo](https://github.com/microsoft/PRISM)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Causal Conv1d](https://github.com/Dao-AILab/causal-conv1d)
- [Mamba SSM](https://github.com/state-spaces/mamba)

---

## Questions?

If you encounter issues not covered here:

1. Check installation logs in `logs/`
2. Run preflight check with verbose: `python scripts/preflight_check.py -v`
3. Review environment snapshot: `cat logs/env_snapshots/latest.txt`
4. Check Slurm job output: `out/` and `err/` directories

For persistent issues, review the setup script ([`scripts/setup_hpc_env.sh`](../scripts/setup_hpc_env.sh)) and adjust paths/modules for your specific HPC cluster.
