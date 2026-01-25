#!/bin/bash
################################################################################
# HPC Environment Setup Script for Masked Diffusion Thesis
# 
# Purpose: One-time comprehensive installation of ALL dependencies including
#          compiled CUDA extensions (flash-attn, causal-conv1d, mamba-ssm).
#
# Usage: Run this on an interactive GPU node or via setup_env_once.sbatch
#        bash scripts/setup_hpc_env.sh
#
# Prerequisites:
#   - Conda environment "masked-diffusion" already created (conda create -n masked-diffusion python=3.9)
#   - Access to HPC modules: gcc-12.2.1/gcc-12, nvidia/cuda-12.4.0
#   - GPU node for compiling CUDA extensions
#
# Output: Creates a snapshot file in logs/ with pip freeze + conda list
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

################################################################################
# Configuration - ADJUST THESE PATHS FOR YOUR HPC CLUSTER
################################################################################

# TODO: Update if your HPC uses different module names
GCC_MODULE="gcc-12.2.1/gcc-12"
CUDA_MODULE="nvidia/cuda-12.4.0"

# TODO: Update if CUDA is installed at a different path
CUDA_HOME_PATH="/opt/share/libs/nvidia/cuda-12.4.0"

# TODO: Update conda environment name if different
CONDA_ENV_NAME="masked-diffusion"

# Compute capability for RTX 2080 Ti
# TODO: Adjust if using different GPUs (3090=8.6, A100=8.0, etc.)
TORCH_CUDA_ARCH_LIST="7.5"

################################################################################
# Setup
################################################################################

echo "========================================"
echo "HPC Environment Setup for Masked Diffusion Thesis"
echo "========================================"
echo "Started at: $(date)"
echo ""

# Get repo root (assuming script is in scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Repository root: $REPO_ROOT"
cd "$REPO_ROOT"

################################################################################
# 1. Load HPC Modules
################################################################################

echo ""
echo "Step 1: Loading HPC modules..."
echo "  - GCC: $GCC_MODULE"
echo "  - CUDA: $CUDA_MODULE"

if command -v module &> /dev/null; then
    module purge
    module load "$GCC_MODULE"
    module load "$CUDA_MODULE"
    echo "Modules loaded successfully"
    module list
else
    echo "WARNING: 'module' command not found. Skipping module loading."
    echo "         Make sure GCC and CUDA are in your PATH."
fi

################################################################################
# 2. Activate Conda Environment
################################################################################

echo ""
echo "Step 2: Activating conda environment: $CONDA_ENV_NAME"

# Initialize conda for bash (adjust path if conda is elsewhere)
# TODO: Update CONDA_BASE if your conda installation is elsewhere
if [ -z "${CONDA_BASE:-}" ]; then
    # Try common locations
    if [ -f "/opt/share/modulefiles/sw/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="/opt/share/modulefiles/sw/miniconda3"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="/opt/conda"
    else
        echo "ERROR: Could not find conda installation."
        echo "       Please set CONDA_BASE environment variable."
        exit 1
    fi
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

################################################################################
# 3. Set Build Environment Variables
################################################################################

echo ""
echo "Step 3: Setting build environment variables..."

# CUDA paths
export CUDA_HOME="$CUDA_HOME_PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Compiler paths (gcc-toolset-12 or similar)
# TODO: Adjust if your HPC uses different gcc paths
export CC=$(which gcc)
export CXX=$(which g++)

# Torch CUDA architecture for RTX 2080 Ti (compute capability 7.5)
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

# Filesystem-safe temp/cache directories
# NOTE: Critical to avoid "Errno 18 Invalid cross-device link" errors
export TMPDIR="$HOME/.tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# HuggingFace cache
export HF_HOME="$HOME/.cache/huggingface"

# Pip cache control (disable for problematic packages)
export PIP_NO_CACHE_DIR=1

# Create temp directories
mkdir -p "$TMPDIR"
mkdir -p "$HF_HOME"

echo "Environment variables set:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  CUDACXX=$CUDACXX"
echo "  CC=$CC"
echo "  CXX=$CXX"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  TMPDIR=$TMPDIR"
echo "  HF_HOME=$HF_HOME"
echo "  PIP_NO_CACHE_DIR=$PIP_NO_CACHE_DIR"

# Verify nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found in PATH. Check CUDA module."
    exit 1
fi
echo "  nvcc: $(which nvcc)"
nvcc --version | head -n 4

################################################################################
# 4. Upgrade pip and build tools
################################################################################

echo ""
echo "Step 4: Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

################################################################################
# 5. Pin packaging to fix lightning compatibility
################################################################################

echo ""
echo "Step 5: Pinning packaging==24.2 (lightning requires <25)..."
pip uninstall -y packaging 2>/dev/null || true
pip install packaging==24.2

################################################################################
# 6. Install PyTorch (if not already installed)
################################################################################

echo ""
echo "Step 6: Ensuring PyTorch is installed..."

# Check if torch is already installed with CUDA
if python -c "import torch; print(torch.__version__, torch.version.cuda)" 2>/dev/null; then
    echo "PyTorch already installed:"
    python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
else
    echo "Installing PyTorch 2.2.2 with CUDA 12.1..."
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
fi

################################################################################
# 7. Install base dependencies from pyproject.toml
################################################################################

echo ""
echo "Step 7: Installing repo package in editable mode..."
pip install -e "$REPO_ROOT"

################################################################################
# 8. Install upstream ReMDM dependencies
################################################################################

echo ""
echo "Step 8: Installing upstream ReMDM dependencies..."

# Core packages from external/remdm/requirements.yml
pip install \
    datasets==2.18.0 \
    einops==0.7.0 \
    fsspec==2024.2.0 \
    h5py==3.10.0 \
    hydra-core==1.3.2 \
    ipdb==0.13.13 \
    lightning==2.2.1 \
    omegaconf==2.3.0 \
    pandas==2.2.1 \
    rich==13.7.1 \
    seaborn==0.13.2 \
    scikit-learn==1.4.0 \
    timm==0.9.16 \
    transformers==4.38.2 \
    triton==2.2.0 \
    wandb==0.13.5 \
    mauve-text==0.4.0

echo "Core dependencies installed."

################################################################################
# 9. Install COMPILED CUDA Extensions (the hard part!)
################################################################################

echo ""
echo "=========================================="
echo "Step 9: Installing COMPILED CUDA extensions"
echo "=========================================="
echo "This may take 10-30 minutes. Be patient!"
echo ""

# 9a. flash-attn (required for DiT backbone)
echo ""
echo "9a. Installing flash-attn==2.5.6..."
echo "    This requires ~15-20 minutes to compile."

pip install flash-attn==2.5.6 \
    --no-build-isolation \
    --no-cache-dir \
    2>&1 | tee "$REPO_ROOT/logs/flash_attn_install.log"

if python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__} installed successfully')" 2>/dev/null; then
    echo "✓ flash-attn installed successfully"
else
    echo "✗ flash-attn installation FAILED. Check logs/flash_attn_install.log"
    exit 1
fi

# 9b. causal-conv1d (required for Mamba/DiMamba backbone)
echo ""
echo "9b. Installing causal-conv1d..."
echo "    This requires ~5-10 minutes to compile."

# Try installing from PyPI first
if pip install causal-conv1d>=1.1.0 \
    --no-build-isolation \
    --no-cache-dir \
    2>&1 | tee "$REPO_ROOT/logs/causal_conv1d_install.log"; then
    echo "✓ causal-conv1d installed from PyPI"
else
    echo "PyPI installation failed. Trying from GitHub..."
    pip install git+https://github.com/Dao-AILab/causal-conv1d.git \
        --no-build-isolation \
        --no-cache-dir \
        2>&1 | tee -a "$REPO_ROOT/logs/causal_conv1d_install.log"
fi

if python -c "import causal_conv1d; print(f'causal-conv1d installed successfully')" 2>/dev/null; then
    echo "✓ causal-conv1d installed successfully"
else
    echo "WARNING: causal-conv1d installation failed."
    echo "         DiMamba backbone will not work, but DiT will."
    echo "         Check logs/causal_conv1d_install.log for details."
fi

# 9c. mamba-ssm (optional, for Mamba/DiMamba backbone)
echo ""
echo "9c. Installing mamba-ssm..."
echo "    This is optional but recommended for DiMamba backbone."

if pip install mamba-ssm \
    --no-build-isolation \
    --no-cache-dir \
    2>&1 | tee "$REPO_ROOT/logs/mamba_ssm_install.log"; then
    echo "✓ mamba-ssm installed successfully"
else
    echo "WARNING: mamba-ssm installation failed."
    echo "         DiMamba backbone will not work, but DiT will."
    echo "         Check logs/mamba_ssm_install.log for details."
fi

################################################################################
# 10. Install additional useful packages
################################################################################

echo ""
echo "Step 10: Installing additional useful packages..."

pip install \
    jupyterlab \
    notebook \
    ipywidgets \
    tensorboard \
    gpustat \
    nvitop

################################################################################
# 11. Create snapshot for reproducibility
################################################################################

echo ""
echo "Step 11: Creating environment snapshot..."

SNAPSHOT_DIR="$REPO_ROOT/logs/env_snapshots"
mkdir -p "$SNAPSHOT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_FILE="$SNAPSHOT_DIR/env_snapshot_${TIMESTAMP}.txt"

{
    echo "=========================================="
    echo "Environment Snapshot"
    echo "=========================================="
    echo "Timestamp: $(date)"
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo ""
    
    echo "=========================================="
    echo "System Info"
    echo "=========================================="
    echo "Python: $(which python)"
    python --version
    echo ""
    echo "GCC: $(which gcc)"
    gcc --version | head -n 1
    echo ""
    echo "NVCC: $(which nvcc)"
    nvcc --version | grep "release"
    echo ""
    
    echo "=========================================="
    echo "PyTorch Info"
    echo "=========================================="
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    echo ""
    
    echo "=========================================="
    echo "Compiled Extensions"
    echo "=========================================="
    python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" 2>&1 || echo "flash-attn: NOT INSTALLED"
    python -c "import causal_conv1d; print('causal-conv1d: installed')" 2>&1 || echo "causal-conv1d: NOT INSTALLED"
    python -c "import mamba_ssm; print('mamba-ssm: installed')" 2>&1 || echo "mamba-ssm: NOT INSTALLED"
    echo ""
    
    echo "=========================================="
    echo "Pip Freeze"
    echo "=========================================="
    pip freeze
    echo ""
    
    echo "=========================================="
    echo "Conda List"
    echo "=========================================="
    conda list
    
} > "$SNAPSHOT_FILE"

echo "Snapshot saved to: $SNAPSHOT_FILE"

# Also create a symlink to latest
ln -sf "env_snapshot_${TIMESTAMP}.txt" "$SNAPSHOT_DIR/latest.txt"

################################################################################
# 12. Run quick verification
################################################################################

echo ""
echo "Step 12: Running quick verification..."

python -c "
import sys
import torch

print('✓ Python:', sys.version.split()[0])
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())

try:
    import flash_attn
    print('✓ flash-attn:', flash_attn.__version__)
except ImportError as e:
    print('✗ flash-attn: FAILED -', e)
    sys.exit(1)

try:
    import causal_conv1d
    print('✓ causal-conv1d: installed')
except ImportError:
    print('⚠ causal-conv1d: not installed (DiMamba will not work)')

try:
    import mamba_ssm
    print('✓ mamba-ssm: installed')
except ImportError:
    print('⚠ mamba-ssm: not installed (DiMamba will not work)')

try:
    import lightning
    print('✓ lightning:', lightning.__version__)
except ImportError as e:
    print('✗ lightning: FAILED -', e)
    sys.exit(1)

try:
    import hydra
    print('✓ hydra-core: installed')
except ImportError as e:
    print('✗ hydra-core: FAILED -', e)
    sys.exit(1)

try:
    import transformers
    print('✓ transformers:', transformers.__version__)
except ImportError as e:
    print('✗ transformers: FAILED -', e)
    sys.exit(1)

print('')
print('Environment setup completed successfully!')
"

################################################################################
# Done
################################################################################

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo "Finished at: $(date)"
echo ""
echo "Next steps:"
echo "  1. Review snapshot: logs/env_snapshots/latest.txt"
echo "  2. Run preflight check: python scripts/preflight_check.py"
echo "  3. Submit test job: sbatch slurm/remdm_smoke.sbatch"
echo ""
