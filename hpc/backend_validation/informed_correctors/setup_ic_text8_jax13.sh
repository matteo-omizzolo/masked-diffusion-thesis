#!/bin/bash
# Idempotent setup script for the informed-correctors/Text8 environment on a
# Bocconi HPC login node (compute nodes have no outbound internet, so this
# script must NOT be invoked from an sbatch job).
#
# This env is dedicated to JAX-on-CUDA13 work for the informed-correctors
# backend-validation track. Do NOT use it for ProSeCo or PyTorch work — that
# stays on `remdm311`. Keeping the environments separated prevents the
# protobuf/CUDA conflicts that broke `remdm311` during the 2026-05-13 setup
# pass.
#
# Usage (from a login node, after `git pull` / `bash hpc/push.sh`):
#   bash hpc/backend_validation/informed_correctors/setup_ic_text8_jax13.sh
#
# Override env name / Python version with environment variables if needed:
#   ENV_NAME=ic_text8_jax13_test PYTHON_VERSION=3.11 bash ...

set -euo pipefail

ENV_NAME="${ENV_NAME:-ic_text8_jax13}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TEXT8_DATA_DIR="${TEXT8_DATA_DIR:-$HOME/mdm/data/text8_md4}"

echo "[setup] target env: $ENV_NAME (python $PYTHON_VERSION)"

if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] loading miniconda3 module..."
    module load miniconda3
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env $ENV_NAME already exists; updating packages in-place."
else
    echo "[setup] creating env $ENV_NAME ..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

echo "[setup] python: $(python --version 2>&1)"
echo "[setup] pip: $(pip --version)"

# Upgrade pip first so newer wheel selectors (e.g. jax[cuda13]) resolve.
pip install --upgrade pip

# JAX with CUDA 13 PJRT plugin (compatible with Bocconi A100 driver
# 580.95.05 / CUDA 13.0). The cuda13 wheels are what avoids the cuda12-vs-13
# driver mismatch that crashed Stage 1 in the earlier remdm311 setup.
echo "[setup] installing jax[cuda13] ..."
pip install --upgrade "jax[cuda13]"

# JAX ecosystem. distrax 0.1.8 (the only currently-published distrax) needs
# JAX >= 0.7; pip will resolve a compatible matrix automatically once jax
# is already installed at a recent version.
echo "[setup] installing JAX-ecosystem packages ..."
pip install --upgrade \
    flax \
    optax \
    orbax-checkpoint \
    clu \
    grain \
    ml_collections \
    absl-py \
    distrax \
    "datasets>=2.21"

# TensorFlow stack required by clu.metric_writers, tensorflow_datasets, and
# tensorflow_probability (used by distrax for the JAX substrate). Note:
# `tf-keras` is needed by tensorflow_probability's lazy-loader validation,
# and `tensorboard` is needed by clu.metric_writers.
echo "[setup] installing TensorFlow stack ..."
pip install --upgrade \
    tensorflow \
    tensorflow-datasets \
    tf-keras \
    tensorflow-probability

# Visualisation / logging used by upstream md4 utilities (utils.py imports
# seaborn at module load; train.py logs to wandb).
echo "[setup] installing visualisation / logging ..."
pip install --upgrade wandb seaborn matplotlib

echo "[setup] staging Text8 zip if missing ..."
mkdir -p "$TEXT8_DATA_DIR/text8"
if [ ! -f "$TEXT8_DATA_DIR/text8/text8.zip" ]; then
    curl -L --fail --silent --show-error \
        http://mattmahoney.net/dc/text8.zip \
        -o "$TEXT8_DATA_DIR/text8/text8.zip"
    echo "[setup] Text8 staged ($(du -h "$TEXT8_DATA_DIR/text8/text8.zip" | cut -f1))"
else
    echo "[setup] Text8 zip already present."
fi

echo "[setup] env smoke (login node — GPU enumeration will fail; that is OK)..."
python - <<'PY'
import importlib
deps = [
    "absl", "clu", "datasets", "distrax", "flax", "grain", "jax",
    "ml_collections", "optax", "orbax.checkpoint", "tensorflow",
    "tensorflow_datasets", "wandb",
]
ok, missing = [], []
for d in deps:
    try:
        importlib.import_module(d)
        ok.append(d)
    except Exception as exc:
        missing.append((d, type(exc).__name__, str(exc)[:80]))
print(f"[setup] imports OK: {len(ok)} of {len(deps)}")
if missing:
    print(f"[setup] imports MISSING: {missing}")
import jax
print(f"[setup] jax version: {jax.__version__}")
try:
    print(f"[setup] jax devices (login): {jax.devices()}")
except Exception as exc:
    print(f"[setup] jax.devices() raised (expected on login node without GPU): {exc!r}")
PY

echo "[setup] done. Submit Stage 0 via sbatch (or override CONDA_ENV)."
