#!/bin/bash
# Setup Python environment on Bocconi HPC

set -e

echo "=========================================="
echo "Setting up Python environment on HPC"
echo "=========================================="

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "To activate manually:"
echo "  source .venv/bin/activate"
