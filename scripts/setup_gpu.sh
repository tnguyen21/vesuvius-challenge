#!/bin/bash
# VM GPU setup script for Vesuvius Challenge
#
# Usage:
#   ./scripts/setup_gpu.sh
#
# This script:
#   1. Checks NVIDIA driver and CUDA availability
#   2. Sets up the Python environment with uv
#   3. Installs JAX with CUDA support
#   4. Verifies GPU is accessible from Python

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Vesuvius Challenge - GPU Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check for NVIDIA driver
echo -e "${YELLOW}Checking NVIDIA driver...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo ""
else
    echo -e "${RED}ERROR: nvidia-smi not found. Is the NVIDIA driver installed?${NC}"
    exit 1
fi

# Check CUDA version
echo -e "${YELLOW}Checking CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "nvcc not in PATH (this is okay if using containerized CUDA)"
fi
echo ""

# Check for uv
echo -e "${YELLOW}Checking uv package manager...${NC}"
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
uv --version
echo ""

# Create/sync virtual environment
echo -e "${YELLOW}Setting up Python environment...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12
fi

# Sync dependencies with training extras (includes JAX CUDA, Keras, medicai)
echo "Installing dependencies with training extras..."
uv sync --extra train

echo ""

# Verify JAX can see GPU
echo -e "${YELLOW}Verifying JAX GPU access...${NC}"
uv run python -c "
import jax
print(f'JAX version: {jax.__version__}')
devices = jax.devices()
print(f'Available devices: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'GPU devices found: {len(gpu_devices)}')
    for i, d in enumerate(gpu_devices):
        print(f'  GPU {i}: {d}')
else:
    print('WARNING: No GPU devices found by JAX!')
    print('You may need to install jax[cuda12] or check CUDA setup')
"

echo ""

# Verify Keras backend
echo -e "${YELLOW}Verifying Keras JAX backend...${NC}"
uv run python -c "
import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras
print(f'Keras version: {keras.__version__}')
print(f'Keras backend: {keras.backend.backend()}')
"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GPU Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run an experiment:"
echo "  ./scripts/run_experiment.sh baseline --epochs 10"
