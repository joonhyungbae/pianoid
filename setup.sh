#!/bin/bash
# Pianoid: Environment Setup Script
# Creates conda environment 'pianoid' with all required dependencies
#
# Usage:
#   bash setup.sh           # Full setup (create env + install packages)
#   bash setup.sh --update  # Update existing environment

set -e

ENV_NAME="pianoid"
PYTHON_VERSION="3.11"  # ARIA requires Python >= 3.11

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Pianoid: Environment Setup${NC}"
echo -e "${GREEN}============================================================${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Parse arguments
UPDATE_ONLY=false
for arg in "$@"; do
    if [ "$arg" == "--update" ]; then
        UPDATE_ONLY=true
    fi
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if environment exists
ENV_EXISTS=$(conda env list | grep "^${ENV_NAME} " || true)

if [ "$UPDATE_ONLY" = true ]; then
    if [ -z "$ENV_EXISTS" ]; then
        echo -e "${RED}Error: Environment '$ENV_NAME' does not exist.${NC}"
        echo "Run 'bash setup.sh' first to create the environment."
        exit 1
    fi
    echo -e "${YELLOW}Updating existing environment '$ENV_NAME'...${NC}"
else
    if [ -n "$ENV_EXISTS" ]; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists.${NC}"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n $ENV_NAME -y
        else
            echo "Updating existing environment instead..."
            UPDATE_ONLY=true
        fi
    fi
fi

# Create new environment if not updating
if [ "$UPDATE_ONLY" = false ]; then
    echo ""
    echo -e "${GREEN}>>> Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION...${NC}"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# Activate environment
echo ""
echo -e "${GREEN}>>> Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch with CUDA support
echo ""
echo -e "${GREEN}>>> Installing PyTorch with CUDA support...${NC}"
# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA..."
    # Install PyTorch with CUDA 11.8 (compatible with most setups)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo ""
echo -e "${GREEN}>>> Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt

# Install additional useful packages
echo ""
echo -e "${GREEN}>>> Installing additional packages...${NC}"
pip install ipython jupyter  # For interactive development

# Install ARIA package (OPTIONAL)
#
# The main Pianoid pipeline (Stages 1-4) and the paper's final numbers do NOT
# use ARIA — the paper's Table 3(D) reports ARIA as a Stage-4 ablation that
# degrades performance, and no production script passes --use_aria. ARIA is
# only required to reproduce that single Table 3(D) ablation row or to rerun
# the L4 input-level variant of Stage 1.
#
# Opt in by setting INSTALL_ARIA=1 before running setup.sh:
#   INSTALL_ARIA=1 bash setup.sh
if [ "${INSTALL_ARIA:-0}" = "1" ]; then
    echo ""
    echo -e "${GREEN}>>> Installing ARIA package (opt-in via INSTALL_ARIA=1)...${NC}"
    if [ -d "vendor/aria" ]; then
        cd vendor/aria
        pip install -e ".[all]"
        cd "$SCRIPT_DIR"
    else
        echo -e "${YELLOW}Warning: vendor/aria not found. ARIA will need to be installed manually.${NC}"
    fi

    echo ""
    echo -e "${GREEN}>>> Downloading ARIA model checkpoint...${NC}"
    mkdir -p models
    if [ ! -f "models/aria-medium-embedding.safetensors" ]; then
        wget -q --show-progress https://huggingface.co/loubb/aria-medium-embedding/resolve/main/model.safetensors \
            -O models/aria-medium-embedding.safetensors
        echo "ARIA model downloaded successfully."
    else
        echo "ARIA model already exists, skipping download."
    fi
else
    echo ""
    echo -e "${YELLOW}>>> Skipping ARIA (opt-in via INSTALL_ARIA=1; only needed for Table 3(D) ablation).${NC}"
fi

# Verify installation
echo ""
echo -e "${GREEN}>>> Verifying installation...${NC}"
python -c "
import torch
import numpy as np
import pandas as pd
import scipy
import matplotlib
import tqdm
import yaml
import wandb
import moviepy

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'SciPy version: {scipy.__version__}')
print(f'Wandb version: {wandb.__version__}')
print(f'MoviePy version: {moviepy.__version__}')

# Check ARIA (optional — only needed for Table 3(D) ablation)
try:
    from aria.model import TransformerEMB
    print('ARIA: installed (optional; only used by Table 3(D) ablation)')
except ImportError:
    print('ARIA: not installed (optional; enable with INSTALL_ARIA=1 bash setup.sh)')

print()
print('All dependencies installed successfully!')
"

# Print activation instructions
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "To activate the environment, run:"
echo -e "  ${YELLOW}conda activate $ENV_NAME${NC}"
echo ""
echo "To reproduce paper numbers (uses released checkpoints, ~10–20 min):"
echo -e "  ${YELLOW}bash inference.sh${NC}"
echo ""
echo "To train from scratch (~3–5 days):"
echo -e "  ${YELLOW}bash train.sh${NC}"
echo ""
