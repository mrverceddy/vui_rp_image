#!/bin/bash
# Quick start script for RunPod pod
# Activates venv and starts the server

set -e
echo "=== StoryGen Pod Server ==="
echo "Started at: $(date)"

VENV_DIR="/workspace/venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Please run 'bash pod_setup.sh' first"
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "Using Python: $(which python)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"

# Check models
if [ ! -d "/workspace/models/Qwen/Qwen-Image-Edit-2511" ]; then
    echo ""
    echo "WARNING: Qwen-Image-Edit-2511 model not found"
    echo "Run 'bash pod_setup.sh' first to download models"
    echo ""
fi

# Create output dir
mkdir -p /workspace/outputs

# Start server
echo ""
echo "=== Starting Server ==="
echo "Server will be at: http://0.0.0.0:8000"
echo ""

cd /workspace/vui_rp_image
export MODEL_DIR="/workspace/models"
export HF_HOME="/workspace/models"

python pod_server.py
