#!/bin/bash
# Quick start script for RunPod pod
# Run after pod reboot to reinstall deps and start server

set -e
echo "=== StoryGen Pod Quick Start ==="
echo "Started at: $(date)"

# Check if deps already installed
if ! python -c "import diffusers" 2>/dev/null; then
    echo "Installing dependencies..."

    pip install -q --upgrade pip

    # PyTorch 2.5 with CUDA 12.1
    pip install -q torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

    # Core deps - use git versions for Qwen2.5-VL compatibility
    pip install -q \
        git+https://github.com/huggingface/diffusers.git \
        git+https://github.com/huggingface/transformers.git \
        "huggingface-hub>=0.32.0,<1.0" \
        "accelerate>=1.2.0" \
        safetensors \
        peft \
        fastapi \
        uvicorn \
        sentencepiece \
        protobuf \
        einops \
        omegaconf

    echo "Dependencies installed!"
else
    echo "Dependencies already installed, skipping..."
fi

# Create output dir
mkdir -p /workspace/outputs

# Check models
if [ ! -d "/workspace/models/qwen-image" ]; then
    echo ""
    echo "WARNING: Models not found in /workspace/models"
    echo "Run 'bash pod_setup.sh' first to download models"
    echo ""
fi

# Start server
echo ""
echo "=== Starting Server ==="
echo "Server will be at: http://0.0.0.0:8000"
echo ""

cd /workspace/vui_rp_image
export MODEL_DIR="/workspace/models"
export HF_HOME="/workspace/models"

python pod_server.py
