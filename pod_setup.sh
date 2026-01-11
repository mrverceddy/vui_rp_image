#!/bin/bash
# RunPod GPU Pod Setup Script for StoryGen
# Run this once when you create the pod

set -e

echo "=== StoryGen Pod Setup ==="

# Create model directory (use /workspace for persistence)
MODEL_DIR="/workspace/models"
mkdir -p $MODEL_DIR

# Install system dependencies
apt-get update && apt-get install -y ffmpeg libsndfile1 git-lfs aria2 portaudio19-dev

# Upgrade pip
pip install --upgrade pip

# Install PyTorch 2.4 with CUDA 12.1 (required for latest diffusers)
echo "Installing PyTorch 2.4..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install diffusers stack with compatible versions
echo "Installing diffusers and transformers..."
pip install \
    diffusers>=0.36.0 \
    transformers>=4.50.0 \
    huggingface_hub>=0.30.0 \
    accelerate>=1.0.0 \
    safetensors \
    peft

# Install other dependencies
pip install \
    fastapi \
    uvicorn \
    sentencepiece \
    protobuf \
    scipy \
    soundfile \
    librosa \
    omegaconf \
    einops \
    toml \
    bitsandbytes \
    aiofiles \
    python-multipart

# Parler-TTS (note: may have transformers version conflict, but still works)
echo "Installing Parler-TTS..."
pip install git+https://github.com/huggingface/parler-tts.git --no-deps
pip install descript-audio-codec

# Fish Speech
echo "Installing Fish Speech..."
pip install fish-speech

# Kohya sd-scripts for LoRA training
if [ ! -d "/workspace/sd-scripts" ]; then
    git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts
    cd /workspace/sd-scripts && pip install -r requirements.txt
fi

echo "=== Downloading Models ==="

# Download models (this takes a while first time)
python3 << 'EOF'
import os
os.environ["HF_HOME"] = "/workspace/models"

from huggingface_hub import snapshot_download

print("Downloading Qwen-Image-2512...")
snapshot_download("Qwen/Qwen-Image-2512", local_dir="/workspace/models/qwen-image")

print("Downloading Wan 2.2 I2V...")
snapshot_download("Wan-AI/Wan2.2-I2V-A14B", local_dir="/workspace/models/wan-i2v")

print("Downloading Parler-TTS...")
snapshot_download("parler-tts/parler-tts-mini-v1", local_dir="/workspace/models/parler-tts")

print("Downloading Fish Speech...")
snapshot_download("fishaudio/fish-speech-1.4", local_dir="/workspace/models/fish-speech")

print("=== All models downloaded! ===")
EOF

echo ""
echo "=== Setup Complete ==="
echo "Models are in: /workspace/models"
echo ""
echo "To start the server, run:"
echo "  python /workspace/vui_rp_image/pod_server.py"
echo ""
