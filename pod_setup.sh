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

# Install Python dependencies
pip install --upgrade pip
pip install \
    fastapi \
    uvicorn \
    torch \
    torchaudio \
    torchvision \
    transformers>=4.45.0 \
    accelerate \
    diffusers>=0.32.0 \
    sentencepiece \
    protobuf \
    scipy \
    soundfile \
    librosa \
    huggingface_hub \
    safetensors \
    omegaconf \
    einops \
    toml \
    bitsandbytes \
    peft \
    aiofiles \
    python-multipart

# Parler-TTS
pip install git+https://github.com/huggingface/parler-tts.git

# Fish Speech
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

print("Downloading Flux Dev...")
snapshot_download("black-forest-labs/FLUX.1-dev", local_dir="/workspace/models/flux-dev")

print("Downloading Wan 2.2 720p...")
snapshot_download("Wan-AI/Wan2.2-I2V-14B-720P", local_dir="/workspace/models/wan-2.2-720p")

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
echo "  python /workspace/pod_server.py"
echo ""
