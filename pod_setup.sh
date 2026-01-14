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

# Install PyTorch 2.5 with CUDA 12.1 (required for Qwen-Image enable_gqa)
echo "Installing PyTorch 2.5..."
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Install diffusers stack with compatible versions
echo "Installing diffusers and transformers..."
# Diffusers from git for QwenImagePipeline support
pip install git+https://github.com/huggingface/diffusers.git
pip install \
    transformers>=4.51.3 \
    huggingface_hub>=0.30.0 \
    accelerate>=1.2.0 \
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

# IMPORTANT: Re-install core ML stack LAST to ensure compatibility
# (other packages like kohya may install older/incompatible versions)
echo "Re-installing latest ML stack (diffusers, transformers, accelerate)..."
pip install --upgrade transformers huggingface-hub accelerate
pip install --force-reinstall git+https://github.com/huggingface/diffusers.git

echo "=== Downloading Models ==="

# Download models (this takes a while first time)
python3 << 'EOF'
import os
os.environ["HF_HOME"] = "/workspace/models"

from huggingface_hub import snapshot_download

print("Downloading Qwen-Image-2512 (text-to-image)...")
snapshot_download("Qwen/Qwen-Image-2512", local_dir="/workspace/models/qwen-image")

print("Downloading Qwen-Image-Edit-2511 (img2img editing)...")
snapshot_download("Qwen/Qwen-Image-Edit-2511", local_dir="/workspace/models/qwen-image-edit")

print("Downloading Wan 2.2 I2V...")
snapshot_download("Wan-AI/Wan2.2-I2V-A14B", local_dir="/workspace/models/wan-i2v")

print("Downloading Parler-TTS...")
snapshot_download("parler-tts/parler-tts-mini-v1", local_dir="/workspace/models/parler-tts")

print("Downloading Fish Speech...")
snapshot_download("fishaudio/fish-speech-1.4", local_dir="/workspace/models/fish-speech")

print("Downloading LoRAs...")

# Multiple Angles LoRA for scene camera control
print("  - Multiple Angles LoRA (scene camera control)...")
snapshot_download(
    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    local_dir="/workspace/models/loras/multiple-angles"
)

# Turbo LoRA for 20x faster text-to-image (4-8 steps instead of 30-40)
print("  - Turbo LoRA (speed boost)...")
snapshot_download(
    "Wuli-art/Qwen-Image-2512-Turbo-LoRA",
    local_dir="/workspace/models/loras/turbo"
)

print("=== All models downloaded! ===")
EOF

echo ""
echo "=== Setup Complete ==="
echo "Models are in: /workspace/models"
echo ""
echo "To start the server, run:"
echo "  python /workspace/vui_rp_image/pod_server.py"
echo ""
