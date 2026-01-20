#!/bin/bash
# RunPod GPU Pod Setup Script for StoryGen
# Run this once when you create the pod

set -e

echo "=== StoryGen Pod Setup ==="

# Create model directory (use /workspace for persistence)
MODEL_DIR="/workspace/models"
mkdir -p $MODEL_DIR

# Install system dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git-lfs aria2 portaudio19-dev libportaudio2 python3-venv

# Create clean virtual environment to isolate from broken system packages
VENV_DIR="/workspace/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating clean virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv for this script
source "$VENV_DIR/bin/activate"
echo "Using Python: $(which python3)"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch 2.5 with CUDA 12.1 (required for Qwen-Image enable_gqa)
echo "Installing PyTorch 2.5..."
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Install diffusers stack with compatible versions
# transformers from git requires huggingface-hub>=1.3.0
echo "Installing diffusers and transformers..."
pip install git+https://github.com/huggingface/diffusers.git
pip install git+https://github.com/huggingface/transformers.git
pip install \
    "huggingface-hub>=1.3.0" \
    "accelerate>=1.2.0" \
    safetensors
# peft from git for compatibility with transformers from git
pip install git+https://github.com/huggingface/peft.git

# Install other dependencies
# NOTE: bitsandbytes removed - causes triton.ops conflict and we don't need 8-bit quantization
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
    aiofiles \
    python-multipart

# Parler-TTS (note: may have transformers version conflict, but still works)
echo "Installing Parler-TTS..."
pip install git+https://github.com/huggingface/parler-tts.git --no-deps
pip install descript-audio-codec

# Fish Speech (requires pyaudio which needs portaudio headers)
echo "Installing Fish Speech..."
# Install pyaudio first (needs portaudio19-dev from apt-get above)
pip install pyaudio || echo "Warning: pyaudio install failed, continuing..."
pip install fish-speech || pip install fish-speech --no-deps

# Kohya sd-scripts for SDXL LoRA training (legacy, kept for compatibility)
if [ ! -d "/workspace/sd-scripts" ]; then
    git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts
    cd /workspace/sd-scripts && pip install -r requirements.txt
fi

# DiffSynth-Studio for Qwen-Image LoRA training (official)
if [ ! -d "/workspace/DiffSynth-Studio" ]; then
    echo "Cloning DiffSynth-Studio for Qwen-Image LoRA training..."
    git clone https://github.com/modelscope/DiffSynth-Studio.git /workspace/DiffSynth-Studio
fi
# Always ensure DiffSynth is installed (in case clone succeeded but install failed)
echo "Installing DiffSynth-Studio..."
cd /workspace/DiffSynth-Studio && pip install -e .

# FlyMyAI LoRA trainer (alternative, simpler)
if [ ! -d "/workspace/flymyai-lora-trainer" ]; then
    echo "Installing FlyMyAI LoRA trainer..."
    git clone https://github.com/FlyMyAI/flymyai-lora-trainer.git /workspace/flymyai-lora-trainer
    cd /workspace/flymyai-lora-trainer && pip install -r requirements.txt
fi

# IMPORTANT: Install ML stack in clean venv (no system package conflicts)
echo "=== Installing ML stack in venv ==="
pip install --no-cache-dir torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Install other ML dependencies
pip install --no-cache-dir "huggingface-hub>=0.30.0" "accelerate>=1.2.0"
# transformers from git for DINOv3 support (required by DiffSynth-Studio)
pip install --no-cache-dir git+https://github.com/huggingface/transformers.git
pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git
# peft from git for compatibility with transformers 5.0.0.dev0
pip install --no-cache-dir git+https://github.com/huggingface/peft.git

# Verify versions are compatible
echo "Verifying ML stack versions..."
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import huggingface_hub; print(f'huggingface_hub: {huggingface_hub.__version__}')"
python3 -c "import torch; print(f'torch: {torch.__version__}')"
python3 -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"

# Test torchvision actually works (catches version mismatch errors)
echo "Testing torchvision import..."
python3 -c "from torchvision import transforms; print('torchvision OK')" || {
    echo "ERROR: torchvision broken, attempting full reinstall..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
    python3 -c "from torchvision import transforms; print('torchvision OK after reinstall')"
}

# CRITICAL: Remove bitsandbytes if installed - causes triton.ops import error
# bitsandbytes tries to import triton.ops which doesn't exist in newer triton versions
echo "Removing bitsandbytes (causes triton compatibility issues)..."
pip uninstall bitsandbytes -y 2>/dev/null || true

echo "=== Downloading Models ==="

# Download models (this takes a while first time)
# Architecture: Unified Qwen-Image-Edit-2511-Lightning for ALL image generation
# - Text-to-image: Use gray (128,128,128) input + prompt
# - Image editing: Use reference image + edit prompt
# - With LoRA: Same model + character/scene LoRA for consistency
python3 << 'EOF'
import os
os.environ["HF_HOME"] = "/workspace/models"

from huggingface_hub import snapshot_download

# Qwen-Image-Edit-2511 with FP8 transformer for faster training
# Download base model (for VAE, text encoder, configs)
print("Downloading Qwen-Image-Edit-2511 base model...")
snapshot_download("Qwen/Qwen-Image-Edit-2511", local_dir="/workspace/models/qwen-edit-2511")

# Download FP8 quantized transformer (replaces BF16 transformer for faster training)
print("Downloading FP8 transformer (20GB, faster training)...")
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="drbaph/Qwen-Image-Edit-2511-FP8",
    filename="qwen_image_edit_2511_fp8_e4m3fn.safetensors",
    local_dir="/workspace/models/qwen-edit-2511-fp8"
)

# Wan 2.1 FLF2V - First-Last-Frame to Video (for scene animation)
print("Downloading Wan 2.1 FLF2V (First-Last-Frame to Video)...")
snapshot_download("Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers", local_dir="/workspace/models/wan-flf2v")

print("Downloading Parler-TTS...")
snapshot_download("parler-tts/parler-tts-mini-v1", local_dir="/workspace/models/parler-tts")

print("Downloading Fish Speech...")
snapshot_download("fishaudio/fish-speech-1.4", local_dir="/workspace/models/fish-speech")

print("Downloading LoRAs...")

# Multiple Angles LoRA for scene camera control (compatible with Lightning)
print("  - Multiple Angles LoRA (scene camera control)...")
snapshot_download(
    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    local_dir="/workspace/models/loras/multiple-angles"
)

print("=== All models downloaded! ===")
print("")
print("Architecture: Qwen-Image-Edit-2511 with FP8 training")
print("  - Base model: BF16 (for inference)")
print("  - FP8 transformer: For faster LoRA training (~16GB VRAM vs 40GB)")
print("  - LoRAs trained work for BOTH text-to-image AND editing")
EOF

echo ""
echo "=== Setup Complete ==="
echo "Models are in: /workspace/models"
echo "Virtual environment: /workspace/venv"
echo ""
echo "To start the server, run:"
echo "  source /workspace/venv/bin/activate && python /workspace/vui_rp_image/pod_server.py"
echo ""
echo "Or use the start script:"
echo "  bash /workspace/vui_rp_image/start.sh"
echo ""
