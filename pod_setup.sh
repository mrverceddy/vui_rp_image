#!/bin/bash
# RunPod GPU Pod Setup Script for StoryGen
# Run this once when you create the pod
#
# Order of operations:
# 1. System dependencies
# 2. Create venv + install PyTorch
# 3. Install custom model libs (parler-tts, fish-speech) with --no-deps
# 4. Install training tools (DiffSynth-Studio)
# 5. Download all models
# 6. Install Qwen pipeline dependencies LAST (transformers, diffusers, peft from git)
# 7. Verify everything works

set -e

echo "=============================================="
echo "=== StoryGen Pod Setup ==="
echo "=============================================="

# Create model directory (use /workspace for persistence)
MODEL_DIR="/workspace/models"
mkdir -p $MODEL_DIR

# =============================================================================
# PHASE 1: System Dependencies
# =============================================================================
echo ""
echo "=== Phase 1: System Dependencies ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git-lfs aria2 portaudio19-dev libportaudio2 python3-venv

# =============================================================================
# PHASE 2: Virtual Environment + PyTorch
# =============================================================================
echo ""
echo "=== Phase 2: Virtual Environment + PyTorch ==="

VENV_DIR="/workspace/venv"

# Always recreate venv to avoid conflicts
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv to ensure clean state..."
    rm -rf "$VENV_DIR"
fi

echo "Creating clean virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Activate venv for this script
source "$VENV_DIR/bin/activate"
echo "Using Python: $(which python)"
echo "Using pip: $(which pip)"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch 2.5 with CUDA 12.1
echo "Installing PyTorch 2.5 with CUDA 12.1..."
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Install base dependencies that everything needs
pip install \
    "huggingface-hub>=0.25.0" \
    "accelerate>=1.2.0" \
    safetensors \
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
    python-multipart \
    pillow

# =============================================================================
# PHASE 3: Custom Model Libraries (with --no-deps to prevent conflicts)
# =============================================================================
echo ""
echo "=== Phase 3: Custom Model Libraries ==="

# Parler-TTS - install without deps to avoid transformers version conflicts
echo "Installing Parler-TTS (--no-deps)..."
pip install git+https://github.com/huggingface/parler-tts.git --no-deps
pip install descript-audio-codec

# Fish Speech - install without deps
echo "Installing Fish Speech (--no-deps)..."
pip install pyaudio || echo "Warning: pyaudio install failed, continuing..."
pip install fish-speech --no-deps || echo "Warning: fish-speech install failed, continuing..."

# =============================================================================
# PHASE 4: Training Tools
# =============================================================================
echo ""
echo "=== Phase 4: Training Tools ==="

# DiffSynth-Studio for Qwen-Image LoRA training
if [ ! -d "/workspace/DiffSynth-Studio" ]; then
    echo "Cloning DiffSynth-Studio..."
    git clone https://github.com/modelscope/DiffSynth-Studio.git /workspace/DiffSynth-Studio
fi
echo "Installing DiffSynth-Studio (--no-deps to prevent conflicts)..."
cd /workspace/DiffSynth-Studio && pip install -e . --no-deps
cd /workspace

# Kohya sd-scripts (optional, legacy)
if [ ! -d "/workspace/sd-scripts" ]; then
    echo "Cloning Kohya sd-scripts..."
    git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts
fi

# =============================================================================
# PHASE 5: Download Models
# =============================================================================
echo ""
echo "=== Phase 5: Downloading Models ==="
echo "This may take a while on first run (~60GB total)..."

python3 << 'EOF'
import os
os.environ["HF_HOME"] = "/workspace/models"

from huggingface_hub import snapshot_download, hf_hub_download

# Qwen-Image-Edit-2511 base model (BF16)
# Provides VAE, text encoder, tokenizer
print("\n[1/6] Downloading Qwen-Image-Edit-2511 base model (~40GB)...")
snapshot_download(
    "Qwen/Qwen-Image-Edit-2511",
    local_dir="/workspace/models/Qwen/Qwen-Image-Edit-2511"
)

# FP8 quantized transformer for inference
print("\n[2/6] Downloading FP8 transformer (~20GB)...")
hf_hub_download(
    repo_id="drbaph/Qwen-Image-Edit-2511-FP8",
    filename="qwen_image_edit_2511_fp8_e4m3fn.safetensors",
    local_dir="/workspace/models/qwen-edit-2511-fp8"
)

# Wan 2.1 FLF2V for video generation
print("\n[3/6] Downloading Wan 2.1 FLF2V...")
snapshot_download(
    "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    local_dir="/workspace/models/wan-flf2v"
)

# Parler-TTS for voice generation
print("\n[4/6] Downloading Parler-TTS...")
snapshot_download(
    "parler-tts/parler-tts-mini-v1",
    local_dir="/workspace/models/parler-tts"
)

# Fish Speech for voice cloning
print("\n[5/6] Downloading Fish Speech...")
snapshot_download(
    "fishaudio/fish-speech-1.4",
    local_dir="/workspace/models/fish-speech"
)

# Multiple Angles LoRA for camera control
print("\n[6/6] Downloading Multiple Angles LoRA...")
snapshot_download(
    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    local_dir="/workspace/models/loras/multiple-angles"
)

print("\n=== All models downloaded! ===")
EOF

# =============================================================================
# PHASE 6: Qwen Pipeline Dependencies (MUST BE LAST)
# =============================================================================
echo ""
echo "=== Phase 6: Qwen Pipeline Dependencies (installing last to prevent overwrites) ==="

# Remove any conflicting packages
pip uninstall bitsandbytes optimum-quanto -y 2>/dev/null || true

# Install latest diffusers, transformers, peft from git
# These MUST be installed last as other packages may try to downgrade them
echo "Installing diffusers from git..."
pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

echo "Installing transformers from git..."
pip install --no-cache-dir git+https://github.com/huggingface/transformers.git

echo "Installing peft from git..."
pip install --no-cache-dir git+https://github.com/huggingface/peft.git

# =============================================================================
# PHASE 7: Verification
# =============================================================================
echo ""
echo "=== Phase 7: Verification ==="

echo "Checking installed versions..."
python3 << 'EOF'
import sys

def check_import(name, package=None):
    package = package or name
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: FAILED - {e}")
        return False

print("\nCore packages:")
check_import("torch")
check_import("torchvision")
check_import("transformers")
check_import("diffusers")
check_import("peft")
check_import("accelerate")
check_import("safetensors")

print("\nQwen pipeline test:")
try:
    from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline
    print("  ✓ QwenImageEditPipeline available")
    print("  ✓ QwenImageEditPlusPipeline available")
except ImportError as e:
    print(f"  ✗ Qwen pipelines: FAILED - {e}")
    sys.exit(1)

print("\nFP8 loading test:")
try:
    from safetensors.torch import load_file
    print("  ✓ safetensors.torch.load_file available")
except ImportError as e:
    print(f"  ✗ safetensors: FAILED - {e}")

print("\nOptional packages:")
check_import("parler_tts", "parler_tts")
check_import("fish_speech", "fish_speech")

print("\n=== Verification complete ===")
EOF

echo ""
echo "=============================================="
echo "=== Setup Complete ==="
echo "=============================================="
echo ""
echo "Models are in: /workspace/models"
echo "Virtual environment: /workspace/venv"
echo ""
echo "Architecture: Qwen-Image-Edit-2511 with FP8 inference"
echo "  - FP8 transformer: ~24GB VRAM"
echo "  - Base BF16 model provides VAE, text encoder, tokenizer"
echo "  - LoRAs work via automatic upcasting"
echo ""
echo "To start the server:"
echo "  source /workspace/venv/bin/activate"
echo "  python /workspace/vui_rp_image/pod_server.py"
echo ""
echo "Or use: bash /workspace/vui_rp_image/start.sh"
echo ""
