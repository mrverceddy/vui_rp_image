#!/bin/bash
# RunPod GPU Pod Setup Script for StoryGen
# Run this once when you create the pod
#
# Install order matters:
# 1. System deps (apt-get)
# 2. Create venv
# 3. PyTorch with CUDA (foundation - must be first)
# 4. Core packages (before HuggingFace stack)
# 5. HuggingFace stack from git (transformers, diffusers, peft) - LAST because Qwen needs newest
# 6. Voice packages
# 7. Training tools (DiffSynth-Studio)
# 8. Download models

set -e

echo "=== StoryGen Pod Setup ==="
echo "Started at: $(date)"

# =============================================================================
# PHASE 1: System Dependencies
# =============================================================================
echo ""
echo "=== Phase 1: System Dependencies ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git-lfs \
    aria2 \
    portaudio19-dev \
    libportaudio2 \
    python3-venv

echo "System dependencies installed."

# =============================================================================
# PHASE 2: Create Virtual Environment
# =============================================================================
echo ""
echo "=== Phase 2: Virtual Environment ==="

VENV_DIR="/workspace/venv"
MODEL_DIR="/workspace/models"

mkdir -p "$MODEL_DIR"

if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv for clean install..."
    rm -rf "$VENV_DIR"
fi

echo "Creating clean virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Activate venv for this script
source "$VENV_DIR/bin/activate"
echo "Using Python: $(which python3)"
echo "Python version: $(python3 --version)"

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# =============================================================================
# PHASE 3: PyTorch with CUDA (MUST BE FIRST - Foundation)
# =============================================================================
echo ""
echo "=== Phase 3: PyTorch 2.5.0 with CUDA 12.1 ==="
echo "This must be installed first as all other ML packages depend on it."

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
echo "Verifying PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# =============================================================================
# PHASE 4: Core Dependencies (before HuggingFace stack)
# =============================================================================
echo ""
echo "=== Phase 4: Core Dependencies ==="

# Web framework
pip install \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles

# Utilities
pip install \
    sentencepiece \
    protobuf \
    scipy \
    soundfile \
    librosa \
    omegaconf \
    einops \
    toml \
    Pillow \
    numpy

echo "Core dependencies installed."

# =============================================================================
# PHASE 5: HuggingFace Stack (MUST BE AFTER PyTorch)
# =============================================================================
echo ""
echo "=== Phase 5: HuggingFace ML Stack ==="
echo "Installing from git for Qwen-Image-Edit-2511 support."
echo "This requires the newest transformers/diffusers with QwenImageEditPlusPipeline."

# First install the supporting packages
pip install \
    "huggingface-hub>=0.30.0" \
    "accelerate>=1.2.0" \
    safetensors

# Then install from git (order matters: transformers -> diffusers -> peft)
echo "Installing transformers from git (for DINOv3, Qwen2.5-VL support)..."
pip install git+https://github.com/huggingface/transformers.git

echo "Installing diffusers from git (for QwenImageEditPlusPipeline)..."
pip install git+https://github.com/huggingface/diffusers.git

echo "Installing peft from git (compatible with transformers git)..."
pip install git+https://github.com/huggingface/peft.git

# Verify HuggingFace stack
echo "Verifying HuggingFace stack..."
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import diffusers; print(f'diffusers: {diffusers.__version__}')"
python3 -c "import peft; print(f'peft: {peft.__version__}')"

# Verify Qwen pipelines are available
echo "Verifying Qwen pipelines..."
python3 -c "from diffusers import QwenImageEditPipeline; print('QwenImageEditPipeline: OK')"
python3 -c "from diffusers import QwenImageEditPlusPipeline; print('QwenImageEditPlusPipeline: OK')"

# =============================================================================
# PHASE 6: Voice Generation Packages
# =============================================================================
echo ""
echo "=== Phase 6: Voice Generation ==="

# Parler-TTS (--no-deps to avoid transformers version conflicts)
echo "Installing Parler-TTS..."
pip install git+https://github.com/huggingface/parler-tts.git --no-deps
pip install descript-audio-codec

# Fish Speech (needs pyaudio which requires portaudio headers)
echo "Installing Fish Speech..."
pip install pyaudio || echo "Warning: pyaudio install failed, Fish Speech voice cloning may not work"
pip install fish-speech || pip install fish-speech --no-deps || echo "Warning: fish-speech install failed"

# =============================================================================
# PHASE 7: Training Tools
# =============================================================================
echo ""
echo "=== Phase 7: Training Tools ==="

# DiffSynth-Studio for Qwen-Image LoRA training (official tool)
if [ ! -d "/workspace/DiffSynth-Studio" ]; then
    echo "Cloning DiffSynth-Studio..."
    git clone https://github.com/modelscope/DiffSynth-Studio.git /workspace/DiffSynth-Studio
fi
echo "Installing DiffSynth-Studio..."
cd /workspace/DiffSynth-Studio && pip install -e .

# Kohya sd-scripts (legacy, kept for compatibility)
if [ ! -d "/workspace/sd-scripts" ]; then
    echo "Cloning Kohya sd-scripts (legacy)..."
    git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts
    cd /workspace/sd-scripts && pip install -r requirements.txt || echo "Warning: sd-scripts requirements failed"
fi

# =============================================================================
# PHASE 8: Cleanup Conflicting Packages
# =============================================================================
echo ""
echo "=== Phase 8: Cleanup ==="

# Remove bitsandbytes - causes triton.ops import error with newer triton versions
echo "Removing bitsandbytes (causes triton compatibility issues)..."
pip uninstall bitsandbytes -y 2>/dev/null || true

# =============================================================================
# PHASE 9: Final Verification
# =============================================================================
echo ""
echo "=== Phase 9: Final Verification ==="

echo "Checking all critical imports..."
python3 << 'VERIFY_EOF'
import sys

def check_import(module, submodule=None):
    try:
        if submodule:
            exec(f"from {module} import {submodule}")
            print(f"  {module}.{submodule}: OK")
        else:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {module}: {version}")
        return True
    except Exception as e:
        print(f"  {module}: FAILED - {e}")
        return False

print("Core ML stack:")
check_import("torch")
check_import("torchvision")
check_import("torchaudio")

print("\nHuggingFace stack:")
check_import("transformers")
check_import("diffusers")
check_import("peft")
check_import("accelerate")
check_import("safetensors")

print("\nQwen pipelines:")
check_import("diffusers", "QwenImageEditPipeline")
check_import("diffusers", "QwenImageEditPlusPipeline")

print("\nVideo pipeline:")
check_import("diffusers", "WanImageToVideoPipeline")

print("\nVoice:")
check_import("parler_tts", "ParlerTTSForConditionalGeneration")

print("\nWeb framework:")
check_import("fastapi")
check_import("uvicorn")
check_import("pydantic")

print("\nUtilities:")
check_import("PIL")
check_import("numpy")
check_import("scipy")

# Test torchvision actually works (catches version mismatch)
print("\nFunctional tests:")
try:
    from torchvision import transforms
    print("  torchvision transforms: OK")
except Exception as e:
    print(f"  torchvision transforms: FAILED - {e}")

try:
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  CUDA: FAILED - {e}")

print("\nVerification complete!")
VERIFY_EOF

# =============================================================================
# PHASE 10: Download Models
# =============================================================================
echo ""
echo "=== Phase 10: Downloading Models ==="
echo "This may take a while (60GB+ total)..."

python3 << 'DOWNLOAD_EOF'
import os
os.environ["HF_HOME"] = "/workspace/models"

from huggingface_hub import snapshot_download, hf_hub_download

print("\n[1/5] Downloading Qwen-Image-Edit-2511 base model...")
snapshot_download(
    "Qwen/Qwen-Image-Edit-2511",
    local_dir="/workspace/models/Qwen/Qwen-Image-Edit-2511"
)

print("\n[2/5] Downloading FP8 transformer (20GB, for faster inference)...")
hf_hub_download(
    repo_id="drbaph/Qwen-Image-Edit-2511-FP8",
    filename="qwen_image_edit_2511_fp8_e4m3fn.safetensors",
    local_dir="/workspace/models/qwen-edit-2511-fp8"
)

print("\n[3/5] Downloading Wan 2.1 FLF2V (First-Last-Frame to Video)...")
snapshot_download(
    "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    local_dir="/workspace/models/wan-flf2v"
)

print("\n[4/5] Downloading Parler-TTS...")
snapshot_download(
    "parler-tts/parler-tts-mini-v1",
    local_dir="/workspace/models/parler-tts"
)

print("\n[5/5] Downloading Fish Speech...")
snapshot_download(
    "fishaudio/fish-speech-1.4",
    local_dir="/workspace/models/fish-speech"
)

print("\n[Bonus] Downloading Multiple Angles LoRA...")
snapshot_download(
    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    local_dir="/workspace/models/loras/multiple-angles"
)

print("\n=== All models downloaded! ===")
DOWNLOAD_EOF

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=========================================="
echo "=== Setup Complete! ==="
echo "=========================================="
echo ""
echo "Models: /workspace/models"
echo "Venv:   /workspace/venv"
echo ""
echo "To start the server:"
echo "  source /workspace/venv/bin/activate && python /workspace/vui_rp_image/pod_server.py"
echo ""
echo "Or use the start script:"
echo "  bash /workspace/vui_rp_image/start.sh"
echo ""
echo "Setup finished at: $(date)"
