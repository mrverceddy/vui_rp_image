#!/bin/bash
# RunPod Bootstrap Script for StoryGen GPU Server
#
# Use this as the "Start Command" when creating a new RunPod pod,
# or run it manually on first setup.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/mrverceddy/vui_rp_image/main/bootstrap.sh | bash

set -e

echo "=============================================="
echo "  StoryGen GPU Server - RunPod Bootstrap"
echo "=============================================="
echo "Started at: $(date)"
echo ""

# Configuration
REPO_URL="${STORYGEN_REPO:-https://github.com/mrverceddy/vui_rp_image.git}"
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/vui_rp_image"

# Create workspace if needed
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# Clone or update repository
if [ -d "$REPO_DIR/.git" ]; then
    echo ">>> Repository exists, pulling latest..."
    cd "$REPO_DIR"
    git fetch origin
    git reset --hard origin/main || git reset --hard origin/master
    git pull
    echo "Repository updated!"
else
    echo ">>> Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    echo "Repository cloned!"
fi

echo ""

# Check if this is first-time setup (models not downloaded)
if [ ! -d "$WORKSPACE/models/qwen-image" ]; then
    echo ">>> First-time setup detected - running full setup..."
    echo "This will install dependencies and download models (~40GB)"
    echo ""
    bash pod_setup.sh
else
    echo ">>> Models found, running quick start..."
    echo ""
    # Just ensure deps are installed and start server
    bash start.sh
fi
