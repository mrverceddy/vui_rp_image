#!/bin/bash
# Quick update script - pulls latest code and restarts server
# Run this when you've pushed changes to GitHub

set -e

echo "=== Updating StoryGen GPU Server ==="
echo ""

cd /workspace/vui_rp_image

# Pull latest
echo "Pulling latest from GitHub..."
git fetch origin
git reset --hard origin/main || git reset --hard origin/master
echo "Updated!"

echo ""
echo "Restarting server..."
echo ""

# Start server
export MODEL_DIR="/workspace/models"
export HF_HOME="/workspace/models"
python pod_server.py
