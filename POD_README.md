# RunPod GPU Pod Setup

Run everything on a single GPU pod (not serverless).

## Step 1: Create Pod

1. Go to [runpod.io/console/pods](https://www.runpod.io/console/pods)
2. Click **+ GPU Pod**
3. Select:
   - **GPU:** A100 80GB (or H100)
   - **Template:** RunPod Pytorch 2.2
   - **Disk:** 100GB (for models)
4. Click **Deploy**

## Step 2: Connect to Pod

Once running, click **Connect** → **Start Web Terminal** (or use SSH)

## Step 3: Upload Setup Files

In the terminal:
```bash
cd /workspace

# Option A: Clone your repo
git clone https://github.com/YOUR_USER/storygen.git
cd storygen/runpod

# Option B: Download files directly
wget https://raw.githubusercontent.com/YOUR_USER/storygen/main/runpod/pod_setup.sh
wget https://raw.githubusercontent.com/YOUR_USER/storygen/main/runpod/pod_server.py
```

## Step 4: Run Setup (Downloads Models)

```bash
chmod +x pod_setup.sh
./pod_setup.sh
```

This takes ~20-30 min (downloads ~40GB of models).

## Step 5: Start Server

```bash
python pod_server.py
```

Server runs at `http://YOUR_POD_IP:8000`

## Step 6: Configure StoryGen

Get your pod's IP from the RunPod dashboard, then in your `.env`:
```bash
RUNPOD_POD_URL=http://YOUR_POD_IP:8000
OPENROUTER_API_KEY=your_key
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate_image` | POST | Flux text-to-image |
| `/generate_video` | POST | Wan 2.2 image-to-video |
| `/generate_voice` | POST | Parler-TTS voice |
| `/synthesize_voice` | POST | Fish Speech clone |
| `/train_lora` | POST | Kohya LoRA training |
| `/health` | GET | Check GPU status |
| `/clear_vram` | POST | Unload models |
| `/download/{file}` | GET | Download output file |

---

## Test It

```bash
# Check health
curl http://YOUR_POD_IP:8000/health

# Generate image
curl -X POST http://YOUR_POD_IP:8000/generate_image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a penguin wearing a top hat", "width": 512, "height": 512}'
```

---

## Keep Server Running

Use `screen` or `tmux` to keep server running after disconnect:

```bash
# Start in screen
screen -S storygen
python pod_server.py

# Detach: Ctrl+A, then D
# Reattach: screen -r storygen
```

---

## Cost

| GPU | Hourly |
|-----|--------|
| A100 80GB | ~$1.89 |
| H100 | ~$3.89 |

Stop the pod when not using it!
