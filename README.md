# VUI RunPod GPU Server

Unified GPU server for image, video, and voice generation.

## Setup on RunPod

1. Create GPU Pod (A100 80GB recommended)
2. Clone this repo:
   ```bash
   cd /workspace
   git clone https://github.com/YOUR_USER/vui_rp_image.git
   cd vui_rp_image
   ```

3. Run setup (downloads models ~40GB):
   ```bash
   chmod +x pod_setup.sh
   ./pod_setup.sh
   ```

4. Start server:
   ```bash
   python pod_server.py
   ```

Server runs at `http://POD_IP:8000`

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /generate_image` | Flux text-to-image |
| `POST /generate_video` | Wan 2.2 image-to-video |
| `POST /generate_voice` | Parler-TTS voice generation |
| `POST /synthesize_voice` | Fish Speech voice cloning |
| `POST /train_lora` | Kohya LoRA training |
| `GET /health` | GPU status |

## Models Included

- Flux Dev (images)
- Wan 2.2 720p (video)
- Parler-TTS (voice)
- Fish Speech (voice cloning)
- Kohya sd-scripts (LoRA training)
