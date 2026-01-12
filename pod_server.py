"""
StoryGen GPU Server for RunPod Pods.

Run with: python pod_server.py
Access at: http://POD_IP:8000

All GPU tasks available via REST API with async job queue.
"""

import base64
import gc
import io
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from uuid import uuid4

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Config
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
OUTPUT_DIR = Path("/workspace/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="StoryGen GPU Server")

# Global model cache
_models = {}

# Job queue system
_jobs = {}  # job_id -> {status, progress, result, error, created_at}
_executor = ThreadPoolExecutor(max_workers=1)  # Single worker for GPU tasks
_job_lock = threading.Lock()


def clear_vram():
    """Unload all models to free VRAM."""
    global _models
    for name in list(_models.keys()):
        del _models[name]
    _models.clear()
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# JOB QUEUE SYSTEM
# =============================================================================

def create_job() -> str:
    """Create a new job and return its ID."""
    job_id = uuid4().hex[:12]
    with _job_lock:
        _jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
    return job_id


def update_job(job_id: str, **kwargs):
    """Update job status."""
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> dict:
    """Get job status."""
    with _job_lock:
        return _jobs.get(job_id, {}).copy()


def cleanup_old_jobs(max_age_seconds: int = 3600):
    """Remove jobs older than max_age_seconds."""
    now = time.time()
    with _job_lock:
        to_remove = [
            jid for jid, job in _jobs.items()
            if now - job["created_at"] > max_age_seconds
        ]
        for jid in to_remove:
            del _jobs[jid]


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and result."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_steps: int = 28
    guidance: float = 3.5
    seed: int = -1


class EditImageRequest(BaseModel):
    image_base64: str
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.7  # How much to change (0.0 = no change, 1.0 = full regeneration)
    width: int = 1024
    height: int = 1024
    num_steps: int = 28
    guidance: float = 3.5
    seed: int = -1


class GenerateVideoRequest(BaseModel):
    start_frame_base64: str
    prompt: str
    end_frame_base64: Optional[str] = None
    num_frames: int = 81
    width: int = 1280
    height: int = 720
    num_steps: int = 30
    guidance: float = 5.0
    seed: int = -1


class GenerateVoiceRequest(BaseModel):
    text: str
    description: str


class SynthesizeVoiceRequest(BaseModel):
    reference_audio_base64: str
    text: str


class TrainLoRARequest(BaseModel):
    dataset_url: str
    lora_name: str
    num_epochs: int = 10
    learning_rate: float = 1e-4
    network_rank: int = 32
    network_alpha: int = 16
    resolution: int = 1024


# =============================================================================
# IMAGE GENERATION (Qwen-Image-2512)
# =============================================================================

def load_qwen_image():
    if "qwen_image" not in _models:
        clear_vram()
        print("Loading Qwen-Image-2512...")
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_DIR / "qwen-image",
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        _models["qwen_image"] = pipe
    return _models["qwen_image"]


def _run_image_generation(job_id: str, req: dict):
    """Background worker for image generation."""
    try:
        update_job(job_id, status="running", progress=5)

        pipe = load_qwen_image()
        update_job(job_id, progress=10)

        seed = req.get("seed", -1)
        generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

        # Create progress callback
        def progress_callback(pipe, step, timestep, callback_kwargs):
            progress = 10 + int((step / req.get("num_steps", 28)) * 85)
            update_job(job_id, progress=progress)
            return callback_kwargs

        image = pipe(
            prompt=req["prompt"],
            negative_prompt=req.get("negative_prompt") or None,
            width=req.get("width", 1024),
            height=req.get("height", 1024),
            num_inference_steps=req.get("num_steps", 28),
            true_cfg_scale=req.get("guidance", 3.5),
            generator=generator,
            callback_on_step_end=progress_callback,
        ).images[0]

        update_job(job_id, progress=95)

        # Save and encode
        path = OUTPUT_DIR / f"image_{job_id}.png"
        image.save(path)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        update_job(
            job_id,
            status="complete",
            progress=100,
            result={"image_base64": image_b64, "path": str(path)},
        )

    except Exception as e:
        update_job(job_id, status="failed", error=str(e))


@app.post("/generate_image")
async def generate_image(req: GenerateImageRequest):
    """Submit image generation job. Returns job_id immediately."""
    job_id = create_job()
    _executor.submit(_run_image_generation, job_id, req.model_dump())
    return {"job_id": job_id, "status": "pending"}


# =============================================================================
# IMAGE EDITING / REPOSING (img2img)
# =============================================================================

def _run_image_edit(job_id: str, req: dict):
    """Background worker for image editing/reposing."""
    try:
        from PIL import Image

        update_job(job_id, status="running", progress=5)

        pipe = load_qwen_image()
        update_job(job_id, progress=10)

        # Decode input image
        input_image = Image.open(io.BytesIO(base64.b64decode(req["image_base64"])))
        input_image = input_image.convert("RGB")
        input_image = input_image.resize((req.get("width", 1024), req.get("height", 1024)))

        seed = req.get("seed", -1)
        generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

        strength = req.get("strength", 0.7)
        # Calculate actual steps based on strength
        total_steps = req.get("num_steps", 28)
        actual_steps = int(total_steps * strength)
        actual_steps = max(1, actual_steps)

        # Create progress callback
        def progress_callback(pipe, step, timestep, callback_kwargs):
            progress = 10 + int((step / actual_steps) * 85)
            update_job(job_id, progress=progress)
            return callback_kwargs

        # img2img: provide image and strength
        # The pipeline will add noise to the image based on strength, then denoise
        image = pipe(
            prompt=req["prompt"],
            negative_prompt=req.get("negative_prompt") or None,
            image=input_image,
            strength=strength,
            width=req.get("width", 1024),
            height=req.get("height", 1024),
            num_inference_steps=total_steps,
            true_cfg_scale=req.get("guidance", 3.5),
            generator=generator,
            callback_on_step_end=progress_callback,
        ).images[0]

        update_job(job_id, progress=95)

        # Save and encode
        path = OUTPUT_DIR / f"edit_{job_id}.png"
        image.save(path)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        update_job(
            job_id,
            status="complete",
            progress=100,
            result={"image_base64": image_b64, "path": str(path)},
        )

    except Exception as e:
        import traceback
        update_job(job_id, status="failed", error=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/edit_image")
async def edit_image(req: EditImageRequest):
    """Submit image edit/repose job. Takes a base image and transforms it based on prompt.

    Use strength to control how much the image changes:
    - 0.3-0.5: Minor changes (expression, small pose adjustments)
    - 0.5-0.7: Moderate changes (different pose, same character)
    - 0.7-0.9: Major changes (significant reposing)
    """
    job_id = create_job()
    _executor.submit(_run_image_edit, job_id, req.model_dump())
    return {"job_id": job_id, "status": "pending"}


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def load_wan():
    if "wan" not in _models:
        clear_vram()
        print("Loading Wan 2.1 FLF2V...")
        from diffusers import WanFLFToVideoPipeline
        pipe = WanFLFToVideoPipeline.from_pretrained(
            MODEL_DIR / "wan-flf2v",
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        _models["wan"] = pipe
    return _models["wan"]


def _run_video_generation(job_id: str, req: dict):
    """Background worker for video generation."""
    try:
        from PIL import Image
        from diffusers.utils import export_to_video

        update_job(job_id, status="running", progress=5)

        pipe = load_wan()
        update_job(job_id, progress=10)

        # Decode frames
        start_img = Image.open(io.BytesIO(base64.b64decode(req["start_frame_base64"])))
        start_img = start_img.resize((req.get("width", 1280), req.get("height", 720)))

        end_img = None
        if req.get("end_frame_base64"):
            end_img = Image.open(io.BytesIO(base64.b64decode(req["end_frame_base64"])))
            end_img = end_img.resize((req.get("width", 1280), req.get("height", 720)))

        seed = req.get("seed", -1)
        generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

        def progress_callback(pipe, step, timestep, callback_kwargs):
            progress = 10 + int((step / req.get("num_steps", 30)) * 85)
            update_job(job_id, progress=progress)
            return callback_kwargs

        output = pipe(
            first_image=start_img,
            last_image=end_img,
            prompt=req["prompt"],
            num_frames=req.get("num_frames", 81),
            width=req.get("width", 1280),
            height=req.get("height", 720),
            num_inference_steps=req.get("num_steps", 30),
            guidance_scale=req.get("guidance", 5.0),
            generator=generator,
            callback_on_step_end=progress_callback,
        )

        update_job(job_id, progress=95)

        frames = output.frames[0]
        path = OUTPUT_DIR / f"video_{job_id}.mp4"
        export_to_video(frames, str(path), fps=16)

        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        update_job(
            job_id,
            status="complete",
            progress=100,
            result={
                "video_base64": video_b64,
                "path": str(path),
                "duration_seconds": req.get("num_frames", 81) / 16.0,
            },
        )

    except Exception as e:
        update_job(job_id, status="failed", error=str(e))


@app.post("/generate_video")
async def generate_video(req: GenerateVideoRequest):
    """Submit video generation job. Returns job_id immediately."""
    job_id = create_job()
    _executor.submit(_run_video_generation, job_id, req.model_dump())
    return {"job_id": job_id, "status": "pending"}


# =============================================================================
# VOICE GENERATION (sync - fast enough)
# =============================================================================

def load_parler():
    if "parler" not in _models:
        clear_vram()
        print("Loading Parler-TTS...")
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        _models["parler"] = ParlerTTSForConditionalGeneration.from_pretrained(
            MODEL_DIR / "parler-tts",
            torch_dtype=torch.float16,
        ).to("cuda")
        _models["parler_tok"] = AutoTokenizer.from_pretrained(MODEL_DIR / "parler-tts")
    return _models["parler"], _models["parler_tok"]


@app.post("/generate_voice")
async def generate_voice(req: GenerateVoiceRequest):
    import torchaudio

    model, tokenizer = load_parler()

    input_ids = tokenizer(req.description, return_tensors="pt").input_ids.to("cuda")
    prompt_ids = tokenizer(req.text, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        audio = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)

    audio_np = audio.cpu().numpy().squeeze()
    sample_rate = model.config.sampling_rate

    path = OUTPUT_DIR / f"voice_{os.urandom(4).hex()}.wav"
    torchaudio.save(str(path), torch.tensor(audio_np).unsqueeze(0), sample_rate)

    with open(path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    return {
        "audio_base64": audio_b64,
        "path": str(path),
        "duration_seconds": len(audio_np) / sample_rate,
    }


# =============================================================================
# VOICE SYNTHESIS (Fish Speech)
# =============================================================================

@app.post("/synthesize_voice")
async def synthesize_voice(req: SynthesizeVoiceRequest):
    clear_vram()

    ref_path = OUTPUT_DIR / f"ref_{os.urandom(4).hex()}.wav"
    ref_path.write_bytes(base64.b64decode(req.reference_audio_base64))

    out_path = OUTPUT_DIR / f"synth_{os.urandom(4).hex()}.wav"

    cmd = [
        "python", "-m", "fish_speech.tools.inference",
        "--checkpoint", str(MODEL_DIR / "fish-speech"),
        "--reference", str(ref_path),
        "--text", req.text,
        "--output", str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if not out_path.exists():
        raise HTTPException(500, f"Synthesis failed: {result.stderr}")

    with open(out_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    return {
        "audio_base64": audio_b64,
        "path": str(out_path),
    }


# =============================================================================
# LORA TRAINING
# =============================================================================

@app.post("/train_lora")
async def train_lora(req: TrainLoRARequest):
    import zipfile
    from urllib.request import urlretrieve

    clear_vram()

    dataset_dir = Path("/tmp/dataset")
    output_dir = Path("/workspace/loras")
    dataset_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    zip_path = dataset_dir / "dataset.zip"
    urlretrieve(req.dataset_url, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dataset_dir / "images")

    cmd = [
        "accelerate", "launch",
        "/workspace/sd-scripts/sdxl_train_network.py",
        "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0",
        "--train_data_dir", str(dataset_dir / "images"),
        "--output_dir", str(output_dir),
        "--output_name", req.lora_name,
        "--max_train_epochs", str(req.num_epochs),
        "--learning_rate", str(req.learning_rate),
        "--network_dim", str(req.network_rank),
        "--network_alpha", str(req.network_alpha),
        "--resolution", f"{req.resolution},{req.resolution}",
        "--mixed_precision", "bf16",
        "--network_module", "networks.lora",
        "--cache_latents",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    lora_path = output_dir / f"{req.lora_name}.safetensors"

    if not lora_path.exists():
        raise HTTPException(500, f"Training failed: {result.stderr[-500:]}")

    return {
        "path": str(lora_path),
        "size_mb": lora_path.stat().st_size / (1024*1024),
        "log": result.stdout[-1000:],
    }


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    cleanup_old_jobs()  # Clean up old jobs on health check
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        "models_loaded": list(_models.keys()),
        "active_jobs": len([j for j in _jobs.values() if j["status"] in ("pending", "running")]),
    }


@app.post("/clear_vram")
async def clear():
    clear_vram()
    return {"status": "cleared"}


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path)


@app.on_event("startup")
async def startup_preload():
    """Preload image model on startup."""
    def preload():
        print("Preloading Qwen-Image-2512...")
        try:
            load_qwen_image()
            print("Model preloaded and ready!")
        except Exception as e:
            print(f"Warning: Failed to preload: {e}")

    thread = threading.Thread(target=preload, daemon=True)
    thread.start()
    print("Model preload started in background...")


if __name__ == "__main__":
    print("Starting StoryGen GPU Server (with job queue)...")
    print(f"Models dir: {MODEL_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
