"""
StoryGen GPU Server for RunPod Pods.

Run with: python pod_server.py
Access at: http://POD_IP:8000

All GPU tasks available via REST API with async job queue.
"""

import asyncio
import base64
import gc
import io
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from uuid import uuid4

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Config
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
OUTPUT_DIR = Path("/workspace/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LORA_OUTPUT_DIR = Path("/workspace/loras")
LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="StoryGen GPU Server")

# Global model cache
_models = {}

# Job queue system
_jobs = {}  # job_id -> {status, progress, result, error, created_at}
_executor = ThreadPoolExecutor(max_workers=1)  # Single worker for GPU tasks
_job_lock = threading.Lock()

# Client tracking for auto-offload
_active_clients = set()  # Set of active client IDs
_offload_task = None  # Asyncio task for delayed offload
IDLE_OFFLOAD_DELAY = 120  # Seconds to wait before offloading after idle

# Model loading state tracking
_model_loading = False
_model_loading_lock = threading.Lock()
_model_ops_lock = threading.Lock()  # Lock for model loading/unloading operations

# Request deduplication
_pending_request_hashes = set()
_request_hash_lock = threading.Lock()


def clear_vram(keep_model: str = None):
    """Unload models to free VRAM.

    Args:
        keep_model: If provided, keep this model loaded (don't clear it).
    """
    global _models
    cleared = []
    for name in list(_models.keys()):
        if keep_model and name == keep_model:
            continue
        del _models[name]
        cleared.append(name)
    if not keep_model:
        _models.clear()
    gc.collect()
    torch.cuda.empty_cache()
    if cleared:
        print(f"[Model Cache] Cleared models: {cleared}, remaining: {list(_models.keys())}")


def set_model_loading(loading: bool):
    """Set model loading state."""
    global _model_loading
    with _model_loading_lock:
        _model_loading = loading


def is_model_loading() -> bool:
    """Check if a model is currently loading."""
    with _model_loading_lock:
        return _model_loading


def get_request_hash(req_dict: dict) -> str:
    """Generate a hash for request deduplication."""
    import hashlib
    import json
    # Only hash the key parameters that define uniqueness
    key_params = {k: v for k, v in req_dict.items() if k not in ("seed",)}
    return hashlib.md5(json.dumps(key_params, sort_keys=True).encode()).hexdigest()[:12]


def add_pending_request(req_hash: str) -> bool:
    """Add request hash if not already pending. Returns True if added (new request)."""
    with _request_hash_lock:
        if req_hash in _pending_request_hashes:
            return False
        _pending_request_hashes.add(req_hash)
        return True


def remove_pending_request(req_hash: str):
    """Remove request hash when job completes."""
    with _request_hash_lock:
        _pending_request_hashes.discard(req_hash)


# =============================================================================
# JOB QUEUE SYSTEM
# =============================================================================

def create_job() -> str:
    """Create a new job and return its ID."""
    _cancel_offload()  # Cancel any pending offload when new work arrives
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
    # Schedule offload check when job completes or fails
    if kwargs.get("status") in ("complete", "failed"):
        _schedule_offload_if_idle()


def get_job(job_id: str) -> dict:
    """Get job status."""
    with _job_lock:
        return _jobs.get(job_id, {}).copy()


def cleanup_old_jobs(max_age_seconds: int = 3600):
    """Remove completed/failed jobs older than max_age_seconds.

    Only removes jobs that are done (complete/failed), never running/pending jobs.
    """
    now = time.time()
    with _job_lock:
        to_remove = [
            jid for jid, job in _jobs.items()
            if now - job["created_at"] > max_age_seconds
            and job["status"] in ("complete", "failed")  # Don't delete running jobs!
        ]
        for jid in to_remove:
            del _jobs[jid]


# =============================================================================
# AUTO-OFFLOAD WHEN IDLE
# =============================================================================

def _has_pending_jobs() -> bool:
    """Check if there are any pending or running jobs."""
    with _job_lock:
        return any(
            job["status"] in ("pending", "running")
            for job in _jobs.values()
        )


async def _delayed_offload():
    """Offload models after idle timeout if no active clients or jobs."""
    global _offload_task
    try:
        await asyncio.sleep(IDLE_OFFLOAD_DELAY)
        if not _active_clients and not _has_pending_jobs():
            with _model_ops_lock:  # Prevent clearing while model is loading
                if not _active_clients and not _has_pending_jobs():  # Double-check after lock
                    print(f"No active clients for {IDLE_OFFLOAD_DELAY}s - offloading models...")
                    clear_vram()
                    print("Models offloaded to free VRAM")
    except asyncio.CancelledError:
        pass  # Offload was cancelled because new work arrived
    finally:
        _offload_task = None


def _cancel_offload():
    """Cancel any pending offload task."""
    global _offload_task
    if _offload_task and not _offload_task.done():
        _offload_task.cancel()
        _offload_task = None


def _schedule_offload_if_idle():
    """Schedule offload if no active clients."""
    global _offload_task
    if not _active_clients and not _has_pending_jobs():
        if _offload_task is None:
            try:
                loop = asyncio.get_event_loop()
                _offload_task = loop.create_task(_delayed_offload())
            except RuntimeError:
                pass  # No event loop running


def register_client(client_id: str):
    """Register an active client and cancel any pending offload."""
    _active_clients.add(client_id)
    _cancel_offload()


def unregister_client(client_id: str):
    """Unregister a client and schedule offload if idle."""
    _active_clients.discard(client_id)
    _schedule_offload_if_idle()


@app.post("/signal_batch_complete")
async def signal_batch_complete(client_id: str = "unknown"):
    """UI signals its batch is complete. Triggers offload if all clients done."""
    unregister_client(client_id)
    return {
        "status": "acknowledged",
        "active_clients": len(_active_clients),
        "offload_scheduled": _offload_task is not None,
    }


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and result."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/jobs")
async def list_jobs():
    """List all jobs with their status."""
    with _job_lock:
        return {
            "jobs": [
                {"job_id": jid, **job}
                for jid, job in sorted(_jobs.items(), key=lambda x: x[1]["created_at"], reverse=True)
            ],
            "total": len(_jobs),
        }


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1664  # 16:9 - Qwen-Image supported resolution
    height: int = 928
    num_steps: int = 50  # Recommended for Qwen-Image
    guidance: float = 4.0  # Recommended true_cfg_scale
    seed: int = -1


class EditImageRequest(BaseModel):
    image_base64: str
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.7  # How much to change (0.0 = no change, 1.0 = full regeneration)
    width: int = 1664  # 16:9 - Qwen supported resolution
    height: int = 928
    num_steps: int = 40
    guidance: float = 4.0
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
    width: int = 1024  # Training image width (pass actual image dimensions)
    height: int = 576  # Training image height (16:9 default)
    dataset_repeat: int = 22  # Reduced from 50 for faster training


class SceneAngleRequest(BaseModel):
    """Request for scene camera angle generation using Multiple Angles LoRA."""
    image_base64: str
    azimuth: str = "front view"  # front view, right side view, back view, left side view, etc.
    elevation: str = "eye-level shot"  # low-angle shot, eye-level shot, elevated shot, high-angle shot
    distance: str = "medium shot"  # close-up, medium shot, wide shot


class GenerateWithLoRARequest(BaseModel):
    """Request for image generation with custom trained LoRA."""
    prompt: str
    lora_name: str  # Name of the trained LoRA (e.g., "myproject_characters")
    lora_strength: float = 0.9
    image_base64: Optional[str] = None  # Optional input image for img2img
    width: int = 1280
    height: int = 720
    num_steps: int = 28
    guidance: float = 3.5
    seed: int = -1


class CharacterVariationRequest(BaseModel):
    """Request for character variation using reference conditioning (WWAA method).

    This generates new images of a character from a reference image.
    The reference is used as CONDITIONING, not img2img - generates from scratch.
    """
    reference_image_base64: str  # Reference image of the character
    prompt: str  # Description of the new pose/scene (e.g., "the character is sitting on a bench")
    negative_prompt: str = ""  # Things to avoid (e.g., "extra arms, wrong clothing")
    enhanced_prompt: str = ""  # VLM-suggested details to add (e.g., "wearing blue dress")
    width: int = 1280  # Match Wan 2.2 video output (16:9)
    height: int = 720
    num_steps: int = 20  # WWAA uses 20 steps
    cfg: float = 4.0  # CFG scale (WWAA uses 4)
    seed: int = -1


# =============================================================================
# IMAGE GENERATION (Qwen-Image-Edit-2511)
# =============================================================================
# ARCHITECTURE: Single Qwen-Image-Edit-2511 for ALL image generation
# - Text-to-image: gray (128,128,128) input + prompt
# - Image editing: reference image + edit prompt
# - With LoRA: Same model + character/scene LoRA = perfect consistency
# - Standard inference (20-40 steps for quality)
# =============================================================================

def load_qwen_lightning():
    """Load Qwen-Image-Edit-2511-Lightning for unified image generation.

    This single model handles BOTH:
    - Text-to-image: Use gray (128,128,128) input image
    - Image editing: Use reference image

    LoRAs trained on this model work for BOTH use cases,
    enabling perfect character consistency across all generated images.
    """
    if "qwen_lightning" not in _models:
        set_model_loading(True)
        try:
            print("Loading Qwen-Image-Edit-2511 (BF16 for inference)...")
            from diffusers import QwenImageEditPipeline

            pipe = QwenImageEditPipeline.from_pretrained(
                MODEL_DIR / "Qwen" / "Qwen-Image-Edit-2511",
                torch_dtype=torch.bfloat16,
            )

            pipe.to("cuda")
            _models["qwen_lightning"] = pipe  # Keep key name for compatibility
            print("Qwen-Image-Edit-2511 loaded (BF16)")
        finally:
            set_model_loading(False)
    return _models["qwen_lightning"]


# Legacy aliases for backwards compatibility
def load_qwen_image():
    """Load unified Lightning model (legacy alias for text-to-image)."""
    return load_qwen_lightning()


def load_qwen_image_edit():
    """Load unified Lightning model (legacy alias for editing)."""
    return load_qwen_lightning()


def _patch_transformers_generation_config():
    """Monkey-patch transformers to fix Qwen2.5-VL dict config bug.

    See: https://github.com/huggingface/transformers/issues/36281
    The bug is in GenerationConfig.from_model_config() which assumes
    decoder_config is a PretrainedConfig but it can be a dict.
    """
    from transformers.generation import configuration_utils
    from transformers import PretrainedConfig

    original_from_model_config = configuration_utils.GenerationConfig.from_model_config

    @classmethod
    def patched_from_model_config(cls, model_config, **kwargs):
        # Fix: convert dict decoder_config to PretrainedConfig
        if hasattr(model_config, 'text_config'):
            if isinstance(model_config.text_config, dict):
                model_config.text_config = PretrainedConfig.from_dict(model_config.text_config)
        if hasattr(model_config, 'decoder') and isinstance(model_config.decoder, dict):
            model_config.decoder = PretrainedConfig.from_dict(model_config.decoder)
        return original_from_model_config.__func__(cls, model_config, **kwargs)

    configuration_utils.GenerationConfig.from_model_config = patched_from_model_config
    print("Patched transformers GenerationConfig.from_model_config()")


# Apply patch on module load
_patch_transformers_generation_config()


def load_qwen_image_edit_plus():
    """Load Lightning model (legacy alias for plus pipeline).

    The Lightning model handles reference conditioning natively.
    For text-to-image: use gray input
    For editing: use reference image
    """
    return load_qwen_lightning()


def load_qwen_lightning_with_angles_lora(lora_strength: float = 0.9):
    """Load Lightning model with Multiple Angles LoRA for camera control.

    Uses the unified Lightning model with the Multiple Angles LoRA.
    """
    with _model_ops_lock:
        if "qwen_lightning_angles" not in _models:
            set_model_loading(True)
            try:
                print(f"[Model Cache] Loading Lightning with Angles LoRA (current cache: {list(_models.keys())})")
                clear_vram(keep_model="qwen_lightning_angles")

                from diffusers import QwenImageEditPipeline

                pipe = QwenImageEditPipeline.from_pretrained(
                    MODEL_DIR / "Qwen" / "Qwen-Image-Edit-2511",
                    torch_dtype=torch.bfloat16,
                )

                pipe.to("cuda")
                print("Qwen-Image-Edit-2511 loaded to GPU")

                # Load the Multiple Angles LoRA
                lora_path = MODEL_DIR / "loras" / "multiple-angles"
                lora_file = lora_path / "qwen-image-edit-2511-multiple-angles-lora.safetensors"

                if lora_file.exists():
                    print(f"Loading Multiple Angles LoRA from {lora_file}...")
                    pipe.load_lora_weights(str(lora_path), weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors")
                    print(f"LoRA loaded, will set strength at inference time")
                    _models["qwen_lightning_angles_lora_loaded"] = True
                else:
                    print(f"Warning: LoRA file not found at {lora_file}, using base model")
                    _models["qwen_lightning_angles_lora_loaded"] = False

                _models["qwen_lightning_angles"] = pipe
                print(f"[Model Cache] Successfully loaded qwen_lightning_angles")
            finally:
                set_model_loading(False)

    pipe = _models["qwen_lightning_angles"]
    if _models.get("qwen_lightning_angles_lora_loaded"):
        pipe.set_adapters(["default_0"], adapter_weights=[lora_strength])
        print(f"LoRA strength set to {lora_strength}")

    return pipe


# Legacy alias
def load_qwen_image_edit_with_angles_lora(lora_strength: float = 0.9):
    """Legacy alias for load_qwen_lightning_with_angles_lora."""
    return load_qwen_lightning_with_angles_lora(lora_strength)


def _run_image_generation(job_id: str, req: dict):
    """Background worker for text-to-image generation using Lightning.

    Uses gray (128,128,128) input image for text-to-image mode.
    The Lightning model treats this as "generate from scratch".
    """
    try:
        from PIL import Image as PILImage

        print(f"[{job_id}] Starting text-to-image generation (Lightning)...")
        update_job(job_id, status="running", progress=5)

        print(f"[{job_id}] Loading Qwen-Image-Edit-Lightning...")
        pipe = load_qwen_lightning()
        print(f"[{job_id}] Model loaded, dtype: {pipe.transformer.dtype}")
        update_job(job_id, progress=10)

        seed = req.get("seed", -1)
        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(actual_seed)
        print(f"[{job_id}] Using seed: {actual_seed}")

        # Lightning model: 4 steps is optimal (vs 28-40 for full model)
        total_steps = req.get("num_steps", 28)
        width = req.get("width", 1024)
        height = req.get("height", 1024)

        # Create gray input image for text-to-image mode
        # The edit model will replace it entirely with the prompted content
        gray_image = PILImage.new("RGB", (width, height), (128, 128, 128))

        print(f"[{job_id}] Prompt: {req['prompt'][:100]}...")
        print(f"[{job_id}] Starting inference with {total_steps} steps...")

        # Create progress callback
        def progress_callback(pipe, step, timestep, callback_kwargs):
            progress = 10 + int((step / total_steps) * 85)
            update_job(job_id, progress=progress)
            return callback_kwargs

        image = pipe(
            prompt=req["prompt"],
            image=gray_image,  # Gray input for text-to-image
            width=width,
            height=height,
            num_inference_steps=total_steps,
            guidance_scale=req.get("guidance", 3.5),
            generator=generator,
            callback_on_step_end=progress_callback,
        ).images[0]

        print(f"[{job_id}] Inference complete!")

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
            result={"image_base64": image_b64, "path": str(path), "seed": actual_seed},
        )

    except Exception as e:
        import traceback
        print(f"[{job_id}] Error: {e}")
        print(traceback.format_exc())
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
    """Background worker for image editing using Lightning model.

    Uses the same model as text-to-image but with reference image input.
    """
    try:
        from PIL import Image

        update_job(job_id, status="running", progress=5)

        # Use unified Lightning model
        pipe = load_qwen_lightning()
        update_job(job_id, progress=10)

        # Decode input image
        input_image = Image.open(io.BytesIO(base64.b64decode(req["image_base64"])))
        input_image = input_image.convert("RGB")
        input_image = input_image.resize((req.get("width", 1024), req.get("height", 1024)))

        seed = req.get("seed", -1)
        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(actual_seed)
        print(f"[{job_id}] Using seed: {actual_seed}")

        # Lightning model: 4 steps is optimal
        total_steps = req.get("num_steps", 28)

        # Create progress callback
        def progress_callback(pipe, step, timestep, callback_kwargs):
            progress = 10 + int((step / total_steps) * 85)
            update_job(job_id, progress=progress)
            return callback_kwargs

        # Lightning model simplified parameters
        image = pipe(
            prompt=req["prompt"],
            image=input_image,
            height=req.get("height", 1024),
            width=req.get("width", 1024),
            num_inference_steps=total_steps,
            guidance_scale=req.get("guidance", 3.5),
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
            result={"image_base64": image_b64, "path": str(path), "seed": actual_seed},
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
# GENERATE WITH CUSTOM LoRA (Character/Scene LoRAs)
# =============================================================================

def _run_generate_with_lora(job_id: str, req: dict):
    """Background worker for image generation with custom trained LoRA.

    Loads base Lightning model + custom LoRA for character/scene consistency.
    Supports both text-to-image (no input image) and img2img (with input image).
    """
    try:
        from PIL import Image

        update_job(job_id, status="running", progress=5)

        lora_name = req["lora_name"]
        lora_strength = req.get("lora_strength", 0.9)

        # Find the LoRA file
        lora_dir = LORA_OUTPUT_DIR / lora_name
        if not lora_dir.exists():
            raise ValueError(f"LoRA directory not found: {lora_dir}")

        # Find the highest epoch or any .safetensors file
        lora_files = list(lora_dir.glob("*.safetensors"))
        if not lora_files:
            raise ValueError(f"No LoRA files found in {lora_dir}")

        # Prefer epoch files, sorted by epoch number
        epoch_files = [f for f in lora_files if "epoch" in f.name]
        if epoch_files:
            lora_file = sorted(epoch_files, key=lambda f: int(f.stem.split("-")[-1]) if f.stem.split("-")[-1].isdigit() else 0)[-1]
        else:
            lora_file = lora_files[0]

        print(f"[{job_id}] Loading LoRA: {lora_file}")
        update_job(job_id, progress=10)

        # Load base Lightning model
        pipe = load_qwen_lightning()
        update_job(job_id, progress=20)

        # Load custom LoRA
        print(f"[{job_id}] Applying LoRA from {lora_file} with strength {lora_strength}...")
        pipe.load_lora_weights(str(lora_dir), weight_name=lora_file.name)
        pipe.set_adapters(["default"], adapter_weights=[lora_strength])
        update_job(job_id, progress=30)

        # Handle input image (for img2img) or create gray placeholder (for text-to-image)
        width = req.get("width", 1280)
        height = req.get("height", 720)

        if req.get("image_base64"):
            # img2img mode
            input_image = Image.open(io.BytesIO(base64.b64decode(req["image_base64"])))
            input_image = input_image.convert("RGB").resize((width, height))
            print(f"[{job_id}] Using input image for img2img")
        else:
            # text-to-image mode: gray placeholder
            input_image = Image.new("RGB", (width, height), (128, 128, 128))
            print(f"[{job_id}] Using gray placeholder for text-to-image")

        seed = req.get("seed", -1)
        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(actual_seed)
        print(f"[{job_id}] Using seed: {actual_seed}")

        total_steps = req.get("num_steps", 28)

        def progress_callback(pipe, step, timestep, callback_kwargs):
            progress = 30 + int((step / total_steps) * 65)
            update_job(job_id, progress=progress)
            return callback_kwargs

        print(f"[{job_id}] Running inference with prompt: {req['prompt'][:80]}...")
        image = pipe(
            prompt=req["prompt"],
            image=input_image,
            height=height,
            width=width,
            num_inference_steps=total_steps,
            guidance_scale=req.get("guidance", 3.5),
            generator=generator,
            callback_on_step_end=progress_callback,
        ).images[0]

        update_job(job_id, progress=95)

        # Unload LoRA to free memory
        pipe.unload_lora_weights()

        # Save and encode
        path = OUTPUT_DIR / f"lora_{lora_name}_{job_id}.png"
        image.save(path)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        update_job(
            job_id,
            status="complete",
            progress=100,
            result={"image_base64": image_b64, "path": str(path), "seed": actual_seed},
        )
        print(f"[{job_id}] LoRA generation complete!")

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{job_id}] ERROR: {error_msg}")
        update_job(job_id, status="failed", error=error_msg)


@app.post("/generate_with_lora")
async def generate_with_lora(req: GenerateWithLoRARequest):
    """Generate image using base model + custom trained LoRA.

    Use this for character-consistent or scene-consistent generation.
    - Without image_base64: text-to-image with LoRA
    - With image_base64: img2img editing with LoRA
    """
    job_id = create_job()
    _executor.submit(_run_generate_with_lora, job_id, req.model_dump())
    return {"job_id": job_id, "status": "pending"}


# =============================================================================
# TEXT-TO-IMAGE V2 (Using working QwenImageEditPlusPipeline with noise reference)
# =============================================================================

def _run_image_generation_v2(job_id: str, req: dict):
    """Text-to-image using QwenImageEditPlusPipeline with noise reference.

    Uses the working edit model instead of broken Qwen-Image-2512.
    Generates from noise/placeholder reference for pure text-to-image.
    """
    try:
        from PIL import Image
        import numpy as np

        print(f"[{job_id}] Starting text-to-image v2 (using edit model)...")
        update_job(job_id, status="running", progress=5)

        # Load the Plus pipeline (reference conditioning)
        print(f"[{job_id}] Loading QwenImageEditPlusPipeline...")
        pipe = load_qwen_image_edit_plus()
        print(f"[{job_id}] Model loaded, dtype: {pipe.transformer.dtype}")
        update_job(job_id, progress=15)

        # Create noise reference image (model will mostly ignore it with good prompt)
        width = req.get("width", 1664)
        height = req.get("height", 928)
        noise_array = np.random.randint(128, 192, (height, width, 3), dtype=np.uint8)
        ref_image = Image.fromarray(noise_array, mode="RGB")

        seed = req.get("seed", -1)
        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(actual_seed)
        print(f"[{job_id}] Using seed: {actual_seed}")

        total_steps = req.get("num_steps", 30)
        print(f"[{job_id}] Prompt: {req['prompt'][:100]}...")
        print(f"[{job_id}] Starting inference with {total_steps} steps...")

        def progress_callback(pipe, step, timestep, callback_kwargs):
            progress = 15 + int((step / total_steps) * 80)
            update_job(job_id, progress=progress)
            if step % 10 == 0:
                print(f"[{job_id}] Step {step}/{total_steps}")
            return callback_kwargs

        neg_prompt = req.get("negative_prompt", "").strip() or " "

        output = pipe(
            image=[ref_image],  # Noise reference (will be mostly overridden by prompt)
            prompt=req["prompt"],
            negative_prompt=neg_prompt,
            num_inference_steps=total_steps,
            guidance_scale=1.0,
            true_cfg_scale=req.get("guidance", 4.0),
            height=height,
            width=width,
            generator=generator,
            num_images_per_prompt=1,
            callback_on_step_end=progress_callback,
        )

        image = output.images[0]
        print(f"[{job_id}] Inference complete!")
        update_job(job_id, progress=95)

        # Save and encode
        path = OUTPUT_DIR / f"image_v2_{job_id}.png"
        image.save(path)
        print(f"[{job_id}] Image saved to {path}")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        update_job(
            job_id,
            status="complete",
            progress=100,
            result={"image_base64": image_b64, "path": str(path), "seed": seed},
        )
        print(f"[{job_id}] Text-to-image v2 complete!")

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{job_id}] ERROR: {error_msg}")
        update_job(job_id, status="failed", error=error_msg)


@app.post("/generate_image_v2")
async def generate_image_v2(req: GenerateImageRequest):
    """Text-to-image using working QwenImageEditPlusPipeline.

    Alternative to /generate_image that uses the working edit model
    with a noise reference instead of broken Qwen-Image-2512.
    """
    job_id = create_job()
    _executor.submit(_run_image_generation_v2, job_id, req.model_dump())
    return {"job_id": job_id, "status": "pending"}


# =============================================================================
# CHARACTER VARIATION (Reference Conditioning - WWAA Method)
# =============================================================================

def _run_character_variation(job_id: str, req: dict):
    """Background worker for character variation using reference conditioning.

    Uses QwenImageEditPlusPipeline - reference image as CONDITIONING, not init image.
    Generates from scratch while maintaining character consistency.
    """
    try:
        from PIL import Image

        update_job(job_id, status="running", progress=5)

        # Load the Plus pipeline (reference conditioning)
        pipe = load_qwen_image_edit_plus()
        update_job(job_id, progress=15)

        # Decode reference image
        ref_image = Image.open(io.BytesIO(base64.b64decode(req["reference_image_base64"])))
        ref_image = ref_image.convert("RGB")

        seed = req.get("seed", -1)
        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(actual_seed)
        print(f"[{job_id}] Using seed: {actual_seed}")

        total_steps = req.get("num_steps", 20)

        def progress_callback(pipe, step, timestep, callback_kwargs):
            progress = 15 + int((step / total_steps) * 80)
            update_job(job_id, progress=progress)
            return callback_kwargs

        # Build prompt with VLM enhancements if provided
        base_prompt = req["prompt"]
        enhanced = req.get("enhanced_prompt", "").strip()
        if enhanced:
            # Prepend VLM-suggested details (e.g., "wearing blue dress with white collar")
            base_prompt = f"{enhanced}, {base_prompt}"

        # Build negative prompt with VLM suggestions if provided
        neg_prompt = req.get("negative_prompt", "").strip()
        if not neg_prompt:
            neg_prompt = " "  # Minimal negative prompt for this model

        # QwenImageEditPlusPipeline: reference image as conditioning
        # Key difference: image is passed as list for conditioning, NOT init latent
        output = pipe(
            image=[ref_image],  # Reference image as conditioning (list!)
            prompt=base_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=total_steps,
            guidance_scale=1.0,  # WWAA uses guidance_scale=1.0
            true_cfg_scale=req.get("cfg", 4.0),  # WWAA uses cfg=4
            height=req.get("height", 720),
            width=req.get("width", 1280),
            generator=generator,
            num_images_per_prompt=1,
            callback_on_step_end=progress_callback,
        )

        image = output.images[0]
        update_job(job_id, progress=95)

        # Save and encode
        path = OUTPUT_DIR / f"charvar_{job_id}.png"
        image.save(path)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        update_job(
            job_id,
            status="complete",
            progress=100,
            result={
                "image_base64": image_b64,
                "path": str(path),
                "prompt": req["prompt"],
                "width": req.get("width", 1280),
                "height": req.get("height", 720),
            },
        )

    except Exception as e:
        import traceback
        update_job(job_id, status="failed", error=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/generate_character_variation")
async def generate_character_variation(req: CharacterVariationRequest):
    """Generate character variation using reference conditioning (WWAA method).

    This is the correct approach for LoRA training datasets:
    - Reference image is used as CONDITIONING (not img2img)
    - Generates completely new image from scratch
    - Maintains character consistency via conditioning
    - Output matches Wan 2.2 video format (1280x720, 16:9)

    Example prompt: "the character is sitting on a wooden bench in a sunny park"
    """
    job_id = create_job()
    _executor.submit(_run_character_variation, job_id, req.model_dump())
    return {"job_id": job_id, "status": "pending"}


# =============================================================================
# SCENE CAMERA ANGLE GENERATION (Multiple Angles LoRA)
# =============================================================================

def _run_scene_angle(job_id: str, req: dict):
    """Background worker for scene camera angle generation using Multiple Angles LoRA.

    Uses the <sks> trigger format: <sks> [azimuth] [elevation] [distance]
    """
    try:
        from PIL import Image

        print(f"[{job_id}] Starting scene angle generation...")
        update_job(job_id, status="running", progress=5)

        # Load model with LoRA
        print(f"[{job_id}] Loading model...")
        pipe = load_qwen_image_edit_with_angles_lora(req.get("lora_strength", 0.9))
        print(f"[{job_id}] Model loaded, preparing input...")
        update_job(job_id, progress=15)

        # Decode input image
        input_image = Image.open(io.BytesIO(base64.b64decode(req["image_base64"])))
        input_image = input_image.convert("RGB")
        input_image = input_image.resize((req.get("width", 1024), req.get("height", 1024)))

        seed = req.get("seed", -1)
        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(actual_seed)
        print(f"[{job_id}] Using seed: {actual_seed}")

        # Build prompt with <sks> trigger
        # Format: <sks> [azimuth] [elevation] [distance]
        azimuth = req.get("azimuth", "front view")
        elevation = req.get("elevation", "eye-level shot")
        distance = req.get("distance", "medium shot")
        additional = req.get("additional_prompt", "")

        prompt = f"<sks> {azimuth} {elevation} {distance}"
        if additional:
            prompt = f"{prompt}, {additional}"

        total_steps = req.get("num_steps", 40)
        print(f"[{job_id}] Prompt: {prompt}")
        print(f"[{job_id}] Starting inference with {total_steps} steps...")

        def progress_callback(pipe, step, timestep, callback_kwargs):
            progress = 15 + int((step / total_steps) * 80)
            update_job(job_id, progress=progress)
            if step % 10 == 0:
                print(f"[{job_id}] Step {step}/{total_steps}")
            return callback_kwargs

        lora_strength = req.get("lora_strength", 0.9)
        print(f"[{job_id}] Running pipeline with LoRA strength {lora_strength}...")
        # LoRA strength is set via set_adapters() in load function
        # QwenImageEditPlusPipeline: image as list for reference conditioning
        image = pipe(
            prompt=prompt,
            negative_prompt=" ",  # Minimal negative prompt for this model
            image=[input_image],  # List for Plus pipeline reference conditioning
            height=req.get("height", 1024),
            width=req.get("width", 1024),
            num_inference_steps=total_steps,
            guidance_scale=1.0,
            true_cfg_scale=req.get("guidance", 4.0),
            generator=generator,
            callback_on_step_end=progress_callback,
        ).images[0]

        print(f"[{job_id}] Inference complete, saving image...")
        update_job(job_id, progress=95)

        # Save and encode
        path = OUTPUT_DIR / f"scene_angle_{job_id}.png"
        image.save(path)
        print(f"[{job_id}] Image saved to {path}")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        update_job(
            job_id,
            status="complete",
            progress=100,
            result={
                "image_base64": image_b64,
                "path": str(path),
                "prompt": prompt,
                "azimuth": azimuth,
                "elevation": elevation,
                "distance": distance,
            },
        )
        print(f"[{job_id}] Scene angle generation complete!")

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{job_id}] ERROR: {error_msg}")
        update_job(job_id, status="failed", error=error_msg)


@app.post("/generate_scene_angle")
async def generate_scene_angle(req: SceneAngleRequest):
    """Generate scene from different camera angle using Multiple Angles LoRA.

    Camera Controls:
    - azimuth: Horizontal rotation
        - "front view" (0°), "front-right quarter view" (45°), "right side view" (90°)
        - "back-right quarter view" (135°), "back view" (180°), "back-left quarter view" (225°)
        - "left side view" (270°), "front-left quarter view" (315°)
    - elevation: Vertical angle
        - "low-angle shot" (-30°), "eye-level shot" (0°), "elevated shot" (30°), "high-angle shot" (60°)
    - distance: Camera distance
        - "close-up" (×0.6), "medium shot" (×1.0), "wide shot" (×1.8)
    """
    job_id = create_job()
    _executor.submit(_run_scene_angle, job_id, req.model_dump())
    return {"job_id": job_id, "status": "pending"}


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def load_wan():
    """Load Wan 2.2 I2V for video generation (supports I2V and first-last-frame modes)."""
    if "wan" not in _models:
        set_model_loading(True)
        try:
            print("Loading Wan 2.2 I2V (Image/First-Last-Frame to Video)...")
            from diffusers import WanImageToVideoPipeline, WanTransformer3DModel

            # Load with bf16 transformers for better quality
            bf16_path = MODEL_DIR / "wan-i2v-bf16"
            base_path = MODEL_DIR / "wan-i2v"

            if bf16_path.exists():
                print("  Using bf16 transformers for better quality...")
                pipe = WanImageToVideoPipeline.from_pretrained(
                    base_path,
                    transformer=WanTransformer3DModel.from_pretrained(
                        bf16_path,
                        subfolder='transformer',
                        torch_dtype=torch.bfloat16,
                    ),
                    transformer_2=WanTransformer3DModel.from_pretrained(
                        bf16_path,
                        subfolder='transformer_2',
                        torch_dtype=torch.bfloat16,
                    ),
                    torch_dtype=torch.bfloat16,
                )
            else:
                print("  Using standard fp16 model...")
                pipe = WanImageToVideoPipeline.from_pretrained(
                    base_path,
                    torch_dtype=torch.float16,
                )

            pipe.to("cuda")
            _models["wan"] = pipe
            print("Wan 2.2 I2V loaded successfully!")
        finally:
            set_model_loading(False)
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
        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(actual_seed)
        print(f"[{job_id}] Using seed: {actual_seed}")

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
        # Don't clear VRAM - A100 80GB can hold multiple models
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
# LORA TRAINING (for Qwen-Image-Edit-2511 with FP8)
# =============================================================================
# Trains LoRAs using FP8 quantization for faster training (~16GB VRAM vs 40GB).
# Same LoRA works for generating first frames AND editing last frames.
# =============================================================================

@app.post("/train_lora")
async def train_lora(req: TrainLoRARequest):
    """Train Qwen-Image-Edit-2511 LoRA from a zip URL.

    The zip should contain images (png/jpg) with matching .txt caption files.
    Uses DiffSynth-Studio with FP8 quantization for faster training.

    Caption format: "{trigger_word}, pose, angle, background, style"
    Example: "project_luna, standing pose, front view, forest, 3D animated"
    NOTE: Do NOT include physical appearance - trigger word encodes that.
    """
    import zipfile
    import shutil
    import os
    from urllib.request import urlretrieve

    clear_vram()

    dataset_dir = Path("/tmp/dataset")
    images_dir = dataset_dir / "images"
    output_dir = Path("/workspace/loras")

    # Clean and recreate
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Download and extract dataset
    zip_path = dataset_dir / "dataset.zip"
    urlretrieve(req.dataset_url, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(images_dir)

    # Count images
    image_count = len(list(images_dir.glob("**/*.png"))) + len(list(images_dir.glob("**/*.jpg")))
    if image_count < 5:
        raise HTTPException(400, f"Need at least 5 images, got {image_count}")

    print(f"Training FP8 LoRA '{req.lora_name}' with {image_count} images...")

    # Check for DiffSynth-Studio
    diffsynth_path = Path("/workspace/DiffSynth-Studio")
    if not diffsynth_path.exists():
        raise HTTPException(500, "DiffSynth-Studio not found")

    # Create metadata CSV for DiffSynth
    metadata_path = images_dir / "metadata.csv"
    with open(metadata_path, "w") as f:
        f.write("image,prompt\n")
        for img_path in images_dir.glob("**/*.png"):
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                caption = caption_path.read_text().strip().replace('"', '""')
                f.write(f'"{img_path.name}","{caption}"\n')
        for img_path in images_dir.glob("**/*.jpg"):
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                caption = caption_path.read_text().strip().replace('"', '""')
                f.write(f'"{img_path.name}","{caption}"\n')

    # Use Qwen-Image-Edit-2511 with FP8 quantization for faster training
    # DiffSynth expects HuggingFace model ID format with DIFFSYNTH_MODEL_BASE_PATH
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/workspace/models"
    model_id = "Qwen/Qwen-Image-Edit-2511"
    model_paths = (
        f"{model_id}:transformer/diffusion_pytorch_model*.safetensors,"
        f"{model_id}:text_encoder/model*.safetensors,"
        f"{model_id}:vae/diffusion_pytorch_model.safetensors"
    )
    model_root = f"/workspace/models/{model_id}"

    # FP8 transformer for faster training (base model frozen, only LoRA trained)
    fp8_transformer = "/workspace/models/qwen-edit-2511-fp8/qwen_image_edit_2511_fp8_e4m3fn.safetensors"
    fp8_path = f"{fp8_transformer}:transformer/diffusion_pytorch_model*.safetensors"

    cmd = [
        "python",
        str(diffsynth_path / "examples" / "qwen_image" / "model_training" / "train.py"),
        "--dataset_base_path", str(images_dir),
        "--dataset_metadata_path", str(metadata_path),
        "--max_pixels", str(req.width * req.height),  # Use actual image dimensions (16:9 support)
        "--dataset_repeat", str(req.dataset_repeat),  # Configurable, default 22
        "--model_id_with_origin_paths", model_paths,
        "--fp8_models", fp8_path,  # Use FP8 for transformer (faster training)
        "--tokenizer_path", f"{model_root}/processor",
        "--processor_path", f"{model_root}/processor",
        "--learning_rate", str(req.learning_rate),
        "--num_epochs", str(req.num_epochs),
        "--remove_prefix_in_ckpt", "pipe.dit.",
        "--output_path", str(output_dir / req.lora_name),
        "--lora_base_model", "dit",
        "--lora_target_modules", "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1",
        "--lora_rank", str(req.network_rank),
        "--use_gradient_checkpointing",
        "--dataset_num_workers", "8",  # Increased from 4 for better throughput
        "--find_unused_parameters",
    ]

    env = os.environ.copy()
    env["DIFFSYNTH_MODEL_BASE_PATH"] = "/workspace/models"

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)

    # Find the LoRA file
    lora_path = output_dir / req.lora_name / f"epoch_{req.num_epochs}.safetensors"
    if not lora_path.exists():
        for pattern in [
            output_dir / req.lora_name / "*.safetensors",
            output_dir / f"{req.lora_name}.safetensors",
        ]:
            matches = list(Path("/").glob(str(pattern).lstrip("/")))
            if matches:
                lora_path = matches[-1]
                break

    if not lora_path.exists():
        raise HTTPException(500, f"Training failed: {result.stderr[-500:]}")

    return {
        "path": str(lora_path),
        "size_mb": lora_path.stat().st_size / (1024*1024),
        "log": result.stdout[-1000:],
        "images_trained": image_count,
        "model": "qwen-edit-2511-fp8",
    }


class TrainLoRABase64Request(BaseModel):
    """Request for LoRA training from base64 encoded images."""
    dataset: list  # List of {"image_base64": str, "caption": str, "filename": str}
    lora_name: str
    trigger_word: str
    num_epochs: int = 10
    learning_rate: float = 1e-4
    network_rank: int = 32
    network_alpha: int = 16
    width: int = 1024  # Training image width (pass actual image dimensions)
    height: int = 576  # Training image height (16:9 default)
    dataset_repeat: int = 22  # Reduced from 50 for faster training


def _run_lora_training_base64(job_id: str, req_dict: dict):
    """Background worker for LoRA training from base64 images.

    Uses DiffSynth-Studio with FP8 quantization for faster training.
    The trained LoRA works for BOTH text-to-image AND image editing.
    """
    import shutil
    import os

    try:
        update_job(job_id, status="running", progress=5)
        print(f"[{job_id}] Starting LoRA training: {req_dict['lora_name']}")

        dataset_dir = Path("/tmp/dataset_b64")
        images_dir = dataset_dir / "images"
        output_dir = Path("/workspace/loras")

        # Count images (already written by endpoint)
        image_count = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
        print(f"[{job_id}] Training with {image_count} images...")
        update_job(job_id, progress=10)

        # Check if DiffSynth-Studio is available
        diffsynth_path = Path("/workspace/DiffSynth-Studio")
        if diffsynth_path.exists():
            print(f"[{job_id}] Using DiffSynth-Studio with FP8 quantization...")

            # Create metadata CSV
            metadata_path = images_dir / "metadata.csv"
            with open(metadata_path, "w") as f:
                f.write("image,prompt\n")
                for img_path in images_dir.glob("*.png"):
                    caption_path = img_path.with_suffix(".txt")
                    if caption_path.exists():
                        caption = caption_path.read_text().strip().replace('"', '""')
                        f.write(f'"{img_path.name}","{caption}"\n')
                for img_path in images_dir.glob("*.jpg"):
                    caption_path = img_path.with_suffix(".txt")
                    if caption_path.exists():
                        caption = caption_path.read_text().strip().replace('"', '""')
                        f.write(f'"{img_path.name}","{caption}"\n')

            update_job(job_id, progress=15)

            # Use Qwen-Image-Edit-2511 with FP8 quantization for faster training
            # DiffSynth expects HuggingFace model ID format with DIFFSYNTH_MODEL_BASE_PATH
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/workspace/models"
            model_id = "Qwen/Qwen-Image-Edit-2511"
            model_paths = (
                f"{model_id}:transformer/diffusion_pytorch_model*.safetensors,"
                f"{model_id}:text_encoder/model*.safetensors,"
                f"{model_id}:vae/diffusion_pytorch_model.safetensors"
            )
            model_root = f"/workspace/models/{model_id}"

            # FP8 transformer for faster training (base model frozen, only LoRA trained)
            fp8_transformer = "/workspace/models/qwen-edit-2511-fp8/qwen_image_edit_2511_fp8_e4m3fn.safetensors"
            fp8_path = f"{fp8_transformer}:transformer/diffusion_pytorch_model*.safetensors"

            cmd = [
                "python",
                str(diffsynth_path / "examples" / "qwen_image" / "model_training" / "train.py"),
                "--dataset_base_path", str(images_dir),
                "--dataset_metadata_path", str(metadata_path),
                "--max_pixels", str(req_dict['width'] * req_dict['height']),  # Use actual image dimensions (16:9 support)
                "--dataset_repeat", str(req_dict.get('dataset_repeat', 22)),  # Configurable, default 22
                "--model_id_with_origin_paths", model_paths,
                "--fp8_models", fp8_path,  # Use FP8 for transformer (faster training)
                "--tokenizer_path", f"{model_root}/processor",
                "--processor_path", f"{model_root}/processor",
                "--learning_rate", str(req_dict['learning_rate']),
                "--num_epochs", str(req_dict['num_epochs']),
                "--remove_prefix_in_ckpt", "pipe.dit.",
                "--output_path", str(output_dir / req_dict['lora_name']),
                "--lora_base_model", "dit",
                "--lora_target_modules", "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1",
                "--lora_rank", str(req_dict['network_rank']),
                "--use_gradient_checkpointing",
                "--dataset_num_workers", "8",  # Increased from 4 for better throughput
                "--find_unused_parameters",
            ]

            env = os.environ.copy()
            env["DIFFSYNTH_MODEL_BASE_PATH"] = "/workspace/models"

            print(f"[{job_id}] Running DiffSynth training with FP8 quantization...")
            update_job(job_id, progress=20)

            # Use Popen to stream output and parse progress
            import re

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                env=env,
            )

            stdout_lines = []
            num_epochs = req_dict['num_epochs']
            current_epoch = 0
            current_step = 0
            total_steps_per_epoch = None
            last_reported_progress = 20

            # Read output line by line
            try:
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    stdout_lines.append(line)

                    # Parse progress from DiffSynth output
                    # Look for patterns like "Epoch 1/10" or "epoch 1" or step counters
                    epoch_match = re.search(r'[Ee]poch\s*[:\s]*(\d+)(?:/(\d+))?', line)
                    step_match = re.search(r'[Ss]tep\s*[:\s]*(\d+)(?:/(\d+))?', line)
                    # Also check for tqdm-style progress: "  5%|" or "100%|"
                    tqdm_match = re.search(r'\s*(\d+)%\|', line)
                    # Check for "X/Y" pattern common in progress bars
                    fraction_match = re.search(r'(\d+)/(\d+)\s*\[', line)

                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        if epoch_match.group(2):
                            num_epochs = int(epoch_match.group(2))

                    if step_match:
                        current_step = int(step_match.group(1))
                        if step_match.group(2):
                            total_steps_per_epoch = int(step_match.group(2))

                    if fraction_match:
                        current_step = int(fraction_match.group(1))
                        total_steps_per_epoch = int(fraction_match.group(2))

                    # Calculate progress (20-90% range for training)
                    if num_epochs > 0:
                        if total_steps_per_epoch and total_steps_per_epoch > 0:
                            # Fine-grained: epoch + step progress
                            epoch_progress = (current_epoch - 1) / num_epochs if current_epoch > 0 else 0
                            step_progress = current_step / total_steps_per_epoch / num_epochs
                            training_progress = epoch_progress + step_progress
                        else:
                            # Coarse: just epoch progress
                            training_progress = current_epoch / num_epochs

                        # Map to 20-90% range
                        progress = 20 + int(training_progress * 70)
                        progress = min(90, max(20, progress))

                        # Update if changed by at least 2%
                        if progress >= last_reported_progress + 2:
                            update_job(job_id, progress=progress)
                            last_reported_progress = progress
                            print(f"[{job_id}] Training progress: {progress}% (epoch {current_epoch}/{num_epochs})")

                    # Also check tqdm percentage directly - but it's EPOCH progress, not overall!
                    if tqdm_match and num_epochs > 0:
                        tqdm_pct = int(tqdm_match.group(1))
                        # tqdm shows progress within current epoch, calculate overall progress
                        # DiffSynth uses 0-based epochs (epoch 0, 1, 2...), so completed = current_epoch
                        completed_epochs = current_epoch  # Fixed: was max(0, current_epoch - 1)
                        epoch_fraction = tqdm_pct / 100.0
                        overall_progress = (completed_epochs + epoch_fraction) / num_epochs
                        progress = 20 + int(overall_progress * 70)  # Map to 20-90% range
                        progress = min(90, max(20, progress))
                        if progress >= last_reported_progress + 2:
                            update_job(job_id, progress=progress)
                            last_reported_progress = progress
                            print(f"[{job_id}] Training progress: {progress}% (epoch {current_epoch}/{num_epochs}, {tqdm_pct}% of epoch)")

                process.wait(timeout=7200)
                returncode = process.returncode
                stdout_text = ''.join(stdout_lines)

            except subprocess.TimeoutExpired:
                process.kill()
                raise

            print(f"[{job_id}] Training subprocess completed with return code: {returncode}")
            if stdout_text:
                print(f"[{job_id}] stdout (last 500 chars): {stdout_text[-500:]}")

            update_job(job_id, progress=90)

            # Store for result
            class Result:
                def __init__(self, rc, out):
                    self.returncode = rc
                    self.stdout = out
                    self.stderr = ""
            result = Result(returncode, stdout_text)

            # Find the LoRA file
            lora_path = output_dir / req_dict['lora_name'] / f"epoch_{req_dict['num_epochs']}.safetensors"
            if not lora_path.exists():
                for pattern in [
                    output_dir / req_dict['lora_name'] / "*.safetensors",
                    output_dir / f"{req_dict['lora_name']}.safetensors",
                ]:
                    matches = list(Path("/").glob(str(pattern).lstrip("/")))
                    if matches:
                        lora_path = matches[-1]
                        break

            if lora_path.exists():
                print(f"[{job_id}] LoRA saved to: {lora_path}")
                lora_b64 = base64.b64encode(lora_path.read_bytes()).decode()
                lora_size_mb = lora_path.stat().st_size / (1024 * 1024)

                update_job(
                    job_id,
                    status="complete",
                    progress=100,
                    result={
                        "lora_base64": lora_b64,
                        "lora_path": str(lora_path),
                        "lora_size_mb": lora_size_mb,
                        "training_log": result.stdout[-2000:] if result.stdout else "",
                    }
                )
                print(f"[{job_id}] Training complete! LoRA size: {lora_size_mb:.1f}MB")
            else:
                error_msg = f"LoRA file not found after training. stderr: {result.stderr[-1000:] if result.stderr else 'none'}"
                print(f"[{job_id}] ERROR: {error_msg}")
                update_job(job_id, status="failed", error=error_msg)
        else:
            update_job(job_id, status="failed", error="DiffSynth-Studio not found")

    except subprocess.TimeoutExpired:
        update_job(job_id, status="failed", error="Training timed out after 2 hours")
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{job_id}] ERROR: {error_msg}")
        update_job(job_id, status="failed", error=error_msg)


@app.post("/train_lora_from_base64")
async def train_lora_from_base64(req: TrainLoRABase64Request):
    """Train Qwen-Image LoRA from base64 encoded images with captions.

    Returns job_id immediately - poll /job/{job_id} for status.

    Accepts images directly as base64 (no URL/zip needed).
    Each item in dataset should have: image_base64, caption, filename
    """
    import shutil

    clear_vram()

    # Setup directories
    dataset_dir = Path("/tmp/dataset_b64")
    images_dir = dataset_dir / "images"
    output_dir = Path("/workspace/loras")

    # Clean previous dataset
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Write images and captions (do this synchronously - it's fast)
    for i, item in enumerate(req.dataset):
        img_b64 = item.get("image_base64", "")
        caption = item.get("caption", "")
        filename = item.get("filename", f"image_{i:03d}.png")

        if not img_b64:
            continue

        base_name = Path(filename).stem
        img_path = images_dir / filename
        img_path.write_bytes(base64.b64decode(img_b64))
        caption_path = images_dir / f"{base_name}.txt"
        caption_path.write_text(caption)

    # Count valid images
    image_count = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
    if image_count < 5:
        raise HTTPException(400, f"Need at least 5 images, got {image_count}")

    print(f"Received {image_count} images for LoRA '{req.lora_name}', starting background training...")

    # Create job and submit to background executor
    job_id = create_job()
    _executor.submit(_run_lora_training_base64, job_id, req.model_dump())

    return {"job_id": job_id, "status": "pending", "image_count": image_count}


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
        "model_loading": is_model_loading(),
        "active_jobs": len([j for j in _jobs.values() if j["status"] in ("pending", "running")]),
    }


@app.get("/queue_status")
async def queue_status():
    """Get detailed queue status for monitoring."""
    with _job_lock:
        pending = len([j for j in _jobs.values() if j["status"] == "pending"])
        running = len([j for j in _jobs.values() if j["status"] == "running"])
        completed = len([j for j in _jobs.values() if j["status"] == "complete"])
        failed = len([j for j in _jobs.values() if j["status"] == "failed"])

    return {
        "pending_jobs": pending,
        "running_jobs": running,
        "completed_jobs": completed,
        "failed_jobs": failed,
        "model_loading": is_model_loading(),
        "models_loaded": list(_models.keys()),
        "active_clients": len(_active_clients),
        "offload_scheduled": _offload_task is not None,
        "pending_request_hashes": len(_pending_request_hashes),
    }


@app.post("/clear_vram")
async def clear():
    clear_vram()
    return {"status": "cleared"}


@app.post("/warmup")
async def warmup_model(model: str = "qwen_image_edit_plus"):
    """Preload a model into GPU memory.

    Call this before batch operations to ensure model is ready.
    Blocks until model is fully loaded.

    Args:
        model: Which model to load. Options:
            - "qwen_image_edit_plus" (default) - for character variations
            - "qwen_image_edit_angles" - for scene angles
    """
    if model == "qwen_image_edit_plus":
        if "qwen_image_edit_plus" in _models:
            return {
                "status": "already_loaded",
                "model": model,
                "cache": list(_models.keys()),
            }
        # Load synchronously (blocking)
        print(f"[Warmup] Loading {model}...")
        try:
            load_qwen_image_edit_plus()
            return {
                "status": "loaded",
                "model": model,
                "cache": list(_models.keys()),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    elif model == "qwen_image_edit_angles":
        if "qwen_image_edit_angles" in _models:
            return {
                "status": "already_loaded",
                "model": model,
                "cache": list(_models.keys()),
            }
        print(f"[Warmup] Loading {model}...")
        try:
            load_qwen_image_edit_with_angles_lora()
            return {
                "status": "loaded",
                "model": model,
                "cache": list(_models.keys()),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    else:
        return {"status": "error", "error": f"Unknown model: {model}"}


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path)


LORA_DIR = Path("/workspace/loras")


@app.get("/list_loras")
async def list_loras():
    """List all trained LoRAs and their epoch files."""
    if not LORA_DIR.exists():
        return {"loras": []}

    loras = []
    for lora_dir in LORA_DIR.iterdir():
        if lora_dir.is_dir():
            epochs = []
            for f in sorted(lora_dir.glob("*.safetensors")):
                epochs.append({
                    "filename": f.name,
                    "size_mb": round(f.stat().st_size / (1024*1024), 2),
                    "modified": f.stat().st_mtime
                })
            if epochs:
                loras.append({
                    "name": lora_dir.name,
                    "epochs": epochs
                })
    return {"loras": loras}


@app.get("/download_lora/{lora_name}/{filename}")
async def download_lora(lora_name: str, filename: str):
    """Download a specific LoRA epoch file."""
    path = LORA_DIR / lora_name / filename
    if not path.exists():
        raise HTTPException(404, f"LoRA file not found: {lora_name}/{filename}")
    return FileResponse(path, filename=filename)


@app.get("/download_lora_all/{lora_name}")
async def download_lora_all(lora_name: str):
    """Download all epoch files for a LoRA as a zip."""
    import zipfile
    lora_path = LORA_DIR / lora_name
    if not lora_path.exists():
        raise HTTPException(404, f"LoRA not found: {lora_name}")

    epoch_files = sorted(lora_path.glob("*.safetensors"))
    if not epoch_files:
        raise HTTPException(404, f"No epoch files found for: {lora_name}")

    # Create zip in memory
    zip_path = OUTPUT_DIR / f"{lora_name}_all_epochs.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in epoch_files:
            zf.write(f, f.name)

    return FileResponse(zip_path, filename=f"{lora_name}_all_epochs.zip")


@app.on_event("startup")
async def startup_preload():
    """Startup event - models loaded on-demand now (no preload)."""
    # Don't preload - let models load on first request
    # This avoids loading wrong model (text2img vs edit)
    print("Server ready - models will load on first request")


if __name__ == "__main__":
    print("Starting StoryGen GPU Server (with job queue)...")
    print(f"Models dir: {MODEL_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
