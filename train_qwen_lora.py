#!/usr/bin/env python3
"""
Qwen-Image-2512 LoRA Training Script.

Uses PEFT to train a LoRA on the Qwen-Image DiT (Diffusion Transformer).
Based on DiffSynth-Studio approach with target modules for Qwen architecture.

Usage:
    python train_qwen_lora.py \
        --dataset_dir /path/to/images \
        --output_dir /workspace/loras \
        --lora_name my_character \
        --trigger_word my_trigger \
        --num_epochs 10 \
        --rank 32
"""

import argparse
import gc
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Qwen-Image DiT target modules for LoRA
# These are the attention and MLP modules in the diffusion transformer
QWEN_TARGET_MODULES = [
    "to_q", "to_k", "to_v",           # Self-attention projections
    "add_q_proj", "add_k_proj", "add_v_proj",  # Cross-attention projections
    "to_out.0", "to_add_out",         # Output projections
    "img_mlp.net.2", "img_mod.1",     # Image MLP layers
    "txt_mlp.net.2", "txt_mod.1",     # Text MLP layers
]


class CaptionedImageDataset(Dataset):
    """Dataset of images with text captions for LoRA training."""

    def __init__(self, image_dir: Path, max_size: int = 1024):
        self.image_dir = Path(image_dir)
        self.max_size = max_size
        self.samples = []

        # Find all images with matching caption files
        for img_path in self.image_dir.glob("*.png"):
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                self.samples.append((img_path, caption_path))

        for img_path in self.image_dir.glob("*.jpg"):
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                self.samples.append((img_path, caption_path))

        print(f"Found {len(self.samples)} image-caption pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption_path = self.samples[idx]

        # Load and resize image
        image = Image.open(img_path).convert("RGB")

        # Resize to max_size while maintaining aspect ratio
        w, h = image.size
        if max(w, h) > self.max_size:
            scale = self.max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            # Round to nearest 64 for model compatibility
            new_w = (new_w // 64) * 64
            new_h = (new_h // 64) * 64
            image = image.resize((new_w, new_h), Image.LANCZOS)

        # Load caption
        caption = caption_path.read_text().strip()

        return {"image": image, "caption": caption, "path": str(img_path)}


def train_qwen_lora(
    dataset_dir: str,
    output_dir: str,
    lora_name: str,
    trigger_word: str,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    rank: int = 32,
    alpha: int = 16,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    max_size: int = 1024,
    save_every_n_epochs: int = 5,
):
    """Train a LoRA for Qwen-Image-2512."""

    from diffusers import QwenImagePipeline
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Training Qwen-Image LoRA: {lora_name}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Trigger word: {trigger_word}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}, Rank: {rank}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = CaptionedImageDataset(Path(dataset_dir), max_size=max_size)
    if len(dataset) < 5:
        raise ValueError(f"Need at least 5 images, found {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load Qwen-Image pipeline
    print("Loading Qwen-Image-2512 model...")
    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=dtype,
    )

    # Get the transformer (DiT) from the pipeline
    transformer = pipe.transformer
    transformer.to(device)
    transformer.train()

    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=QWEN_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )

    # Apply LoRA to transformer
    print("Applying LoRA to transformer...")
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Freeze VAE and text encoder
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Move to device
    pipe.vae.to(device)
    pipe.text_encoder.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Training loop
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Process batch
            images = batch["image"]
            captions = batch["caption"]

            # Encode images with VAE
            with torch.no_grad():
                # Process each image in batch
                latents_list = []
                for img in images:
                    # Convert PIL to tensor
                    img_tensor = torch.from_numpy(
                        np.array(img).astype(np.float32) / 127.5 - 1.0
                    ).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)

                    latent = pipe.vae.encode(img_tensor).latent_dist.sample()
                    latent = latent * pipe.vae.config.scaling_factor
                    latents_list.append(latent)

                latents = torch.cat(latents_list, dim=0)

                # Encode text
                text_embeds = pipe.text_encoder(captions[0])  # Batch size 1 for now

            # Sample noise
            noise = torch.randn_like(latents)

            # Sample timestep
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()

            # Add noise to latents
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise with transformer
            noise_pred = transformer(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeds,
            ).sample

            # MSE loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss = loss / gradient_accumulation
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * gradient_accumulation
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation:.4f}"})

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every_n_epochs == 0 or epoch == num_epochs - 1:
            checkpoint_name = f"{lora_name}_epoch{epoch+1}"
            save_path = output_path / f"{checkpoint_name}.safetensors"

            # Save LoRA weights
            transformer.save_pretrained(output_path / checkpoint_name)
            print(f"Saved checkpoint: {save_path}")

    # Save final LoRA
    final_path = output_path / f"{lora_name}.safetensors"
    transformer.save_pretrained(output_path / lora_name)

    # Also save in single file format for easier loading
    from safetensors.torch import save_file
    lora_state_dict = {
        k: v for k, v in transformer.state_dict().items()
        if "lora" in k.lower()
    }
    save_file(lora_state_dict, final_path)

    print(f"Training complete! LoRA saved to: {final_path}")

    # Cleanup
    del transformer, pipe
    gc.collect()
    torch.cuda.empty_cache()

    return str(final_path)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen-Image LoRA")
    parser.add_argument("--dataset_dir", required=True, help="Directory with images and captions")
    parser.add_argument("--output_dir", required=True, help="Output directory for LoRA")
    parser.add_argument("--lora_name", required=True, help="Name for the LoRA")
    parser.add_argument("--trigger_word", required=True, help="Trigger word for the LoRA")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_size", type=int, default=1024, help="Max image size")
    parser.add_argument("--save_every_n_epochs", type=int, default=5, help="Save checkpoint every N epochs")

    args = parser.parse_args()

    train_qwen_lora(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        lora_name=args.lora_name,
        trigger_word=args.trigger_word,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        rank=args.rank,
        alpha=args.alpha,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_size=args.max_size,
        save_every_n_epochs=args.save_every_n_epochs,
    )


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues
    main()
