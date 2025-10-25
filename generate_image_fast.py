#!/usr/bin/env python
# coding: utf-8

import torch  # Import PyTorch for handling computations
from transformers import CLIPTextModel, CLIPTokenizer  # Import CLIP text model and tokenizer for text encoding
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler  # Import Stable Diffusion pipeline and fast scheduler
import PIL.Image as Image  # Import PIL (Python Imaging Library) for image processing

# Check if a GPU is available and set the device to CUDA if it is, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory} GB")

# Load the Stable Diffusion pipeline with optimizations
print("📦 Loading pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    low_cpu_mem_usage=True,  # Use accelerate for memory optimization
    safety_checker=None,  # Disable safety checker for speed (optional)
    requires_safety_checker=False
)

# Convert to half precision for GPU speed optimization
if device == "cuda":
    pipe = pipe.to(device, dtype=torch.float16)
else:
    pipe = pipe.to(device)

# Use DPM++ scheduler for faster generation (fewer steps needed)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Enable memory efficient attention (if available)
if hasattr(pipe, "enable_attention_slicing"):
    pipe.enable_attention_slicing()

# Enable memory efficient VAE (if available) 
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()

# Enable memory-efficient attention for RTX 3060 (5GB)
if hasattr(pipe, "enable_model_cpu_offload"):
    try:
        pipe.enable_model_cpu_offload()
        print("✅ Model CPU offload enabled for memory efficiency")
    except:
        print("⚠️ Model CPU offload not available")

# Try to enable xformers for faster attention (if available)
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✅ Xformers enabled for faster attention")
except:
    print("⚠️ Xformers not available, using standard attention")

print("✅ Pipeline loaded with GPU optimizations!")

# Function to generate an image from a text prompt
def generate_image_from_prompt(prompt, num_inference_steps=20, guidance_scale=7.5):
    """
    This function takes a text prompt as input and generates an image using the Stable Diffusion model.
    
    Args:
    - prompt (str): The text description to generate the image from.
    - num_inference_steps (int): Number of denoising steps (default: 20, original was 50)
    - guidance_scale (float): How closely to follow the prompt (default: 7.5)

    Returns:
    - PIL.Image: The generated image.
    """
    print(f"Generating image for: '{prompt}'")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    
    # Generate the image from the prompt without computing gradients (saves memory and computation)
    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,  # Reduced from 50 to 20 for speed
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(42)  # For reproducible results
        ).images[0]
    return image

# Example usage
if __name__ == "__main__":
    prompt = "mountain sunset"  # The text prompt for the image
    
    import time
    start_time = time.time()
    
    generated_image = generate_image_from_prompt(prompt, num_inference_steps=15)  # Optimized for RTX 3060
    
    end_time = time.time()
    print(f"⚡ Generation completed in {end_time - start_time:.2f} seconds")
    
    # Save the generated image to a file
    output_path = "ai-generated-image-gpu-fast.png"
    generated_image.save(output_path)
    print(f"💾 Image saved to: {output_path}")
    
    # Display GPU memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"📊 GPU Memory used: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")
    
    # Display the generated image
    generated_image.show()