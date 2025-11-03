#!/usr/bin/env python
# coding: utf-8

import torch  # Import PyTorch for handling computations
from transformers import CLIPTextModel, CLIPTokenizer  # Import CLIP text model and tokenizer for text encoding
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler  # Import Stable Diffusion pipeline and fast scheduler
import PIL.Image as Image  # Import PIL (Python Imaging Library) for image processing
import os
import json
from pathlib import Path

# Check if a GPU is available and set the device to CUDA if it is, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory} GB")

# Dossier des modèles personnalisés
CUSTOM_MODELS_DIR = Path(__file__).parent.parent / "models" / "custom-models"
MODELS_CONFIG_FILE = CUSTOM_MODELS_DIR / "models_config.json"

def load_custom_models():
    """Charge les modèles personnalisés depuis models/custom-models/"""
    custom_models = {}
    
    # Charger depuis le fichier de configuration s'il existe
    if MODELS_CONFIG_FILE.exists():
        try:
            with open(MODELS_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                for key, info in config.items():
                    model_name = info.get('name', key)
                    model_path = info.get('path', '')
                    if os.path.exists(model_path):
                        custom_models[f"📦 {model_name}"] = model_path
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la config des modèles: {e}")
    
    # Scanner également le dossier pour les modèles non enregistrés
    if CUSTOM_MODELS_DIR.exists():
        for item in CUSTOM_MODELS_DIR.iterdir():
            if item.is_dir() and (item / "model_index.json").exists():
                model_name = f"📦 {item.name}"
                if model_name not in custom_models:
                    custom_models[model_name] = str(item)
    
    return custom_models

# Modèles disponibles (pré-installés + personnalisés)
AVAILABLE_MODELS = {
    "SD 1.4 (Défaut)": "CompVis/stable-diffusion-v1-4",
    "SD 1.5": "runwayml/stable-diffusion-v1-5",
    "SD 2.1": "stabilityai/stable-diffusion-2-1",
    "Realistic Vision": "SG161222/Realistic_Vision_V5.1_noVAE",
    "DreamShaper": "Lykon/DreamShaper",
    "Anything V5": "stablediffusionapi/anything-v5",
}

# Ajouter les modèles personnalisés
AVAILABLE_MODELS.update(load_custom_models())

# Pipeline global
pipe = None
current_model = None

def load_model(model_name_or_path="CompVis/stable-diffusion-v1-4"):
    """
    Charge un modèle Stable Diffusion
    
    Args:
        model_name_or_path: Nom du modèle sur HuggingFace ou chemin local
    
    Returns:
        Pipeline Stable Diffusion chargé
    """
    global pipe, current_model
    
    # Si le même modèle est déjà chargé, ne pas recharger
    if current_model == model_name_or_path and pipe is not None:
        print(f"✅ Modèle déjà chargé: {model_name_or_path}")
        return pipe
    
    print(f"📦 Chargement du modèle: {model_name_or_path}...")
    
    # Vérifier si c'est un chemin local
    is_local = os.path.exists(model_name_or_path)
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=is_local  # Utiliser uniquement les fichiers locaux si c'est un chemin
        )
        
        # Convert to half precision for GPU speed optimization
        if device == "cuda":
            pipe = pipe.to(device, dtype=torch.float16)
        else:
            pipe = pipe.to(device)
        
        # Use DPM++ scheduler for faster generation
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            print("✅ Model CPU offload enabled for memory efficiency")
        
        current_model = model_name_or_path
        print(f"✅ Modèle chargé avec succès!")
        
        return pipe
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print(f"💡 Tentative de téléchargement depuis HuggingFace...")
        
        # Réessayer sans local_files_only
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if device == "cuda":
            pipe = pipe.to(device, dtype=torch.float16)
        else:
            pipe = pipe.to(device)
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        
        current_model = model_name_or_path
        print(f"✅ Modèle téléchargé et chargé!")
        
        return pipe

# Load the default Stable Diffusion pipeline
print("📦 Loading default pipeline...")
pipe = load_model("CompVis/stable-diffusion-v1-4")

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

print("✅ Pipeline loaded with GPU optimizations!")

# Function to generate an image from a text prompt
def generate_image_from_prompt(prompt, num_inference_steps=20, guidance_scale=7.5, width=512, height=512, model_name=None, seed=None, return_seed=False, negative_prompt=None):
    """
    This function takes a text prompt as input and generates an image using the Stable Diffusion model.
    
    Args:
    - prompt (str): The text description to generate the image from.
    - num_inference_steps (int): Number of denoising steps (default: 20, original was 50)
    - guidance_scale (float): How closely to follow the prompt (default: 7.5)
    - width (int): Width of the generated image in pixels (default: 512, must be multiple of 8)
    - height (int): Height of the generated image in pixels (default: 512, must be multiple of 8)
    - model_name (str): Nom du modèle à utiliser (None = utiliser le modèle actuellement chargé)
    - seed (int): Seed pour la génération aléatoire (None = seed aléatoire, -1 = seed aléatoire)
    - return_seed (bool): Si True, retourne un tuple (image, seed)
    - negative_prompt (str): Ce que vous ne voulez PAS voir dans l'image (défauts à éviter)

    Returns:
    - PIL.Image: The generated image (ou tuple (image, seed) si return_seed=True).
    """
    global pipe
    
    # Charger le modèle si spécifié
    if model_name and model_name != current_model:
        pipe = load_model(model_name)
    
    # Gérer le seed
    if seed is None or seed == -1:
        # Seed aléatoire
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        print(f"🎲 Using random seed: {seed}")
    else:
        print(f"🔢 Using seed: {seed}")
    
    print(f"Generating image for: '{prompt}'")
    if negative_prompt:
        print(f"🚫 Negative prompt: '{negative_prompt}'")
    print(f"Model: {current_model}")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Size: {width}x{height}")
    
    # Ensure dimensions are multiples of 8 (required by Stable Diffusion)
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    # Créer le générateur avec le seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate the image from the prompt without computing gradients (saves memory and computation)
    with torch.no_grad():
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,  # Specify what to avoid in the image
            num_inference_steps=num_inference_steps,  # Reduced from 50 to 20 for speed
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator  # Use seed for controllable randomness
        ).images[0]
    
    if return_seed:
        return image, seed
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