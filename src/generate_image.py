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
    configured_paths = set()  # Pour éviter les doublons
    
    # Charger depuis le fichier de configuration s'il existe
    if MODELS_CONFIG_FILE.exists():
        try:
            with open(MODELS_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                for key, info in config.items():
                    model_name = info.get('name', key)
                    model_path = info.get('path', '')
                    # Convertir en chemin absolu si c'est un chemin relatif
                    if not os.path.isabs(model_path):
                        model_path = str(CUSTOM_MODELS_DIR / model_path)
                    if os.path.exists(model_path):
                        custom_models[f"📦 {model_name}"] = model_path
                        configured_paths.add(os.path.normpath(model_path))
                        print(f"✅ Modèle personnalisé chargé: {model_name} -> {model_path}")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la config des modèles: {e}")
    
    # Scanner également le dossier pour les modèles non enregistrés
    if CUSTOM_MODELS_DIR.exists():
        for item in CUSTOM_MODELS_DIR.iterdir():
            # Ignorer les fichiers de configuration et README
            if item.name in ['models_config.json', 'models_config.json.example', 'README.md', 'QUICK_START.md']:
                continue
            
            # Vérifier si ce fichier/dossier n'est pas déjà configuré
            item_path = os.path.normpath(str(item))
            if item_path in configured_paths:
                continue
            
            # Dossiers avec model_index.json (modèles Diffusers complets)
            if item.is_dir() and (item / "model_index.json").exists():
                model_name = f"📦 {item.name}"
                if model_name not in custom_models:
                    custom_models[model_name] = str(item)
                    print(f"✅ Modèle dossier trouvé: {item.name}")
            
            # Fichiers .safetensors ou .ckpt (checkpoints uniques)
            elif item.is_file() and item.suffix in ['.safetensors', '.ckpt']:
                # Extraire le nom sans extension
                model_name = f"📦 {item.stem}"
                if model_name not in custom_models:
                    custom_models[model_name] = str(item)
                    print(f"✅ Checkpoint trouvé: {item.name}")
    
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
        model_name_or_path: Nom du modèle sur HuggingFace, chemin vers un dossier ou fichier .safetensors/.ckpt
    
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
    is_single_file = is_local and os.path.isfile(model_name_or_path)
    
    # Détecter si c'est un modèle SDXL
    is_sdxl = "xl" in model_name_or_path.lower() or "sdxl" in model_name_or_path.lower()
    
    try:
        # Charger depuis un fichier .safetensors ou .ckpt unique
        if is_single_file and model_name_or_path.endswith(('.safetensors', '.ckpt')):
            print(f"🔧 Chargement depuis un checkpoint unique...")
            
            # Utiliser le pipeline approprié selon le type de modèle
            if is_sdxl:
                from diffusers import StableDiffusionXLPipeline
                print(f"⚡ Détection d'un modèle SDXL - utilisation du pipeline XL")
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_name_or_path,
                    use_safetensors=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    variant="fp16" if device == "cuda" else None
                )
            else:
                pipe = StableDiffusionPipeline.from_single_file(
                    model_name_or_path,
                    use_safetensors=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
        # Charger depuis un dossier ou HuggingFace
        else:
            if is_sdxl:
                from diffusers import StableDiffusionXLPipeline
                print(f"⚡ Détection d'un modèle SDXL - utilisation du pipeline XL")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_name_or_path,
                    use_safetensors=True,
                    local_files_only=is_local,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    variant="fp16" if device == "cuda" else None
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_name_or_path,
                    use_safetensors=True,
                    local_files_only=is_local,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
        
        # Convert to half precision for GPU speed optimization
        if device == "cuda":
            pipe = pipe.to(device, dtype=torch.float16)
        else:
            pipe = pipe.to(device)
        
        # Use DPM++ 2M Karras scheduler for faster and better generation
        print("⚡ Configuration du scheduler DPM++ 2M Karras...")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
        
        # Enable memory optimizations conditionally
        if device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # CPU offload seulement si VRAM limitée ou modèle SDXL
            if total_vram_gb <= 6 or is_sdxl:
                pipe.enable_model_cpu_offload()
                print("✅ Model CPU offload enabled (GPU 6GB ou SDXL)")
            else:
                # Garder le modèle entièrement sur GPU pour plus de vitesse
                print("✅ Modèle gardé sur GPU (VRAM suffisante)")
            
            # Optimisations mémoire supplémentaires
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing(1)  # Slice size = 1 pour économiser VRAM
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
        
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

def is_sdxl_model(model_path):
    """Vérifie si un modèle est SDXL"""
    if not model_path:
        return False
    return "xl" in model_path.lower() or "sdxl" in model_path.lower()

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
    
    # Ajuster les dimensions pour SDXL selon la VRAM disponible
    if is_sdxl_model(current_model):
        # Vérifier la VRAM disponible
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Ajuster la résolution selon la VRAM
            if total_vram_gb <= 6:
                # RTX 3060 Laptop - Réduire pour éviter OOM et accélérer
                recommended_size = 768
            elif total_vram_gb <= 8:
                recommended_size = 896
            else:
                recommended_size = 1024
        else:
            recommended_size = 768
        
        if width < recommended_size or height < recommended_size:
            print(f"⚡ SDXL optimisé : ajustement automatique de {width}x{height} → {recommended_size}x{recommended_size} pour GPU {total_vram_gb:.0f}GB")
            width = recommended_size
            height = recommended_size
        
        # Optimiser les steps pour SDXL sur GPU limité
        if num_inference_steps == 20 and total_vram_gb <= 6:
            num_inference_steps = 15
            print(f"⚡ Steps réduits à {num_inference_steps} pour accélérer SDXL sur GPU 6GB")
    
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