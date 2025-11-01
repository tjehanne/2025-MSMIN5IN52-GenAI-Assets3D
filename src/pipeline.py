#!/usr/bin/env python
# coding: utf-8

"""
Pipeline complet : Génération d'image 2D à partir de texte puis conversion en modèle 3D
Combine generate_image_fast.py et generate_3d_from_image.py
"""

import os
import sys
import time
import argparse

# Importer les générateurs
from src.generate_image import generate_image_from_prompt, device as image_device
from src.generate_3d import TripoSR3DGenerator, device as model_3d_device

def text_to_3d_pipeline(
    prompt, 
    output_dir="output/pipeline",
    image_steps=15,
    image_guidance=7.5,
    image_width=512,
    image_height=512,
    model_3d_resolution=256,
    save_format="obj",
    render_video=False,
    keep_intermediate=True,
    sd_model=None,
    apply_texture=True
):
    """
    Pipeline complet : Texte -> Image 2D -> Modèle 3D
    
    Args:
        prompt (str): Description textuelle pour générer l'image
        output_dir (str): Dossier de sortie
        image_steps (int): Nombre d'étapes pour la génération d'image
        image_guidance (float): Guidance scale pour la génération d'image
        image_width (int): Largeur de l'image générée en pixels (défaut: 512)
        image_height (int): Hauteur de l'image générée en pixels (défaut: 512)
        model_3d_resolution (int): Résolution du modèle 3D
        save_format (str): Format de sauvegarde ('obj' ou 'glb')
        render_video (bool): Si True, génère une vidéo de rendu
        keep_intermediate (bool): Si True, garde les fichiers intermédiaires
        sd_model (str): Nom ou chemin du modèle Stable Diffusion (None = modèle par défaut)
        apply_texture (bool): Si True, applique la texture de l'image au modèle 3D
    
    Returns:
        dict: Chemins vers les fichiers générés
    """
    print("\n" + "="*70)
    print("🚀 TEXT-TO-3D PIPELINE")
    print("="*70)
    print(f"📝 Prompt: '{prompt}'")
    print(f"💾 Output: {output_dir}")
    print("="*70 + "\n")
    
    total_start = time.time()
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # === ÉTAPE 1 : Génération de l'image 2D ===
    print("🎨 STEP 1/2: Generating 2D image from text...")
    print("-" * 70)
    
    image_start = time.time()
    generated_image = generate_image_from_prompt(
        prompt, 
        num_inference_steps=image_steps,
        guidance_scale=image_guidance,
        width=image_width,
        height=image_height,
        model_name=sd_model
    )
    image_time = time.time() - image_start
    
    # Sauvegarder l'image générée
    image_path = os.path.join(output_dir, "generated_image.png")
    generated_image.save(image_path)
    
    print(f"✅ Image generated in {image_time:.2f}s")
    print(f"💾 Image saved: {image_path}\n")
    
    # === ÉTAPE 2 : Génération du modèle 3D ===
    print("🎯 STEP 2/2: Converting 2D image to 3D model...")
    print("-" * 70)
    
    model_3d_start = time.time()
    
    # Créer le générateur 3D
    generator = TripoSR3DGenerator(
        device=model_3d_device,
        mc_resolution=model_3d_resolution
    )
    
    # Générer le modèle 3D
    mesh_path = generator.generate_3d_model(
        image_path=image_path,
        output_dir=output_dir,
        remove_bg=True,
        foreground_ratio=0.85,
        save_format=save_format,
        render_video=render_video,
        apply_texture=apply_texture
    )
    
    model_3d_time = time.time() - model_3d_start
    
    # === RÉSUMÉ ===
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"⏱️  Image generation: {image_time:.2f}s")
    print(f"⏱️  3D model generation: {model_3d_time:.2f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print("-" * 70)
    print(f"📸 2D Image: {image_path}")
    print(f"🎲 3D Model: {mesh_path}")
    
    if render_video:
        video_path = os.path.join(output_dir, "render.mp4")
        if os.path.exists(video_path):
            print(f"🎥 Render Video: {video_path}")
    
    print("="*70 + "\n")
    
    # Retourner les chemins des fichiers générés
    results = {
        "image_path": image_path,
        "mesh_path": mesh_path,
        "output_dir": output_dir,
        "prompt": prompt,
        "total_time": total_time
    }
    
    if render_video and os.path.exists(os.path.join(output_dir, "render.mp4")):
        results["video_path"] = os.path.join(output_dir, "render.mp4")
    
    return results


def batch_text_to_3d(prompts, base_output_dir="output/batch_pipeline", **kwargs):
    """
    Génère plusieurs modèles 3D à partir d'une liste de prompts
    
    Args:
        prompts (list): Liste de descriptions textuelles
        base_output_dir (str): Dossier de base pour les sorties
        **kwargs: Arguments supplémentaires pour text_to_3d_pipeline
    
    Returns:
        list: Liste des résultats pour chaque prompt
    """
    print("\n" + "="*70)
    print(f"🚀 BATCH TEXT-TO-3D PIPELINE ({len(prompts)} prompts)")
    print("="*70 + "\n")
    
    all_results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'#'*70}")
        print(f"Processing {i+1}/{len(prompts)}")
        print(f"{'#'*70}")
        
        output_dir = os.path.join(base_output_dir, f"model_{i:03d}")
        
        try:
            result = text_to_3d_pipeline(
                prompt=prompt,
                output_dir=output_dir,
                **kwargs
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error processing prompt '{prompt}': {e}")
            all_results.append(None)
    
    # Résumé final
    successful = sum(1 for r in all_results if r is not None)
    print("\n" + "="*70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(prompts) - successful}")
    print("="*70 + "\n")
    
    return all_results


def main():
    """Fonction principale avec interface en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Pipeline complet : Texte -> Image 2D -> Modèle 3D"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Description textuelle pour générer le modèle 3D (si non fourni, demandé interactivement)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/pipeline",
        help="Dossier de sortie"
    )
    parser.add_argument(
        "--image-steps",
        type=int,
        default=25,  # Augmenté de 15 à 25 pour meilleure qualité
        help="Nombre d'étapes pour la génération d'image (10-50, défaut: 25 pour haute qualité)"
    )
    parser.add_argument(
        "--image-guidance",
        type=float,
        default=7.5,
        help="Guidance scale pour la génération d'image (défaut: 7.5)"
    )
    parser.add_argument(
        "--resolution-3d",
        type=int,
        default=320,  # Augmenté de 256 à 320 pour plus de détails
        help="Résolution du modèle 3D (128-512, défaut: 320 pour haute qualité)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["obj", "glb"],
        default="obj",
        help="Format de sauvegarde du modèle 3D (défaut: obj)"
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Générer une vidéo de rendu du modèle 3D"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Mode batch : traiter plusieurs prompts prédéfinis"
    )
    
    args = parser.parse_args()
    
    # Si pas de prompt fourni et pas en mode batch, demander interactivement
    if not args.batch and args.prompt is None:
        print("\n" + "="*70)
        print("🎨 GÉNÉRATEUR DE MODÈLES 3D")
        print("="*70)
        print("Décrivez ce que vous voulez générer en 3D.")
        print("Exemples :")
        print("  - a futuristic robot head, metallic, detailed")
        print("  - a dragon skull, fantasy art, ancient")
        print("  - a medieval sword with magical glow")
        print("  - a vintage camera, black metal, photorealistic")
        print("-"*70)
        
        args.prompt = input("📝 Votre description : ").strip()
        
        if not args.prompt:
            print("❌ Aucun prompt fourni. Utilisation d'un exemple par défaut.")
            args.prompt = "a futuristic robot head, detailed, sci-fi"
        
        print("="*70 + "\n")
    
    if args.batch:
        # Mode batch avec plusieurs prompts
        prompts = [
            "a majestic dragon head, fantasy art",
            "a cute robot companion, kawaii style",
            "a medieval helmet, iron and steel",
            "a futuristic car, sleek design",
            "a fantasy sword, magical glow"
        ]
        
        batch_text_to_3d(
            prompts=prompts,
            base_output_dir="output/batch_pipeline",
            image_steps=args.image_steps,
            image_guidance=args.image_guidance,
            model_3d_resolution=args.resolution_3d,
            save_format=args.format,
            render_video=args.render_video
        )
    else:
        # Mode simple prompt
        text_to_3d_pipeline(
            prompt=args.prompt,
            output_dir=args.output_dir,
            image_steps=args.image_steps,
            image_guidance=args.image_guidance,
            model_3d_resolution=args.resolution_3d,
            save_format=args.format,
            render_video=args.render_video
        )
    
    print("🎊 All done! Check your output folder for the generated models.")


if __name__ == "__main__":
    main()
