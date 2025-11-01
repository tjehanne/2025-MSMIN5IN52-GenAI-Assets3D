#!/usr/bin/env python
# coding: utf-8

"""
Programme de génération de modèles 3D à partir d'images 2D avec TripoSR
Utilise le modèle TripoSR pour créer des meshes 3D à partir d'images
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
import rembg
import trimesh

# Ajouter le dossier TripoSR au path pour importer les modules
triposr_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "TripoSR")
sys.path.insert(0, triposr_path)

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# Vérifier la disponibilité du GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory} GB")

class TripoSR3DGenerator:
    """
    Classe pour générer des modèles 3D à partir d'images 2D
    """
    
    def __init__(self, device="cuda:0", chunk_size=8192, mc_resolution=320):
        """
        Initialise le générateur 3D
        
        Args:
            device (str): Device à utiliser ('cuda:0' ou 'cpu')
            chunk_size (int): Taille des chunks pour l'extraction de surface (réduit la VRAM)
            mc_resolution (int): Résolution de la grille marching cubes (défaut: 320 pour haute qualité)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.chunk_size = chunk_size
        self.mc_resolution = mc_resolution
        self.model = None
        self.rembg_session = None
        
        print("📦 Initializing TripoSR model...")
        self._load_model()
        print("✅ Model loaded successfully!")
    
    def _load_model(self):
        """Charge le modèle TripoSR depuis HuggingFace"""
        start_time = time.time()
        
        # Charger le modèle pré-entraîné
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        
        # Configurer le chunk size pour optimiser la mémoire
        self.model.renderer.set_chunk_size(self.chunk_size)
        
        # Déplacer sur le device approprié
        self.model.to(self.device)
        
        # Initialiser la session rembg pour la suppression de fond
        self.rembg_session = rembg.new_session()
        
        load_time = time.time() - start_time
        print(f"⚡ Model loaded in {load_time:.2f} seconds")
    
    def preprocess_image(self, image_path, remove_bg=True, foreground_ratio=0.85):
        """
        Prétraite l'image d'entrée
        
        Args:
            image_path (str): Chemin vers l'image d'entrée
            remove_bg (bool): Si True, supprime automatiquement l'arrière-plan
            foreground_ratio (float): Ratio de la taille du premier plan
        
        Returns:
            PIL.Image: Image prétraitée
        """
        print(f"🖼️  Processing image: {image_path}")
        
        # Charger l'image
        image = Image.open(image_path)
        
        if remove_bg:
            # Supprimer l'arrière-plan
            print("  🎨 Removing background...")
            image = remove_background(image, self.rembg_session)
            
            # Redimensionner le premier plan
            image = resize_foreground(image, foreground_ratio)
            
            # Convertir en RGB avec fond gris
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
        else:
            # Convertir simplement en RGB
            image = image.convert("RGB")
        
        return image
    
    def apply_texture_to_mesh(self, mesh, texture_image_path, output_path, uv_config="default"):
        """
        Applique une texture à un mesh 3D en utilisant un mapping UV simple
        
        Args:
            mesh: Mesh trimesh
            texture_image_path (str): Chemin vers l'image de texture
            output_path (str): Chemin de sortie pour le mesh texturé
            uv_config (str): Configuration UV à utiliser (ex: "default", "flip_u", "flip_v", etc.)
        
        Returns:
            str: Chemin vers le mesh texturé
        """
        print(f"  🎨 Applying texture with config: {uv_config}...")
        
        # Le modèle est déjà dans la bonne orientation (rotation appliquée lors de l'extraction)
        # On applique directement la texture
        
        # Charger l'image de texture
        texture_image = Image.open(texture_image_path)
        
        # Sauvegarder la texture au format compatible
        base_name = os.path.splitext(output_path)[0]
        texture_filename = f"{base_name}_texture.png"
        texture_image.save(texture_filename)
        
        # Créer un mapping UV aligné avec la vue frontale de l'image
        vertices = mesh.vertices
        
        # Normaliser les vertices
        center = vertices.mean(axis=0)
        vertices_centered = vertices - center
        
        # Calculer les coordonnées UV selon la configuration
        r = np.sqrt(np.sum(vertices_centered**2, axis=1))
        
        # =====================================================================
        # PROJECTION UV OPTIMALE : proj2_flip_v
        # =====================================================================
        # Configuration identifiée : Projection XZ avec miroir vertical
        # - Projection 2 : theta = arctan2(X, Z), phi = arccos(Y/r)
        # - Transformation : flip_v (miroir vertical, v = 1.0 - v)
        #
        # Pour tester d'autres configurations, modifiez uv_config="proj2_flip_v"
        # lors de l'appel à apply_texture_to_mesh()
        # =====================================================================
        
        if "proj1" in uv_config:
            # Projection 1 : X, Y standard
            theta = np.arctan2(vertices_centered[:, 1], vertices_centered[:, 0])
            phi = np.arccos(np.clip(vertices_centered[:, 2] / (r + 1e-8), -1, 1))
        elif "proj2" in uv_config or uv_config == "default":
            # Projection 2 : X, Z (CONFIGURATION OPTIMALE)
            theta = np.arctan2(vertices_centered[:, 0], vertices_centered[:, 2])
            phi = np.arccos(np.clip(vertices_centered[:, 1] / (r + 1e-8), -1, 1))
        elif "proj3" in uv_config:
            # Projection 3 : Y, Z
            theta = np.arctan2(vertices_centered[:, 1], vertices_centered[:, 2])
            phi = np.arccos(np.clip(vertices_centered[:, 0] / (r + 1e-8), -1, 1))
        elif "proj4" in uv_config:
            # Projection 4 : X, -Y (compensé pour rotation Z)
            theta = np.arctan2(vertices_centered[:, 0], -vertices_centered[:, 1])
            phi = np.arccos(np.clip(vertices_centered[:, 2] / (r + 1e-8), -1, 1))
        elif "proj5" in uv_config:
            # Projection 5 : Y, -X
            theta = np.arctan2(vertices_centered[:, 1], -vertices_centered[:, 0])
            phi = np.arccos(np.clip(vertices_centered[:, 2] / (r + 1e-8), -1, 1))
        elif "proj6" in uv_config:
            # Projection 6 : -X, Y
            theta = np.arctan2(-vertices_centered[:, 0], vertices_centered[:, 1])
            phi = np.arccos(np.clip(vertices_centered[:, 2] / (r + 1e-8), -1, 1))
        else:
            # Par défaut : utiliser proj2 (optimale)
            theta = np.arctan2(vertices_centered[:, 0], vertices_centered[:, 2])
            phi = np.arccos(np.clip(vertices_centered[:, 1] / (r + 1e-8), -1, 1))
        
        # Convertir en coordonnées UV [0, 1]
        u = (theta + np.pi) / (2 * np.pi)
        v = phi / np.pi
        
        # Appliquer les transformations UV selon la configuration
        # Par défaut (ou "default"), appliquer flip_v (configuration optimale)
        if uv_config == "default" or "flip_v" in uv_config:
            v = 1.0 - v  # Miroir vertical (TRANSFORMATION OPTIMALE)
        
        if "flip_u" in uv_config and uv_config != "default":
            u = 1.0 - u
        
        if uv_config == "flip_both":
            u = 1.0 - u
            v = 1.0 - v
        elif uv_config == "rotate_90":
            u, v = v, 1.0 - u
        elif uv_config == "rotate_180":
            u = 1.0 - u
            v = 1.0 - v
        elif uv_config == "rotate_270":
            u, v = 1.0 - v, u
        elif uv_config == "swap_uv":
            u, v = v, u
        elif uv_config == "swap_flip_u":
            u, v = v, u
            u = 1.0 - u
        elif uv_config == "swap_flip_v":
            u, v = v, u
            v = 1.0 - v
        # "default" : pas de modification
        
        # Créer les coordonnées UV
        uv_coords = np.stack([u, v], axis=1)
        
        # Créer le matériau avec texture
        material = trimesh.visual.material.SimpleMaterial(
            image=texture_image,
            ambient=[0.8, 0.8, 0.8, 1.0],
            diffuse=[1.0, 1.0, 1.0, 1.0]
        )
        
        # Appliquer les UV au mesh
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv_coords,
            image=texture_image,
            material=material
        )
        
        # Exporter le mesh avec texture
        mesh.export(output_path)
        
        print(f"  ✅ Textured mesh saved: {output_path}")
        print(f"  🖼️  Texture file: {texture_filename}")
        
        return output_path
    
    def generate_3d_model(self, image_path, output_dir="output", 
                         remove_bg=True, foreground_ratio=0.85,
                         save_format="obj", render_video=False,
                         apply_texture=True):
        """
        Génère un modèle 3D à partir d'une image
        
        Args:
            image_path (str): Chemin vers l'image d'entrée
            output_dir (str): Dossier de sortie
            remove_bg (bool): Si True, supprime l'arrière-plan
            foreground_ratio (float): Ratio du premier plan
            save_format (str): Format de sauvegarde ('obj' ou 'glb')
            render_video (bool): Si True, génère une vidéo de rendu
            apply_texture (bool): Si True, applique la texture de l'image au modèle 3D
        
        Returns:
            str: Chemin vers le modèle 3D généré
        """
        print(f"\n{'='*60}")
        print(f"🎯 Starting 3D generation from image")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Créer le dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Prétraiter l'image
        preprocess_start = time.time()
        image = self.preprocess_image(image_path, remove_bg, foreground_ratio)
        preprocess_time = time.time() - preprocess_start
        print(f"  ⏱️  Preprocessing: {preprocess_time:.2f}s")
        
        # Sauvegarder l'image prétraitée
        processed_image_path = os.path.join(output_dir, "input_processed.png")
        image.save(processed_image_path)
        print(f"  💾 Processed image saved: {processed_image_path}")
        
        # Générer le code de scène 3D
        print("  🧠 Generating 3D scene code...")
        model_start = time.time()
        with torch.no_grad():
            scene_codes = self.model([image], device=self.device)
        model_time = time.time() - model_start
        print(f"  ⏱️  Model inference: {model_time:.2f}s")
        
        # Extraire le mesh
        print("  🔨 Extracting 3D mesh...")
        mesh_start = time.time()
        meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=self.mc_resolution)
        mesh_time = time.time() - mesh_start
        print(f"  ⏱️  Mesh extraction: {mesh_time:.2f}s")
        
        # =====================================================================
        # RÉORIENTATION DU MODÈLE 3D
        # =====================================================================
        # Le modèle généré par TripoSR n'est pas toujours dans la même 
        # orientation que l'image 2D source. On applique une rotation pour
        # aligner le modèle 3D avec l'orientation de l'image.
        #
        # CONFIGURATION ACTUELLE : Rotation Z de 90°
        # Cette orientation a été testée et validée.
        #
        # Pour tester d'autres orientations, utilisez :
        #   python scripts/test_orientations.py chemin/vers/model.obj
        #
        # Voir docs/CONFIG_ORIENTATION.md pour plus de détails
        # =====================================================================
        
        print("  🔄 Rotating model to match 2D image orientation...")
        mesh = meshes[0]
        
        # Rotation de 90° autour de l'axe Z (rotation dans le plan)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.pi / 2,  # 90 degrés (modifiable)
            direction=[0, 0, 1],  # Axe Z (options: [1,0,0]=X, [0,1,0]=Y, [0,0,1]=Z)
            point=[0, 0, 0]
        )
        mesh.apply_transform(rotation_matrix)
        
        # Sauvegarder le mesh réorienté
        export_start = time.time()
        mesh_filename = f"model.{save_format}"
        mesh_path = os.path.join(output_dir, mesh_filename)
        mesh.export(mesh_path)
        export_time = time.time() - export_start
        print(f"  ⏱️  Mesh export: {export_time:.2f}s")
        
        # Appliquer la texture si demandé
        if apply_texture:
            try:
                # =====================================================================
                # CONFIGURATION OPTIMALE IDENTIFIÉE : proj2_flip_v
                # =====================================================================
                # Après tests exhaustifs :
                # - Projection : proj2 (arctan2(X, Z) - vue de côté XZ)
                # - Transformation : flip_v (miroir vertical, v = 1.0 - v)
                #
                # Cette configuration aligne parfaitement la texture avec le modèle 3D
                # qui a été préalablement tourné de 90° autour de l'axe Z.
                #
                # Pour tester d'autres configurations, décommentez la section
                # "TEST MULTIPLE VARIANTS" ci-dessous.
                # =====================================================================
                
                print(f"\n  🎨 Applying optimized texture (proj2_flip_v)...")
                
                # Charger le mesh avec trimesh
                mesh_trimesh = trimesh.load(mesh_path)
                
                # Créer le nom de fichier pour le modèle texturé
                textured_mesh_filename = f"model_textured.{save_format}"
                textured_mesh_path = os.path.join(output_dir, textured_mesh_filename)
                
                # Appliquer la texture avec la configuration optimale
                self.apply_texture_to_mesh(mesh_trimesh, image_path, textured_mesh_path, uv_config="proj2_flip_v")
                
                print(f"  ✅ Texture applied successfully")
                print(f"  📦 Textured model: {textured_mesh_path}")
                
                # Mettre à jour le chemin du mesh pour le retour
                mesh_path = textured_mesh_path
                
                # =====================================================================
                # TEST MULTIPLE VARIANTS (Décommenter pour debug/test)
                # =====================================================================
                # Pour générer toutes les variantes (24 configurations) comme avant,
                # décommentez le code ci-dessous. Cela peut être utile pour :
                # - Tester de nouvelles images
                # - Déboguer des problèmes d'orientation
                # - Comparer différentes configurations
                # =====================================================================
                """
                uv_configs = {
                    "proj1_default": "Projection XY, Z vertical",
                    "proj1_flip_u": "Projection XY, miroir horizontal",
                    "proj1_flip_v": "Projection XY, miroir vertical",
                    "proj2_default": "Projection XZ, Y vertical",
                    "proj2_flip_u": "Projection XZ, miroir horizontal",
                    "proj2_flip_v": "Projection XZ, miroir vertical ✅ OPTIMAL",
                    "proj3_default": "Projection YZ, X vertical",
                    "proj3_flip_u": "Projection YZ, miroir horizontal",
                    "proj3_flip_v": "Projection YZ, miroir vertical",
                    "proj4_default": "Projection X,-Y (compensation rotation)",
                    "proj4_flip_u": "Projection X,-Y, miroir horizontal",
                    "proj4_flip_v": "Projection X,-Y, miroir vertical",
                    "proj5_default": "Projection Y,-X",
                    "proj5_flip_u": "Projection Y,-X, miroir horizontal",
                    "proj5_flip_v": "Projection Y,-X, miroir vertical",
                    "proj6_default": "Projection -X,Y",
                    "proj6_flip_u": "Projection -X,Y, miroir horizontal",
                    "proj6_flip_v": "Projection -X,Y, miroir vertical",
                    "default": "Config actuelle (aucune transformation)",
                    "flip_u": "Config actuelle, miroir horizontal",
                    "flip_v": "Config actuelle, miroir vertical",
                    "flip_both": "Config actuelle, miroir H+V",
                }
                
                print(f"\n  🎨 Generating {len(uv_configs)} texture projection variants...")
                textured_paths = []
                
                for config_name, config_desc in uv_configs.items():
                    mesh_copy = mesh_trimesh.copy()
                    variant_filename = f"model_tex_{config_name}.{save_format}"
                    variant_path = os.path.join(output_dir, variant_filename)
                    self.apply_texture_to_mesh(mesh_copy, image_path, variant_path, uv_config=config_name)
                    textured_paths.append((config_name, config_desc, variant_path))
                    print(f"    ✅ {config_name}")
                
                # Créer un guide pour les variantes
                uv_guide_path = os.path.join(output_dir, "UV_PROJECTIONS_GUIDE.txt")
                with open(uv_guide_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("🎨 GUIDE DES PROJECTIONS UV - VARIANTES DE TEST\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Configurations générées : {len(uv_configs)}\n")
                    f.write("✅ Configuration optimale identifiée : proj2_flip_v\n\n")
                    f.write("-" * 80 + "\n\n")
                    for config_name, config_desc, config_path in textured_paths:
                        f.write(f"  • {os.path.basename(config_path)}\n")
                        f.write(f"    → {config_desc}\n\n")
                    f.write("=" * 80 + "\n")
                print(f"  📋 Variants guide: {uv_guide_path}")
                """
                
            except Exception as e:
                print(f"  ⚠️  Could not apply texture: {e}")
                print(f"  💡 Using mesh without texture")
        
        # Générer une vidéo de rendu si demandé
        if render_video:
            print("  🎬 Rendering video...")
            render_start = time.time()
            from tsr.utils import save_video
            
            render_images = self.model.render(scene_codes, n_views=30, return_type="pil")
            video_path = os.path.join(output_dir, "render.mp4")
            save_video(render_images[0], video_path, fps=30)
            render_time = time.time() - render_start
            print(f"  ⏱️  Video rendering: {render_time:.2f}s")
            print(f"  🎥 Video saved: {video_path}")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ 3D model generated successfully!")
        print(f"💾 Model saved: {mesh_path}")
        print(f"� Orientation: Z-axis rotation (90°)")
        print(f"⚡ Total time: {total_time:.2f}s")
        print(f"{'='*60}\n")
        
        # Afficher l'utilisation de la mémoire GPU
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU Memory: {memory_used:.2f}/{memory_total:.2f} GB")
        
        return mesh_path


def main():
    """Fonction principale pour tester le générateur 3D"""
    
    # Configuration
    input_image = "ai-generated-image-gpu-fast.png"  # Image générée par generate_image_fast.py
    output_directory = "output/3d_model"
    
    # Vérifier si l'image existe
    if not os.path.exists(input_image):
        print(f"⚠️  Warning: {input_image} not found!")
        print("   Please run generate_image_fast.py first to generate an image,")
        print("   or provide your own image path.")
        
        # Chercher une image d'exemple dans le dossier output
        if os.path.exists("output"):
            images = [f for f in os.listdir("output") if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                input_image = os.path.join("output", images[0])
                print(f"   Using: {input_image}")
            else:
                print("   No image found in output folder.")
                return
        else:
            print("   No output folder found.")
            return
    
    # Créer le générateur 3D
    generator = TripoSR3DGenerator(
        device=device,
        chunk_size=8192,  # Optimisé pour RTX 3060
        mc_resolution=320  # Résolution augmentée pour meilleure qualité
    )
    
    # Générer le modèle 3D
    mesh_path = generator.generate_3d_model(
        image_path=input_image,
        output_dir=output_directory,
        remove_bg=True,  # Supprimer l'arrière-plan automatiquement
        foreground_ratio=0.85,
        save_format="obj",  # Format OBJ (compatible avec la plupart des logiciels 3D)
        render_video=False  # Mettre à True pour générer une vidéo de rendu
    )
    
    print(f"\n🎉 Done! You can open {mesh_path} in Blender, MeshLab, or any 3D viewer.")


if __name__ == "__main__":
    main()
