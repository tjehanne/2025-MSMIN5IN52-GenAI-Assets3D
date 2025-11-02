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
    
    def generate_3d_model(self, image_path, output_dir="output", 
                         remove_bg=True, foreground_ratio=0.85,
                         save_format="obj", render_video=False):
        """
        Génère un modèle 3D à partir d'une image
        
        Args:
            image_path (str): Chemin vers l'image d'entrée
            output_dir (str): Dossier de sortie
            remove_bg (bool): Si True, supprime l'arrière-plan
            foreground_ratio (float): Ratio du premier plan
            save_format (str): Format de sauvegarde ('obj' ou 'glb')
            render_video (bool): Si True, génère une vidéo de rendu
        
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
        meshes = self.model.extract_mesh(scene_codes, has_vertex_color=False, resolution=self.mc_resolution)
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
