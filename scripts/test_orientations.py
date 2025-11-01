#!/usr/bin/env python
# coding: utf-8

"""
Script pour tester rapidement les différentes orientations d'un modèle 3D existant
"""

import os
import sys
import numpy as np
import trimesh

# Ajouter le dossier src au path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
sys.path.insert(0, src_path)

def generate_orientation_variants(input_mesh_path, output_dir=None):
    """
    Génère plusieurs versions d'un modèle 3D avec différentes orientations
    
    Args:
        input_mesh_path (str): Chemin vers le modèle 3D d'entrée
        output_dir (str): Dossier de sortie (optionnel)
    """
    print(f"📦 Loading mesh: {input_mesh_path}")
    
    # Charger le mesh
    mesh = trimesh.load(input_mesh_path)
    
    # Déterminer le dossier de sortie
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_mesh_path), "orientations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtenir l'extension du fichier
    file_ext = os.path.splitext(input_mesh_path)[1]
    
    # Définir toutes les rotations possibles
    rotations = {
        "original": None,
        "x_90": (np.pi / 2, [1, 0, 0]),
        "x_-90": (-np.pi / 2, [1, 0, 0]),
        "x_180": (np.pi, [1, 0, 0]),
        "y_90": (np.pi / 2, [0, 1, 0]),
        "y_-90": (-np.pi / 2, [0, 1, 0]),
        "y_180": (np.pi, [0, 1, 0]),
        "z_90": (np.pi / 2, [0, 0, 1]),
        "z_-90": (-np.pi / 2, [0, 0, 1]),
        "z_180": (np.pi, [0, 0, 1]),
    }
    
    print(f"\n🔄 Generating {len(rotations)} orientation variants...\n")
    
    mesh_paths = []
    for rot_name, rot_params in rotations.items():
        # Copier le mesh
        mesh_copy = mesh.copy()
        
        # Appliquer la rotation si spécifiée
        if rot_params is not None:
            angle, direction = rot_params
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle=angle,
                direction=direction,
                point=[0, 0, 0]
            )
            mesh_copy.apply_transform(rotation_matrix)
        
        # Sauvegarder
        output_filename = f"model_{rot_name}{file_ext}"
        output_path = os.path.join(output_dir, output_filename)
        mesh_copy.export(output_path)
        mesh_paths.append((rot_name, output_path))
        print(f"  ✅ {output_filename}")
    
    # Créer un fichier guide
    guide_path = os.path.join(output_dir, "README.txt")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("🔄 GUIDE DES ORIENTATIONS\n")
        f.write("=" * 70 + "\n\n")
        f.write("Ce dossier contient le modèle 3D avec différentes orientations.\n")
        f.write("Ouvrez chaque fichier pour trouver celui qui correspond le mieux.\n\n")
        
        for rot_name, rot_path in mesh_paths:
            f.write(f"\n📦 {os.path.basename(rot_path)}\n")
            
            if rot_name == "original":
                f.write("   → Orientation originale\n")
            elif "x_" in rot_name:
                angle = rot_name.split("_")[1]
                f.write(f"   → Rotation de {angle}° autour de l'axe X\n")
                f.write("      (bascule avant/arrière - comme hocher la tête)\n")
            elif "y_" in rot_name:
                angle = rot_name.split("_")[1]
                f.write(f"   → Rotation de {angle}° autour de l'axe Y\n")
                f.write("      (rotation gauche/droite - comme secouer la tête)\n")
            elif "z_" in rot_name:
                angle = rot_name.split("_")[1]
                f.write(f"   → Rotation de {angle}° autour de l'axe Z\n")
                f.write("      (rotation dans le plan - comme tourner une photo)\n")
    
    print(f"\n{'='*60}")
    print(f"✅ Génération terminée !")
    print(f"📂 Dossier de sortie: {output_dir}")
    print(f"📋 Guide: {guide_path}")
    print(f"{'='*60}\n")


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Génère plusieurs orientations d'un modèle 3D"
    )
    parser.add_argument(
        "input",
        help="Chemin vers le fichier 3D d'entrée (.obj, .glb, etc.)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Dossier de sortie (par défaut: ./orientations)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Vérifier que le fichier existe
    if not os.path.exists(args.input):
        print(f"❌ Erreur: Le fichier {args.input} n'existe pas")
        return 1
    
    # Générer les variantes
    generate_orientation_variants(args.input, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
