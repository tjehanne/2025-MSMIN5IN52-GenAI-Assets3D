#!/usr/bin/env python
# coding: utf-8

"""
Interface graphique Gradio pour la génération de modèles 3D
Permet de saisir un prompt et ajuster les paramètres de qualité
"""

import os
import time
import threading
import numpy as np
import torch
import gradio as gr
from src.pipeline import text_to_3d_pipeline

# Fonction pour recharger les modèles disponibles (y compris les modèles personnalisés)
def get_available_models():
    """Recharge la liste des modèles disponibles"""
    from src.generate_image import load_custom_models, AVAILABLE_MODELS as BASE_MODELS
    
    models = dict(BASE_MODELS)
    models.update(load_custom_models())
    return models

# Charger les modèles au démarrage
STABLE_DIFFUSION_MODELS = get_available_models()

# Variable globale pour gérer l'interruption
generation_cancelled = False
generation_lock = threading.Lock()

def generate_3d_from_text(
    prompt,
    sd_model,
    image_steps,
    image_guidance,
    image_width,
    image_height,
    resolution_3d,
    save_format,
    render_video,
    progress=gr.Progress()
):
    """
    Fonction appelée par l'interface Gradio pour la génération à partir de texte
    """
    global generation_cancelled
    
    # Réinitialiser le flag d'annulation
    with generation_lock:
        generation_cancelled = False
    
    if not prompt or prompt.strip() == "":
        return None, None, None, None, "❌ Veuillez entrer une description !"
    
    try:
        # Vérifier l'annulation avant de commencer
        with generation_lock:
            if generation_cancelled:
                return None, None, None, None, "🛑 **Génération annulée par l'utilisateur**"
        
        progress(0.05, desc="🔄 Initialisation...")
        
        # Vérifier l'annulation
        with generation_lock:
            if generation_cancelled:
                return None, None, None, None, "🛑 **Génération annulée avant le démarrage**"
        
        progress(0.1, desc="🎨 Génération de l'image 2D...")
        
        # Convertir le nom du modèle en chemin
        model_path = STABLE_DIFFUSION_MODELS.get(sd_model, sd_model) if sd_model else None
        
        # Vérifier l'annulation avant la génération
        with generation_lock:
            if generation_cancelled:
                return None, None, None, None, "🛑 **Génération annulée par l'utilisateur**"
        
        progress(0.2, desc="🎨 Chargement du modèle d'IA...")
        
        # Vérifier l'annulation
        with generation_lock:
            if generation_cancelled:
                return None, None, None, None, "🛑 **Génération annulée pendant le chargement**"
        
        progress(0.3, desc="🖼️ Création de l'image 2D...")
        
        # Générer le modèle 3D
        result = text_to_3d_pipeline(
            prompt=prompt.strip(),
            output_dir="output/gradio",
            image_steps=int(image_steps),
            image_guidance=float(image_guidance),
            image_width=int(image_width),
            image_height=int(image_height),
            model_3d_resolution=int(resolution_3d),
            save_format=save_format,
            render_video=render_video,
            sd_model=model_path
        )
        
        # Vérifier l'annulation après la génération
        with generation_lock:
            if generation_cancelled:
                return None, None, None, None, "🛑 **Génération annulée par l'utilisateur**"
        
        progress(1.0, desc="✅ Terminé !")
        
        # Préparer les fichiers de sortie
        image_path = result['image_path'] if os.path.exists(result['image_path']) else None
        mesh_path = result['mesh_path'] if os.path.exists(result['mesh_path']) else None
        video_path = result.get('video_path') if render_video and 'video_path' in result else None
        
        # Message de succès
        message = f"""
✅ **Génération réussie !**

📝 Prompt : {result['prompt']}
⏱️ Temps total : {result['total_time']:.1f}s

📂 Fichiers générés :
- 📸 Image 2D : {os.path.basename(image_path)}
- 🎲 Modèle 3D : {os.path.basename(mesh_path)}
{'- 🎬 Vidéo : ' + os.path.basename(video_path) if video_path else ''}

💡 Utilisez le viewer 3D interactif ou téléchargez le modèle pour l'ouvrir dans Blender/MeshLab.
"""
        
        return image_path, mesh_path, mesh_path, video_path, message
        
    except Exception as e:
        import traceback
        error_msg = f"❌ **Erreur lors de la génération :**\n\n```\n{str(e)}\n```\n\n**Détails :**\n```\n{traceback.format_exc()}\n```"
        return None, None, None, None, error_msg


def generate_3d_from_image(
    input_image,
    resolution_3d,
    save_format,
    render_video,
    progress=gr.Progress()
):
    """
    Fonction appelée par l'interface Gradio pour la génération à partir d'une image
    """
    global generation_cancelled
    
    # Réinitialiser le flag d'annulation
    with generation_lock:
        generation_cancelled = False
    
    if input_image is None:
        return None, None, None, "❌ Veuillez déposer une image !"
    
    try:
        from src.generate_3d import TripoSR3DGenerator
        
        # Vérifier l'annulation avant de commencer
        with generation_lock:
            if generation_cancelled:
                return None, None, None, "🛑 **Génération annulée par l'utilisateur**"
        
        progress(0.1, desc="🔄 Initialisation...")
        
        # Créer le dossier de sortie
        os.makedirs("output/gradio", exist_ok=True)
        
        # Sauvegarder l'image uploadée
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = f"output/gradio/uploaded_{timestamp}.png"
        
        # Si input_image est un chemin de fichier (string)
        if isinstance(input_image, str):
            from PIL import Image
            img = Image.open(input_image)
            img.save(image_path)
        else:
            # Si c'est déjà une image PIL ou numpy array
            from PIL import Image
            if isinstance(input_image, np.ndarray):
                img = Image.fromarray(input_image)
            else:
                img = input_image
            img.save(image_path)
        
        progress(0.2, desc="🎨 Chargement du modèle 3D...")
        
        # Vérifier l'annulation
        with generation_lock:
            if generation_cancelled:
                return None, None, None, "🛑 **Génération annulée**"
        
        # Initialiser le générateur 3D
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        generator = TripoSR3DGenerator(
            device=device,
            mc_resolution=int(resolution_3d)
        )
        
        progress(0.4, desc="🎲 Génération du modèle 3D...")
        
        # Vérifier l'annulation
        with generation_lock:
            if generation_cancelled:
                return None, None, None, "🛑 **Génération annulée**"
        
        # Générer le modèle 3D
        start_time = time.time()
        mesh_path = generator.generate_3d_model(
            image_path,
            output_dir="output/gradio",
            remove_bg=True,
            foreground_ratio=0.85,
            save_format=save_format,
            render_video=render_video
        )
        
        total_time = time.time() - start_time
        
        # Vérifier si une vidéo a été générée
        video_path = None
        if render_video:
            potential_video = os.path.join("output/gradio", "render.mp4")
            if os.path.exists(potential_video):
                video_path = potential_video
        
        progress(1.0, desc="✅ Terminé !")
        
        # Message de succès
        message = f"""
✅ **Génération réussie !**

⏱️ Temps total : {total_time:.1f}s

📂 Fichiers générés :
- 📸 Image source : {os.path.basename(image_path)}
- 🎲 Modèle 3D : {os.path.basename(mesh_path)}
{'- 🎬 Vidéo : ' + os.path.basename(video_path) if video_path else ''}
"""
        
        return mesh_path, mesh_path, video_path, message
        
    except Exception as e:
        import traceback
        error_msg = f"❌ **Erreur lors de la génération :**\n\n```\n{str(e)}\n```\n\n**Détails :**\n```\n{traceback.format_exc()}\n```"
        return None, None, None, error_msg


def cancel_generation():
    """
    Fonction pour annuler la génération en cours
    """
    global generation_cancelled
    with generation_lock:
        generation_cancelled = True
    return "⚠️ **Annulation en cours...** La génération sera interrompue dès que possible."


# Exemples de prompts
example_prompts = [
    ["a futuristic robot head, metallic chrome, detailed", "SD 1.4 (Défaut)", 25, 7.5, 512, 512, 320, "obj", False],
    ["a dragon skull, ancient bone, fantasy art, detailed teeth", "SD 1.4 (Défaut)", 25, 7.5, 512, 512, 320, "obj", False],
    ["a magical crystal ball on brass stand, glowing blue", "DreamShaper", 25, 7.5, 512, 512, 320, "obj", False],
    ["a medieval sword with runes, steel blade, ornate handle", "SD 1.5", 25, 7.5, 512, 768, 320, "obj", False],
    ["a steampunk clockwork mechanism, brass gears, intricate", "Realistic Vision", 30, 7.5, 512, 512, 384, "obj", False],
]

# Créer l'interface Gradio
with gr.Blocks(title="Générateur de Modèles 3D", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Générateur de Modèles 3D avec IA")
    
    with gr.Tabs() as input_tabs:
        # Onglet 1: Génération par texte
        with gr.Tab("📝 Texte → 3D"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Description")
                    
                    prompt_input = gr.Textbox(
                        label="Décrivez votre modèle 3D",
                        placeholder="Ex: a futuristic robot head, metallic, detailed",
                        lines=3
                    )
                    
                    gr.Markdown("### 🎨 Modèle de Génération")
                    
                    # Bouton pour recharger les modèles
                    reload_btn = gr.Button("🔄 Recharger les modèles", size="sm", variant="secondary")
                    
                    sd_model_selector = gr.Dropdown(
                        choices=list(STABLE_DIFFUSION_MODELS.keys()),
                        value="SD 1.4 (Défaut)",
                        label="Modèle Stable Diffusion"
                    )
                    
                    def reload_models():
                        """Recharge la liste des modèles disponibles"""
                        global STABLE_DIFFUSION_MODELS
                        STABLE_DIFFUSION_MODELS = get_available_models()
                        return gr.Dropdown(choices=list(STABLE_DIFFUSION_MODELS.keys()))
                    
                    reload_btn.click(
                        fn=reload_models,
                        outputs=[sd_model_selector]
                    )
                    
                    gr.Markdown("### 🎚️ Profils rapides")
                    
                    with gr.Row():
                        fast_btn = gr.Button("⚡ Rapide", size="sm")
                        standard_btn = gr.Button("⭐ Standard", size="sm")
                        quality_btn = gr.Button("💎 Haute Qualité", size="sm")
                        max_btn = gr.Button("🏆 Maximum", size="sm")
                    
                    gr.Markdown("### ⚙️ Paramètres de Qualité")
                    
                    with gr.Accordion("📐 Paramètres d'image 2D", open=False):
                        gr.Markdown("#### 🖼️ Dimensions de l'image")
                        
                        with gr.Row():
                            image_width = gr.Slider(
                                minimum=256,
                                maximum=1024,
                                value=512,
                                step=64,
                                label="Largeur (pixels)"
                            )
                            
                            image_height = gr.Slider(
                                minimum=256,
                                maximum=1024,
                                value=512,
                                step=64,
                                label="Hauteur (pixels)"
                            )
                        
                        # Boutons de presets de dimensions
                        with gr.Row():
                            square_btn = gr.Button("⬛ Carré 512x512", size="sm")
                            portrait_btn = gr.Button("📱 Portrait 512x768", size="sm")
                            landscape_btn = gr.Button("🖼️ Paysage 768x512", size="sm")
                            hd_btn = gr.Button("📺 HD 768x768", size="sm")
                        
                        gr.Markdown("#### 🎨 Qualité de génération 2D")
                        
                        image_steps = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=25,
                            step=5,
                            label="Steps (qualité de l'image)"
                        )
                        
                        image_guidance = gr.Slider(
                            minimum=5.0,
                            maximum=15.0,
                            value=7.5,
                            step=0.5,
                            label="🎯 Guidance (fidélité au prompt)"
                        )
                    
                    with gr.Accordion("🎲 Paramètres de modèle 3D", open=False):
                        resolution_3d = gr.Slider(
                            minimum=128,
                            maximum=512,
                            value=320,
                            step=64,
                            label="Résolution du modèle 3D"
                        )
                        
                        save_format = gr.Radio(
                            choices=["obj", "glb"],
                            value="obj",
                            label="💾 Format de sortie"
                        )
                        
                        render_video = gr.Checkbox(
                            label="🎬 Générer une vidéo de rendu (rotation 360°)",
                            value=False
                        )
                    
                    with gr.Row():
                        generate_btn = gr.Button("🚀 Générer le modèle 3D", variant="primary", size="lg", scale=3)
                        cancel_btn = gr.Button("🛑 Arrêter", variant="stop", size="lg", scale=1)
        
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Résultats")
                    
                    status_output = gr.Markdown("")
                    
                    with gr.Tabs():
                        with gr.Tab("📸 Image 2D"):
                            image_output = gr.Image(label="Image générée", type="filepath")
                        
                        with gr.Tab("🎲 Viewer 3D Interactif"):
                            model_viewer = gr.Model3D(
                                label="Modèle 3D interactif",
                                clear_color=[0.1, 0.1, 0.1, 1.0],
                                camera_position=[90, 90, 3.5],
                                height=500
                            )
                        
                        with gr.Tab("💾 Téléchargement"):
                            mesh_output = gr.File(label="Télécharger le modèle 3D")
                        
                        with gr.Tab("🎬 Vidéo"):
                            video_output = gr.Video(label="Vidéo de rendu")
            
            # Exemples
            gr.Examples(
                examples=example_prompts,
                inputs=[prompt_input, sd_model_selector, image_steps, image_guidance, image_width, image_height, resolution_3d, save_format, render_video]
            )
        
        # Onglet 2: Génération à partir d'une image
        with gr.Tab("🖼️ Image → 3D"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ Image Source")
                    
                    image_input = gr.Image(
                        label="Déposez votre image ici",
                        type="filepath",
                        height=300
                    )
                    
                    gr.Markdown("### 🎚️ Profils rapides")
                    
                    with gr.Row():
                        fast_btn_img = gr.Button("⚡ Rapide", size="sm")
                        standard_btn_img = gr.Button("⭐ Standard", size="sm")
                        quality_btn_img = gr.Button("💎 Haute Qualité", size="sm")
                        max_btn_img = gr.Button("🏆 Maximum", size="sm")
                    
                    gr.Markdown("### ⚙️ Paramètres de Qualité")
                    
                    with gr.Accordion("🎲 Paramètres avancés", open=False):
                        resolution_3d_img = gr.Slider(
                            minimum=128,
                            maximum=512,
                            value=320,
                            step=64,
                            label="Résolution du modèle 3D"
                        )
                        
                        save_format_img = gr.Radio(
                            choices=["obj", "glb"],
                            value="obj",
                            label="💾 Format de sortie"
                        )
                        
                        render_video_img = gr.Checkbox(
                            label="🎬 Générer une vidéo de rendu (rotation 360°)",
                            value=False
                        )
                    
                    with gr.Row():
                        generate_btn_img = gr.Button("🚀 Générer le modèle 3D", variant="primary", size="lg", scale=3)
                        cancel_btn_img = gr.Button("🛑 Arrêter", variant="stop", size="lg", scale=1)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Résultats")
                    
                    status_output_img = gr.Markdown("")
                    
                    with gr.Tabs():
                        with gr.Tab("🎲 Viewer 3D Interactif"):
                            model_viewer_img = gr.Model3D(
                                label="Modèle 3D interactif",
                                clear_color=[0.1, 0.1, 0.1, 1.0],
                                camera_position=[90, 90, 3.5],
                                height=500
                            )
                        
                        with gr.Tab("💾 Téléchargement"):
                            mesh_output_img = gr.File(label="Télécharger le modèle 3D")
                        
                        with gr.Tab("🎬 Vidéo"):
                            video_output_img = gr.Video(label="Vidéo de rendu")
    
    # Événements des boutons de profil (Texte → 3D)
    fast_btn.click(
        lambda: [10, 7.5, 512, 512, 192],
        outputs=[image_steps, image_guidance, image_width, image_height, resolution_3d]
    )
    
    standard_btn.click(
        lambda: [15, 7.5, 512, 512, 256],
        outputs=[image_steps, image_guidance, image_width, image_height, resolution_3d]
    )
    
    quality_btn.click(
        lambda: [25, 7.5, 512, 512, 320],
        outputs=[image_steps, image_guidance, image_width, image_height, resolution_3d]
    )
    
    max_btn.click(
        lambda: [35, 7.5, 768, 768, 384],
        outputs=[image_steps, image_guidance, image_width, image_height, resolution_3d]
    )
    
    # Événements des boutons de dimensions
    square_btn.click(
        lambda: [512, 512],
        outputs=[image_width, image_height]
    )
    
    portrait_btn.click(
        lambda: [512, 768],
        outputs=[image_width, image_height]
    )
    
    landscape_btn.click(
        lambda: [768, 512],
        outputs=[image_width, image_height]
    )
    
    hd_btn.click(
        lambda: [768, 768],
        outputs=[image_width, image_height]
    )
    
    # Événement de génération
    generate_event = generate_btn.click(
        fn=generate_3d_from_text,
        inputs=[
            prompt_input,
            sd_model_selector,
            image_steps,
            image_guidance,
            image_width,
            image_height,
            resolution_3d,
            save_format,
            render_video
        ],
        outputs=[image_output, mesh_output, model_viewer, video_output, status_output]
    )
    
    # Événement d'annulation
    cancel_btn.click(
        fn=cancel_generation,
        inputs=None,
        outputs=status_output,
        cancels=[generate_event]  # Annule l'événement de génération en cours
    )
    
    # Événements des boutons de profil (Image → 3D)
    fast_btn_img.click(
        lambda: 192,
        outputs=[resolution_3d_img]
    )
    
    standard_btn_img.click(
        lambda: 256,
        outputs=[resolution_3d_img]
    )
    
    quality_btn_img.click(
        lambda: 320,
        outputs=[resolution_3d_img]
    )
    
    max_btn_img.click(
        lambda: 384,
        outputs=[resolution_3d_img]
    )
    
    # Événement de génération (Image → 3D)
    generate_event_img = generate_btn_img.click(
        fn=generate_3d_from_image,
        inputs=[
            image_input,
            resolution_3d_img,
            save_format_img,
            render_video_img
        ],
        outputs=[mesh_output_img, model_viewer_img, video_output_img, status_output_img]
    )
    
    # Événement d'annulation (Image → 3D)
    cancel_btn_img.click(
        fn=cancel_generation,
        inputs=None,
        outputs=status_output_img,
        cancels=[generate_event_img]
    )
    
if __name__ == "__main__":
    print("🚀 Lancement de l'interface Gradio...")
    print("📱 Une fenêtre de navigateur va s'ouvrir automatiquement")
    print("🌐 Ou accédez manuellement à l'URL affichée ci-dessous\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,  # Trouve automatiquement un port libre
        share=False,
        inbrowser=True
    )
