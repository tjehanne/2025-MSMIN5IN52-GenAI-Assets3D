# Générateur de Modèles 3D par IA

Projet de génération de modèles 3D à partir de texte ou d'images, utilisant TripoSR et Stable Diffusion.

## 🎯 Fonctionnalités

- **Génération d'images 2D** à partir de descriptions textuelles (Stable Diffusion)
- **Support de modèles personnalisés** Stable Diffusion
- **Conversion d'images en modèles 3D** avec TripoSR
- **Pipeline complet** : texte → image → modèle 3D
- **Interface web intuitive** avec Gradio

## 📁 Structure du Projet

```
.
├── src/                          # Code source principal
│   ├── __init__.py
│   ├── generate_image.py         # Génération d'images 2D
│   ├── generate_3d.py            # Conversion image → 3D
│   ├── pipeline.py               # Pipeline complet
│   ├── interface/                # Interfaces utilisateur
│   │   ├── __init__.py
│   │   └── gradio_app.py         # Interface web Gradio
│   └── utils/                    # Utilitaires
│
├── scripts/                      # Scripts utilitaires
│   └── launch_gradio.bat         # Lanceur interface web
│
├── models/                       # Modèles IA
│   ├── TripoSR/                 # Modèle TripoSR (3D)
│   └── custom-models/           # Modèles Stable Diffusion personnalisés
│
├── output/                       # Fichiers générés
│   └── gradio/                  # Sorties de l'interface Gradio
│
├── requirements.txt              # Dépendances Python principales
├── requirements-3d.txt           # Dépendances spécifiques 3D
└── README.md                     # Ce fichier
```

## 🚀 Installation Rapide

### 1. Cloner le dépôt
```bash
git clone https://github.com/tjehanne/2025-MSMIN5IN52-GenAI-Assets3D.git
cd 2025-MSMIN5IN52-GenAI-Assets3D
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
pip install -r requirements-3d.txt
```

### 4. Télécharger les modèles
Le modèle TripoSR sera téléchargé automatiquement au premier lancement.
Pour utiliser des modèles Stable Diffusion personnalisés, placez les fichiers `.safetensors` dans `models/custom-models/`.

## 💻 Utilisation

### Interface Web (Recommandé)
```bash
# Windows
scripts\launch_gradio.bat

# Linux/Mac
chmod +x scripts/launch_gradio.bat
./scripts/launch_gradio.bat
```
Puis ouvrez http://localhost:7860 dans votre navigateur.

**L'interface permet** :
- 🎨 Génération d'images 2D avec Stable Diffusion
- 🧊 Conversion en modèles 3D avec TripoSR
- 📦 Export en formats OBJ, GLB, STL

### Pipeline Python Direct
```python
from src.pipeline import generate_3d_from_text

# Génération complète texte → 3D
generate_3d_from_text(
    prompt="a futuristic robot head, metallic, detailed",
    output_dir="output/my_model"
)
```

## 🎨 Exemples de Résultats

**Prompt**: "a dragon skull, fantasy art, ancient bone"
- Génération d'image 2D avec Stable Diffusion
- Conversion en modèle 3D avec TripoSR
- Formats de sortie : OBJ, GLB, STL

## ⚙️ Configuration Requise

- **Python** : 3.8+
- **GPU** : NVIDIA avec CUDA (recommandé, 4GB+ VRAM)
- **RAM** : 8GB minimum, 16GB recommandé
- **Espace disque** : ~5GB pour les modèles IA

## 🛠️ Technologies Utilisées

- **TripoSR** : Génération de modèles 3D
- **Stable Diffusion** : Génération d'images 2D
- **PyTorch** : Framework de deep learning
- **Gradio** : Interface web interactive
- **Diffusers** : Pipeline de diffusion

## 📝 License

Ce projet utilise plusieurs bibliothèques open-source. Consultez les fichiers LICENSE respectifs dans le dossier `models/TripoSR/`.

## 👥 Auteurs

Projet développé dans le cadre du cours d'IA Générative - EPF 2025

---

**Projet #12 : Créateur d'Assets 3D pour le Prototypage**

Ce projet implémente une application qui génère rapidement des modèles 3D à partir de textes ou d'images pour une utilisation dans des moteurs de jeu ou applications 3D.

Technologies utilisées :
- Modèles Image-to-3D (TripoSR)
- Stable Diffusion pour la génération d'images

---