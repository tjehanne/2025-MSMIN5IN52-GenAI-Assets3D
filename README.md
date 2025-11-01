# Générateur de Modèles 3D par IA

Projet de génération de modèles 3D à partir de texte ou d'images, utilisant TripoSR et Stable Diffusion.

## 🎯 Fonctionnalités

- **Génération d'images 2D** à partir de descriptions textuelles (Stable Diffusion)
- **6 modèles Stable Diffusion** intégrés + support de modèles personnalisés
- **Téléchargement facile** de modèles depuis Hugging Face et Civitai
- **Conversion d'images en modèles 3D** avec TripoSR
- **Pipeline complet** : texte → image → modèle 3D
- **Interface web intuitive** avec Gradio
- **Contrôle des dimensions** d'image (préréglages + personnalisé)
- **Bouton Stop** pour annuler les générations en cours
- **Interface en ligne de commande** interactive
- **Préréglages de qualité** (Fast, Standard, High Quality, Maximum)

## 📁 Structure du Projet

```
.
├── src/                          # Code source principal
│   ├── __init__.py
│   ├── generate_image.py         # Génération d'images 2D
│   ├── generate_3d.py            # Conversion image → 3D
│   ├── pipeline.py               # Pipeline complet texte → 3D
│   └── interface/                # Interfaces utilisateur
│       ├── __init__.py
│       ├── gradio_app.py         # Interface web Gradio
│       └── cli.py                # Interface ligne de commande
│
├── scripts/                      # Scripts utilitaires
│   ├── demo_3d.py               # Démonstrations
│   ├── test_installation.py     # Test des dépendances
│   ├── launch_gradio.bat        # Lanceur interface web
│   ├── download_models.py       # Téléchargeur de modèles
│   ├── download_models_menu.bat # Menu interactif de téléchargement
│   └── test_model.py            # Tester un modèle téléchargé
│
├── docs/                         # Documentation
│   ├── GUIDE_3D.md              # Guide détaillé
│   ├── INSTALL.md               # Instructions d'installation
│   ├── QUICK_START_3D.md        # Démarrage rapide
│   ├── INTERFACE_GRADIO.md      # Guide interface Gradio
│   ├── PROMPT_EXAMPLES.md       # Exemples de prompts
│   └── QUALITY_SETTINGS.md      # Réglages de qualité
│
├── models/                       # Modèles IA
│   ├── TripoSR/                 # Modèle TripoSR
│   └── custom-models/           # Vos modèles personnalisés
│       ├── README.md            # Guide des modèles personnalisés
│       └── models_config.json   # Configuration automatique
│
├── output/                       # Fichiers générés
│
├── requirements.txt              # Dépendances Python principales
├── requirements-3d.txt           # Dépendances spécifiques 3D
└── README.md                     # Ce fichier
```

## 🚀 Installation Rapide

### 1. Cloner le dépôt
```bash
git clone https://github.com/votre-repo/2025-MSMIN5IN52-GenAI-Assets3D.git
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

### 4. Vérifier l'installation
```bash
python scripts/test_installation.py
```

## 💻 Utilisation

### Interface Web (Recommandé)
```bash
# Windows
scripts\launch_gradio.bat

# Linux/Mac
cd scripts && ./launch_gradio.bat
```
Puis ouvrez http://localhost:7860 dans votre navigateur.

**Nouveauté** : L'interface inclut maintenant :
- 🎨 Sélecteur de modèles Stable Diffusion
- 📐 Contrôle des dimensions d'image (préréglages + sliders)
- 🛑 Bouton Stop pour annuler les générations
- 🔄 Bouton pour recharger les modèles personnalisés

### Télécharger des Modèles Personnalisés

**Menu interactif** (Windows) :
```bash
scripts\download_models_menu.bat
```

**Ligne de commande** :
```bash
# Voir les modèles recommandés
python scripts\download_models.py --list-recommended

# Télécharger un modèle recommandé
python scripts\download_models.py --model realistic-vision

# Depuis Hugging Face
python scripts\download_models.py --hf-id runwayml/stable-diffusion-v1-5

# Convertir un fichier .safetensors
python scripts\download_models.py --convert models\custom-models\mon-modele.safetensors

# Tester un modèle téléchargé
python scripts\test_model.py "models/custom-models/mon-modele"
```

📚 **Guide complet** : Voir [docs/GUIDE_MODELES_PERSONNALISES.md](docs/GUIDE_MODELES_PERSONNALISES.md)

### Interface Ligne de Commande
```bash
python -m src.interface.cli
```

### Pipeline Python Direct
```python
from src.pipeline import text_to_3d_pipeline

text_to_3d_pipeline(
    prompt="a futuristic robot head, metallic, detailed",
    image_steps=25,
    model_3d_resolution=320
)
```

## 📖 Documentation

- **[Guide Complet](docs/GUIDE_3D.md)** - Documentation détaillée
- **[Guide Modèles Personnalisés](docs/GUIDE_MODELES_PERSONNALISES.md)** - ⭐ Télécharger et utiliser des modèles externes
- **[Installation](docs/INSTALL.md)** - Instructions d'installation pas à pas
- **[Démarrage Rapide](docs/QUICK_START_3D.md)** - Premiers pas
- **[Interface Gradio](docs/INTERFACE_GRADIO.md)** - Utilisation de l'interface web
- **[Exemples de Prompts](docs/PROMPT_EXAMPLES.md)** - Exemples de descriptions
- **[Réglages de Qualité](docs/QUALITY_SETTINGS.md)** - Optimisation des paramètres
- **[Modèles Personnalisés (README)](models/custom-models/README.md)** - Info sur le dossier custom-models

## 🎨 Exemples de Résultats

**Prompt**: "a dragon skull, fantasy art, ancient bone"
- Temps de génération : ~25 secondes
- Formats de sortie : OBJ, GLB, STL
- Résolution 3D : 320 (standard)

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
- **Transformers** : Modèles de langage
- **Rembg** : Suppression d'arrière-plan

## 📝 License

Ce projet utilise plusieurs bibliothèques open-source. Consultez les fichiers LICENSE respectifs.

## 👥 Auteurs

Projet développé dans le cadre du cours d'IA Générative - EPF 2025

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

---

Pour plus d'informations, consultez la [documentation complète](docs/GUIDE_3D.md).
    *   Description : Créez un agent qui scrape les sites de concurrents et synthétise les informations clés dans un rapport de veille hebdomadaire.
    *   Technologies clés : Scraping web, analyse et synthèse de texte.
    *   Difficulté : ⭐⭐⭐ (Avancé)

8.  **Assistant de Réponse à Appel d'Offres**
    *   Description : Concevez un système qui génère une première ébauche de réponse technique à un appel d'offres en se basant sur le cahier des charges et une base de connaissances interne.
    *   Technologies clés : RAG, génération de texte long format.
    *   Difficulté : ⭐⭐⭐⭐ (Très avancé)

### Catégorie : Génération Multimédia et Créative

9.  **Générateur d'histoires multimodales**
    *   Description : Développer une application qui génère une histoire courte et illustre chaque paragraphe avec une image générée.
    *   Technologies clés : API OpenAI (GPT-4o, DALL-E 3) ou modèles locaux.
    *   Difficulté : ⭐⭐⭐ (Avancé)

10. **Compositeur de Bandes Sonores d'Ambiance**
    *   Description : Créez une application qui génère des boucles musicales instrumentales pour des ambiances spécifiques (ex: "forêt mystérieuse", "cyberpunk sous la pluie").
    *   Technologies clés : API de génération musicale (Suno, Udio, Stable Audio).
    *   Difficulté : ⭐⭐⭐ (Avancé)

11. **Générateur de Storyboards Vidéo**
    *   Description : Développez un outil qui prend un court scénario et le transforme en une séquence de clips vidéo courts (storyboard animé).
    *   Technologies clés : LLM pour la scénarisation, API de génération vidéo (Luma Dream Machine).
    *   Difficulté : ⭐⭐⭐⭐ (Très avancé)

12. **Créateur d'Assets 3D pour le Prototypage**
    *   Description : Concevez une application qui génère rapidement des modèles 3D simples à partir d'images ou de textes pour une utilisation dans un moteur de jeu.
    *   Technologies clés : Modèles Image-to-3D (TripoSR) ou Text-to-3D (Luma Genie).
    *   Difficulté : ⭐⭐⭐ (Avancé)

### Catégorie : Outils de Développement et d'Analyse

13. **Auditeur de biais dans les LLMs**
    *   Description : Concevoir un outil qui évalue les biais d'un modèle de langage en lui soumettant des prompts standardisés et en analysant les réponses.
    *   Technologies clés : Prompt engineering, analyse de texte, visualisation de données.
    *   Difficulté : ⭐⭐ (Intermédiaire)

14. **Générateur de Contenu Structuré (CV, Facture, Rapport)**
    *   Description : Développez un workflow multi-agents qui prend des informations en langage naturel et génère un document structuré au format PDF.
    *   Technologies clés : Semantic Kernel, ReportLab (pour PDF), gestion de workflow.
    *   Difficulté : ⭐⭐⭐ (Avancé)

---
Pour toutes les autres informations (planning, critères d'évaluation détaillés), veuillez vous référer au document de modalités fourni dans le dossier du cours.

Bon projet à tous !