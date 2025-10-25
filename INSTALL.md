# 🚀 Installation Guide pour generate_image_fast.py

## 📋 Prérequis
- Python 3.8 ou supérieur
- GPU NVIDIA avec support CUDA (testé avec RTX 3060)
- 8 GB de RAM minimum
- 10 GB d'espace disque libre

## ⚡ Installation Rapide

### Option 1 : Script automatique (Recommandé)
```bash
install.bat
```

### Option 2 : Installation manuelle
```bash
# 1. PyTorch avec CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Packages AI
pip install -r requirements.txt
```

### Option 3 : Installation complète
```bash
pip install -r requirements-gpu.txt
```

## 🧪 Test de l'installation
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 🎯 Utilisation
```bash
cd versions
python generate_image_fast.py
```

## 📊 Performance attendue
- **Temps de génération**: 5-6 secondes (RTX 3060)  
- **Qualité**: Haute (15 steps optimisés)
- **Mémoire GPU**: ~2-3 GB utilisés

## 🔧 Dépannage

### GPU non détecté
1. Vérifiez les drivers NVIDIA
2. Réinstallez PyTorch avec CUDA
3. Redémarrez VS Code/terminal

### Erreur de mémoire
1. Fermez autres applications utilisant le GPU  
2. Réduisez `num_inference_steps` à 10-12
3. Utilisez `pipe.enable_model_cpu_offload()`

## 📦 Packages principaux
- `torch 2.5.1+cu121` - PyTorch avec CUDA
- `diffusers 0.35.2` - Stable Diffusion
- `transformers 4.57.1` - Modèles Hugging Face
- `accelerate 1.11.0` - Optimisations mémoire