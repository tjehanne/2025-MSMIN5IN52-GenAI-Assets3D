# 🌀 Projet : Génération rapide de modèles 3D à partir d’images ou de texte  

## 📌 Description  
Ce projet a pour objectif de concevoir une application permettant de **générer rapidement des modèles 3D simples** à partir :  
- d’une **image** (via un pipeline *Image-to-3D*, ex. **TripoSR**)  
- d’un **texte** (via un pipeline *Text-to-3D*, ex. **Luma Genie**)  

Les modèles produits sont directement exploitables dans un moteur de jeu (Unity, Unreal Engine, Godot…).  

L’application se veut simple d’utilisation et vise à **accélérer la création d’assets 3D** pour le prototypage et le développement de jeux vidéo.  

---

## 🚀 Fonctionnalités principales  
- 🖼️ **Image-to-3D** : Générer un objet 3D à partir d’une image.  
- ✍️ **Text-to-3D** : Générer un objet 3D à partir d’une description textuelle.  
- 💾 Export des modèles en formats standards (.glb, .fbx, .obj).  
- 🔧 Interface simple (CLI ou Web) pour tester rapidement les générations.  
- ⚡ Optimisation pour **RTX 3060 Laptop GPU** (CUDA).  

---

## 🛠️ Technologies utilisées  
- **Python 3.10+**  
- **TripoSR** (Image-to-3D)  
- **Luma Genie** (Text-to-3D)  
- **PyTorch** (accélération GPU avec CUDA)  
- **Blender / trimesh** (traitement et export des modèles)  
- (optionnel) **FastAPI + React** pour une interface Web  

---

## 📂 Structure du projet  
```bash
.
├── README.md              # Documentation du projet
├── requirements.txt       # Dépendances Python
├── app/
│   ├── main.py            # Point d’entrée principal
│   ├── image_to_3d.py     # Génération 3D depuis image (TripoSR)
│   ├── text_to_3d.py      # Génération 3D depuis texte (Luma Genie)
│   ├── utils/             # Fonctions utilitaires (export, nettoyage, etc.)
│   └── outputs/           # Modèles générés
└── web/                   # (optionnel) Frontend Web
