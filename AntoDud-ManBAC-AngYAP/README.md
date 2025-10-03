# Projet Générateur d'histoires interactives avec image

## 📋 Description du projet

Ce projet est une extrapolation du sujet n°9 qui consiste à générer une histoire avec des images par paragraphe. 

Nous avons imaginé une version où, de manière similaire aux premières utilisations d'IA générative avec AIDungeon, on pourrait interagir dynamiquement avec l'histoire, avec des images qui accompagnent au fur et à mesure. 

Le concept se base sur une expérience narrative interactive où l'utilisateur peut influencer le déroulement de l'histoire en temps réel, combinant génération textuelle et visuelle pour créer une expérience immersive de jeu de rôle.

## 🎯 Objectifs

- Créer une expérience narrative interactive et immersive
- Générer du contenu textuel cohérent avec maintien du contexte
- Produire des illustrations visuelles correspondant à chaque séquence
- Permettre à l'utilisateur d'influencer le déroulement de l'histoire
- Maintenir la cohérence narrative sur de longues sessions

## ✨ Fonctionnalités principales

- **Génération d'histoire adaptative** : L'IA s'adapte aux choix et actions du joueur
- **Illustrations dynamiques** : Chaque scène est accompagnée d'une image générée automatiquement
- **Système de mémoire** : Maintien du contexte et de la cohérence narrative
- **Interface interactive** : Permet au joueur de saisir ses actions et décisions
- **Sauvegarde de progression** : Possibilité de reprendre une histoire en cours

## 🔄 Architecture et Workflow

### Workflow principal

1. **Initialisation** : L'utilisateur choisit un genre d'histoire (fantasy, sci-fi, horreur, etc.)
2. **Génération du contexte initial** : Création du cadre narratif et de la première scène
3. **Boucle interactive** :
   - Génération du texte narratif
   - Création de l'image correspondante
   - Présentation à l'utilisateur
   - Saisie de l'action/choix de l'utilisateur
   - Mise à jour du contexte et de la mémoire
   - Retour à la génération du texte

### Composants techniques

- **Modèle de génération textuelle** : Qwen 4B (ou supérieur selon les besoins en performance)
  - Gestion de la narration et du dialogue
  - Adaptation aux actions du joueur
  - Maintien de la cohérence narrative

- **Système de mémoire** : Fichier de contexte dynamique
  - Résumé de l'histoire en cours
  - Personnages et lieux importants
  - Actions précédentes du joueur
  - État actuel du monde narratif

- **Génération d'images** : Qwen-Image ou API externe
  - Illustration de chaque scène importante
  - Adaptation au style narratif choisi
  - Optimisation pour la vitesse de génération

## 🛠 Technologies envisagées

- **Backend** : Python avec FastAPI
- **Modèles IA** : 
  - Texte : Qwen3 4B/7B ou Llama 3.2
  - Images : Stable Diffusion XL, Qwen-Image ou API (OpenAI DALL-E, Midjourney)
- **Frontend** : Interface web (Next.js)
- **Orchestration** : LangChain ou Semantic Kernel pour la gestion des workflows
