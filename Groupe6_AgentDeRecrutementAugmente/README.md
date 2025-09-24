# Agent de Recrutement Augmenté

## Description du Projet

Ce projet vise à développer une **application web** intelligente capable d'analyser un ensemble de CVs en les comparant à une fiche de poste donnée, afin de produire un classement justifié des candidats. L'objectif est d'assister les recruteurs dans leur processus de sélection en automatisant la première étape de tri, tout en fournissant des explications claires sur les décisions prises.

Concrètement, imaginez que vous avez reçu 100 candidatures pour un poste de développeur Python. Lire et analyser chaque CV manuellement prendrait des heures. Notre outil va automatiser cette tâche en lisant chaque CV, en extrayant les informations clés (compétences, expérience, formation), puis en les comparant aux exigences du poste. À la fin du processus, vous obtenez une liste ordonnée des candidats les plus pertinents, accompagnée d'une explication pour chaque position dans le classement.

L'application web permettra aux recruteurs d'uploader facilement les CVs et la fiche de poste via une interface graphique intuitive, puis de visualiser les résultats directement dans leur navigateur, sans avoir à utiliser la ligne de commande.

## Technologies Clés

- **RAG (Retrieval-Augmented Generation)** : Cette technologie permet au système de "lire" les CVs et la fiche de poste, d'extraire les informations pertinentes, puis de générer des réponses basées sur ces documents. C'est ce qui permet de comparer objectivement les candidats aux critères du poste.
- **Extraction d'entités** : Cette technique permet d'identifier automatiquement des éléments spécifiques dans un texte, comme les noms de compétences (Python, SQL, Django), les noms d'entreprises, les dates d'expérience, ou les diplômes. Cela transforme un CV en données structurées que l'ordinateur peut analyser.
- **Pandas** : C'est une bibliothèque Python très populaire pour manipuler et analyser des données. Elle sera utilisée pour stocker les informations extraites des CVs, effectuer des calculs (comme un score de correspondance), et générer des statistiques.
- **Modèle de langage (LLM)** : Un grand modèle de langage (comme ceux qui alimentent les chatbots) sera utilisé pour générer des explications naturelles et compréhensibles. Par exemple : "Ce candidat est classé premier car il possède 5 ans d'expérience en Python, dont 2 ans avec Django, ce qui correspond exactement aux exigences du poste."
- **Framework Web (Flask/Django)** : Pour créer l'interface web, permettant aux utilisateurs d'interagir avec l'outil via un navigateur.
- **HTML/CSS/JavaScript** : Pour concevoir une interface utilisateur moderne et réactive.

## Fonctionnalités Principales

1. **Interface Web** : Une page d'accueil intuitive où le recruteur peut uploader les CVs et la fiche de poste.
2. **Analyse des CVs** : L'outil peut lire des CVs au format PDF ou texte, extraire les informations clés (compétences, expérience professionnelle, formation, etc.) et les structurer dans une base de données.
3. **Comparaison avec la fiche de poste** : L'outil lit la fiche de poste et identifie les compétences et expériences requises. Il compare ensuite chaque candidat à ces critères.
4. **Classement des candidats** : En fonction de la correspondance entre le profil du candidat et le poste, l'outil attribue un score à chaque candidat et les classe du plus au moins pertinent.
5. **Justification du classement** : Pour chaque candidat, l'outil génère un petit rapport expliquant pourquoi il a obtenu ce score. Cela permet au recruteur de comprendre la logique du classement et de prendre une décision éclairée.
6. **Visualisation des Résultats** : Affichage du classement dans un tableau interactif sur la page web, avec la possibilité de cliquer sur chaque candidat pour voir son rapport détaillé.

## Structure du Projet

```
Groupe6_AgentDeRecrutementAugmente/
│
├── README.md                 # Ce fichier, qui explique le projet
├── app.py                    # Point d'entrée de l'application web (Flask)
├── src/                      # Dossier contenant la logique métier
│   ├── cv_parser.py          # Module qui lit et analyse les CVs
│   ├── job_matcher.py        # Module qui compare les CVs à la fiche de poste
│   ├── report_generator.py   # Module qui crée les rapports de justification
│   └── data_processor.py     # Module pour la manipulation des données (Pandas)
├── templates/                # Dossier pour les pages HTML
│   ├── index.html            # Page d'accueil avec les formulaires d'upload
│   └── results.html          # Page d'affichage des résultats
├── static/                   # Dossier pour les fichiers statiques
│   ├── css/                  # Feuilles de style (style.css)
│   ├── js/                   # Scripts côté client (script.js)
│   └── uploads/              # Dossier temporaire pour les fichiers uploadés
├── data/                     # Dossier pour les données d'entrée (optionnel)
│   ├── cvs/                  # Placez ici les CVs si besoin d'exemples
│   └── job_descriptions/     # Placez ici la fiche de poste si besoin d'exemple
└── output/                   # Dossier où seront sauvegardés les résultats
    ├── ranking.csv           # Fichier avec la liste des candidats et leurs scores
    └── reports/              # Dossier contenant un rapport détaillé pour chaque candidat
```

## Utilisation

1. **Lancer l'application** : Ouvrez un terminal, placez-vous dans le dossier du projet, puis tapez la commande : `python app.py`
2. **Accéder à l'interface** : Ouvrez votre navigateur web et rendez-vous à l'adresse `http://localhost:5000`
3. **Uploader les fichiers** : Sur la page web, utilisez les boutons pour sélectionner et envoyer les CVs (dossier ou fichiers individuels) et la fiche de poste.
4. **Lancer l'analyse** : Cliquez sur le bouton "Analyser les candidatures".
5. **Consulter les résultats** : L'application affichera automatiquement le classement des candidats et les rapports de justification sur la page web.

## Difficulté

⭐⭐⭐ (Avancé)

## Équipe

Groupe 6 - Projet de Fin de Cours sur l'IA Générative 2025

### Membres du Groupe

- KOUNDJO BRENDA
- SOUZA MARILSON
- TALA Lamyae