# Générateur de Contenu Structuré

## Description
Application multi-agents qui prend des informations en langage naturel et génère des documents structurés au format PDF (CV, facture, rapport).

## Technologies
- Python 3.8-3.12
- Semantic Kernel
- ReportLab
- LangChain
- OpenAI API

## Installation

1. Clonez le dépôt
2. Créez un environnement virtuel:
```bash
python -m venv venv
```

3. Activez l'environnement:
```bash
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

4. Installez les dépendances:
```bash
pip install -r requirements.txt
```

5. Configurez les variables d'environnement:
```bash
cp .env.example .env
```

Éditez le fichier `.env` et ajoutez votre clé API OpenAI:
```
OPENAI_API_KEY=votre_clé_api_ici
OPENAI_MODEL=gpt-3.5-turbo
```

## Utilisation

### Méthode 1: Utilisation directe
```python
from src.orchestrator import generate_document_from_text

# Générer un document à partir d'un texte
texte = "Jean Dupont, développeur Python avec 5 ans d'expérience..."
chemin_pdf = await generate_document_from_text(texte, "output/cv_jean.pdf")
```

### Méthode 2: Exécution des exemples
```bash
python test_examples.py
```

### Méthode 3: Utilisation en ligne de commande
```bash
python -m src.orchestrator
```

## Structure du projet
```
projet/
├── src/
│   ├── agents/
│   │   ├── text_analyzer.py    # Analyse du texte avec Semantic Kernel
│   │   ├── structure_generator.py # Génération de la structure de document
│   │   └── pdf_generator.py    # Génération du PDF avec ReportLab
│   ├── models.py              # Modèles de données
│   └── orchestrator.py        # Coordination du workflow
├── output/                    # Documents générés
├── requirements.txt           # Dépendances
├── .env.example               # Exemple de fichier de configuration
└── test_examples.py           # Exemples de test
```

## Agents

### 1. Agent Analyseur de Texte
- Utilise Semantic Kernel et OpenAI pour analyser le texte d'entrée
- Classifie le type de document (CV, facture, rapport)
- Extrait les données structurées du texte naturel

### 2. Agent Générateur de Structure
- Transforme les données extraites en structure de document
- Crée des objets CVData, FactureData ou RapportData
- Valide et normalise les données

### 3. Agent Générateur PDF
- Génère des documents PDF à partir des structures de données
- Utilise ReportLab pour la mise en page
- Supporte différents formats (CV, facture, rapport)

### 4. Agent Orchestrateur
- Coordonne le workflow complet
- Gère la séquence: analyse → structure → PDF
- Fournit une interface simple pour l'utilisation

## Configuration OpenAI
Le système utilise l'API OpenAI pour l'analyse sémantique. Assurez-vous d'avoir une clé API valide.

## Limitations
- Nécessite une connexion Internet pour l'API OpenAI
- Performance dépendante de la qualité du texte d'entrée
- Formatage PDF basique (peut être amélioré)

## Auteurs
Projet développé dans le cadre du cours IA Générative 2025.