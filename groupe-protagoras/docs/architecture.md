# Conception Architecturale

Ce document décrit l'architecture globale du système "Agent d'Analyse d'Arguments Hybride", structurée en un pipeline modulaire permettant une intégration fluide entre les composants d'IA générative et symbolique.

## Architecture Globale (Pipeline)

Le système suit un flux en plusieurs étapes :

1. **Entrée** : Un discours ou débat est fourni en texte brut.
2. **Prétraitement** : Segmentation en unités argumentatives (prémisses, conclusions).
3. **Analyse Informelle (LLM)** : Détection des sophismes via LangChain.
4. **Transformation Logique** : Normalisation des énoncés en logique formelle.
5. **Analyse Formelle (TweetyProject)** : Validation de la structure logique.
6. **Fusion & Résultat** : Corrélation des analyses et génération d'un verdict.
7. **Sortie** : Rapport détaillé avec explications et visualisations.

## Interfaces entre Composants

- **LLM ↔ Prétraitement** : Le LLM reçoit le texte brut et retourne une structure JSON contenant les unités argumentatives identifiées.
- **Prétraitement → Transformation Logique** : Les énoncés sont normalisés (ex. : "Tous les chats sont mignons" → `∀x (Chat(x) → Mignon(x))`).
- **Transformation Logique → TweetyProject** : Les formules logiques sont transmises via une API ou appel direct à la bibliothèque.
- **TweetyProject → Fusion** : Retourne un objet JSON avec statut de validité, cohérence, et implications.
- **Fusion → Sortie** : Génère un rapport unifié combinant les faiblesses informelles et les erreurs formelles.

## Formats d'Entrée/Sortie

### Format d'Entrée
```json
{
  "debate_id": "string",
  "text": "Texte complet du débat"
}
```

### Format de Sortie
```json
{
  "analysis_id": "string",
  "overall_validity": "boolean",
  "informal_fallacies": [
    {
      "type": "string",
      "location": "string",
      "explanation": "string"
    }
  ],
  "formal_validity": {
    "is_valid": "boolean",
    "inconsistencies": ["string"],
    "logical_implications": ["string"]
  },
  "final_verdict": "string"
}
```

Cette architecture permet une extensibilité future (ajout de détecteurs de biais, multilinguisme) tout en assurant une traçabilité complète des décisions prises par le système.