"""
Module de prétraitement des données pour l'analyse d'arguments.
Responsable de la segmentation du discours, de l'extraction des unités argumentatives
et de la normalisation en propositions logiques atomiques.
"""

from typing import List, Dict, Any
import re

def segmenter_discours(texte: str) -> List[str]:
    """
    Segmente un discours en unités argumentatives (phrases ou propositions).
    """
    # Séparation par points, points d'interrogation, points d'exclamation
    phrases = re.split(r'[.!?]+', texte)
    # Nettoyage et suppression des éléments vides
    return [phrase.strip() for phrase in phrases if phrase.strip()]

def extraire_premisses_conclusions(texte: str, llm_client) -> Dict[str, Any]:
    """
    Utilise un LLM pour extraire les prémisses, conclusions et relations.
    Le client LLM doit implémenter une méthode `generate` prenant un prompt.
    """
    prompt = f"""
    Analyse le discours suivant et extrait :
    1. Les prémisses (affirmations servant de base à l'argument)
    2. La ou les conclusions (affirmations déduites)
    3. Les relations implicites entre elles

    Discours : {texte}

    Retourne un JSON avec les clés 'premises', 'conclusions', 'relations'.
    """
    
    reponse = llm_client.generate(prompt)
    # Ici, on suppose que le LLM retourne un JSON valide
    return reponse

def normaliser_en_logique_atomique(unites: List[str]) -> List[str]:
    """
    Normalise les énoncés en propositions logiques atomiques.
    Exemple : "Tous les chats sont mignons" → "Chat(x) → Mignon(x)"
    """
    propositions = []
    for unite in unites:
        # Remplacements simples pour la démonstration
        proposition = unite
        proposition = re.sub(r"Tous les (\w+) sont (\w+)", r"\1(x) → \2(x)", proposition)
        proposition = re.sub(r"Aucun (\w+) n'est (\w+)", r"\1(x) → ¬\2(x)", proposition)
        proposition = re.sub(r"(\w+) est (\w+)", r"\1 → \2", proposition)
        propositions.append(proposition.strip())
    
    return propositions

# Exemple d'utilisation
if __name__ == "__main__":
    texte = "Tous les hommes sont mortels. Socrate est un homme. Donc Socrate est mortel."
    
    # 1. Segmentation
    unites = segmenter_discours(texte)
    print("Unités segmentées:", unites)
    
    # 2. Extraction (simulée sans LLM réel)
    # extraits = extraire_premisses_conclusions(texte, llm_client)
    
    # 3. Normalisation
    propositions = normaliser_en_logique_atomique(unites[:-1])  # Exclure la conclusion
    print("Propositions logiques:", propositions)