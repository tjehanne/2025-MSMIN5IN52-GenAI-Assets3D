"""
Modèles de données pour les différents types de documents
"""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class CVData:
    """Structure pour un CV"""
    nom: str
    email: Optional[str] = None
    telephone: Optional[str] = None
    adresse: Optional[str] = None
    experiences: List[str] = None
    formations: List[str] = None
    competences: List[str] = None
    
    def __post_init__(self):
        if self.experiences is None:
            self.experiences = []
        if self.formations is None:
            self.formations = []
        if self.competences is None:
            self.competences = []

@dataclass
class FactureData:
    """Structure pour une facture"""
    numero_facture: str
    date: datetime
    client_nom: str
    client_adresse: Optional[str] = None
    services: List[dict] = None  # [{"description": str, "quantite": int, "prix_unitaire": float}]
    montant_total: float = 0.0
    tva: float = 20.0  # % de TVA par défaut
    
    def __post_init__(self):
        if self.services is None:
            self.services = []
        if self.montant_total == 0.0 and self.services:
            # Calculer le montant total automatiquement
            self.montant_total = sum(service['quantite'] * service['prix_unitaire'] for service in self.services)

@dataclass
class RapportData:
    """Structure pour un rapport"""
    titre: str
    auteur: str
    date: datetime
    resume: Optional[str] = None
    sections: List[dict] = None  # [{"titre": str, "contenu": str}]
    conclusions: Optional[str] = None
    
    def __post_init__(self):
        if self.sections is None:
            self.sections = []

@dataclass
class ExtractedData:
    """Container pour les données extraites"""
    document_type: str  # "cv", "facture", "rapport"
    confidence_score: float
    data: object  # CVData, FactureData, ou RapportData
    raw_text: str