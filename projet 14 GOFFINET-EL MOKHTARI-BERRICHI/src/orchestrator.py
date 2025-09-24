"""
Agent Orchestrateur - Coordonne le workflow complet de génération de documents
"""
import os
from typing import Any
import asyncio
from datetime import datetime
from .agents.text_analyzer import TextAnalyzerAgent
from .agents.structure_generator import StructureGeneratorAgent
from .agents.pdf_generator import PDFGeneratorAgent
from .models import ExtractedData

class DocumentOrchestrator:
    """Agent orchestrateur qui coordonne le workflow complet"""
    
    def __init__(self):
        self.text_analyzer = TextAnalyzerAgent()
        self.structure_generator = StructureGeneratorAgent()
        self.pdf_generator = PDFGeneratorAgent()
    
    async def generate_document(self, input_text: str, output_path: str = None) -> str:
        """
        Génère un document structuré à partir d'un texte en langage naturel
        
        Args:
            input_text: Texte en langage naturel décrivant le document à générer
            output_path: Chemin de sortie pour le PDF (optionnel)
            
        Returns:
            Chemin du fichier PDF généré
        """
        # Étape 1: Analyse du texte
        print("Analyse du texte d'entrée...")
        extracted_data = await self.text_analyzer.analyze_text(input_text)
        
        print(f"Document identifié comme: {extracted_data.document_type.upper()} (confiance: {extracted_data.confidence_score:.2f})")
        
        # Étape 2: Génération de la structure
        print("Génération de la structure de document...")
        structured_data = self.structure_generator.generate_structure(extracted_data)
        
        # Étape 3: Génération du PDF
        print("Génération du document PDF...")
        
        # Déterminer le nom de fichier de sortie si non fourni
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_type = extracted_data.document_type.lower()
            output_path = f"output/{doc_type}_{timestamp}.pdf"
        
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Générer le PDF
        pdf_path = self.pdf_generator.generate_pdf(structured_data, output_path)
        
        print(f"Document généré avec succès: {pdf_path}")
        
        return pdf_path

# Fonction utilitaire pour une utilisation simple
async def generate_document_from_text(text: str, output_path: str = None) -> str:
    """
    Fonction utilitaire pour générer un document à partir d'un texte
    
    Args:
        text: Texte en langage naturel
        output_path: Chemin de sortie (optionnel)
        
    Returns:
        Chemin du fichier PDF généré
    """
    orchestrator = DocumentOrchestrator()
    return await orchestrator.generate_document(text, output_path)

# Exemple d'utilisation
if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    # Exemple de texte pour un CV
    cv_text = """
    Je m'appelle Jean Dupont. Voici mon CV.
    Email: jean.dupont@email.com
    Téléphone: 06 12 34 56 78
    
    Expériences professionnelles:
    - Développeur Python chez TechCorp (2020-2023): Développement d'applications web avec Django et Flask
    - Stagiaire chez StartupXYZ (2019): Maintenance de bases de données et scripts d'automatisation
    
    Formations:
    - Master en Informatique, Université Paris (2019)
    - Licence en Sciences, Université Lyon (2017)
    
    Compétences:
    Python, Django, Flask, SQL, Git, Linux
    """
    
    # Exemple de texte pour une facture
    invoice_text = """
    Facture n°FACT-2023-001
    Date: 2023-12-15
    
    Client: Marie Martin
    Adresse: 123 Rue de Paris, 75001 Paris
    
    Services:
    - Développement d'application web: 40 heures à 100€/heure
    - Maintenance mensuelle: 10 heures à 80€/heure
    
    Montant total: 4800€
    TVA: 20%
    """
    
    # Exemple de texte pour un rapport
    report_text = """
    Rapport d'analyse trimestriel
    Auteur: Sophie Bernard
    Date: 2023-12-10
    
    Résumé:
    Ce rapport présente l'analyse des performances du trimestre Q4 2023.
    
    Sections:
    - Performance financière: Le chiffre d'affaires a augmenté de 15% par rapport au trimestre précédent.
    - Analyse du marché: La demande pour nos produits a augmenté de 25% dans la région européenne.
    - Défis opérationnels: Nous avons rencontré des retards dans la chaîne d'approvisionnement.
    
    Conclusions:
    Malgré les défis, les perspectives pour 2024 sont positives avec une croissance attendue de 20%.
    """
    
    # Créer l'orchestrateur
    orchestrator = DocumentOrchestrator()
    
    # Générer les documents
    asyncio.run(orchestrator.generate_document(cv_text, "output/cv_test.pdf"))
    asyncio.run(orchestrator.generate_document(invoice_text, "output/facture_test.pdf"))
    asyncio.run(orchestrator.generate_document(report_text, "output/rapport_test.pdf"))