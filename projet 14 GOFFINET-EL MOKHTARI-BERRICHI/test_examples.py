"""
Exemples de test pour le générateur de documents structurés
"""
import asyncio
import os
from datetime import datetime

# Créer le répertoire de sortie s'il n'existe pas
os.makedirs("output", exist_ok=True)

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

# Exemple de texte pour un CV plus complexe
complex_cv_text = """
CV de Sophie Martin

Coordonnées:
- Email: sophie.martin@protonmail.com
- Téléphone: 07 89 01 23 45
- Adresse: 456 Avenue des Champs, 75008 Paris

Profil professionnel:
Développeuse full-stack expérimentée avec 8 ans d'expérience dans le développement d'applications web et mobiles.

Expériences professionnelles:
1. Lead Developer chez Innovatech (2020-présent)
   - Direction d'une équipe de 5 développeurs
   - Conception et développement d'applications React et Node.js
   - Mise en place de pipelines CI/CD avec GitHub Actions

2. Développeuse Senior chez WebSolutions (2017-2020)
   - Développement d'applications Angular et Spring Boot
   - Optimisation des performances des applications
   - Collaboration avec les équipes produit et design

3. Développeuse Junior chez TechStart (2015-2017)
   - Développement de sites web avec WordPress et PHP
   - Maintenance des serveurs Linux
   - Support technique aux clients

Formations:
- Diplôme d'ingénieur en informatique, École Polytechnique (2015)
- Baccalauréat Scientifique, Lycée Henri IV (2012)

Compétences techniques:
- Langages: JavaScript, TypeScript, Python, Java, PHP
- Frameworks: React, Angular, Node.js, Spring Boot
- Bases de données: PostgreSQL, MongoDB, MySQL
- Outils: Git, Docker, Kubernetes, AWS
- Méthodologies: Agile, Scrum, DevOps

Langues:
- Français: Langue maternelle
- Anglais: Courant
- Espagnol: Intermédiaire

Centres d'intérêt:
- Open source contribution
- Running
- Photographie
"""

# Exemple de texte pour une facture détaillée
detailed_invoice_text = """
Facture n°FACT-2023-002
Date: 2023-12-20

Entreprise: Digital Solutions SARL
Adresse: 789 Boulevard Voltaire, 75011 Paris
SIRET: 123 456 789 00012
TVA intracommunautaire: FR40 123456789

Client: E-Commerce Pro
Adresse: 321 Rue de la Paix, 75002 Paris

Période de facturation: 1er décembre 2023 - 31 décembre 2023

Détails des services:
1. Développement de fonctionnalités e-commerce
   - Création de nouvelles pages produits: 20 heures à 120€/heure
   - Intégration du système de paiement: 15 heures à 120€/heure
   - Optimisation du panier d'achat: 10 heures à 120€/heure

2. Maintenance et support technique
   - Support technique hebdomadaire: 4 heures à 90€/heure
   - Corrections de bugs: 8 heures à 90€/heure

3. Hébergement mensuel
   - Serveur cloud (2 vCPU, 8GB RAM): 100€/mois

Conditions de paiement:
- Paiement à réception
- Virement bancaire
- IBAN: FR76 1234 5678 9012 3456 7890 123
- BIC: AGRIFRPPXXX

Montant total HT: 6 270,00 €
TVA (20%): 1 254,00 €
Montant total TTC: 7 524,00 €
"""

async def test_document_generation():
    """Teste la génération de différents types de documents"""
    try:
        # Importer l'orchestrateur
        from src.orchestrator import generate_document_from_text
        
        # Générer un CV simple
        print("Test 1: Génération d'un CV simple...")
        cv_path = await generate_document_from_text(cv_text, "output/test_cv_simple.pdf")
        print(f"✓ CV simple généré: {cv_path}")
        
        # Générer une facture simple
        print("\nTest 2: Génération d'une facture simple...")
        invoice_path = await generate_document_from_text(invoice_text, "output/test_facture_simple.pdf")
        print(f"✓ Facture simple générée: {invoice_path}")
        
        # Générer un rapport simple
        print("\nTest 3: Génération d'un rapport simple...")
        report_path = await generate_document_from_text(report_text, "output/test_rapport_simple.pdf")
        print(f"✓ Rapport simple généré: {report_path}")
        
        # Générer un CV complexe
        print("\nTest 4: Génération d'un CV complexe...")
        complex_cv_path = await generate_document_from_text(complex_cv_text, "output/test_cv_complexe.pdf")
        print(f"✓ CV complexe généré: {complex_cv_path}")
        
        # Générer une facture détaillée
        print("\nTest 5: Génération d'une facture détaillée...")
        detailed_invoice_path = await generate_document_from_text(detailed_invoice_text, "output/test_facture_detaillée.pdf")
        print(f"✓ Facture détaillée générée: {detailed_invoice_path}")
        
        print("\n✅ Tous les tests ont été exécutés avec succès!")
        print(f"Documents générés dans le répertoire 'output/'")
        
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        print("Assurez-vous que le module src.orchestrator est correctement installé et que PYTHONPATH est configuré.")
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution des tests: {e}")

if __name__ == "__main__":
    asyncio.run(test_document_generation())