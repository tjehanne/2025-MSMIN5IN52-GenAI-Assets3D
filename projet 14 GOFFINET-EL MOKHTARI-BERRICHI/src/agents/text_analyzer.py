"""
Agent Analyseur de Texte - Utilise Semantic Kernel pour analyser le texte d'entrée
"""
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import KernelArguments
import os
from dotenv import load_dotenv
from typing import Tuple, Optional
from ..models import ExtractedData
import json

load_dotenv()

class TextAnalyzerAgent:
    """Agent responsable de l'analyse et de la classification du texte d'entrée"""
    
    def __init__(self):
        self.kernel = sk.Kernel()
        
        # Configuration du service OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = api_key and api_key != "your_openai_api_key_here"
        
        if self.use_openai:
            try:
                service_id = "chat-gpt"
                self.kernel.add_service(
                    OpenAIChatCompletion(
                        service_id=service_id,
                        ai_model_id=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                        api_key=api_key,
                    )
                )
                # Get the chat completion service for direct use
                self.chat_completion = self.kernel.get_service(service_id)
                print("✓ Connexion OpenAI configurée")
            except Exception as e:
                print(f"⚠ Erreur de connexion OpenAI: {e}")
                self.use_openai = False
        else:
            print("⚠ Clé OpenAI non configurée, utilisation du mode hors ligne")
        
        # Prompts pour l'analyse
        self.classification_prompt = """
        Analysez le texte suivant et déterminez s'il s'agit d'informations pour créer un CV, une facture, ou un rapport.

        Règles de classification:
        - CV: contient des informations personnelles, expériences professionnelles, formations, compétences
        - Facture: contient des informations de facturation, services/produits, montants, dates
        - Rapport: contient un titre, des sections, des analyses, des conclusions

        Texte à analyser:
        {{$input}}

        Répondez avec un seul mot: CV, FACTURE, ou RAPPORT
        Ajoutez aussi un score de confiance entre 0 et 1.
        
        Format de réponse: TYPE|SCORE
        """
        
        self.extraction_prompt = """
        Extrayez les informations structurées du texte suivant pour créer un {{$document_type}}.

        Texte d'entrée:
        {{$input}}

        Pour un CV, extrayez:
        - nom (obligatoire)
        - email, téléphone, adresse (si disponibles)
        - expériences professionnelles (liste)
        - formations (liste)
        - compétences (liste)

        Pour une FACTURE, extrayez:
        - numéro de facture
        - date
        - nom du client
        - adresse du client (si disponible)
        - liste des services/produits avec quantité et prix
        - montant total

        Pour un RAPPORT, extrayez:
        - titre
        - auteur
        - date
        - résumé (si disponible)
        - sections avec titres et contenu
        - conclusions (si disponibles)

        Répondez au format JSON structuré selon le type de document.
        """

    def _classify_offline(self, text: str) -> Tuple[str, float]:
        """Classification hors ligne basée sur des mots-clés"""
        text_lower = text.lower()
        
        # Mots-clés pour CV
        cv_keywords = ["cv", "curriculum", "expérience", "formation", "compétence", "diplôme", "poste", "développeur", "ingénieur"]
        cv_score = sum(1 for keyword in cv_keywords if keyword in text_lower)
        
        # Mots-clés pour facture
        facture_keywords = ["facture", "invoice", "montant", "prix", "€", "euro", "tva", "client", "service", "produit"]
        facture_score = sum(1 for keyword in facture_keywords if keyword in text_lower)
        
        # Mots-clés pour rapport
        rapport_keywords = ["rapport", "report", "analyse", "résumé", "section", "conclusion", "auteur", "date"]
        rapport_score = sum(1 for keyword in rapport_keywords if keyword in text_lower)
        
        # Déterminer le type avec le score le plus élevé
        scores = {
            "cv": cv_score / len(cv_keywords),
            "facture": facture_score / len(facture_keywords),
            "rapport": rapport_score / len(rapport_keywords)
        }
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, min(confidence, 0.9)  # Limiter la confiance à 90%

    async def classify_document_type(self, text: str) -> Tuple[str, float]:
        """
        Classifie le type de document basé sur le texte d'entrée
        
        Args:
            text: Le texte à analyser
            
        Returns:
            Tuple[str, float]: (type_document, score_confiance)
        """
        if not self.use_openai:
            return self._classify_offline(text)
        
        try:
            # Créer un prompt pour la classification
            prompt = self.classification_prompt.replace("{{$input}}", text)
            
            # Créer l'historique de chat
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Exécuter la classification
            result = await self.chat_completion.get_chat_message_contents(
                chat_history=chat_history,
                settings=None
            )
            
            response = str(result[0].content).strip() if result else ""
            
            # Parser la réponse
            if "|" in response:
                doc_type, score_str = response.split("|", 1)
                doc_type = doc_type.strip().lower()
                try:
                    score = float(score_str.strip())
                except ValueError:
                    score = 0.5
            else:
                doc_type = response.lower()
                score = 0.5
            
            # Normaliser le type de document
            if doc_type in ["cv", "curriculum", "resume"]:
                return "cv", score
            elif doc_type in ["facture", "invoice", "bill"]:
                return "facture", score
            elif doc_type in ["rapport", "report"]:
                return "rapport", score
            else:
                return "rapport", 0.3  # Par défaut, rapport avec faible confiance
                
        except Exception as e:
            print(f"Erreur lors de la classification: {e}, utilisation du mode hors ligne")
            return self._classify_offline(text)

    def _extract_offline(self, text: str, document_type: str) -> dict:
        """Extraction hors ligne basée sur des expressions régulières simples"""
        import re
        from datetime import datetime
        
        if document_type == "cv":
            return self._extract_cv_offline(text)
        elif document_type == "facture":
            return self._extract_facture_offline(text)
        else:  # rapport
            return self._extract_rapport_offline(text)
    
    def _extract_cv_offline(self, text: str) -> dict:
        """Extraction hors ligne pour CV"""
        import re
        
        # Extraire le nom (première ligne souvent)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        nom = "Nom non spécifié"
        for line in lines[:5]:  # Chercher dans les 5 premières lignes
            if not any(keyword in line.lower() for keyword in ['email', 'téléphone', 'adresse', 'cv', 'expérience']):
                if len(line.split()) <= 4:  # Probablement un nom
                    nom = line
                    break
        
        # Extraire l'email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        email = email_match.group() if email_match else None
        
        # Extraire le téléphone
        tel_match = re.search(r'(?:téléphone|tel|phone)[\s:]*([0-9\s\-\.]{10,})', text, re.IGNORECASE)
        telephone = tel_match.group(1).strip() if tel_match else None
        
        # Extraire les expériences (lignes après "expérience")
        experiences = []
        exp_section = False
        for line in text.split('\n'):
            line = line.strip()
            if 'expérience' in line.lower():
                exp_section = True
                continue
            if exp_section and line.startswith('-'):
                experiences.append(line[1:].strip())
            elif exp_section and ('formation' in line.lower() or 'compétence' in line.lower()):
                break
        
        # Extraire les formations
        formations = []
        form_section = False
        for line in text.split('\n'):
            line = line.strip()
            if 'formation' in line.lower():
                form_section = True
                continue
            if form_section and line.startswith('-'):
                formations.append(line[1:].strip())
            elif form_section and ('compétence' in line.lower() or 'langue' in line.lower()):
                break
        
        # Extraire les compétences
        competences = []
        comp_section = False
        for line in text.split('\n'):
            line = line.strip()
            if 'compétence' in line.lower():
                comp_section = True
                continue
            if comp_section and line:
                if not line.startswith('-'):
                    # Diviser par virgules
                    competences.extend([comp.strip() for comp in line.split(',') if comp.strip()])
                else:
                    competences.append(line[1:].strip())
            elif comp_section and any(keyword in line.lower() for keyword in ['langue', 'centre', 'intérêt']):
                break
        
        return {
            "nom": nom,
            "email": email,
            "telephone": telephone,
            "experiences": experiences,
            "formations": formations,
            "competences": competences
        }
    
    def _extract_facture_offline(self, text: str) -> dict:
        """Extraction hors ligne pour facture"""
        import re
        from datetime import datetime
        
        # Extraire numéro de facture
        facture_match = re.search(r'(?:facture|invoice)[\s\w]*[:\s]*([A-Z0-9-]+)', text, re.IGNORECASE)
        numero_facture = facture_match.group(1) if facture_match else "FACT-001"
        
        # Extraire date
        date_match = re.search(r'date[\s:]*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
        date = datetime.now().isoformat() if not date_match else date_match.group(1)
        
        # Extraire client
        client_match = re.search(r'client[\s:]*([^\n]+)', text, re.IGNORECASE)
        client_nom = client_match.group(1).strip() if client_match else "Client non spécifié"
        
        # Extraire montant total
        montant_match = re.search(r'(?:montant|total)[\s\w]*[:\s]*([0-9,\s]+)[\s]*€', text, re.IGNORECASE)
        montant_total = float(re.sub(r'[^\d]', '', montant_match.group(1))) if montant_match else 0.0
        
        return {
            "numero_facture": numero_facture,
            "date": date,
            "client_nom": client_nom,
            "montant_total": montant_total,
            "services": []
        }
    
    def _extract_rapport_offline(self, text: str) -> dict:
        """Extraction hors ligne pour rapport"""
        import re
        from datetime import datetime
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Titre (première ligne non vide)
        titre = lines[0] if lines else "Titre non spécifié"
        
        # Auteur
        auteur_match = re.search(r'auteur[\s:]*([^\n]+)', text, re.IGNORECASE)
        auteur = auteur_match.group(1).strip() if auteur_match else "Auteur non spécifié"
        
        # Date
        date_match = re.search(r'date[\s:]*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
        date = datetime.now().isoformat() if not date_match else date_match.group(1)
        
        # Résumé
        resume_match = re.search(r'résumé[\s:]*([^\n]+(?:\n[^-\n]+)*)', text, re.IGNORECASE)
        resume = resume_match.group(1).strip() if resume_match else None
        
        # Sections
        sections = []
        current_section = None
        for line in lines:
            if line.startswith('-') and ':' in line:
                if current_section:
                    sections.append(current_section)
                parts = line[1:].split(':', 1)
                current_section = {
                    "titre": parts[0].strip(),
                    "contenu": parts[1].strip() if len(parts) > 1 else ""
                }
        if current_section:
            sections.append(current_section)
        
        # Conclusions
        conclusions_match = re.search(r'conclusion[s]?[\s:]*([^\n]+(?:\n[^-\n]+)*)', text, re.IGNORECASE)
        conclusions = conclusions_match.group(1).strip() if conclusions_match else None
        
        return {
            "titre": titre,
            "auteur": auteur,
            "date": date,
            "resume": resume,
            "sections": sections,
            "conclusions": conclusions
        }

    async def extract_structured_data(self, text: str, document_type: str) -> dict:
        """
        Extrait les données structurées du texte selon le type de document
        
        Args:
            text: Le texte à analyser
            document_type: Le type de document identifié
            
        Returns:
            dict: Les données extraites sous forme de dictionnaire
        """
        if not self.use_openai:
            return self._extract_offline(text, document_type)
        
        try:
            # Créer le prompt d'extraction avec les variables remplacées
            prompt = self.extraction_prompt.replace("{{$input}}", text)
            prompt = prompt.replace("{{$document_type}}", document_type.upper())
            
            # Créer l'historique de chat
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Exécuter l'extraction
            result = await self.chat_completion.get_chat_message_contents(
                chat_history=chat_history,
                settings=None
            )
            
            response = str(result[0].content).strip() if result else ""
            
            # Tenter de parser en JSON
            try:
                extracted_data = json.loads(response)
                return extracted_data
            except json.JSONDecodeError:
                # Si ce n'est pas du JSON valide, retourner une structure basique
                return {
                    "raw_response": response,
                    "extraction_method": "fallback"
                }
                
        except Exception as e:
            print(f"Erreur lors de l'extraction: {e}, utilisation du mode hors ligne")
            return self._extract_offline(text, document_type)

    async def analyze_text(self, text: str) -> ExtractedData:
        """
        Analyse complète du texte: classification + extraction
        
        Args:
            text: Le texte à analyser
            
        Returns:
            ExtractedData: Objet contenant toutes les informations extraites
        """
        # Étape 1: Classification
        doc_type, confidence = await self.classify_document_type(text)
        
        # Étape 2: Extraction des données
        extracted_dict = await self.extract_structured_data(text, doc_type)
        
        # Retourner l'objet ExtractedData
        return ExtractedData(
            document_type=doc_type,
            confidence_score=confidence,
            data=extracted_dict,  # Sera transformé en objet spécifique par l'agent suivant
            raw_text=text
        )