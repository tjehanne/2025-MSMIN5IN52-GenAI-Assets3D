"""
Agent Générateur PDF - Génère des documents PDF à partir des structures de données
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import os
from typing import Any
from ..models import CVData, FactureData, RapportData

class PDFGeneratorAgent:
    """Agent responsable de la génération de documents PDF"""
    
    def __init__(self):
        # Enregistrer des polices (optionnel - utiliser les polices par défaut si non disponibles)
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'DejaVuSans-Bold.ttf'))
            self.default_font = 'DejaVuSans'
            self.bold_font = 'DejaVuSans-Bold'
        except:
            self.default_font = 'Helvetica'
            self.bold_font = 'Helvetica-Bold'
        
        # Styles
        self.styles = getSampleStyleSheet()
        self.custom_styles = {
            'Title': ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # Centré
                fontName=self.bold_font
            ),
            'Subtitle': ParagraphStyle(
                'CustomSubtitle',
                parent=self.styles['Heading2'],
                fontSize=18,
                spaceAfter=20,
                fontName=self.bold_font
            ),
            'Header': ParagraphStyle(
                'CustomHeader',
                parent=self.styles['Heading3'],
                fontSize=14,
                spaceBefore=12,
                spaceAfter=6,
                fontName=self.bold_font
            ),
            'Body': ParagraphStyle(
                'CustomBody',
                parent=self.styles['BodyText'],
                fontSize=12,
                spaceAfter=12,
                fontName=self.default_font
            ),
            'Small': ParagraphStyle(
                'Small',
                parent=self.styles['BodyText'],
                fontSize=10,
                spaceAfter=6,
                fontName=self.default_font
            )
        }
    
    def generate_pdf(self, data: Any, output_path: str) -> str:
        """
        Génère un document PDF à partir des données structurées
        
        Args:
            data: Données structurées (CVData, FactureData, ou RapportData)
            output_path: Chemin de sortie pour le fichier PDF
            
        Returns:
            Chemin du fichier PDF généré
        """
        # Déterminer le type de document
        if isinstance(data, CVData):
            return self._generate_cv_pdf(data, output_path)
        elif isinstance(data, FactureData):
            return self._generate_invoice_pdf(data, output_path)
        elif isinstance(data, RapportData):
            return self._generate_report_pdf(data, output_path)
        else:
            raise ValueError(f"Type de données non supporté: {type(data)}")
    
    def _generate_cv_pdf(self, cv_data: CVData, output_path: str) -> str:
        """Génère un PDF pour un CV"""
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        story = []
        
        # Titre (Nom)
        title = Paragraph(cv_data.nom.upper(), self.custom_styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Informations de contact (si disponibles)
        contact_info = []
        if cv_data.email:
            contact_info.append(f"Email: {cv_data.email}")
        if cv_data.telephone:
            contact_info.append(f"Téléphone: {cv_data.telephone}")
        if cv_data.adresse:
            contact_info.append(f"Adresse: {cv_data.adresse}")
            
        if contact_info:
            contact_text = " | ".join(contact_info)
            contact_para = Paragraph(contact_text, self.custom_styles['Small'])
            story.append(contact_para)
            story.append(Spacer(1, 20))
        
        # Expériences professionnelles
        if cv_data.experiences:
            header = Paragraph("EXPÉRIENCES PROFESSIONNELLES", self.custom_styles['Header'])
            story.append(header)
            
            for exp in cv_data.experiences:
                exp_para = Paragraph(exp, self.custom_styles['Body'])
                story.append(exp_para)
                story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 12))
        
        # Formations
        if cv_data.formations:
            header = Paragraph("FORMATIONS", self.custom_styles['Header'])
            story.append(header)
            
            for formation in cv_data.formations:
                formation_para = Paragraph(formation, self.custom_styles['Body'])
                story.append(formation_para)
                story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 12))
        
        # Compétences
        if cv_data.competences:
            header = Paragraph("COMPÉTENCES", self.custom_styles['Header'])
            story.append(header)
            
            skills_text = ", ".join(cv_data.competences)
            skills_para = Paragraph(skills_text, self.custom_styles['Body'])
            story.append(skills_para)
            story.append(Spacer(1, 12))
        
        # Générer le PDF
        doc.build(story)
        return output_path
    
    def _generate_invoice_pdf(self, invoice_data: FactureData, output_path: str) -> str:
        """Génère un PDF pour une facture"""
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        story = []
        
        # En-tête
        title = Paragraph("FACTURE", self.custom_styles['Title'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Informations de la facture
        invoice_info = [
            [Paragraph("<b>Numéro de facture:</b>", self.custom_styles['Body']), 
             Paragraph(invoice_data.numero_facture, self.custom_styles['Body'])],
            [Paragraph("<b>Date:</b>", self.custom_styles['Body']), 
             Paragraph(invoice_data.date.strftime("%d/%m/%Y"), self.custom_styles['Body'])]
        ]
        
        invoice_table = Table(invoice_info, colWidths=[150, 300])
        invoice_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(invoice_table)
        story.append(Spacer(1, 20))
        
        # Informations du client
        if invoice_data.client_nom or invoice_data.client_adresse:
            client_header = Paragraph("CLIENT", self.custom_styles['Subtitle'])
            story.append(client_header)
            
            client_info = []
            if invoice_data.client_nom:
                client_info.append([Paragraph("<b>Nom:</b>", self.custom_styles['Body']), 
                                  Paragraph(invoice_data.client_nom, self.custom_styles['Body'])])
            if invoice_data.client_adresse:
                client_info.append([Paragraph("<b>Adresse:</b>", self.custom_styles['Body']), 
                                  Paragraph(invoice_data.client_adresse, self.custom_styles['Body'])])
            
            client_table = Table(client_info, colWidths=[150, 300])
            client_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(client_table)
            story.append(Spacer(1, 20))
        
        # Détails des services
        if invoice_data.services:
            services_header = Paragraph("DÉTAILS DES SERVICES", self.custom_styles['Subtitle'])
            story.append(services_header)
            
            # Tableau des services
            service_data = [["Description", "Quantité", "Prix Unitaire", "Total"]]
            
            for service in invoice_data.services:
                description = service.get("description", "Service non spécifié")
                quantite = service.get("quantite", 1)
                prix_unitaire = service.get("prix_unitaire", 0.0)
                total = quantite * prix_unitaire
                
                service_data.append([
                    Paragraph(description, self.custom_styles['Body']),
                    Paragraph(str(quantite), self.custom_styles['Body']),
                    Paragraph(f"{prix_unitaire:.2f} €", self.custom_styles['Body']),
                    Paragraph(f"{total:.2f} €", self.custom_styles['Body'])
                ])
            
            # Créer le tableau
            col_widths = [250, 80, 100, 100]
            service_table = Table(service_data, colWidths=col_widths)
            
            # Style du tableau
            service_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), self.bold_font),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(service_table)
            story.append(Spacer(1, 20))
        
        # Total
        total_data = [
            ["", "", "TOTAL HT:", Paragraph(f"{invoice_data.montant_total:.2f} €", self.custom_styles['Body'])],
            ["", "", f"TVA ({invoice_data.tva}%)", Paragraph(f"{invoice_data.montant_total * invoice_data.tva / 100:.2f} €", self.custom_styles['Body'])],
            ["", "", "TOTAL TTC:", Paragraph(f"{invoice_data.montant_total * (1 + invoice_data.tva / 100):.2f} €", self.custom_styles['Body'])]
        ]
        
        total_table = Table(total_data, colWidths=[250, 80, 100, 100])
        total_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('FONTNAME', (2, 0), (3, -1), self.bold_font),
            ('FONTSIZE', (2, 0), (3, -1), 14),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(total_table)
        story.append(Spacer(1, 20))
        
        # Générer le PDF
        doc.build(story)
        return output_path
    
    def _generate_report_pdf(self, report_data: RapportData, output_path: str) -> str:
        """Génère un PDF pour un rapport"""
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        story = []
        
        # Titre
        title = Paragraph(report_data.titre, self.custom_styles['Title'])
        story.append(title)
        
        # Sous-titre (Auteur et Date)
        subtitle_text = f"Par {report_data.auteur} - {report_data.date.strftime('%d/%m/%Y')}"
        subtitle = Paragraph(subtitle_text, self.custom_styles['Subtitle'])
        story.append(subtitle)
        story.append(Spacer(1, 30))
        
        # Résumé (si disponible)
        if report_data.resume:
            summary_header = Paragraph("RÉSUMÉ", self.custom_styles['Header'])
            story.append(summary_header)
            
            summary_para = Paragraph(report_data.resume, self.custom_styles['Body'])
            story.append(summary_para)
            story.append(Spacer(1, 20))
        
        # Sections
        for section in report_data.sections:
            section_title = section.get("titre", "Section")
            section_content = section.get("contenu", "")
            
            # Titre de la section
            header = Paragraph(section_title, self.custom_styles['Header'])
            story.append(header)
            
            # Contenu de la section
            content_para = Paragraph(section_content, self.custom_styles['Body'])
            story.append(content_para)
            story.append(Spacer(1, 12))
        
        # Conclusions (si disponibles)
        if report_data.conclusions:
            conclusions_header = Paragraph("CONCLUSIONS", self.custom_styles['Header'])
            story.append(conclusions_header)
            
            conclusions_para = Paragraph(report_data.conclusions, self.custom_styles['Body'])
            story.append(conclusions_para)
            story.append(Spacer(1, 12))
        
        # Générer le PDF
        doc.build(story)
        return output_path