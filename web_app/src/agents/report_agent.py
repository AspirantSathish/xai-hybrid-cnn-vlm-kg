# src/agents/report_utils.py
import re
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from utils.logging_utils import setup_logger
import json

class ReportAgent:

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__, "report_agent.log")
        pass
    
    def normalize_text(self, text):

        sections = [
            "Morphological findings:",
            "Blast characteristics:",
            "Additional observations:",
            "Auer rods:",
            "Final Diagnosis:"
        ]

        # ensure every section starts on new line
        for sec in sections:
            text = text.replace(sec, f"\n{sec}")

        # normalize bullet characters
        text = text.replace("–", "-")
        text = text.replace("/", "")
        text = text.replace("  ", " ")

        return text

    def extract_section(self, text, section):

        pattern = rf"{section}:(.*?)(?=\n[A-Za-z ]+:|$)"

        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return ""

    def format_bullets(self, text, indent="        "):  # 8 spaces

        if not text:
            return ""

        items = text.split("-")

        cleaned = []

        for item in items:
            item = item.strip()
            if item:
                cleaned.append(f"{indent}- {item}")
            
        
        return "\n".join(cleaned)
    
    def parse_report(self, raw_text):

        raw_text = self.normalize_text(raw_text)

        morphological = self.extract_section(raw_text, "Morphological findings")
        blast = self.extract_section(raw_text, "Blast characteristics")
        additional = self.extract_section(raw_text, "Additional observations")

        auer_match = re.search(r"Auer rods:\s*(.*)", raw_text, re.IGNORECASE)
        auer = auer_match.group(1).strip() if auer_match else "Not reported"

        diagnosis_match = re.search(r"Final Diagnosis:\s*(.*)", raw_text, re.IGNORECASE)
        diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else "Not reported"

        morphological = self.format_bullets(morphological)
        blast = self.format_bullets(blast)
        additional = self.format_bullets(additional)
        
        return {
            "morphological": morphological or "Not reported",
            "blast": blast or "Not reported",
            "additional": additional or "",
            "auer": auer or "Not reported",
            "diagnosis": diagnosis or "Not reported"
        }


    # -----------------------------
    # STREAMLIT UI OUTPUT
    # -----------------------------
    def generate_ui_text(self, raw_text):
        self.logger.info(f"Generating UI text from raw VLM output. {raw_text[:100]}...")  # Log first 100 chars for debugging
        data = self.parse_report(raw_text)
        
        ui_text = f"""
        Morphological findings:
        \n{data['morphological']}

        Blast characteristics:
        \n{data['blast']}
        """

        if data["additional"]:
            ui_text += f"""

        Additional observations:
        \n{data['additional']}
        """

        ui_text += f"""

        Presence of Auer rods:
            {data['auer']}

        Final Diagnosis:
            {data['diagnosis']}
        """
        self.logger.info("UI text generated successfully.")
        return ui_text



    # -----------------------------
    # PDF REPORT GENERATION
    # -----------------------------
    def generate_pdf(self, raw_text,predicted_label,confidence, pdf_path="hematology_report.pdf"):
        self.logger.info(f"Generating PDF report from raw VLM output. {raw_text[:100]}...")  # Log first 100 chars for debugging
        data = self.parse_report(raw_text)

        morph = data["morphological"] or "Not reported"
        blast = data["blast"] or "Not reported"
        additional = data["additional"] or ""
        auer = data["auer"] or "Not reported"
        diagnosis = data["diagnosis"] or "Not reported"

        styles = getSampleStyleSheet()

        story = []
        
        story.append(Paragraph("Leukemia Diagnostic Report", styles["Title"]))
        story.append(Paragraph("Explainable Leukemia Diagnosis Platform", styles["Italic"]))
        story.append(Spacer(1,20))

        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", styles["Normal"]))
        story.append(Spacer(1, 20))
        confidence_pct = confidence * 100
        story.append(Paragraph(f"<b>Predicted Subtype: {predicted_label} and Confidence Score: {confidence_pct:.2f}% </b>", styles["Heading3"]))        
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>Morphological findings:</b>", styles["Heading3"]))
        story.append(Paragraph(morph, styles["Normal"]))
        story.append(Spacer(1, 10))


        story.append(Paragraph("<b>Blast characteristics:</b>", styles["Heading3"]))
        story.append(Paragraph(blast, styles["Normal"]))
        story.append(Spacer(1, 10))
        
        # Add Additional observations only if present
        if data["additional"]:
            story.append(Paragraph("<b>Additional observations:</b>", styles["Heading3"]))
            story.append(Paragraph(additional, styles["Normal"]))
            story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Presence of Auer rods:</b>", styles["Heading3"]))
        story.append(Paragraph(auer, styles["Normal"]))
        story.append(Spacer(1, 10))


        story.append(Paragraph("<b>Final Diagnosis:</b>", styles["Heading3"]))
        story.append(Paragraph(diagnosis, styles["Normal"]))
        story.append(Spacer(1, 20))


        # Interpretation
        interpretation = """
Based on the morphological characteristics observed in the microscopic field,
including nuclear morphology, chromatin distribution, and blast cell presence,
the findings are suggestive of the reported hematological condition.

Further laboratory investigations including immunophenotyping,
cytogenetics, and molecular studies are recommended for confirmation.
"""

        story.append(Paragraph("<b>Interpretation:</b>", styles["Heading3"]))
        story.append(Paragraph(interpretation, styles["Normal"]))
        story.append(Spacer(1, 20))


        # AI Disclaimer
        disclaimer = """
This report was generated using an Artificial Intelligence based analysis system.
The results are intended to support clinical evaluation and should not be used
as the sole basis for medical diagnosis.

Please consult a qualified hematologist or medical professional
for final clinical interpretation and diagnosis.
"""

        story.append(Paragraph("<b>AI Generated Report Disclaimer:</b>", styles["Heading3"]))
        story.append(Paragraph(disclaimer, styles["Normal"]))


        pdf = SimpleDocTemplate(pdf_path, pagesize=A4)
        pdf.build(story)
        self.logger.info(f"PDF report generated successfully at {pdf_path}")
        return pdf_path
    


    def create_diagnostic_report(self, report_id, patient_id, performer_name, subtype, confidence, morphology, kg_validation, pdf_url):
        report = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "HEM",
                    "display": "Hematology"
                }]
            }],
            "code": { "text": "Leukemia Subtype Diagnosis" },
            "subject": { "reference": f"Patient/{patient_id}" },
            "effectiveDateTime": datetime.now().isoformat(),
            "issued": datetime.now().isoformat(),
            "performer": [{
                "reference": f"Practitioner/{performer_name.lower().replace(' ', '')}",
                "display": performer_name
            }],
            "result": [
                { "subtypeClassification": subtype, "confidenceScore": f"{confidence}%" },
                { "textDescription": morphology },
                { "kgReasoning": kg_validation }
            ],
            "conclusion": f"Subtype: {subtype}. Morphology: {morphology}. KG validation: {kg_validation}.",
            "presentedForm": [
                { "contentType": "application/pdf", "url": pdf_url }
            ]
        }
        return report

    def to_json(self, report):
        return json.dumps(report, indent=2)
