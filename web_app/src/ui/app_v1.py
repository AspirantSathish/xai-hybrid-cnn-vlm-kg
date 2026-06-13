import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import streamlit as st
from PIL import Image
import time
import json
from agents.cnn_model import CNNModel  # adjust path if needed
from agents.vlm_model import VLMModel
from utils.image_utils import ImageProcessor
from agents.report_agent import ReportAgent
import tempfile


processor = ImageProcessor(save_dir=os.path.join("web_app", "processed_images"))
# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(layout="wide", page_title="HemaXplain AI", page_icon="🩸")

# ----------------------------
# PROFESSIONAL CLINICAL UI STYLING
# ----------------------------
st.markdown("""
<style>
/* Core Layout */
.block-container {
  padding: 1.5rem;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  min-height: 100vh;
}

* {
  font-family: 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
}

/* Typography */
h1 { color: #0f172a; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }
h2 { color: #1e293b; font-size: 1.5rem; font-weight: 600; margin-top: 1.5rem; }
h3 { color: #334155; font-size: 1.1rem; font-weight: 600; }

/* Card System - Professional */
.card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  /* padding: 2px; */
/* margin-bottom: 1px; */
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  transition: all 0.2s ease;
}

.card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}

.card-header {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 2px solid #f1f5f9;
}

.card-header h3 {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1.1rem;
}

/* Metric Boxes - Clinical */
.metric-box {
  background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
  color: white;
  padding: 16px;
  border-radius: 10px;
  text-align: center;
  font-weight: 600;
  transition: transform 0.2s;
}

.metric-box:hover {
  transform: translateY(-2px);
}

.metric-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0.9;
  margin-bottom: 4px;
}

.metric-value {
  font-size: 24px;
  font-weight: 700;
}

/* Color-Coded Risk Levels */
.risk-critical { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
.risk-high { background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); }
.risk-moderate { background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%); }
.risk-low { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }

/* Content Boxes */
.content-box {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-left: 4px solid #0ea5e9;
  padding: 14px;
  border-radius: 8px;
  margin: 12px 0;
}

.content-box.vlm { border-left-color: #06b6d4; }
.content-box.kg { border-left-color: #10b981; }
.content-box.warning { border-left-color: #f59e0b; }

/* Buttons */
.btn-primary {
  background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
  color: white;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15);
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(15, 23, 42, 0.25);
}

.btn-primary:active {
  transform: translateY(0);
}

.btn-secondary {
  background: white;
  color: #0ea5e9;
  padding: 10px 20px;
  border: 2px solid #0ea5e9;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-secondary:hover {
  background: #f0f9ff;
}

.btn-success {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
}

.btn-success:hover {
  transform: translateY(-2px);
}

/* Status Indicators */
.status-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.status-pending { background: #fef3c7; color: #92400e; }
.status-approved { background: #dcfce7; color: #166534; }
.status-correction { background: #fee2e2; color: #991b1b; }
.status-review { background: #dbeafe; color: #1e40af; }

/* Summary Card */
.summary-card {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border: 2px solid #0ea5e9;
  border-radius: 12px;
  /* padding: 20px; */
  /* margin-bottom: 16px; */
}

.summary-title {
  color: #0ea5e9;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 700;
  margin-bottom: 12px;
}

.summary-finding {
  color: #0f172a;
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 8px;
}

.summary-meta {
  display: flex;
  gap: 16px;
  font-size: 13px;
  color: #64748b;
}

/* Graph Visualization Placeholder */
.graph-container {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  min-height: 300px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.graph-container svg {
  max-width: 100%;
  height: auto;
}

/* Modal Overlay */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

/* Modal Dialog */
.modal-dialog {
  background: white;
  border-radius: 12px;
  padding: 0;
  max-width: 600px;
  width: 90%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.modal-header {
  padding: 24px;
  border-bottom: 1px solid #e2e8f0;
  background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
  color: white;
  font-size: 18px;
  font-weight: 700;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-close {
  background: none;
  border: none;
  color: white;
  font-size: 24px;
  cursor: pointer;
  opacity: 0.8;
  transition: opacity 0.2s;
}

.modal-close:hover {
  opacity: 1;
}

.modal-body {
  padding: 24px;
}

.modal-footer {
  padding: 16px 24px;
  border-top: 1px solid #e2e8f0;
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  background: #f8fafc;
}

/* Form Elements */
.form-group {
  margin-bottom: 16px;
}

.form-label {
  display: block;
  font-weight: 600;
  color: #1e293b;
  margin-bottom: 8px;
  font-size: 14px;
}

.form-control {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  font-size: 14px;
  transition: all 0.2s;
}

.form-control:focus {
  outline: none;
  border-color: #0ea5e9;
  box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
}

textarea.form-control {
  resize: vertical;
  min-height: 100px;
  font-family: inherit;
}

/* Radio Options */
.radio-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 16px;
}

.radio-option {
  display: flex;
  align-items: center;
  padding: 12px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.radio-option:hover {
  border-color: #0ea5e9;
  background: #f0f9ff;
}

.radio-option input[type="radio"] {
  margin-right: 12px;
  cursor: pointer;
  accent-color: #0ea5e9;
}

/* Patient Info Header */
.patient-info {
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 16px;
  font-size: 13px;
  color: #0369a1;
}

.patient-info strong {
  color: #0c4a6e;
}

/* Divider */
hr {
  border: none;
  border-top: 1px solid #e2e8f0;
  margin: 20px 0;
}

/* Disabled State */
.disabled-form {
  opacity: 0.6;
  pointer-events: none;
}

/* Responsive */
@media (max-width: 768px) {
  .block-container { padding: 1rem; }
  .modal-dialog { width: 95%; }
  h1 { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div style="text-align: center;">
    <h1 style="margin: 0; color: #0f172a;">🩸 Explainable Leukemia Diagnosis Platform</h1>
    <p style="color: #64748b; margin: 2px 0 0 0; font-size: 14px;">Hybrid CNN–VLM with Knowledge Graph Reasoning for Automated Clinical Reporting</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "review_status" not in st.session_state:
    st.session_state.review_status = "pending"
if "review_comments" not in st.session_state:
    st.session_state.review_comments = ""
if "analysis_required" not in st.session_state:
    st.session_state.analysis_required = False
if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = 30
if "patient_sex" not in st.session_state:
    st.session_state.patient_sex = "Unknown"
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False
if "review_submitted" not in st.session_state:
    st.session_state.review_submitted = False
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False
# ----------------------------
# MAIN LAYOUT: 25% LEFT, 75% RIGHT
# ----------------------------
left_col, right_col = st.columns([0.25, 0.75])

# ============================
# LEFT COLUMN: INPUT CONTROLS (25%)
# ============================
with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 📥 Input")

    inputs_disabled = st.session_state.report_generated

    model_choice = st.selectbox(
        "Select Model",
        ["blip2-leukemia-xl"],
        disabled=inputs_disabled
    )

    with st.expander("Patient Details", expanded=False):

        patient_id = st.text_input(
            "Patient ID",
            value=st.session_state.patient_id,
            disabled=inputs_disabled
        )

        patient_age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=st.session_state.patient_age,
            disabled=inputs_disabled
        )

        patient_sex = st.selectbox(
            "Sex",
            ["Unknown", "Male", "Female"],
            index=["Unknown","Male","Female"].index(st.session_state.patient_sex),
            disabled=inputs_disabled
        )

        symptoms = st.text_area(
            "Symptoms",
            disabled=inputs_disabled
        )

    uploaded_file = st.file_uploader(
        "Upload Blood Smear Image",
        type=["jpg", "jpeg", "png"],
        disabled=inputs_disabled
    )

      # Image preview
    if uploaded_file is not None:
        st.markdown("### Uploaded Image")
        image = Image.open(uploaded_file)
               
        with st.popover(f"📄 {uploaded_file.name}"):
            st.image(image, use_container_width=True)

    generate_report = st.button(
        "Generate Report",
        use_container_width=True,
        type="primary",
        disabled=inputs_disabled
    )
    
    # Initialize model once
    cnn_model_path = os.path.join(os.path.dirname(__file__), '../../model/best_model_DenseNet121.h5')
    
    cnn = CNNModel(cnn_model_path)

    @st.cache_resource(show_spinner=False)
    def load_vlm_model(model_path):
        st.session_state.logs.append("Loading VLM model…")
        return VLMModel(model_path)
    report_agent = ReportAgent()
    # Placeholder model path (update after adding your model)
    VLM_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../model/blip2-leukemia-xl')
    

    st.markdown("</div>", unsafe_allow_html=True)
# ============================
# RIGHT COLUMN: ANALYSIS RESULTS (75%)
# ============================

with right_col:
    if st.session_state.last_report is None:
        # Placeholder state
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #94a3b8;">
            <p style="font-size: 18px; margin-bottom: 12px;">📤 Upload a blood smear image and click "Generate Report" to begin analysis</p>
            <p style="font-size: 13px; color: #cbd5e1;">Results will appear here with CNN classification, VLM analysis, knowledge graph reasoning, and structured reporting.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        r = st.session_state.last_report
        
        # TOP SECTION: Clinical Summary
        #st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        #st.markdown(f"""
        #<div style="display: flex; justify-content: space-between; align-items: start;">
        #    <div>
        #        <div class="summary-title">📊 Clinical Impression</div>
        #        <div class="summary-finding">{r['label']}</div>
        #        <div class="summary-meta">
        #            <span>🎯 Confidence: <strong>{r['confidence']*100:.1f}%</strong></span>
        #            <span>⚠️ Risk: <strong>{r['risk']}</strong></span>
        #        </div>
        #    </div>
        #    <div>
        #        <span class="status-badge status-review">📋 PENDING REVIEW</span>
        #    </div>
        #</div>
        #""", unsafe_allow_html=True)
        #st.markdown('</div>', unsafe_allow_html=True)"""
        
        # Patient Info Display
        if st.session_state.patient_id:
            st.markdown(f"""
            <div class="patient-info">
                👤 <strong>Patient:</strong> {st.session_state.patient_id} | 
                <strong>Age:</strong> {st.session_state.patient_age} | 
                <strong>Sex:</strong> {st.session_state.patient_sex}
            </div>
            """, unsafe_allow_html=True)
        
        # ROW 1: CNN + VLM (LEFT) | KG REASONING (RIGHT)
        row1_left, row1_right = st.columns(2)
        
        # LEFT: CNN Classification + VLM Analysis
        with row1_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><h3>🔬 CNN Classification + 🤖 VLM Analysis</h3></div>', unsafe_allow_html=True)
            
            # Metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Predicted</div><div class="metric-value">{r["label"].split()[0]}</div></div>', unsafe_allow_html=True)
            with col_m2:
                confidence_pct = r['confidence'] * 100
                risk_class = "risk-high" if confidence_pct > 80 else "risk-moderate"
                st.markdown(f'<div class="metric-box {risk_class}"><div class="metric-label">Confidence</div><div class="metric-value">{confidence_pct:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.markdown("**Morphological Summary:**")
            st.markdown(f'<div class="content-box">{r["summary"]}', unsafe_allow_html=True)
            
            #st.markdown('</div>', unsafe_allow_html=True)
        
        # RIGHT: Knowledge Graph Reasoning
        with row1_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><h3>🧠 Knowledge Graph</h3></div>', unsafe_allow_html=True)
                                  
            st.markdown("**Reasoning Summary:**")
            st.markdown("""
            <div class="content-box kg">
                <p>Knowledge graph reasoning is currently under development. Future updates will provide detailed insights into the relationships between morphological features, blast characteristics, and clinical outcomes.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True) 
                 
            if st.session_state.review_status == "approved":

                st.divider()

                #st.success("human review approved ✅", icon="✅")
                st.info("human review loop development in progress.", icon="ℹ️")                
                
                # Two columns layout
                col1, col2 = st.columns(2)

                with col1:
                    confidence_pct = r['confidence'] * 100
                    report = report_agent.create_diagnostic_report(
                        report_id=r["report_id"],
                        patient_id=st.session_state.patient_id,
                        performer_name="Dr.Explainable AI",
                        subtype=r["label"],
                        confidence=confidence_pct + "%",
                        morphology=r["summary"],
                        kg_validation="NIL",
                        pdf_url=f"https://yourapp/report/leukemia-report-{r['report_id']}.pdf"
                    )
                    
                    #st.json(report)
                    st.download_button(
                        label="⬇️ Download Diagnostic Report JSON",
                        data=report_agent.to_json(report),
                        file_name=f"leukemia-report-{r['report_id']}.json",
                        mime="application/json"
                    )
                with col2: 
                                      
                    if st.button("⬇️ Generate PDF"):                        
                        pdf_file = report_agent.generate_pdf(r['summary'],r['label'],r['confidence'])
                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                label="Download Hematology Report",
                                data=f,
                                file_name=f"hematology_report_{r['report_id']}.pdf",
                                mime="application/pdf"
                            ) 

# ============================
# GENERATE REPORT ACTION
# ============================
if generate_report:
    if uploaded_file is None:
        st.error("❌ Please upload an image first.")
    else:
        st.session_state.logs.append("Starting analysis workflow...")
        with st.status("🔄 Analyzing image... This may take a few moments.", expanded=True) as status:
            time.sleep(0.3)
            st.write("📤 Loading CNN model...")  
            # Save resized image with unique ID
            save_path, filename, report_id = processor.resize_and_save(image)
            st.write(f"✅ Image saved at {save_path}")
            st.write(f"📂 Filename: {filename}")          
            time.sleep(0.5)
            
            st.write("🎯 Performing image classification...")
            time.sleep(0.5)
            class_name, confidence = cnn.predict(image)
            
            st.write("🤖 VLM analysis in progress...")
            # VLM inference if Finetuned Model selected
            if model_choice == "blip2-leukemia-xl":
                # 
                st.session_state.logs.append("Running VLM inference…")
                prompt = """
                Analyze the microscopy image and provide the following:
                Morphological findings:
                Blast characteristics:
                Presence of Auer rods:
                """
                try:
                    vlm = load_vlm_model(VLM_MODEL_PATH)
                    st.session_state.logs.append("Model loaded, generating report…")
                    generated_text = vlm.predict(save_path,class_name, prompt)
                    st.session_state.logs.append("Report generated. Parsing output…")
                    label = generated_text.split(". ")[0] if generated_text else "Result unavailable"
                    generated_text += f"\n\nFinal Diagnosis: {class_name}"
                    st.session_state.logs.append("Parsing complete.")
                    summary = report_agent.generate_ui_text(generated_text)
                    st.session_state.logs.append(f"UI text generated. {summary[-50:]}")  # Log last 50 chars of summary for debugging
                except Exception as e:
                    label = "[VLM inference failed]"
                    summary = str(e)
                    st.session_state.logs.append("Failed to generate report.")
                        
                finally:
                    if label:
                        st.session_state.logs.append(f"VLM label: {label}")
                    st.session_state.logs.append("VLM analysis complete.")
            else:
                st.session_state.logs.append("No model selected, generating placeholder report…")                                
                summary = "NIL"
            time.sleep(0.3)
            
            st.write("🧠 Building knowledge graph...")
            time.sleep(0.3)
            
            status.update(label="✅ Analysis complete!", state="complete")
        
        # Store report
        st.session_state.last_report = {
            "report_id": report_id,
            "model": model_choice,
            "label": class_name,
            "confidence": confidence,
            "risk": "Moderate" if 0.6 <= confidence <= 0.8 else ("High" if confidence > 0.8 else "Low"),
            "summary": summary,
            "raw_output": generated_text if 'generated_text' in locals() else "NIL"
        }
        #st.session_state.report_generated = True
        st.session_state.review_submitted = False
        st.session_state.review_status = "pending"
        st.session_state.review_status = "approved"  # For demonstration, auto-approve the report
        st.session_state.review_comments = ""
        st.session_state.logs.append(f"✅ Classification complete: {class_name} (confidence: {confidence:.2f})")
        
        st.rerun()