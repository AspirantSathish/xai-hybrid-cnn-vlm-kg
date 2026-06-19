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

# Initialize model once
cnn_model_path = os.path.join(os.path.dirname(__file__), '../../model/best_model_DenseNet121.h5')
cnn = CNNModel(cnn_model_path)

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(layout="wide", page_title="HemaXplain AI", page_icon="🩸")

# ----------------------------
# SIMPLE CSS FOR BETTER GUI
# ----------------------------
st.markdown("""
<style>
/* Make main container wider */
.block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}

/* Card look */
.card {
  background: #0f172a10;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 4px 6px;
  margin-bottom: 2px;
}

/* Badge */
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
}

/* Chat bubbles */
.chat-user {
  background: #2563eb12;
  border: 1px solid #2563eb30;
  padding: 10px 12px;
  border-radius: 14px;
  margin: 6px 0;
}
.chat-assistant {
  background: #16a34a10;
  border: 1px solid #16a34a30;
  padding: 10px 12px;
  border-radius: 14px;
  margin: 6px 0;
}
.small-muted {color:#6b7280; font-size:12px;}
hr {margin: 0.6rem 0;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.title("🩸 Explainable Leukemia Diagnosis Platform")
st.caption("Hybrid CNN–VLM with Knowledge Graph Reasoning for Automated Clinical Reporting")

# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "text": "Upload a blood smear image and click **Generate Report**. I’ll show report + explainability placeholders."}
    ]
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None

# ----------------------------
# LAYOUT
# ----------------------------
col1, col2 = st.columns([1, 2])

# ============================
# LEFT: CONTROLS
# ============================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input")

    model_choice = st.selectbox("Select Model", ["Finetuned Model"])

    # Optional patient details (EHR placeholder)
    with st.expander("Patient Details (optional)", expanded=False):
        pid = st.text_input("Patient ID", placeholder="e.g., P-1029")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Sex", ["Unknown", "Male", "Female", "Other"])
        symptoms = st.text_area("Symptoms / Notes", placeholder="e.g., fatigue, fever, bruising...")

    uploaded_file = st.file_uploader("Upload Blood Smear Image", type=["jpg", "jpeg", "png"],accept_multiple_files=False)
    # Image preview
    if uploaded_file is not None:
        st.markdown("### Uploaded Image")
        image = Image.open(uploaded_file)
               
        with st.popover(f"📄 {uploaded_file.name}"):
            st.image(image, use_container_width=True)
        
    generate_report = st.button("Generate Report", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Human feedback loop (UI only)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Human Feedback Loop")
    feedback_action = st.radio("Doctor validation", ["Pending", "Approved ✅", "Needs Correction ✍️"], horizontal=True)
    correction = st.text_area("Correction / Comments (optional)", placeholder="e.g., change AML → ALL, note atypical lymphocytes...")
    st.button("Submit Feedback", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================
# RIGHT: OUTPUT
# ============================
with col2:
    # TOP: Tabs for output organization
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Output")

    tab_classification, tab_explain, tab_report, tab_logs = st.tabs([
        "🔬 Classification + VLM Output",
        "🧠 Explainability (KG Reasoning)",
        "📄 Final Report",
        "🧾 Logs"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------
    # BUTTON ACTION: GENERATE
    # ----------------------------
    if generate_report:
        if uploaded_file is None:
            st.error("Please upload an image first.")
        else:
            st.session_state.logs.append("Starting workflow…")
            steps = []
            with st.status("Running pipeline steps…", expanded=True) as status:
                for s in steps:
                    st.write(s)
                    time.sleep(0.25)
                status.update(label="Done", state="complete")
                class_name, confidence = cnn.predict(image)
                #st.write(f"🩸 Predicted Class: **{class_name}**")
                #st.write(f"📊 Confidence: {confidence:.2f}")
            # Placeholder results
            mock_conf = 0.82
            label = "Suspected AML (placeholder)"
            risk = "Moderate risk (placeholder)"

            st.session_state.last_report = {
                "model": model_choice,
                "label": class_name,
                "confidence": confidence,
                "risk": risk,
                "summary": "Morphology suggests blast-like cells with high N:C ratio (placeholder)."
            }

            # Add to chat
            st.session_state.chat.append({"role": "user", "text": f"Generate report using {model_choice}."})
            st.session_state.chat.append({"role": "assistant", "text": f"Report generated. Predicted: {label} with confidence {mock_conf:.2f} (placeholder)."})
            st.session_state.logs.append("Workflow completed.")

            st.rerun()

    # ----------------------------
    # TAB CONTENTS
    # ----------------------------
    with tab_classification:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        #st.markdown("### Classification + VLM Output")
        if st.session_state.last_report is None:
            st.info("Generate a report to see output here.")
        else:
            r = st.session_state.last_report
            st.markdown(f"**Predicted Label:** {r['label']}")
            st.markdown(f"**Confidence:** {r['confidence']*100:.2f}%")
            st.markdown(f"**Risk:** {r['risk']}")
            st.markdown(f"**Summary:** {r['summary']}")
            st.markdown("\n*VLM output placeholder: Add visual-language model results here.*")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_explain:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Explainability (KG Reasoning)")
        if st.session_state.last_report is None:
            st.info("Generate a report to see explainability here.")
        else:
            st.markdown("**Knowledge Graph Reasoning Trace:**")
            st.write("- Entity: *Blast-like cell* → Relation: *has high N:C ratio* → Conclusion: *AML suspected* (placeholder)")
            st.write("- Entity: *No Auer rods* → Relation: *not detected* → Conclusion: *AML subtype undetermined* (placeholder)")
            st.markdown("\n*Add more KG-based reasoning details here as available.*")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_report:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Final Clinical Report")
        if st.session_state.last_report is None:
            st.info("Generate a report to see output here.")
        else:
            r = st.session_state.last_report
            badge_text = f"Confidence: {r['confidence']*100:.2f}%"
            st.markdown(f'<span class="badge">{badge_text}</span>', unsafe_allow_html=True)
            st.markdown(f"### Result\n**{r['label']}**")
            st.write(r["summary"])
            st.markdown("### Clinical Notes (placeholder)")
            st.write("- Blast-like cells: *suggested*\n- N:C ratio: *high*\n- Nucleoli: *possible*\n- Auer rods: *not seen (placeholder)*")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(r, indent=2),
                    file_name="medagent_report.json",
                    mime="application/json",
                    use_container_width=True
                )
            with c2:
                st.button("⬇️ Generate PDF (placeholder)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_logs:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Simple run logs for debugging your pipeline.")
        if st.session_state.logs:
            for line in st.session_state.logs[-50:]:
                st.write("• " + line)
        else:
            st.write("No logs yet.")
        st.markdown('</div>', unsafe_allow_html=True)