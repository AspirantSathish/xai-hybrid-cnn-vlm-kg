import streamlit as st
from PIL import Image

st.set_page_config(page_title="Explainable Leukemia Diagnosis Platform", layout="wide")

# --- Layout ---
left, right = st.columns([1, 2])

# --- Left Panel: Patient Info & Upload ---
with left:
    st.header("🩸 Patient Information")
    patient_id = st.text_input("Patient ID", "P12345")
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    uploaded_file = st.file_uploader("Upload Blood Smear Image", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("Upload a sample image to begin analysis.")

    if st.button("Start Analysis"):
        st.session_state["stage"] = "analysis"

    if st.button("Restart Workflow"):
        st.session_state.clear()
        st.experimental_rerun()

# --- Right Panel: Dynamic Workflow ---
with right:
    st.header("⚙️ AI Workflow")

    stage = st.session_state.get("stage", "idle")

    # Stage: Analysis Progress
    if stage == "analysis":
        st.subheader("AI Analysis Progress")
        st.write("Preprocessing → CNN → VLM → KG → Draft Report")
        st.success("✅ Draft Report Ready")

        if st.button("Review AI Report"):
            st.session_state["stage"] = "review"
            st.experimental_rerun()

        if st.button("Back"):
            st.session_state["stage"] = "idle"
            st.experimental_rerun()

    # Stage: Human Review
    elif stage == "review":
        st.subheader("🩺 Hematologist Review")
        st.write("AI Report Draft: Subtype ALL, Morphological explanation, KG evidence")

        review = st.radio("Validation:", ["Approve ✅", "Needs Correction ✍️", "Reject ❌"])
        comments = st.text_area("Comments (optional)", "e.g., note atypical lymphocytes...")

        if st.button("Submit Review"):
            st.session_state["review_status"] = review
            st.session_state["comments"] = comments
            st.session_state["stage"] = "final"
            st.experimental_rerun()

        if st.button("Back"):
            st.session_state["stage"] = "analysis"
            st.experimental_rerun()

    # Stage: Final Report
    elif stage == "final":
        st.subheader("📄 Final Diagnosis Report")
        st.success("Leukemia Diagnosis Complete")
        st.write("Subtype: Acute Lymphoblastic Leukemia (ALL)")
        st.write("Validation Status:", st.session_state["review_status"])
        st.write("Comments:", st.session_state["comments"])
        st.download_button("Download Report", "Dummy PDF Content", file_name="Leukemia_Report.pdf")

        if st.button("Back"):
            st.session_state["stage"] = "review"
            st.experimental_rerun()
