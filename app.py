import streamlit as st
from PIL import Image
from predict import predict

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Face Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Custom CSS (Professional Typography & Layout)
# -------------------------------------------------------
st.markdown("""
<style>
    /* Global Overrides */
    .reportview-container {
        background: #f8fafc;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 8px;
    }
    .sub-title {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 24px;
    }
    
    /* Metrics and Status */
    .prediction-header {
        font-size: 20px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .status-text {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Progress Bar Labels */
    .prob-label {
        font-size: 14px;
        font-weight: 500;
        color: #475569;
        margin-top: 10px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar (Model Meta-Data)
# -------------------------------------------------------
with st.sidebar:
    st.title("📌 Project Overview")
    st.markdown("---")

    st.markdown("### Architecture")
    st.info("**Vision Transformer (ViT)**\n\n`Google ViT-Base-Patch16-224`")

    st.markdown("### Dataset Details")
    st.caption("Trained on **140K** Real & Fake Face Datasets")
    
    st.markdown("### Model Performance")
    col_acc, col_auc = st.columns(2)
    with col_acc:
        st.metric(label="Test Accuracy", value="97.50%")
    with col_auc:
        st.metric(label="ROC-AUC", value="0.997")

    st.markdown("---")
    st.markdown("### Developer")
    st.markdown("**Neehar S**  \n*M.Sc Data Science*")

# -------------------------------------------------------
# Main Header
# -------------------------------------------------------
st.markdown("<div class='main-title'>🛡️ Deepfake Face Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>An advanced Vision Transformer (ViT) deployment for distinguishing authentic facial imagery from AI-generated variants.</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------
# Core Application Logic
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a facial image for analysis (PNG, JPG, JPEG)",
    type=["jpg", "jpeg", "png"],
    help="Ensure the face is well-lit and clearly visible for optimal classification accuracy."
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        
        # UI Columns Split
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("Submitted Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Analysis & Classification")
            
            with st.spinner("Executing neural network inference..."):
                result = predict(image)

            # Extract metrics safely
            label = result.get("label", "Unknown")
            confidence = result.get("confidence", 0.0)
            fake_prob = result.get("fake_probability", 0.0)
            real_prob = result.get("real_probability", 0.0)

            # Display Result Indicator
            if label == "Real":
                st.success("### ✅ AUTHENTIC FACE")
            else:
                st.error("### ❌ AI-GENERATED / FAKE FACE")

            # Quantitative metrics display
            st.markdown("---")
            st.metric(
                label="Classification Confidence", 
                value=f"{confidence:.2f}%",
                help="The mathematical confidence level of the model's finalized decision."
            )

            # Probability Breakdowns
            st.markdown("<div class='prob-label'>Real Face Probability</div>", unsafe_allow_html=True)
            st.progress(real_prob / 100.0)
            st.caption(f"**{real_prob:.2f}%** probability of authenticity")

            st.markdown("<div class='prob-label'>Fake Face Probability</div>", unsafe_allow_html=True)
            st.progress(fake_prob / 100.0)
            st.caption(f"**{fake_prob:.2f}%** probability of synthetic manipulation")

    except Exception as e:
        st.error(f"An error occurred while processing the image. Please verify file integrity. Details: {e}")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.caption(
    "Deepfake Face Detection Architecture | Vision Transformer (ViT) Deployment Pipeline | MSc Data Science Project"
)