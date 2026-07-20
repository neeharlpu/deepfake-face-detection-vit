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
# Emerald Green Professional CSS
# -------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@700&display=swap');

    .stApp {
        background: #0a0f1c;
        color: #e0f2fe;
    }

    /* Main Title - Emerald Gradient */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 17.5px;
        color: #94a3b8;
        max-width: 700px;
    }

    /* Sidebar */
    .stSidebar {
        background: #111827;
        border-right: 1px solid #334155;
    }
    
    .sidebar-header {
        font-size: 22px;
        font-weight: 600;
        color: #34d399;
        margin-bottom: 20px;
    }

    /* Analysis Card */
    .analysis-card {
        background: linear-gradient(145deg, #1e2937, #0f172a);
        padding: 32px;
        border-radius: 20px;
        border: 1px solid #334155;
        box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.4);
    }

    /* Result Styling */
    .success-box {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #34d399;
        color: #34d399;
        padding: 20px;
        border-radius: 16px;
        font-size: 26px;
        font-weight: 700;
    }
    
    .error-box {
        background: rgba(248, 113, 113, 0.15);
        border: 1px solid #f87171;
        color: #f87171;
        padding: 20px;
        border-radius: 16px;
        font-size: 26px;
        font-weight: 700;
    }

    /* Progress Bars - Emerald */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981, #34d399) !important;
    }
    
    .prob-label {
        font-weight: 600;
        color: #cbd5e1;
        margin-top: 20px;
        margin-bottom: 8px;
        font-size: 15px;
    }

    /* Image Styling */
    .stImage img {
        border-radius: 18px;
        border: 2px solid #334155;
    }

    /* Metric Enhancement */
    .stMetric [data-testid="stMetricValue"] {
        color: #6ee7b7;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 class='sidebar-header'>🛡️ Deepfake Detector</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### Model")
    st.info("**Vision Transformer (ViT-Base)**\n\n`Patch16-224`")
    
    st.markdown("### Training")
    st.caption("140,000+ Real & AI-Generated Faces")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "97.5%")
    with col2:
        st.metric("ROC-AUC", "0.997")

# -------------------------------------------------------
# Main Header
# -------------------------------------------------------
st.markdown("<div class='main-title'>Deepfake Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced AI-powered system that identifies real faces from synthetic and manipulated imagery using state-of-the-art Vision Transformer technology.</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------
# Upload & Analysis
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a clear facial image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Best results with well-lit, front-facing portraits."
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1.1, 1], gap="large")

        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🔍 Analysis Result")
            
            with st.spinner("Analyzing image with Vision Transformer..."):
                result = predict(image)

            label = result.get("label", "Unknown")
            confidence = result.get("confidence", 0.0)
            fake_prob = result.get("fake_probability", 0.0)
            real_prob = result.get("real_probability", 0.0)

            # Result Display
            if label == "Real":
                st.markdown('<div class="success-box">✅ AUTHENTIC FACE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ SYNTHETIC / DEEPFAKE</div>', unsafe_allow_html=True)

            st.markdown("---")
            
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.1f}%"
            )

            st.markdown("<div class='prob-label'>Real Face Probability</div>", unsafe_allow_html=True)
            st.progress(real_prob / 100.0)
            st.caption(f"**{real_prob:.1f}%**")

            st.markdown("<div class='prob-label'>Fake Face Probability</div>", unsafe_allow_html=True)
            st.progress(fake_prob / 100.0)
            st.caption(f"**{fake_prob:.1f}%**")

    except Exception as e:
        st.error(f"Failed to process image: {e}")

# Footer
st.markdown("---")
st.caption("Vision Transformer • Emerald AI Security")