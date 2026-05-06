"""
LeafLens — Streamlit Frontend
Run: streamlit run streamlit_app.py
"""

import io
import base64
import requests
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="LeafLens — Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:5000"

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f0; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a5c2a, #2d8a44);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p  { font-size: 1.1rem; margin: 0.5rem 0 0; opacity: 0.9; }

    /* Cards */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #2d8a44;
    }
    .warning-card {
        background: #fff8e1;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #f9a825;
    }
    .danger-card {
        background: #ffebee;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #c62828;
    }
    .healthy-card {
        background: #e8f5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1b5e20;
    }

    /* Severity badge */
    .severity-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-none     { background: #c8e6c9; color: #1b5e20; }
    .badge-moderate { background: #ffe0b2; color: #e65100; }
    .badge-high     { background: #ffcdd2; color: #b71c1c; }
    .badge-critical { background: #b71c1c; color: white; }

    /* Confidence bar */
    .conf-bar-outer {
        background: #e0e0e0;
        border-radius: 8px;
        height: 18px;
        width: 100%;
        overflow: hidden;
    }
    .conf-bar-inner {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #43a047, #1b5e20);
        transition: width 0.5s ease;
    }

    /* Metrics row */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    /* Upload area */
    .uploadedFile { border-radius: 8px; }

    /* Sidebar */
    .css-1d391kg { background-color: #1a5c2a; }
    section[data-testid="stSidebar"] { background-color: #1a3a22; }
    section[data-testid="stSidebar"] * { color: #d4edda !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: #d4edda !important; }

    /* How-it-works steps */
    .step-box {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .step-num {
        background: #2d8a44;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
SEVERITY_CARD = {
    "None":     "healthy-card",
    "Moderate": "warning-card",
    "High":     "danger-card",
    "Critical": "danger-card",
}
SEVERITY_BADGE = {
    "None":     "badge-none",
    "Moderate": "badge-moderate",
    "High":     "badge-high",
    "Critical": "badge-critical",
}
SEVERITY_ICON = {
    "None":     "✅",
    "Moderate": "⚠️",
    "High":     "🔴",
    "Critical": "🚨",
}

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_api(b64_image: str) -> dict:
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"image": b64_image},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to Flask API. Make sure it is running on port 5000."}
    except requests.exceptions.Timeout:
        return {"error": "API request timed out. Please try again."}
    except Exception as e:
        return {"error": str(e)}


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 LeafLens")
    st.markdown("---")

    # API status
    api_ok = check_api_health()
    if api_ok:
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline\n\nRun: `python backend/app.py`")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    LeafLens uses **MobileNetV2** + **transfer learning** to diagnose crop diseases
    from leaf images across **38 disease classes**.
    """)

    st.markdown("---")
    st.markdown("### Supported Crops")
    crops = ["🍎 Apple", "🍒 Cherry", "🌽 Corn", "🍇 Grape",
             "🍊 Orange", "🍑 Peach", "🫑 Pepper", "🥔 Potato",
             "🍓 Strawberry", "🍅 Tomato"]
    for c in crops:
        st.markdown(f"- {c}")

    st.markdown("---")
    st.markdown("### Tips for best results")
    st.markdown("""
    - Use **close-up** images of the leaf
    - Ensure **good lighting** (natural light preferred)
    - Include **both sides** of symptomatic leaves
    - Image size ≥ 200×200 pixels
    """)

    confidence_threshold = st.slider(
        "Confidence threshold (%)", 30, 90, 50,
        help="Predictions below this are shown as uncertain"
    )

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 LeafLens</h1>
    <p>AI-Powered Crop Disease Detection — Upload a leaf image for instant diagnosis</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAIN TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Diagnose", "📊 How It Works", "ℹ️ About"])

# ────────── TAB 1: DIAGNOSE ──────────
with tab1:
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.markdown("### Upload Leaf Image")
        upload_mode = st.radio(
            "Input method",
            ["Upload from device", "Use camera"],
            horizontal=True,
            label_visibility="collapsed"
        )

        uploaded_file = None
        if upload_mode == "Upload from device":
            uploaded_file = st.file_uploader(
                "Choose a leaf image",
                type=["jpg", "jpeg", "png", "webp"],
                help="Supported formats: JPG, PNG, WebP"
            )
        else:
            uploaded_file = st.camera_input("Take a photo of the leaf")

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded leaf", use_column_width=True)

            # Show image metadata
            st.caption(f"Size: {img.width}×{img.height}px  |  Mode: {img.mode}")

            st.markdown("---")
            analyze_btn = st.button(
                "🔬 Analyze Leaf",
                type="primary",
                use_container_width=True,
                disabled=not api_ok
            )

            if not api_ok:
                st.warning("Start the Flask API first to enable analysis.")

    with col_result:
        st.markdown("### Diagnosis Result")

        if not uploaded_file:
            st.info("👆 Upload a leaf image on the left to get started.")

            # How-it-works quick steps
            st.markdown("#### How it works")
            steps = [
                ("Upload", "Choose or take a photo of any crop leaf."),
                ("Analyze", "MobileNetV2 CNN processes the image in milliseconds."),
                ("Diagnose", "Get disease name, confidence score, and treatment plan."),
            ]
            for i, (title, desc) in enumerate(steps, 1):
                st.markdown(f"""
                <div class="step-box">
                    <div class="step-num">{i}</div>
                    <div>
                        <strong>{title}</strong><br>
                        <span style="color:#555; font-size:0.9rem;">{desc}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        elif uploaded_file and "analyze_btn" in dir() and analyze_btn:
            with st.spinner("🔬 Analyzing leaf…"):
                b64 = image_to_base64(img)
                result = call_api(b64)

            if "error" in result:
                st.error(f"❌ {result['error']}")

            elif result.get("status") == "uncertain":
                st.warning(f"""
                **Low confidence ({result['confidence']}%)**

                {result['message']}
                """)
                if "top3" in result:
                    st.markdown("##### Top guesses:")
                    for item in result["top3"]:
                        name = item["class"].replace("___", " — ").replace("_", " ")
                        st.markdown(f"- **{name}** — {item['confidence']}%")

            elif result.get("status") == "success":
                sev   = result.get("severity", "Unknown")
                icon  = SEVERITY_ICON.get(sev, "🔵")
                card  = SEVERITY_CARD.get(sev, "result-card")
                badge = SEVERITY_BADGE.get(sev, "badge-moderate")
                conf  = result["confidence"]

                # Main result card
                st.markdown(f"""
                <div class="{card}">
                    <h3>{icon} {result['display_name']}</h3>
                    <span class="severity-badge {badge}">Severity: {sev}</span>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                bar_color = "#43a047" if conf >= 80 else "#fb8c00" if conf >= 60 else "#e53935"
                st.markdown(f"""
                <p style="margin:0.5rem 0 4px; font-size:0.9rem; color:#444;">
                    Confidence: <strong>{conf}%</strong>
                </p>
                <div class="conf-bar-outer">
                    <div class="conf-bar-inner" style="width:{conf}%; background:{bar_color};"></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Treatment + Prevention
                col_t, col_p = st.columns(2)
                with col_t:
                    st.markdown("#### 💊 Treatment")
                    st.info(result["treatment"])
                with col_p:
                    st.markdown("#### 🛡️ Prevention")
                    st.success(result["prevention"])

                st.markdown("---")

                # Top-3
                st.markdown("#### 📊 Top-3 Predictions")
                for i, item in enumerate(result.get("top3", []), 1):
                    name = item["class"].replace("___", " — ").replace("_", " ")
                    c    = item["confidence"]
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:10px; margin:6px 0;">
                        <span style="width:24px; font-weight:700; color:#555;">#{i}</span>
                        <span style="flex:1; font-size:0.9rem;">{name}</span>
                        <span style="font-weight:600; min-width:52px; text-align:right;">{c}%</span>
                    </div>
                    <div class="conf-bar-outer" style="height:10px; margin-bottom:8px;">
                        <div class="conf-bar-inner" style="width:{min(c,100)}%;"></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Download report
                st.markdown("---")
                report = f"""LeafLens Diagnosis Report
========================
Disease:    {result['display_name']}
Severity:   {result['severity']}
Confidence: {result['confidence']}%

Treatment:
{result['treatment']}

Prevention:
{result['prevention']}

Top-3 Predictions:
{chr(10).join([f"  {i+1}. {x['class']} — {x['confidence']}%" for i, x in enumerate(result.get('top3',[]))])}
"""
                st.download_button(
                    "📥 Download Report",
                    data=report,
                    file_name="leaflens_report.txt",
                    mime="text/plain"
                )

# ────────── TAB 2: HOW IT WORKS ──────────
with tab2:
    st.markdown("### System Architecture")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        #### Frontend (Streamlit)
        - Image upload / camera input
        - Image preview & preprocessing
        - Results visualization
        - Report download
        """)
    with c2:
        st.markdown("""
        #### Backend (Flask API)
        - REST endpoint `/predict`
        - Base64 image decoding
        - Model inference
        - Treatment lookup
        """)
    with c3:
        st.markdown("""
        #### AI Model (TensorFlow)
        - MobileNetV2 backbone
        - Transfer learning
        - 38-class softmax output
        - Confidence scoring
        """)

    st.markdown("---")
    st.markdown("### ML Pipeline")
    st.code("""
Image Input (any size)
    ↓ PIL.Image resize to 224×224
    ↓ Normalize pixels: /255.0
    ↓ np.expand_dims → shape (1, 224, 224, 3)
    ↓ MobileNetV2 feature extraction (frozen)
    ↓ GlobalAveragePooling2D → (1, 1280)
    ↓ Dense(256, relu) → Dropout(0.4)
    ↓ Dense(128, relu) → Dropout(0.3)
    ↓ Dense(38, softmax)
    ↓ argmax → predicted class index
    ↓ class_names[index] → disease name
Output: disease, confidence, treatment
    """, language="text")

    st.markdown("### Dataset")
    st.markdown("""
    | Property | Value |
    |---|---|
    | Name | PlantVillage |
    | Total images | ~54,000 |
    | Classes | 38 (disease + healthy) |
    | Crops | Apple, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato |
    | Image size | 256×256 (resized to 224×224) |
    | Source | [Kaggle PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) |
    """)

# ────────── TAB 3: ABOUT ──────────
with tab3:
    st.markdown("### LeafLens")
    st.markdown("""
    LeafLens is an AI-powered crop disease detection web application built as part of a final year
    B.E. Artificial Intelligence & Data Science project.

    **Tech Stack:**
    - Model: TensorFlow 2.x, MobileNetV2 (transfer learning)
    - Backend: Flask, Python 3.10+
    - Frontend: Streamlit
    - Dataset: PlantVillage (38 disease classes)

    **Developer:** Mayuri Rajendra Dorale
    **Institution:** Dr. D Y Patil Institute of Engineering, Management & Research, Akurdi, Pune
    """)