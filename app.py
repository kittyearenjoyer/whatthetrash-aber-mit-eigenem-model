import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ----------------
MODEL_PATH = "mein model für die app.h5"

st.set_page_config(
    page_title="What the Trash",
    page_icon="♻️",
    layout="wide",
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* ---------- Hintergrund ---------- */
.main {
    background: linear-gradient(180deg, #f8fafc 0%, #e5e7eb 100%);
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

/* ---------- Alle Texte schwarz ---------- */
html, body, p, span, div, label, h1, h2, h3, h4, h5, h6 {
    color: black !important;
}

/* ---------- Hero ---------- */
.hero-box {
    padding: 2.5rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #22c55e, #86efac);
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    margin-bottom: 1.5rem;
}

/* ---------- Cards ---------- */
.glass-card {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 22px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

.metric-box {
    background: rgba(255,255,255,0.7);
    padding: 1rem;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(0,0,0,0.08);
}

.small-text {
    font-size: 0.95rem;
}

/* ---------- Upload + Kamera ---------- */
[data-testid="stFileUploader"],
[data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.65);
    padding: 1rem;
    border-radius: 18px;
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
    color: black !important;
}

div[data-testid="stTabs"] + div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LADEN ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- KLASSEN ANPASSEN ----------------
# HIER DEINE REIHENFOLGE EINTRAGEN
class_names = [
    "plastic",
    "paper",
    "glass",
    "metal",
    "organic",
    "trash"
]

# ---------------- FUNKTIONEN ----------------
def preprocess_image(image):
    img = image.resize((224, 224))   # falls dein Modell andere Größe braucht ändern
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_disposal(label):
    disposal_map = {
        "plastic": "Gelbe Tonne 🟡",
        "paper": "Blaue Tonne 🔵",
        "cardboard": "Blaue Tonne 🔵",
        "glass": "Glascontainer 🟢",
        "metal": "Gelbe Tonne 🟡",
        "organic": "Biotonne 🟤",
        "trash": "Restmüll ⚫",
    }

    for key in disposal_map:
        if key in label.lower():
            return disposal_map[key]

    return "Lokal prüfen 📍"

def run_prediction(image):
    img = preprocess_image(image)
    pred = model.predict(img, verbose=0)[0]

    idx = np.argmax(pred)
    label = class_names[idx]
    score = float(pred[idx])

    return label, score

# ---------------- HERO ----------------
st.markdown("""
<div class="hero-box">
    <h1>♻️ What the Trash</h1>
    <p>KI erkennt deinen Müll per Upload oder Webcam.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- STATS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="metric-box"><h3>6+</h3><div class="small-text">Kategorien</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-box"><h3>AI</h3><div class="small-text">TensorFlow Modell</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="metric-box"><h3>Webcam</h3><div class="small-text">Live Nutzung</div></div>', unsafe_allow_html=True)

st.write("")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📁 Upload", "📷 Webcam"])

# ---------- Upload ----------
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        with st.spinner("KI analysiert Bild..."):
            label, score = run_prediction(image)

        st.success(f"Erkannt: {label}")
        st.info(f"Sicherheit: {score:.1%}")
        st.warning(f"Entsorgung: {get_disposal(label)}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Webcam ----------
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    camera = st.camera_input("Foto aufnehmen")

    if camera:
        image = Image.open(camera).convert("RGB")
        st.image(image, use_container_width=True)

        with st.spinner("KI analysiert Bild..."):
            label, score = run_prediction(image)

        st.success(f"Erkannt: {label}")
        st.info(f"Sicherheit: {score:.1%}")
        st.warning(f"Entsorgung: {get_disposal(label)}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("")
st.caption("What the Trash • TensorFlow Waste Classifier")
