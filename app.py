import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile
import os
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MüllScan – KI-Mülltrennung",
    page_icon="♻️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d1f0f;
    color: #e8f5e9;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #69f0ae;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 1.1rem;
    color: #a5d6a7;
    margin-bottom: 2rem;
}

.bin-card {
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    border: 2px solid transparent;
    transition: transform 0.2s;
}

.bin-card:hover { transform: translateY(-2px); }

.bin-yellow  { background: #2a2700; border-color: #fdd835; color: #fdd835; }
.bin-blue    { background: #001a2e; border-color: #29b6f6; color: #29b6f6; }
.bin-brown   { background: #1a0e00; border-color: #a1887f; color: #a1887f; }
.bin-green   { background: #001a06; border-color: #66bb6a; color: #66bb6a; }
.bin-gray    { background: #1a1a1a; border-color: #9e9e9e; color: #9e9e9e; }
.bin-red     { background: #1a0000; border-color: #ef5350; color: #ef5350; }

.result-box {
    background: #132b14;
    border: 2px solid #2e7d32;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
}

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #69f0ae;
}

.result-conf {
    font-size: 0.9rem;
    color: #81c784;
    margin-top: 0.2rem;
}

.badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

.divider {
    border: none;
    border-top: 1px solid #1b5e20;
    margin: 2rem 0;
}

.tip-box {
    background: #0a1f0b;
    border-left: 4px solid #69f0ae;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.92rem;
    color: #c8e6c9;
    margin-top: 0.8rem;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Bin mapping ────────────────────────────────────────────────────────────────
BIN_INFO = {
    "plastic":      {"label": "Gelbe Tonne",   "css": "bin-yellow", "emoji": "🟡", "tip": "Leere Plastikverpackungen, Dosen, Verbundmaterialien. Bitte kurz ausspülen."},
    "paper":        {"label": "Papiertonne",   "css": "bin-blue",   "emoji": "📘", "tip": "Zeitungen, Kartons, Pappe. Kein beschmutztes Papier."},
    "glass":        {"label": "Glascontainer", "css": "bin-green",  "emoji": "🟢", "tip": "Nach Farbe trennen: weiß, braun, grün. Keine Glasscherben oder Pfandflaschen."},
    "organic":      {"label": "Biotonne",      "css": "bin-brown",  "emoji": "🟤", "tip": "Lebensmittelreste, Kaffeesatz, Gartenabfälle. Kein Fleisch bei offener Biotonne."},
    "residual":     {"label": "Restmülltonne", "css": "bin-gray",   "emoji": "⚫", "tip": "Alles, was nicht recycelt werden kann. So wenig wie möglich hierher."},
    "hazardous":    {"label": "Sondermüll",    "css": "bin-red",    "emoji": "🔴", "tip": "Batterien, Farben, Chemikalien → zum Wertstoffhof bringen!"},
}

def map_prediction_to_bin(class_name: str) -> dict:
    """Map roboflow class names to German bin categories."""
    c = class_name.lower()
    if any(k in c for k in ["plastic","bottle","can","metal","tin","foil","packaging","verpackung"]):
        return BIN_INFO["plastic"]
    if any(k in c for k in ["paper","cardboard","carton","newspaper","box","papier","karton"]):
        return BIN_INFO["paper"]
    if any(k in c for k in ["glass","glas","jar"]):
        return BIN_INFO["glass"]
    if any(k in c for k in ["food","organic","fruit","vegetable","bio","compost","coffee"]):
        return BIN_INFO["organic"]
    if any(k in c for k in ["battery","chemical","paint","hazard","sonder"]):
        return BIN_INFO["hazardous"]
    return BIN_INFO["residual"]

# ── Roboflow client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="7o2Yvo2UzYCr4sKiJwSV"
    )

def run_inference(image_path: str):
    client = get_client()
    result = client.run_workflow(
        workspace_name="juliuss-workspace-gdhwh",
        workflow_id="detect-and-classify-4",
        images={"image": image_path},
        use_cache=True
    )
    return result

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">♻️ MüllScan</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Foto hochladen – KI erkennt den richtigen Mülleimer.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Bild auswählen (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

camera = st.camera_input("… oder direkt mit der Kamera fotografieren")

image_source = uploaded or camera

if image_source:
    img = Image.open(image_source)
    st.image(img, use_container_width=True, caption="Dein Bild")

    if st.button("🔍 Jetzt analysieren", use_container_width=True, type="primary"):
        with st.spinner("KI analysiert das Bild …"):
            # save to temp file
            suffix = ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                img_rgb = img.convert("RGB")
                img_rgb.save(tmp.name, format="JPEG")
                tmp_path = tmp.name

            try:
                result = run_inference(tmp_path)
            except Exception as e:
                st.error(f"Fehler bei der API-Anfrage: {e}")
                st.stop()
            finally:
                os.unlink(tmp_path)

        # ── Parse result ──────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 📊 Ergebnis")

        # Roboflow workflow results can vary by workflow type.
        # We try several common output shapes.
        detections = []
        try:
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                # shape A: list of dicts with 'predictions'
                if "predictions" in first:
                    preds = first["predictions"]
                    if isinstance(preds, list):
                        detections = preds
                    elif isinstance(preds, dict) and "predictions" in preds:
                        detections = preds["predictions"]
                # shape B: direct list of prediction dicts
                elif "class" in first:
                    detections = result
            elif isinstance(result, dict):
                if "predictions" in result:
                    detections = result["predictions"]
        except Exception:
            pass

        if detections:
            # Sort by confidence descending
            try:
                detections = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)
            except Exception:
                pass

            top = detections[0]
            class_name = top.get("class", top.get("label", "Unbekannt"))
            confidence = top.get("confidence", 0)

            bin_info = map_prediction_to_bin(class_name)

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">{bin_info['emoji']} {bin_info['label']}</div>
                <div class="result-conf">Erkannt als: <b>{class_name}</b> &nbsp;·&nbsp; Konfidenz: <b>{confidence*100:.1f}%</b></div>
                <div class="tip-box">💡 {bin_info['tip']}</div>
            </div>
            """, unsafe_allow_html=True)

            if len(detections) > 1:
                st.markdown("#### Weitere erkannte Objekte")
                for det in detections[1:4]:
                    cn = det.get("class", det.get("label", "?"))
                    cf = det.get("confidence", 0)
                    bi = map_prediction_to_bin(cn)
                    st.markdown(f"""
                    <div class="bin-card {bi['css']}">
                        {bi['emoji']} {cn} → {bi['label']} &nbsp;<span style="opacity:0.6;font-weight:300">({cf*100:.0f}%)</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.warning("Kein Objekt erkannt. Versuche ein deutlicheres Foto.")

            with st.expander("🔧 Debug: Rohe API-Antwort"):
                st.json(result)

else:
    st.markdown("### 🗑️ Welche Tonne ist die richtige?")
    for key, info in BIN_INFO.items():
        st.markdown(f'<div class="bin-card {info["css"]}">{info["emoji"]} <b>{info["label"]}</b> — {info["tip"]}</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#2e7d32;font-size:0.8rem;font-family:Space Mono,monospace;">MüllScan · Powered by Roboflow & Streamlit</p>', unsafe_allow_html=True)
