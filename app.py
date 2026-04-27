import streamlit as st
from PIL import Image
import requests
import base64
import io

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
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1f0f; color: #e8f5e9; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.hero-title { font-family: 'Space Mono', monospace; font-size: 2.8rem; font-weight: 700; color: #69f0ae; line-height: 1.1; margin-bottom: 0.2rem; }
.hero-sub { font-weight: 300; font-size: 1.1rem; color: #a5d6a7; margin-bottom: 2rem; }
.bin-card { border-radius: 12px; padding: 1.2rem 1.5rem; margin: 0.5rem 0; font-size: 1rem; font-weight: 600; border: 2px solid transparent; }
.bin-yellow { background: #2a2700; border-color: #fdd835; color: #fdd835; }
.bin-blue   { background: #001a2e; border-color: #29b6f6; color: #29b6f6; }
.bin-brown  { background: #1a0e00; border-color: #a1887f; color: #a1887f; }
.bin-green  { background: #001a06; border-color: #66bb6a; color: #66bb6a; }
.bin-gray   { background: #1a1a1a; border-color: #9e9e9e; color: #9e9e9e; }
.bin-red    { background: #1a0000; border-color: #ef5350; color: #ef5350; }
.result-box { background: #132b14; border: 2px solid #2e7d32; border-radius: 14px; padding: 1.5rem 2rem; margin-top: 1rem; }
.result-label { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #69f0ae; }
.result-conf { font-size: 0.9rem; color: #81c784; margin-top: 0.2rem; }
.tip-box { background: #0a1f0b; border-left: 4px solid #69f0ae; border-radius: 0 8px 8px 0; padding: 0.9rem 1.2rem; font-size: 0.92rem; color: #c8e6c9; margin-top: 0.8rem; }
.divider { border: none; border-top: 1px solid #1b5e20; margin: 2rem 0; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Bin mapping ────────────────────────────────────────────────────────────────
BIN_INFO = {
    "plastic":   {"label": "Gelbe Tonne",   "css": "bin-yellow", "emoji": "🟡", "tip": "Leere Plastikverpackungen, Dosen, Verbundmaterialien. Bitte kurz ausspülen."},
    "paper":     {"label": "Papiertonne",   "css": "bin-blue",   "emoji": "📘", "tip": "Zeitungen, Kartons, Pappe. Kein beschmutztes Papier."},
    "glass":     {"label": "Glascontainer", "css": "bin-green",  "emoji": "🟢", "tip": "Nach Farbe trennen: weiss, braun, gruen. Keine Glasscherben oder Pfandflaschen."},
    "organic":   {"label": "Biotonne",      "css": "bin-brown",  "emoji": "🟤", "tip": "Lebensmittelreste, Kaffeesatz, Gartenabfaelle. Kein Fleisch bei offener Biotonne."},
    "residual":  {"label": "Restmuelltonne","css": "bin-gray",   "emoji": "⚫", "tip": "Alles, was nicht recycelt werden kann. So wenig wie moeglich hierher."},
    "hazardous": {"label": "Sondermuell",   "css": "bin-red",    "emoji": "🔴", "tip": "Batterien, Farben, Chemikalien → zum Wertstoffhof bringen!"},
}

def map_to_bin(class_name):
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

# ── Roboflow REST API (kein SDK, kein opencv) ──────────────────────────────────
API_KEY     = "6FXxln7g0fLOOAVQkBTg"
WORKSPACE   = "juliuss-workspace-gdhwh"
WORKFLOW_ID = "detect-and-classify-4"
API_URL     = f"https://serverless.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW_ID}?api_key={API_KEY}"

def img_to_b64(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_inference(img):
    payload = {"inputs": {"image": {"type": "base64", "value": img_to_b64(img)}}}
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def parse_detections(result):
    """Try to extract a flat list of {class, confidence} dicts from various Roboflow response shapes."""
    if isinstance(result, list):
        if result and "class" in result[0]:
            return result
        if result and "predictions" in result[0]:
            p = result[0]["predictions"]
            return p if isinstance(p, list) else p.get("predictions", [])
    if isinstance(result, dict):
        for o in result.get("outputs", [result]):
            if "predictions" in o:
                p = o["predictions"]
                return p if isinstance(p, list) else p.get("predictions", [])
    return []

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">&#9851; MuellScan</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Foto hochladen - KI erkennt den richtigen Muelleimer.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded = st.file_uploader("Bild auswaehlen (JPG, PNG, WEBP)", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
camera   = st.camera_input("... oder direkt mit der Kamera fotografieren")
image_source = uploaded or camera

if image_source:
    img = Image.open(image_source)
    st.image(img, use_container_width=True, caption="Dein Bild")

    if st.button("Jetzt analysieren", use_container_width=True, type="primary"):
        with st.spinner("KI analysiert das Bild..."):
            try:
                result = run_inference(img)
            except requests.exceptions.HTTPError as e:
                st.error(f"API-Fehler {e.response.status_code}: {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Fehler: {e}")
                st.stop()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### Ergebnis")

        detections = parse_detections(result)
        if detections:
            try:
                detections = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)
            except Exception:
                pass

            top = detections[0]
            cn  = top.get("class", top.get("label", "Unbekannt"))
            cf  = top.get("confidence", 0)
            bi  = map_to_bin(cn)

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">{bi['emoji']} {bi['label']}</div>
                <div class="result-conf">Erkannt als: <b>{cn}</b> &nbsp;·&nbsp; Konfidenz: <b>{cf*100:.1f}%</b></div>
                <div class="tip-box">&#128161; {bi['tip']}</div>
            </div>""", unsafe_allow_html=True)

            if len(detections) > 1:
                st.markdown("#### Weitere erkannte Objekte")
                for det in detections[1:4]:
                    cn2 = det.get("class", det.get("label","?"))
                    cf2 = det.get("confidence", 0)
                    bi2 = map_to_bin(cn2)
                    st.markdown(
                        f'<div class="bin-card {bi2["css"]}">{bi2["emoji"]} {cn2} &rarr; {bi2["label"]} '
                        f'<span style="opacity:0.6;font-weight:300">({cf2*100:.0f}%)</span></div>',
                        unsafe_allow_html=True)
        else:
            st.warning("Kein Objekt erkannt. Versuche ein deutlicheres Foto.")

        with st.expander("Debug: Rohe API-Antwort"):
            st.json(result)

else:
    st.markdown("### Welche Tonne ist die richtige?")
    for info in BIN_INFO.values():
        st.markdown(f'<div class="bin-card {info["css"]}">{info["emoji"]} <b>{info["label"]}</b> — {info["tip"]}</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#2e7d32;font-size:0.8rem;font-family:Space Mono,monospace;">MuellScan · Powered by Roboflow &amp; Streamlit</p>', unsafe_allow_html=True)
