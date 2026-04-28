import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MuellScan",
    page_icon="♻️",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1f0f; color: #e8f5e9; }
h1,h2,h3 { font-family: 'Space Mono', monospace; }
.hero-title { font-family: 'Space Mono', monospace; font-size: 2.6rem; font-weight: 700; color: #69f0ae; line-height: 1.1; margin-bottom: 0.2rem; }
.hero-sub   { font-weight: 300; font-size: 1.1rem; color: #a5d6a7; margin-bottom: 2rem; }
.bin-card   { border-radius: 12px; padding: 1.1rem 1.4rem; margin: 0.4rem 0; font-size: 1rem; font-weight: 600; border: 2px solid transparent; }
.bin-yellow { background:#2a2700; border-color:#fdd835; color:#fdd835; }
.bin-blue   { background:#001a2e; border-color:#29b6f6; color:#29b6f6; }
.bin-brown  { background:#1a0e00; border-color:#a1887f; color:#a1887f; }
.bin-green  { background:#001a06; border-color:#66bb6a; color:#66bb6a; }
.bin-gray   { background:#1a1a1a; border-color:#9e9e9e; color:#9e9e9e; }
.bin-red    { background:#1a0000; border-color:#ef5350; color:#ef5350; }
.result-box   { background:#132b14; border:2px solid #2e7d32; border-radius:14px; padding:1.5rem 2rem; margin-top:1rem; }
.result-label { font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700; color:#69f0ae; }
.result-conf  { font-size:0.9rem; color:#81c784; margin-top:0.2rem; }
.tip-box { background:#0a1f0b; border-left:4px solid #69f0ae; border-radius:0 8px 8px 0; padding:0.9rem 1.2rem; font-size:0.92rem; color:#c8e6c9; margin-top:0.8rem; }
.bar-wrap { background:#0a1f0b; border-radius:8px; height:10px; margin-top:6px; }
.bar-fill { background:#69f0ae; border-radius:8px; height:10px; }
.divider  { border:none; border-top:1px solid #1b5e20; margin:2rem 0; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Klassen & Tonnen-Mapping ───────────────────────────────────────────────────
# MobileNetV3 mit 10 Klassen – standard Trash-Klassifizierungs-Datensatz
# Indizes 0-9 werden den wahrscheinlichsten Klassen zugeordnet.
# Falls dein Modell andere Klassen hat, werden sie im Debug-Bereich angezeigt.
CLASS_NAMES = [
    "cardboard",   # 0
    "glass",       # 1
    "metal",       # 2
    "paper",       # 3
    "plastic",     # 4
    "trash",       # 5
    "battery",     # 6
    "biological",  # 7
    "clothes",     # 8
    "shoes",       # 9
]

BIN_INFO = {
    "cardboard":  {"label": "Papiertonne",    "css": "bin-blue",   "emoji": "📦", "tip": "Kartons zusammenfalten und in die Papiertonne."},
    "glass":      {"label": "Glascontainer",  "css": "bin-green",  "emoji": "🍶", "tip": "Nach Farbe trennen: weiss, braun, gruen. Keine Pfandflaschen."},
    "metal":      {"label": "Gelbe Tonne",    "css": "bin-yellow", "emoji": "🥫", "tip": "Dosen, Alufolie und Metallverpackungen in die Gelbe Tonne."},
    "paper":      {"label": "Papiertonne",    "css": "bin-blue",   "emoji": "📰", "tip": "Zeitungen, Hefte, Pappe – kein beschmutztes Papier."},
    "plastic":    {"label": "Gelbe Tonne",    "css": "bin-yellow", "emoji": "🧴", "tip": "Leere Plastikverpackungen in die Gelbe Tonne. Kurz ausspuelen."},
    "trash":      {"label": "Restmuelltonne", "css": "bin-gray",   "emoji": "🗑️", "tip": "Alles was nicht recycelt werden kann. So wenig wie moeglich."},
    "battery":    {"label": "Sondermuell",    "css": "bin-red",    "emoji": "🔋", "tip": "Batterien zum Wertstoffhof oder in die Sammelbox im Supermarkt."},
    "biological": {"label": "Biotonne",       "css": "bin-brown",  "emoji": "🍌", "tip": "Lebensmittelreste, Kaffeesatz, Gartenabfaelle in die Biotonne."},
    "clothes":    {"label": "Altkleidercontainer", "css": "bin-green", "emoji": "👕", "tip": "Saubere Kleidung in den Altkleidercontainer oder Secondhand."},
    "shoes":      {"label": "Altkleidercontainer", "css": "bin-green", "emoji": "👟", "tip": "Schuhe paarweise zusammenbinden, in den Altkleidercontainer."},
    "unknown":    {"label": "Restmuelltonne", "css": "bin-gray",   "emoji": "❓", "tip": "Wenn unsicher: Restmuelltonne. Im Zweifel beim Wertstoffhof nachfragen."},
}

def get_bin(class_name: str) -> dict:
    return BIN_INFO.get(class_name.lower(), BIN_INFO["unknown"])

# ── Modell laden (MobileNetV3 Large, 10 Klassen) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    # Gewichte von HuggingFace runterladen
    weights_path = hf_hub_download(
        repo_id="KittyEarEnjoyer/schule_",
        filename="model.pth"
    )
    # Architektur aufbauen
    model = models.mobilenet_v3_large(weights=None)
    # Letzten Classifier auf 10 Klassen anpassen
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 10)
    # Gewichte laden
    state = torch.load(weights_path, map_location="cpu")
    # state_dict kann direkt oder unter einem Key gespeichert sein
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model

# ── Preprocessing ──────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def classify(img: Image.Image):
    model = load_model()
    tensor = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=-1)[0]
    top5  = probs.argsort(descending=True)[:5].tolist()
    return [{"class": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"klasse_{i}",
              "index": i,
              "score": probs[i].item()} for i in top5]

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">&#9851;&#65039; WhatTheTrash</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Foto hochladen – KI erkennt den richtigen Muelleimer.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded = st.file_uploader("Bild auswaehlen (JPG, PNG, WEBP)", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
camera   = st.camera_input("... oder direkt mit der Kamera fotografieren")
image_source = uploaded or camera

if image_source:
    img = Image.open(image_source)
    st.image(img, use_container_width=True, caption="Dein Bild")

    if st.button("Jetzt analysieren", use_container_width=True, type="primary"):
        with st.spinner("Modell wird geladen und analysiert..."):
            try:
                predictions = classify(img)
            except Exception as e:
                st.error(f"Fehler bei der Analyse: {e}")
                st.stop()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### Ergebnis")

        top = predictions[0]
        bi  = get_bin(top["class"])
        pct = top["score"] * 100

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">{bi['emoji']} {bi['label']}</div>
            <div class="result-conf">Erkannt als: <b>{top['class']}</b> &nbsp;·&nbsp; Konfidenz: <b>{pct:.1f}%</b></div>
            <div class="bar-wrap"><div class="bar-fill" style="width:{min(pct,100):.1f}%"></div></div>
            <div class="tip-box">&#128161; {bi['tip']}</div>
        </div>""", unsafe_allow_html=True)

        if len(predictions) > 1:
            st.markdown("#### Weitere Klassen (Top 5)")
            for pred in predictions[1:]:
                bi2  = get_bin(pred["class"])
                pct2 = pred["score"] * 100
                st.markdown(
                    f'<div class="bin-card {bi2["css"]}">'
                    f'{bi2["emoji"]} {pred["class"]} &rarr; {bi2["label"]} '
                    f'<span style="opacity:0.6;font-weight:300">({pct2:.1f}%)</span>'
                    f'</div>', unsafe_allow_html=True)

        with st.expander("🔧 Debug: Rohe Modell-Ausgabe"):
            st.json(predictions)

else:
    st.markdown("### Welche Tonne ist die richtige?")
    shown = set()
    for info in BIN_INFO.values():
        if info["label"] not in shown:
            st.markdown(f'<div class="bin-card {info["css"]}">{info["emoji"]} <b>{info["label"]}</b> — {info["tip"]}</div>', unsafe_allow_html=True)
            shown.add(info["label"])

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#2e7d32;font-size:0.8rem;font-family:Space Mono,monospace;">MuellScan &middot; KittyEarEnjoyer/schule_ &middot; MobileNetV3</p>', unsafe_allow_html=True)
