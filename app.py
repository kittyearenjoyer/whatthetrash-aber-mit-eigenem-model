import streamlit as st
from PIL import Image
import io

st.set_page_config(
    page_title="MuellScan – KI-Muelltrennung",
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

# ── Label → Tonne mapping ──────────────────────────────────────────────────────
# Will be populated after model loads (uses model's actual label names)
BIN_INFO = {
    "plastic":     {"label":"Gelbe Tonne",    "css":"bin-yellow","emoji":"🟡","tip":"Leere Plastikverpackungen, Dosen, Verbundmaterialien. Bitte kurz ausspuelen."},
    "paper":       {"label":"Papiertonne",    "css":"bin-blue",  "emoji":"📘","tip":"Zeitungen, Kartons, Pappe. Kein beschmutztes Papier."},
    "glass":       {"label":"Glascontainer",  "css":"bin-green", "emoji":"🟢","tip":"Nach Farbe trennen: weiss, braun, gruen. Keine Glasscherben."},
    "organic":     {"label":"Biotonne",       "css":"bin-brown", "emoji":"🟤","tip":"Lebensmittelreste, Kaffeesatz, Gartenabfaelle."},
    "metal":       {"label":"Gelbe Tonne",    "css":"bin-yellow","emoji":"🟡","tip":"Dosen, Alufolie, Metallverpackungen in die Gelbe Tonne."},
    "cardboard":   {"label":"Papiertonne",    "css":"bin-blue",  "emoji":"📘","tip":"Kartons zusammenfalten, dann in die Papiertonne."},
    "trash":       {"label":"Restmuelltonne", "css":"bin-gray",  "emoji":"⚫","tip":"Alles, was nicht recycelt werden kann."},
    "battery":     {"label":"Sondermuell",    "css":"bin-red",   "emoji":"🔴","tip":"Batterien zum Wertstoffhof oder Rueckgabeboxen im Supermarkt."},
    "default":     {"label":"Restmuelltonne", "css":"bin-gray",  "emoji":"⚫","tip":"Wenn unsicher: Restmuelltonne. So wenig wie moeglich hierher."},
}

def map_label_to_bin(label: str) -> dict:
    l = label.lower().strip()
    for key in BIN_INFO:
        if key in l:
            return BIN_INFO[key]
    # fuzzy matches
    if any(k in l for k in ["kunststoff","verpackung","dose","folie","pet"]):
        return BIN_INFO["plastic"]
    if any(k in l for k in ["papier","zeitung","karton","pappe"]):
        return BIN_INFO["paper"]
    if any(k in l for k in ["glas","flasche"]):
        return BIN_INFO["glass"]
    if any(k in l for k in ["bio","kompost","essen","obst","gemuese","rest"]):
        return BIN_INFO["organic"]
    if any(k in l for k in ["metall","alu","eisen","stahl","blech"]):
        return BIN_INFO["metal"]
    if any(k in l for k in ["batterie","akku","elektronik","sonder"]):
        return BIN_INFO["battery"]
    return BIN_INFO["default"]

# ── Model loading (cached so it only runs once) ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import ViTForImageClassification, ViTImageProcessor
    import torch
    model_id = "KittyEarEnjoyer/schule_muellklassifizierung"
    processor = ViTImageProcessor.from_pretrained(model_id)
    model     = ViTForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model

# ── Inference ──────────────────────────────────────────────────────────────────
def classify(img: Image.Image):
    import torch
    processor, model = load_model()
    inputs  = processor(images=img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1)[0]
    top5_idx = probs.argsort(descending=True)[:5].tolist()
    results = []
    for idx in top5_idx:
        label = model.config.id2label[idx]
        conf  = probs[idx].item()
        results.append({"label": label, "score": conf})
    return results

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">&#9851;&#65039; MuellScan</div>', unsafe_allow_html=True)
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
        bi  = map_label_to_bin(top["label"])
        pct = top["score"] * 100

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">{bi['emoji']} {bi['label']}</div>
            <div class="result-conf">Modell-Klasse: <b>{top['label']}</b> &nbsp;·&nbsp; Konfidenz: <b>{pct:.1f}%</b></div>
            <div class="bar-wrap"><div class="bar-fill" style="width:{min(pct,100):.1f}%"></div></div>
            <div class="tip-box">&#128161; {bi['tip']}</div>
        </div>""", unsafe_allow_html=True)

        if len(predictions) > 1:
            st.markdown("#### Weitere Klassen (Top 5)")
            for pred in predictions[1:]:
                bi2  = map_label_to_bin(pred["label"])
                pct2 = pred["score"] * 100
                st.markdown(
                    f'<div class="bin-card {bi2["css"]}">'
                    f'{bi2["emoji"]} {pred["label"]} &rarr; {bi2["label"]} '
                    f'<span style="opacity:0.6;font-weight:300">({pct2:.1f}%)</span>'
                    f'</div>', unsafe_allow_html=True)

else:
    # Show bin guide when no image
    st.markdown("### Welche Tonne ist die richtige?")
    shown = {}
    for info in BIN_INFO.values():
        if info["label"] not in shown:
            st.markdown(f'<div class="bin-card {info["css"]}">{info["emoji"]} <b>{info["label"]}</b> — {info["tip"]}</div>', unsafe_allow_html=True)
            shown[info["label"]] = True

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#2e7d32;font-size:0.8rem;font-family:Space Mono,monospace;">MuellScan &middot; KittyEarEnjoyer/schule_muellklassifizierung &middot; Streamlit</p>', unsafe_allow_html=True)
