import os
import streamlit as st
import numpy as np
from PIL import Image
import tf_keras as keras

# ── Konfiguration ──────────────────────────────────────────────────────────────
MODEL_PATH = "https://drive.google.com/file/d/1TG0SjD8Df3U4_RgJ8O5gUqBPq-2Cz2NL/view?usp=drive_link"
IMG_SIZE   = (224, 224)          # anpassen falls dein Modell eine andere Größe erwartet

# Klassen in der Reihenfolge, die dein Modell gelernt hat.
# Passe diese Liste an deine echten Trainings-Labels an!
KLASSEN = [
    "Glas",
    "Metall",
    "Papier",
    "Pappe",
    "Plastik",
    "Restmüll",
]

# Welche Tonne gehört zu welcher Klasse?
TONNE = {
    "Glas":     ("🟫 Altglascontainer", "#8B5E3C"),
    "Metall":   ("🟡 Gelbe Tonne / Gelber Sack", "#F5C518"),
    "Papier":   ("🔵 Blaue Tonne",  "#1E90FF"),
    "Pappe":    ("🔵 Blaue Tonne",  "#1E90FF"),
    "Plastik":  ("🟡 Gelbe Tonne / Gelber Sack", "#F5C518"),
    "Restmüll": ("⚫ Restmülltonne", "#555555"),
}

# ── Modell laden (gecacht) ─────────────────────────────────────────────────────
@st.cache_resource
def lade_modell():
    return keras.models.load_model(MODEL_PATH)

# ── Hilfsfunktionen ────────────────────────────────────────────────────────────
def bild_vorbereiten(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1, H, W, 3)

def klassifizieren(model, img: Image.Image):
    x     = bild_vorbereiten(img)
    preds = model.predict(x, verbose=0)[0]      # shape: (num_classes,)
    idx   = int(np.argmax(preds))
    return KLASSEN[idx], float(preds[idx]), preds

# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mülltrennungs-KI",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ Mülltrennungs-KI")
st.write("Lade ein Foto deines Abfalls hoch und die KI sagt dir, in welche Tonne er gehört.")

modell = lade_modell()

hochgeladen = st.file_uploader(
    "Bild auswählen (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if hochgeladen:
    bild = Image.open(hochgeladen)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(bild, caption="Hochgeladenes Bild", use_container_width=True)

    with col2:
        with st.spinner("Analysiere …"):
            klasse, konfidenz, alle_preds = klassifizieren(modell, bild)

        tonne_text, tonne_farbe = TONNE.get(klasse, ("❓ Unbekannt", "#999999"))

        st.markdown("### Ergebnis")
        st.markdown(
            f"""
            <div style="
                background:{tonne_farbe}22;
                border-left: 6px solid {tonne_farbe};
                padding: 12px 16px;
                border-radius: 6px;
                margin-bottom: 12px;
            ">
                <h3 style="margin:0; color:{tonne_farbe}">{klasse}</h3>
                <p style="margin:0">{tonne_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Konfidenz", f"{konfidenz*100:.1f} %")

    st.markdown("---")
    st.markdown("**Alle Wahrscheinlichkeiten**")
    for k, p in sorted(zip(KLASSEN, alle_preds), key=lambda x: -x[1]):
        st.progress(float(p), text=f"{k}: {p*100:.1f} %")

st.markdown(
    """
    ---
    <small>Modell: lokales Keras-Modell · Datenschutz: Bilder werden nicht gespeichert</small>
    """,
    unsafe_allow_html=True,
)
