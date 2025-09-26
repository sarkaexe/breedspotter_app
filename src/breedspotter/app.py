# src/breedspotter/app.py
import os
import sys
from pathlib import Path
import streamlit as st

# --- make sure "src" is on sys.path when run as a script (Streamlit Cloud/local) ---
SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from breedspotter.infer import Predictor  # absolute import

st.set_page_config(page_title="BreedSpotter", page_icon="🐶", layout="centered")
st.title("🐶 BreedSpotter")
st.write("Wgraj zdjęcie psa, a model spróbuje rozpoznać rasę.")

ckpt_default = os.path.join("checkpoints", "best.pt")
ckpt = st.text_input("Ścieżka do checkpointu (.pt)", ckpt_default)

uploaded = st.file_uploader("Zdjęcie (JPG/PNG)", type=["jpg","jpeg","png"])
k = st.slider("Top-k", 1, 10, 5)

if st.button("Klasyfikuj") and uploaded and ckpt:
    with st.spinner("Inferencja..."):
        img_path = "/tmp/_upload.png"
        with open(img_path, "wb") as f:
            f.write(uploaded.read())
        predictor = Predictor(ckpt)
        preds = predictor.predict(img_path, k=k)
    st.image(img_path, caption="Wejściowy obraz")
    st.subheader("Wyniki")
    for label, p in preds:
        st.write(f"- **{label}**: {p:.3f}")
