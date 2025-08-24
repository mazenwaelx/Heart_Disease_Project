import json
from pathlib import Path
import subprocess, sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "final_model.pkl"
METRICS_PATH = BASE / "results" / "evaluation_metrics.txt"

st.set_page_config(page_title="Heart Disease (UCI) Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease (UCI) — Predictor")
st.caption("Dataset loaded via `ucimlrepo` (UCI id=45).")

def ensure_model():
    if not MODEL_PATH.exists():
        with st.spinner("Training model — this will also download the dataset on first run..."):
            res = subprocess.run([sys.executable, str(BASE / "train.py")], capture_output=True, text=True)
            st.code(res.stdout or "", language="bash")
            if res.returncode != 0:
                st.error("Training failed.")
                st.code(res.stderr or "", language="bash")
                return False
    return True

def load_model():
    obj = joblib.load(MODEL_PATH)
    return obj["pipeline"], obj.get("feature_names")

with st.expander("Train / Re-train model"):
    if st.button("Run training now"):
        ensure_model()
        st.success("Training finished (or already trained).")

# Input form (map to common UCI features; unknowns will be imputed)
with st.form("form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("age", 1, 120, 55)
        sex = st.selectbox("sex (1=male, 0=female)", [1, 0], index=0)
        cp = st.selectbox("cp (chest pain type 0-3)", [0,1,2,3], index=1)
        trestbps = st.number_input("trestbps", 60, 250, 130)
        chol = st.number_input("chol", 100, 700, 240)
        fbs = st.selectbox("fbs (>120mg/dl)", [0,1], index=0)
    with c2:
        restecg = st.selectbox("restecg", [0,1], index=1)
        thalach = st.number_input("thalach", 60, 250, 150)
        exang = st.selectbox("exang", [0,1], index=0)
        oldpeak = st.number_input("oldpeak", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("slope", [0,1,2], index=1)
        ca = st.selectbox("ca (# major vessels)", [0,1,2,3,4], index=0)
        thal = st.selectbox("thal", [0,1,2,3], index=2)
    submitted = st.form_submit_button("Predict")

if submitted:
    if ensure_model():
        pipe, _ = load_model()
        row = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        X = pd.DataFrame([row])
        pred = pipe.predict(X)[0]
        proba = None
        try:
            proba = pipe.predict_proba(X)[0,1]
        except Exception:
            pass
        st.subheader("Result")
        st.write("**Prediction:**", "Heart disease **likely**" if pred==1 else "Heart disease **unlikely**")
        if proba is not None:
            st.write(f"**Probability:** {proba:.2%}")
        st.caption("Educational use only — not medical advice.")

st.divider()
st.subheader("Latest training metrics")
if METRICS_PATH.exists():
    try:
        st.json(json.loads(METRICS_PATH.read_text()))
    except Exception:
        st.code(METRICS_PATH.read_text()[:2000])
else:
    st.info("No metrics yet. Click 'Run training now' above.")