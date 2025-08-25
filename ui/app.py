
import json
from pathlib import Path
import subprocess, sys

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease (UCI) — Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease (UCI) — Predictor")

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "final_model.pkl"
DATA_PATH = BASE / "data" / "heart_disease.csv"

def ensure_data():
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        return
    try:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        X = heart.data.features.copy()
        y = heart.data.targets.copy()
        y = (y.iloc[:,0] > 0).astype(int).rename("target")
        df = pd.concat([X, y], axis=1)
        df.to_csv(DATA_PATH, index=False)
    except Exception as e:
        st.error(f"Could not fetch dataset automatically: {e}")

def train_and_save():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    ensure_data()
    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns and "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df.drop(columns=["num"], inplace=True)

    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    num = X.select_dtypes(include=[float,int]).columns.tolist()
    cat = [c for c in X.columns if c not in num]

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])

    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=600))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    try:
        auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:,1])
    except Exception:
        auc = None

    joblib.dump({"pipeline": pipe, "feature_names": X.columns.tolist()}, MODEL_PATH)
    return {"accuracy": float(accuracy_score(yte, pipe.predict(Xte))), "roc_auc": auc}

def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    with st.spinner("Training model for the first time..."):
        metrics = train_and_save()
        st.success(f"Training complete. Accuracy={metrics['accuracy']:.3f}" + (f", AUC={metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else ""))

ensure_model()
obj = joblib.load(MODEL_PATH)
pipe = obj["pipeline"]

with st.form("input_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("age", 1, 120, 55)
        sex = st.selectbox("sex (1=male,0=female)", [1,0], index=0)
        cp = st.selectbox("cp (0-3)", [0,1,2,3], index=1)
        trestbps = st.number_input("trestbps", 60, 250, 130)
        chol = st.number_input("chol", 100, 700, 240)
        fbs = st.selectbox("fbs (>120)", [0,1], index=0)
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
    row = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    X = pd.DataFrame([row])
    pred = pipe.predict(X)[0]
    try:
        proba = pipe.predict_proba(X)[0,1]
    except Exception:
        proba = None
    st.subheader("Result")
    st.write("**Prediction:**", "Heart disease **likely**" if pred==1 else "Heart disease **unlikely**")
    if proba is not None:
        st.write(f"**Probability:** {proba:.2%}")
    st.caption("Educational use only — not medical advice.")
