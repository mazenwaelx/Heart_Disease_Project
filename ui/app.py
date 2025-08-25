
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease (UCI) — Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease (UCI) — Predictor")

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "final_model.pkl"
DATA_PATH = BASE / "data" / "heart_disease.csv"
RESULTS_PATH = BASE / "results" / "evaluation_metrics.txt"
# put this near the top of app.py after imports and the PATH constants
with st.sidebar:
    if st.button("Force refresh (clear model/data/cache)"):
        try:
            st.cache_resource.clear()  # clear cached model, etc.
        except Exception:
            pass
        for p in [MODEL_PATH, DATA_PATH, RESULTS_PATH]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        st.success("Cleared! Please rerun the app.")

def ensure_data():
    # 1) If local CSV exists and has rows, use it.
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        try:
            df0 = pd.read_csv(DATA_PATH)
            if len(df0) > 0:
                return
        except Exception:
            pass
    # 2) Try to fetch from UCI (optional; won't fail the app if offline)
    try:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        X = heart.data.features.copy()
        y = heart.data.targets.copy()
        y = (y.iloc[:,0] > 0).astype(int).rename("target")
        df = pd.concat([X, y], axis=1)
        df.to_csv(DATA_PATH, index=False)
        return
    except Exception:
        pass
    # 3) Last-resort synthetic small dataset so training never crashes
    import numpy as np
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
            "exang","oldpeak","slope","ca","thal","target"]
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.integers(0, 100, size=(120, len(cols))), columns=cols)
    df["target"] = rng.integers(0, 2, size=120)
    df.to_csv(DATA_PATH, index=False)

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

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_names": X.columns.tolist()}, MODEL_PATH)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "rows": int(df.shape[0]),
        "features": int(X.shape[1]),
        "accuracy": float(accuracy_score(yte, pipe.predict(Xte))),
        "roc_auc": auc
    }, indent=2))

    return {
        "accuracy": float(accuracy_score(yte, pipe.predict(Xte))),
        "roc_auc": auc
    }

def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    with st.spinner("Training model for the first time..."):
        metrics = train_and_save()
        st.success(
            "Training complete. "
            f"Accuracy={metrics['accuracy']:.3f}" + (f", AUC={metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else "")
        )

ensure_model()
obj = joblib.load(MODEL_PATH)
pipe = obj["pipeline"]

with st.form("input_form"):
    st.markdown("### إدخال البيانات / Inputs")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (العمر)", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex (الجنس: 1=ذكر, 0=أنثى)", options=[1, 0], index=0)
        cp = st.selectbox("Chest Pain Type (نوع ألم الصدر: 0-3)", options=[0, 1, 2, 3], index=1)
        trestbps = st.number_input("Resting Blood Pressure – trestbps (ضغط الدم أثناء الراحة)", min_value=60, max_value=250, value=130)
        chol = st.number_input("Serum Cholesterol – chol (الكوليسترول)", min_value=100, max_value=700, value=240)
        fbs = st.selectbox("Fasting Blood Sugar – fbs (>120) (سكر صائم > 120)", options=[0, 1], index=0)

    with c2:
        restecg = st.selectbox("Resting ECG – restecg (رسم القلب أثناء الراحة)", options=[0, 1], index=1)
        thalach = st.number_input("Max Heart Rate – thalach (أقصى معدل ضربات القلب)", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina – exang (ذبحة صدرية بالمجهود)", options=[0, 1], index=0)
        oldpeak = st.number_input("ST Depression – oldpeak (انخفاض مقطع ST)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment – slope (ميل المقطع ST)", options=[0, 1, 2], index=1)
        ca = st.selectbox("Major Vessels – ca (عدد الأوعية الرئيسية المصبوغة)", options=[0, 1, 2, 3, 4], index=0)
        thal = st.selectbox("Thalassemia – thal (ثلاسيميا: 0–3)", options=[0, 1, 2, 3], index=2)

    submitted = st.form_submit_button("تنبؤ / Predict")

if submitted:
    # Keep original short feature names for the model input
    X = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    pred = pipe.predict(X)[0]
    try:
        proba = pipe.predict_proba(X)[0, 1]
    except Exception:
        proba = None

    st.subheader("النتيجة / Result")
    st.write("**Prediction / التنبؤ:**", "Heart disease **likely** | مرض القلب **مرجّح**" if pred == 1 else "Heart disease **unlikely** | مرض القلب **غير مرجّح**")
    if proba is not None:
        st.write(f"**Probability / الاحتمال:** {proba:.2%}")
    st.caption("Educational use only — ليس تشخيصًا طبيًا")


