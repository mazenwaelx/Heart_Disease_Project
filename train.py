import json
from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

BASE = Path(__file__).resolve().parents[0]
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_from_ucimlrepo():
    # UCI Heart Disease (id=45)
    heart = fetch_ucirepo(id=45)
    X = heart.data.features.copy()
    y = heart.data.targets.copy()

    # The 'num' column is 0..4; convert to binary target (0 no disease, >0 disease)
    if "num" in y.columns:
        y_bin = (y["num"] > 0).astype(int)
    else:
        # Some mirrors name it differently; try first column
        first = y.columns[0]
        y_bin = (y[first] > 0).astype(int)

    # Harmonize column names (lowercase, no spaces)
    X.columns = [str(c).strip().lower() for c in X.columns]
    return X, y_bin.rename("target")

def build_pipeline(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols),
        ]
    )
    # Simple strong baseline
    clf = LogisticRegression(max_iter=500, n_jobs=None) if hasattr(LogisticRegression, "n_jobs") else LogisticRegression(max_iter=500)
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    return pipe, numeric_cols, categorical_cols

def main():
    X, y = load_from_ucimlrepo()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe, numeric_cols, categorical_cols = build_pipeline(X)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Probabilities for AUC if available
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    # CV AUC
    cv_auc = None
    try:
        cv_auc = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc").mean()
    except Exception:
        pass

    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model
    model_path = MODELS_DIR / "final_model.pkl"
    joblib.dump({"pipeline": pipe, "feature_names": X.columns.tolist()}, model_path)

    # Save metrics
    metrics = {
        "accuracy": acc,
        "roc_auc": auc,
        "cv_roc_auc": cv_auc,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "classification_report": report,
    }
    (RESULTS_DIR / "evaluation_metrics.txt").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"accuracy": acc, "roc_auc": auc, "cv_roc_auc": cv_auc}, indent=2))
    print(f"Saved model to: {model_path}")

if __name__ == "__main__":
    main()