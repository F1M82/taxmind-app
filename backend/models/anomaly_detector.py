"""
models/anomaly_detector.py
─────────────────────────────────────────────────────────────────────────────
Tax Anomaly / Fraud Detection Model

Uses an ensemble of:
  - Isolation Forest (unsupervised, no labels needed)
  - XGBoost Classifier (supervised, requires is_anomaly labels)

Both models are trained and served; the API exposes their combined score.
"""

from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))
IF_PATH = MODEL_DIR / "anomaly_iso_forest.joblib"
XGB_PATH = MODEL_DIR / "anomaly_xgb.joblib"
PREP_PATH = MODEL_DIR / "anomaly_preprocessor.joblib"

NUMERIC_FEATURES = [
    "gross_income", "investment_income", "business_income",
    "total_deductions", "retirement_contributions", "dependents",
    "taxable_income", "effective_tax_rate",
]
CATEGORICAL_FEATURES = ["filing_status", "state"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ─── Shared preprocessor ─────────────────────────────────────────────────────

def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL_FEATURES),
    ])


# ─── Training ────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, save: bool = True) -> dict[str, Any]:
    """
    Train both Isolation Forest and XGBoost anomaly detectors.

    Args:
        df: DataFrame including ALL_FEATURES + 'is_anomaly' column.
        save: Whether to persist models.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X = df[ALL_FEATURES].copy()
    y = df["is_anomaly"].values

    # Build and fit preprocessor
    preprocessor = _build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # ── Isolation Forest (unsupervised) ──────────────────────────────────────
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_transformed)
    iso_scores = iso.decision_function(X_transformed)  # lower = more anomalous

    # ── XGBoost Classifier (supervised) ──────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = float((y_tr == 0).sum()) / float((y_tr == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb.fit(X_tr, y_tr)

    xgb_preds = xgb.predict(X_te)
    xgb_probas = xgb.predict_proba(X_te)[:, 1]

    metrics = {
        "xgb_auc": round(float(roc_auc_score(y_te, xgb_probas)), 4),
        "xgb_report": classification_report(y_te, xgb_preds, output_dict=True),
        "iso_anomaly_rate_detected": round(
            float((iso.predict(X_transformed) == -1).mean()), 4
        ),
    }

    if save:
        joblib.dump(preprocessor, PREP_PATH)
        joblib.dump(iso, IF_PATH)
        joblib.dump(xgb, XGB_PATH)
        print(f"✅ Anomaly models saved → {MODEL_DIR}")

    print(f"📊 Anomaly XGB — AUC: {metrics['xgb_auc']:.4f}")
    print(f"📊 Isolation Forest anomaly rate: {metrics['iso_anomaly_rate_detected']:.2%}")
    return {"metrics": metrics}


# ─── Inference ───────────────────────────────────────────────────────────────

def _load_models():
    for p in [PREP_PATH, IF_PATH, XGB_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Model not found at {p}. Run training first.")
    return (
        joblib.load(PREP_PATH),
        joblib.load(IF_PATH),
        joblib.load(XGB_PATH),
    )


def detect(features: dict) -> dict:
    """
    Detect if a tax record is anomalous.

    Returns:
        dict with is_anomaly flag, risk_score, and individual model signals.
    """
    preprocessor, iso, xgb = _load_models()

    df = pd.DataFrame([features])
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"

    X = preprocessor.transform(df[ALL_FEATURES])

    # Isolation Forest: decision_function score (negative = anomalous)
    iso_score = float(iso.decision_function(X)[0])
    iso_flag = iso.predict(X)[0] == -1  # True if anomaly

    # XGBoost fraud probability
    xgb_proba = float(xgb.predict_proba(X)[0][1])
    xgb_flag = xgb_proba >= 0.5

    # Combined risk score: blend both signals
    # Normalize iso_score: typical range [-0.5, 0.5]; invert so higher = riskier
    iso_normalized = max(0.0, min(1.0, (-iso_score + 0.5) / 1.0))
    combined_risk = round(0.4 * iso_normalized + 0.6 * xgb_proba, 4)

    risk_level = (
        "HIGH" if combined_risk >= 0.7 else
        "MEDIUM" if combined_risk >= 0.4 else
        "LOW"
    )

    return {
        "is_anomaly": bool(iso_flag or xgb_flag),
        "risk_score": combined_risk,
        "risk_level": risk_level,
        "signals": {
            "isolation_forest": {
                "flagged": bool(iso_flag),
                "decision_score": round(iso_score, 4),
            },
            "xgboost_classifier": {
                "flagged": bool(xgb_flag),
                "fraud_probability": round(xgb_proba, 4),
            },
        },
        "explanation": _explain(features, combined_risk),
    }


def _explain(features: dict, risk_score: float) -> str:
    """Generate a simple rule-based explanation for the risk score."""
    reasons = []
    income = float(features.get("gross_income", 0))
    deductions = float(features.get("total_deductions", 0))
    if income > 0 and deductions / income > 0.6:
        reasons.append("unusually high deduction-to-income ratio")
    if float(features.get("tax_liability", 1)) == 0 and income > 50000:
        reasons.append("zero tax liability on significant income")
    eff_rate = float(features.get("effective_tax_rate", 0))
    if eff_rate < 0.01 and income > 100000:
        reasons.append("very low effective tax rate for income level")
    if not reasons:
        return "No specific red flags identified."
    return "Potential red flags: " + "; ".join(reasons) + "."


# ─── CLI training ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = Path("data/synthetic/tax_records.csv")
    if not DATA_PATH.exists():
        print("⚠️  Synthetic data not found. Run: python data/generate_synthetic.py")
    else:
        df = pd.read_csv(DATA_PATH)
        train(df)
