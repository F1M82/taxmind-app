"""
models/tax_predictor.py
─────────────────────────────────────────────────────────────────────────────
Tax Liability Prediction Model (XGBoost Regression)

Predicts estimated tax liability from financial features.
Supports both synthetic and real data pipelines.
"""

from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))
MODEL_PATH = MODEL_DIR / "tax_predictor.joblib"

NUMERIC_FEATURES = [
    "gross_income", "investment_income", "business_income",
    "total_deductions", "retirement_contributions", "dependents", "age",
]
CATEGORICAL_FEATURES = ["filing_status", "state"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "tax_liability"


# ─── Preprocessing pipeline ──────────────────────────────────────────────────

def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL_FEATURES),
    ])


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])


# ─── Training ────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, save: bool = True) -> dict[str, Any]:
    """
    Train the tax prediction model.

    Args:
        df: DataFrame with all required columns.
        save: Whether to save the trained pipeline to disk.

    Returns:
        dict with metrics and pipeline.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, preds)), 2),
        "r2": round(float(r2_score(y_test, preds)), 4),
        "test_samples": len(y_test),
    }

    if save:
        joblib.dump(pipeline, MODEL_PATH)
        print(f"✅ Tax predictor saved → {MODEL_PATH}")

    print(f"📊 Tax Predictor — MAE: ${metrics['mae']:,.2f} | R²: {metrics['r2']:.4f}")
    return {"pipeline": pipeline, "metrics": metrics}


# ─── Inference ───────────────────────────────────────────────────────────────

def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run training first."
        )
    return joblib.load(MODEL_PATH)


def predict(features: dict) -> dict:
    """
    Predict tax liability for a single taxpayer.

    Args:
        features: Dict with keys matching ALL_FEATURES.

    Returns:
        dict with predicted_tax_liability and effective_rate.
    """
    pipeline = load_model()
    df = pd.DataFrame([features])

    # Default missing optional fields
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"

    predicted = float(pipeline.predict(df[ALL_FEATURES])[0])
    gross = float(features.get("gross_income", 1))
    effective_rate = round(predicted / gross, 4) if gross > 0 else 0.0

    return {
        "predicted_tax_liability": round(max(predicted, 0), 2),
        "effective_tax_rate": effective_rate,
        "effective_tax_rate_pct": f"{effective_rate * 100:.2f}%",
    }


# ─── CLI training ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = Path("data/synthetic/tax_records.csv")
    if not DATA_PATH.exists():
        print("⚠️  Synthetic data not found. Run: python data/generate_synthetic.py")
    else:
        df = pd.read_csv(DATA_PATH)
        train(df)
