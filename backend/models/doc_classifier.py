"""
models/doc_classifier.py - Fixed version
Tax Document Classification Model (TF-IDF + Logistic Regression)
"""

from __future__ import annotations
import os
import joblib
import pandas as pd
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))
MODEL_PATH = MODEL_DIR / "doc_classifier.joblib"

CLASSES = ["W2", "1099", "invoice", "receipt", "tax_return"]


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def train(df: pd.DataFrame, save: bool = True) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X = df["text"].values
    y = df["document_type"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = round(float(accuracy_score(y_test, preds)), 4)
    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "accuracy": acc,
        "per_class_f1": {
            cls: round(report[cls]["f1-score"], 4)
            for cls in CLASSES if cls in report
        },
        "test_samples": len(y_test),
    }

    if save:
        joblib.dump(pipeline, MODEL_PATH)
        print(f"✅ Document classifier saved → {MODEL_PATH}")

    print(f"📊 Doc Classifier — Accuracy: {acc:.4f}")
    return {"pipeline": pipeline, "metrics": metrics}


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)


def classify(text: str) -> dict:
    pipeline = load_model()
    probas = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    top_idx = int(probas.argmax())

    return {
        "predicted_class": classes[top_idx],
        "confidence": round(float(probas[top_idx]), 4),
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(classes, probas)
        },
    }


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
    except ImportError:
        raise ImportError("Run: pip install pdfplumber")


if __name__ == "__main__":
    DATA_PATH = Path("data/synthetic/documents.csv")
    if not DATA_PATH.exists():
        print("Run: python data/generate_synthetic.py")
    else:
        df = pd.read_csv(DATA_PATH)
        train(df)