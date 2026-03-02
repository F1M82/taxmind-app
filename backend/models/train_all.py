"""
models/train_all.py - Fixed version
"""

import sys
import os
from pathlib import Path

# Fix import paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import time
import json
import pandas as pd


def train_all(real_data_path=None, skip=None):
    skip = skip or []
    start = time.time()
    print("=" * 60)
    print("🚀 TaxMind — Model Training Pipeline")
    print("=" * 60)

    # ── Step 1: Ensure data exists ─────────────────────────────────────────
    synth_dir = ROOT / "data" / "synthetic"
    synth_records = synth_dir / "tax_records.csv"
    synth_docs = synth_dir / "documents.csv"
    synth_qa = synth_dir / "qa_knowledge_base.json"

    if not all([synth_records.exists(), synth_docs.exists(), synth_qa.exists()]):
        print("\n📊 Generating synthetic data...")
        os.system(f"python {ROOT / 'data' / 'generate_synthetic.py'}")

    # ── Step 2: Load data ─────────────────────────────────────────────────
    if real_data_path:
        print(f"\n📂 Loading real data from: {real_data_path}")
        tax_df = pd.read_csv(real_data_path)
    else:
        print("\n📂 Loading synthetic tax records...")
        tax_df = pd.read_csv(synth_records)

    doc_df = pd.read_csv(synth_docs)
    qa_pairs = json.loads(synth_qa.read_text())

    results = {}

    # ── Step 3: Train Tax Predictor ───────────────────────────────────────
    if "predictor" not in skip:
        print("\n" + "─" * 40)
        print("🔢 Training Tax Liability Predictor...")
        from models.tax_predictor import train as train_predictor
        r = train_predictor(tax_df)
        results["tax_predictor"] = r["metrics"]

    # ── Step 4: Train Document Classifier ─────────────────────────────────
    if "classifier" not in skip:
        print("\n" + "─" * 40)
        print("📄 Training Document Classifier...")
        from models.doc_classifier import train as train_classifier
        r = train_classifier(doc_df)
        results["doc_classifier"] = r["metrics"]

    # ── Step 5: Train Anomaly Detector ────────────────────────────────────
    if "anomaly" not in skip:
        print("\n" + "─" * 40)
        print("🚨 Training Anomaly Detector...")
        from models.anomaly_detector import train as train_anomaly
        r = train_anomaly(tax_df)
        results["anomaly_detector"] = r["metrics"]

    # ── Step 6: Build Q&A Vector Store ────────────────────────────────────
    if "qa" not in skip:
        print("\n" + "─" * 40)
        print("💬 Building Q&A Vector Store...")
        from models.tax_qa import build_vector_store
        build_vector_store(qa_pairs)
        results["tax_qa"] = {"knowledge_base_size": len(qa_pairs)}

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = round(time.time() - start, 1)
    print("\n" + "=" * 60)
    print(f"✅ Training complete in {elapsed}s")
    print("─" * 60)
    for model, metrics in results.items():
        print(f"  {model}: {metrics}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all TaxMind ML models")
    parser.add_argument("--real-data", type=str, default=None)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["predictor", "classifier", "anomaly", "qa"])
    args = parser.parse_args()
    train_all(real_data_path=args.real_data, skip=args.skip)
