#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from src.pipeline.predictor import TestFailurePredictor
from src.utils.helpers import load_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--data", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    predictor = TestFailurePredictor.load(args.artifacts, args.config)
    data_path = args.data or cfg["paths"]["raw_data"]
    df = pd.read_csv(data_path)
    if len(df) > 600:
        df = df.sample(600, random_state=42)
    results = predictor.predict_batch(df["full_text"].tolist())
    preds = [r["prediction"] for r in results]
    print("Accuracy:", accuracy_score(df["outcome"], preds))
    print(classification_report(df["outcome"], preds, digits=3))

if __name__ == "__main__":
    main()
