#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

from src.data.generator import generate_test_dataset
from src.data.preprocessing import TestPreprocessor
from src.models.embeddings import EmbeddingModel
from src.models.classifier import FailureClassifier
from src.models.similarity import SimilarityIndex
from src.utils.helpers import load_config, ensure_dirs, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["data"]["random_seed"])
    ensure_dirs(cfg["paths"]["data_dir"], cfg["paths"]["artifacts_dir"])

    data_path = Path(args.data or cfg["paths"]["raw_data"])
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_test_dataset(cfg["data"]["n_samples"], cfg["data"]["fail_ratio"], cfg["data"]["random_seed"])
        df.to_csv(data_path, index=False)

    pre = TestPreprocessor(max_text_length=cfg["preprocessing"]["max_text_length"])
    df = pre.transform_df(df)
    train_df, test_df = train_test_split(df, test_size=cfg["data"]["test_size"],
                                         random_state=cfg["data"]["random_seed"], stratify=df["outcome"])
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    emb_cfg = cfg["embedding"]
    embedder = EmbeddingModel(model_name=emb_cfg["model_name"], device=emb_cfg.get("device"),
                              normalize=emb_cfg.get("normalize", True))
    X_train = embedder.encode(train_df["cleaned_text"].tolist(), batch_size=emb_cfg.get("batch_size", 64))
    X_test = embedder.encode(test_df["cleaned_text"].tolist(), batch_size=emb_cfg.get("batch_size", 64))

    clf_cfg = cfg["classifier"]
    clf = FailureClassifier(classifier_type=clf_cfg["type"], max_iter=clf_cfg.get("max_iter", 1000),
                            class_weight=clf_cfg.get("class_weight", "balanced"),
                            n_estimators=clf_cfg.get("n_estimators", 200),
                            random_state=clf_cfg.get("random_state", 42))
    clf.fit(X_train, train_df["outcome"].tolist())
    y_pred = clf.predict(X_test)
    y_true = test_df["outcome"].tolist()
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))
    try:
        # fail class probability
        proba = clf.predict_proba(X_test)
        fail_idx = list(clf.classes_).index("fail")
        auc = roc_auc_score([1 if y == "fail" else 0 for y in y_true], proba[:, fail_idx])
        print(f"ROC-AUC (fail): {auc:.4f}")
    except Exception as e:
        print("AUC:", e)

    cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
    plt.title("Confusion Matrix - Test Failure Prediction")
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["artifacts_dir"]) / "confusion_matrix.png", dpi=120)
    plt.close()

    clf.save(cfg["paths"]["classifier"])
    sim = SimilarityIndex(metric=cfg["similarity"]["metric"], top_k=cfg["similarity"]["top_k"])
    meta_cols = ["test_id", "test_name", "outcome", "module"]
    sim.build(X_train, train_df[meta_cols])
    sim.save(Path(cfg["paths"]["artifacts_dir"]) / "faiss.index",
             Path(cfg["paths"]["artifacts_dir"]) / "metadata.csv")
    print("✅ Done. Artifacts in", cfg["paths"]["artifacts_dir"])

if __name__ == "__main__":
    main()
