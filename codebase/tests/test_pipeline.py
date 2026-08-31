from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.generator import generate_test_dataset
from src.data.preprocessing import TestPreprocessor
from src.models.embeddings import EmbeddingModel
from src.models.classifier import FailureClassifier

def test_gen():
    df = generate_test_dataset(100, seed=1)
    assert len(df) == 100
    assert set(df["outcome"].unique()) <= {"pass", "fail"}

def test_tiny():
    df = generate_test_dataset(200, seed=2)
    pre = TestPreprocessor()
    df = pre.transform_df(df)
    emb = EmbeddingModel(device="cpu")
    X = emb.encode(df["cleaned_text"].tolist(), batch_size=32, show_progress=False)
    clf = FailureClassifier()
    clf.fit(X[:150], df["outcome"].iloc[:150].tolist())
    assert len(clf.predict(X[150:])) == 50

if __name__ == "__main__":
    test_gen()
    test_tiny()
    print("OK")
