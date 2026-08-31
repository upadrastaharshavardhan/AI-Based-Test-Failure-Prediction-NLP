from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from src.data.preprocessing import TestPreprocessor
from src.models.embeddings import EmbeddingModel
from src.models.classifier import FailureClassifier
from src.models.similarity import SimilarityIndex

class TestFailurePredictor:
    def __init__(self, embedder, classifier, similarity, preprocessor):
        self.embedder = embedder
        self.classifier = classifier
        self.similarity = similarity
        self.preprocessor = preprocessor

    def predict(self, test_name: str = "", module: str = "", commit_message: str = "",
                changed_files: str = "", historical_fail_rate: float = 0.0,
                full_text: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        if full_text is None:
            full_text = (f"Test: {test_name}\nModule: {module}\nCommit: {commit_message}\n"
                         f"Changed: {changed_files}\nHistoricalFailRate: {historical_fail_rate}")
        cleaned = self.preprocessor.clean(full_text)
        emb = self.embedder.encode([cleaned], show_progress=False)
        pred = self.classifier.predict_with_confidence(emb)[0]
        similar = self.similarity.search(emb, top_k=top_k)[0]
        return {
            "prediction": pred["prediction"],
            "confidence": pred["confidence"],
            "similar_tests": similar,
            "cleaned_input": cleaned[:300],
        }

    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[Dict]:
        cleaned = self.preprocessor.transform(texts)
        embs = self.embedder.encode(cleaned, show_progress=True)
        preds = self.classifier.predict_with_confidence(embs)
        sims = self.similarity.search(embs, top_k=top_k)
        return [{"prediction": p["prediction"], "confidence": p["confidence"],
                 "similar_tests": s} for p, s in zip(preds, sims)]

    @classmethod
    def load(cls, artifacts_dir: Union[str, Path], config_path: Optional[Union[str, Path]] = None):
        artifacts_dir = Path(artifacts_dir)
        if config_path is None:
            config_path = Path("config/config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        emb_cfg = cfg.get("embedding", {})
        embedder = EmbeddingModel(model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                                  device=emb_cfg.get("device"), normalize=emb_cfg.get("normalize", True))
        classifier = FailureClassifier.load(artifacts_dir / "classifier.joblib")
        similarity = SimilarityIndex(metric=cfg.get("similarity", {}).get("metric", "cosine"),
                                     top_k=cfg.get("similarity", {}).get("top_k", 5))
        similarity.load(artifacts_dir / "faiss.index", artifacts_dir / "metadata.csv")
        preprocessor = TestPreprocessor(max_text_length=cfg.get("preprocessing", {}).get("max_text_length", 800))
        return cls(embedder, classifier, similarity, preprocessor)
