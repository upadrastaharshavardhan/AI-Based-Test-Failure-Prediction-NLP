from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class FailureClassifier:
    def __init__(self, classifier_type: str = "logistic", max_iter: int = 1000,
                 class_weight: str = "balanced", n_estimators: int = 200, random_state: int = 42):
        self.classifier_type = classifier_type
        self.label_encoder = LabelEncoder()
        if classifier_type == "logistic":
            self.model = LogisticRegression(max_iter=max_iter, class_weight=class_weight,
                                            random_state=random_state, n_jobs=-1)
        else:
            self.model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight,
                                                random_state=random_state, n_jobs=-1)

    def fit(self, X: np.ndarray, y: List[str]) -> "FailureClassifier":
        y_enc = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_enc)
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        return self.label_encoder.inverse_transform(self.model.predict(X)).tolist()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_with_confidence(self, X: np.ndarray) -> List[Dict]:
        probs = self.predict_proba(X)
        idxs = np.argmax(probs, axis=1)
        labels = self.label_encoder.inverse_transform(idxs)
        return [{"prediction": lab, "confidence": float(probs[i, idxs[i]])}
                for i, lab in enumerate(labels)]

    @property
    def classes_(self):
        return self.label_encoder.classes_.tolist()

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "label_encoder": self.label_encoder,
                     "classifier_type": self.classifier_type}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FailureClassifier":
        data = joblib.load(path)
        obj = cls(classifier_type=data["classifier_type"])
        obj.model = data["model"]
        obj.label_encoder = data["label_encoder"]
        return obj
