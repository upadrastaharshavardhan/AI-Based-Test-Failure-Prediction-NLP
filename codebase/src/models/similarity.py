from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors

class SimilarityIndex:
    def __init__(self, metric: str = "cosine", top_k: int = 5):
        self.metric = metric
        self.top_k = top_k
        self.index = None
        self.metadata = None
        self._use_faiss = FAISS_AVAILABLE

    def build(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> "SimilarityIndex":
        self.metadata = metadata.reset_index(drop=True)
        emb = np.ascontiguousarray(embeddings.astype(np.float32))
        if self._use_faiss:
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
        else:
            self.index = NearestNeighbors(n_neighbors=self.top_k, metric="cosine")
            self.index.fit(emb)
        return self

    def search(self, query: np.ndarray, top_k: Optional[int] = None) -> List[List[dict]]:
        k = top_k or self.top_k
        q = np.ascontiguousarray(query.astype(np.float32))
        results = []
        if self._use_faiss:
            scores, indices = self.index.search(q, k)
            for sc, ix in zip(scores, indices):
                row = []
                for s, i in zip(sc, ix):
                    if i < 0: continue
                    m = self.metadata.iloc[i]
                    row.append({"similarity": float(s), "test_id": m.get("test_id", ""),
                                "test_name": m.get("test_name", ""), "outcome": m.get("outcome", "")})
                results.append(row)
        else:
            dist, indices = self.index.kneighbors(q, n_neighbors=k)
            for drow, irow in zip(dist, indices):
                row = []
                for d, i in zip(drow, irow):
                    m = self.metadata.iloc[i]
                    row.append({"similarity": float(1 - d), "test_id": m.get("test_id", ""),
                                "test_name": m.get("test_name", ""), "outcome": m.get("outcome", "")})
                results.append(row)
        return results

    def save(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        if self._use_faiss:
            faiss.write_index(self.index, str(index_path))
        else:
            import joblib
            joblib.dump(self.index, str(index_path) + ".sklearn")
        self.metadata.to_csv(metadata_path, index=False)

    def load(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> "SimilarityIndex":
        self.metadata = pd.read_csv(metadata_path)
        if self._use_faiss and Path(index_path).exists():
            self.index = faiss.read_index(str(index_path))
        else:
            import joblib
            self.index = joblib.load(str(index_path) + ".sklearn")
            self._use_faiss = False
        return self
