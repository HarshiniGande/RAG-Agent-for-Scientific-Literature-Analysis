"""
vector_store.py
------------------
Thin wrapper around FAISS for storing chunk embeddings and their metadata
(source file, position, text), with save/load to disk so the index only
needs to be built once per corpus version.
"""
import json
import os

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product on normalized vectors
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]):
        assert vectors.shape[0] == len(metadata), "vectors/metadata length mismatch"
        assert vectors.shape[1] == self.dim, f"expected dim {self.dim}, got {vectors.shape[1]}"
        self.index.add(vectors.astype("float32"))
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        query_vector = query_vector.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)
        return results

    def all_vectors(self) -> np.ndarray:
        """Reconstruct all stored vectors -- used by the MMR re-ranker."""
        n = self.index.ntotal
        return np.vstack([self.index.reconstruct(i) for i in range(n)])

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "metadata.json"), "w") as f:
            json.dump({"dim": self.dim, "metadata": self.metadata}, f, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "VectorStore":
        with open(os.path.join(index_dir, "metadata.json"), "r") as f:
            data = json.load(f)
        store = cls(dim=data["dim"])
        store.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        store.metadata = data["metadata"]
        return store
