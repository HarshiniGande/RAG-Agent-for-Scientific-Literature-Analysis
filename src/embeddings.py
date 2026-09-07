"""
embeddings.py
--------------
Pluggable embedding backends for the RAG pipeline.

- LSAEmbedder: TF-IDF + Truncated SVD ("latent semantic analysis"). Fully
  local and offline -- no API keys or model downloads required -- which is
  what makes this repo runnable end-to-end as a portfolio demo / in CI.
- AnthropicEmbedder / SentenceTransformerEmbedder: thin stubs showing how a
  production deployment would swap in a hosted or open-weights embedding
  model with the same `.fit()` / `.embed()` interface, so the rest of the
  pipeline (vector_store, retriever, agent) needs zero changes to switch.
"""
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class BaseEmbedder:
    def fit(self, texts: list[str]):
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseEmbedder":
        raise NotImplementedError


class LSAEmbedder(BaseEmbedder):
    """TF-IDF -> Truncated SVD dense embedding. Deterministic, offline,
    and fast -- a reasonable stand-in for a dense encoder when the goal is
    to demonstrate the retrieval architecture rather than squeeze out the
    last point of semantic recall."""

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, texts: list[str]):
        n_components = min(self.n_components, max(2, len(texts) - 1))
        if n_components != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .embed()")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        return normalize(dense, axis=1).astype("float32")

    def save(self, path: str):
        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd, "n_components": self.n_components}, path)

    @classmethod
    def load(cls, path: str) -> "LSAEmbedder":
        data = joblib.load(path)
        obj = cls(n_components=data["n_components"])
        obj.vectorizer = data["vectorizer"]
        obj.svd = data["svd"]
        obj._fitted = True
        return obj


class AnthropicEmbedder(BaseEmbedder):
    """Stub for a hosted embedding backend. Anthropic's API is primarily a
    generation (messages) API rather than a dedicated embeddings endpoint,
    so in production this would typically call a dedicated embeddings
    provider (e.g. Voyage AI, which Anthropic recommends for embeddings) --
    swap the API call below for whichever provider your infra uses. The
    interface matches LSAEmbedder so no other module needs to change.
    """

    def __init__(self, model: str = "voyage-2"):
        self.model = model
        self._fitted = True  # hosted models need no local fitting step

    def fit(self, texts: list[str]):
        return self  # no-op: hosted embedding models are pre-trained

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "Wire this up to your embedding provider's API (e.g. Voyage AI, "
            "OpenAI, or Cohere) and return an (n_texts, dim) float32 array."
        )

    def save(self, path: str):
        joblib.dump({"model": self.model}, path)

    @classmethod
    def load(cls, path: str) -> "AnthropicEmbedder":
        data = joblib.load(path)
        return cls(model=data["model"])


def get_embedder(backend: str = "lsa") -> BaseEmbedder:
    if backend == "lsa":
        return LSAEmbedder()
    elif backend == "hosted":
        return AnthropicEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")


def load_embedder(path: str, backend: str = "lsa") -> BaseEmbedder:
    if backend == "lsa":
        return LSAEmbedder.load(path)
    elif backend == "hosted":
        return AnthropicEmbedder.load(path)
    raise ValueError(f"Unknown embedding backend: {backend}")
