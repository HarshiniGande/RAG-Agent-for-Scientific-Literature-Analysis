"""
retriever.py
--------------
Retrieval logic on top of the vector store: plain top-k similarity search,
plus Maximal Marginal Relevance (MMR) re-ranking so retrieved chunks aren't
all near-duplicates of the same passage from the same paper.
"""
import numpy as np

from vector_store import VectorStore


def top_k_search(store: VectorStore, embedder, query: str, k: int = 5) -> list[dict]:
    query_vec = embedder.embed([query])[0]
    return store.search(query_vec, k=k)


def mmr_rerank(
    store: VectorStore,
    embedder,
    query: str,
    fetch_k: int = 20,
    top_k: int = 5,
    lambda_mult: float = 0.6,
) -> list[dict]:
    """Maximal Marginal Relevance: iteratively picks the candidate that
    maximizes `lambda * relevance_to_query - (1 - lambda) * max_similarity_to_already_selected`,
    trading off relevance against redundancy so results span more of the
    corpus instead of clustering on one passage.
    """
    query_vec = embedder.embed([query])[0]
    candidates = store.search(query_vec, k=min(fetch_k, len(store.metadata)))
    if len(candidates) <= top_k:
        return candidates

    candidate_texts = [c["text"] for c in candidates]
    candidate_vecs = embedder.embed(candidate_texts)

    selected_idx = [0]  # start with the top relevance hit
    remaining_idx = list(range(1, len(candidates)))

    while len(selected_idx) < top_k and remaining_idx:
        best_score = -np.inf
        best_i = None
        for i in remaining_idx:
            relevance = float(np.dot(candidate_vecs[i], query_vec))
            redundancy = max(float(np.dot(candidate_vecs[i], candidate_vecs[j])) for j in selected_idx)
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i
        selected_idx.append(best_i)
        remaining_idx.remove(best_i)

    return [candidates[i] for i in selected_idx]
