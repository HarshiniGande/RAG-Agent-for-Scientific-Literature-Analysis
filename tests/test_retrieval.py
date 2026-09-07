import os
import sys
import shutil
import tempfile

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest import chunk_document, extract_title_and_abstract, build_index
from embeddings import LSAEmbedder, get_embedder
from vector_store import VectorStore
from retriever import top_k_search, mmr_rerank
from agent import extractive_answer, answer_question
from corpus_audit import (
    parse_paper_metadata,
    audit_source_imbalance,
    audit_recency,
    audit_length_outliers,
    run_audit,
)

SAMPLE_TEXT = """Title: A Test Paper About Widgets
Authors: A. Author
Year: 2022
Journal: Journal of Widget Studies

Abstract: Widgets are useful. This paper studies widgets in depth. Widgets \
can be red or blue. We find that red widgets perform better than blue \
widgets in most conditions. Blue widgets remain popular for aesthetic \
reasons despite lower performance.
"""

SAMPLE_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_papers")


@pytest.fixture
def temp_index_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_extract_title_and_abstract():
    title, abstract = extract_title_and_abstract(SAMPLE_TEXT)
    assert title == "A Test Paper About Widgets"
    assert abstract.startswith("Widgets are useful.")
    assert "Journal:" not in abstract


def test_chunk_document_produces_overlapping_chunks():
    chunks = chunk_document(SAMPLE_TEXT, source_file="test.txt", max_sentences=2, overlap=1)
    assert len(chunks) > 1
    assert all(c.title == "A Test Paper About Widgets" for c in chunks)
    assert all(c.source_file == "test.txt" for c in chunks)


def test_lsa_embedder_fit_and_embed():
    texts = ["red widgets are fast", "blue widgets are slow", "widgets come in many colors"]
    embedder = LSAEmbedder(n_components=2)
    embedder.fit(texts)
    vectors = embedder.embed(texts)
    assert vectors.shape[0] == 3
    # embeddings should be L2-normalized (unit norm) for cosine similarity via inner product
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_get_embedder_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_embedder("not_a_real_backend")


def test_vector_store_add_and_search():
    texts = ["red widgets are fast", "blue widgets are slow", "cats are unrelated animals"]
    embedder = LSAEmbedder(n_components=2)
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, metadata=[{"text": t, "source_file": f"doc{i}.txt"} for i, t in enumerate(texts)])

    query_vec = embedder.embed(["fast red widget"])[0]
    results = store.search(query_vec, k=2)
    assert len(results) == 2
    assert "score" in results[0]


def test_vector_store_save_and_load(temp_index_dir):
    texts = ["alpha beta gamma", "delta epsilon zeta"]
    embedder = LSAEmbedder(n_components=2)
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, metadata=[{"text": t} for t in texts])
    store.save(temp_index_dir)

    loaded = VectorStore.load(temp_index_dir)
    assert loaded.dim == store.dim
    assert len(loaded.metadata) == 2


def test_mmr_rerank_returns_requested_count():
    texts = [f"widget type {i} performance test" for i in range(10)]
    embedder = LSAEmbedder(n_components=4)
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, metadata=[{"text": t, "source_file": f"doc{i}.txt"} for i, t in enumerate(texts)])

    results = mmr_rerank(store, embedder, "widget performance", fetch_k=8, top_k=3)
    assert len(results) == 3


def test_extractive_answer_limits_to_max_chunks():
    chunks = [
        {"source_file": f"doc{i}.txt", "title": f"Paper {i}", "score": 0.9 - i * 0.1, "text": f"content {i}"}
        for i in range(5)
    ]
    answer = extractive_answer("some question?", chunks, max_chunks=2)
    assert "doc0.txt" in answer
    assert "doc1.txt" in answer
    assert "doc2.txt" not in answer


def test_build_index_and_answer_question_end_to_end(temp_index_dir):
    build_index(SAMPLE_CORPUS_DIR, temp_index_dir, backend="lsa")
    assert os.path.exists(os.path.join(temp_index_dir, "faiss.index"))
    assert os.path.exists(os.path.join(temp_index_dir, "embedder.joblib"))

    result = answer_question(
        "How does retrieval augmentation reduce hallucination?", temp_index_dir, top_k=3, use_llm=False
    )
    assert "answer" in result
    assert len(result["sources"]) > 0


def test_parse_paper_metadata():
    meta = parse_paper_metadata(SAMPLE_TEXT)
    assert meta["title"] == "A Test Paper About Widgets"
    assert meta["year"] == "2022"
    assert meta["journal"] == "Journal of Widget Studies"


def test_audit_source_imbalance_flags_dominant_journal():
    papers = [{"journal": "Journal A"} for _ in range(8)] + [{"journal": "Journal B"} for _ in range(2)]
    result = audit_source_imbalance(papers, dominance_threshold=0.5)
    assert "Journal A" in result["flagged_dominant_journals"]


def test_audit_recency_flags_stale_corpus():
    papers = [{"year": "1990"} for _ in range(8)] + [{"year": "2024"} for _ in range(2)]
    result = audit_recency(papers, stale_threshold_years=10, current_year=2026)
    assert result["flagged"] is True


def test_audit_length_outliers_detects_outlier():
    papers = [{"file": f"p{i}.txt", "length_words": 100} for i in range(9)]
    papers.append({"file": "outlier.txt", "length_words": 5000})
    outliers = audit_length_outliers(papers, z_threshold=2.0)
    assert any(o["file"] == "outlier.txt" for o in outliers)


def test_run_audit_on_sample_corpus():
    results = run_audit(SAMPLE_CORPUS_DIR)
    assert results["n_papers"] == 10
    assert "source_imbalance" in results
    assert "recency" in results
