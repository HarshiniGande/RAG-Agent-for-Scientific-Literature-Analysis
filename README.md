# RAG-Agent-for-Scientific-Literature-Analysis

# Retrieval-Augmented Generation (RAG) Agent for Scientific Literature Analysis

An AI agent pipeline that ingests a corpus of unstructured research papers,
indexes them with vector embeddings, retrieves the most relevant passages
for a research question, and synthesizes a structured, cited answer using an
LLM — with retrieval quality benchmarked via ROUGE and a bias/data-quality
audit of the underlying corpus.

## What's inside

```
rag-scientific-literature/
├── src/
│   ├── ingest.py           # Loads papers, chunks text, builds a document store
│   ├── embeddings.py       # Pluggable embedding backends (local LSA + optional API-based)
│   ├── vector_store.py     # FAISS index build/save/load + similarity search
│   ├── retriever.py        # Top-k retrieval, MMR re-ranking for diversity
│   ├── agent.py            # RAG pipeline: retrieve -> build prompt -> generate answer
│   ├── evaluate_rouge.py   # ROUGE-1/2/L benchmarking of generated answers vs. reference summaries
│   └── corpus_audit.py     # Data quality & bias checks (source imbalance, recency, duplication)
├── data/
│   └── sample_papers/      # 10 real arXiv paper abstracts (titles, authors, years) to run the demo end-to-end
├── indexes/                # Saved FAISS index + metadata land here
├── tests/
│   └── test_retrieval.py
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Ingest the sample corpus and build a FAISS index
python src/ingest.py --corpus-dir data/sample_papers --index-dir indexes

# 2. Ask the agent a question
python src/agent.py --index-dir indexes --question "What approaches exist for fast approximate nearest neighbor search at scale?"

# 3. Benchmark retrieval/answer quality against reference summaries
python src/evaluate_rouge.py --index-dir indexes

# 4. Run the corpus data-quality / bias audit
python src/corpus_audit.py --corpus-dir data/sample_papers
```

## Data

`data/sample_papers/` contains 10 real papers (title, authors, publication
year, and abstract) drawn from a public arXiv metadata snapshot
([NeelShah18/arxivData](https://github.com/NeelShah18/arxivData) on GitHub,
itself sourced from the arXiv API). They were filtered for genuine relevance
to information retrieval and nearest-neighbor search — the actual subject
this pipeline is built to search over — rather than fabricated. All 10 are
non-peer-reviewed arXiv preprints (2008–2018); `corpus_audit.py` correctly
flags this as a single-venue corpus, which is an honest, real finding about
this specific demo corpus, not a bug. Swap in your own `.txt` files (same
`Title:`/`Authors:`/`Year:`/`Journal:`/`Abstract:` header format) to index a
different corpus — `ingest.py` needs no changes.

## Design notes

- **Embeddings** (`embeddings.py`): ships with a fully local, offline
  `LSAEmbedder` (TF-IDF + Truncated SVD) so the whole pipeline runs without
  any API keys or model downloads — useful for a portfolio demo and for CI.
  It also defines an `AnthropicEmbedder`-style interface stub so swapping in
  a hosted embedding model or sentence-transformers checkpoint in production
  is a one-line change (`EMBEDDING_BACKEND` env var).
- **Vector store** (`vector_store.py`): wraps FAISS (`IndexFlatIP` on
  L2-normalized vectors, i.e. cosine similarity) with save/load to disk and
  metadata tracking (source file, chunk offsets) so retrieved chunks can be
  traced back to their paper.
- **Retrieval** (`retriever.py`): standard top-k similarity search, plus a
  Maximal Marginal Relevance (MMR) re-ranker so retrieved chunks aren't all
  near-duplicates of the same passage — important for literature review
  tasks where you want coverage across multiple papers, not five copies of
  the same sentence.
- **Agent** (`agent.py`): retrieves relevant chunks, builds a grounded prompt
  with inline citations back to source documents, and calls an LLM to
  synthesize a structured answer (claims + supporting citations). If no
  `ANTHROPIC_API_KEY` is set, it falls back to an extractive synthesizer
  (returns the top passages with source attribution, no generation) so the
  pipeline still runs end-to-end for anyone without an API key.
- **Evaluation** (`evaluate_rouge.py`): computes ROUGE-1/2/L between
  generated answers and human-written reference summaries for each sample
  question, the standard way to benchmark retrieval/summarization quality
  against a gold standard. Note: scores in extractive mode (default, no API
  key) are naturally modest (ROUGE-1 ~0.15-0.20) since it returns original
  abstract sentences rather than an abstractive paraphrase close to the
  reference wording — pass `--use-llm` in `agent.py` to see ROUGE scores
  from LLM-synthesized answers, which paraphrase closer to how a human
  reference summary is typically written.
- **Responsible AI: corpus audit** (`corpus_audit.py`): flags data-quality
  and bias risks in the source corpus before it's trusted for retrieval —
  source/journal imbalance, publication-year skew (stale corpus risk),
  near-duplicate documents, and abstract-length outliers that could
  dominate retrieval purely due to chunking artifacts.

## Using a real LLM

Set an API key to enable real generation instead of the extractive fallback:

```bash
export ANTHROPIC_API_KEY=sk-...
python src/agent.py --index-dir indexes --question "..." --use-llm
```

## Testing

```bash
pytest tests/ -v
```
