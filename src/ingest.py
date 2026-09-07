"""
ingest.py
----------
Loads a directory of research paper text files, splits each into overlapping
chunks (sentence-boundary aware), embeds the chunks, and builds a FAISS
index + metadata store on disk.
"""
import argparse
import json
import os
import re
from dataclasses import dataclass, asdict

from embeddings import get_embedder
from vector_store import VectorStore


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    text: str
    position: int  # order of this chunk within its source document
    title: str = ""


def split_into_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (avoids pulling in a heavy NLP dependency
    for a portfolio demo). Splits on '.', '!', '?' followed by whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def extract_title_and_abstract(raw_text: str) -> tuple[str, str]:
    """Splits the plain-text paper format used in data/sample_papers into
    its Title line and Abstract body, so embedding/retrieval/ROUGE scoring
    operate on the actual scientific content rather than header metadata
    (Authors/Year/Journal), which would otherwise dilute both semantic
    similarity and n-gram overlap metrics.
    """
    title_match = re.search(r"^Title:\s*(.+)$", raw_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""

    abstract_match = re.search(r"Abstract:\s*(.+)", raw_text, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else raw_text.strip()

    return title, abstract


def chunk_document(text: str, source_file: str, max_sentences: int = 3, overlap: int = 1) -> list[Chunk]:
    """Groups consecutive sentences of the paper's abstract into overlapping
    chunks, so each chunk keeps enough local context to be interpretable on
    its own, while overlap prevents information from being lost at chunk
    boundaries. Header metadata (title/authors/year/journal) is parsed out
    separately rather than chunked, so it doesn't compete with the actual
    scientific content during retrieval.
    """
    title, abstract = extract_title_and_abstract(text)
    sentences = split_into_sentences(abstract)
    chunks = []
    start = 0
    position = 0
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk_text = " ".join(sentences[start:end])
        chunk_id = f"{os.path.basename(source_file)}::chunk_{position}"
        chunks.append(
            Chunk(chunk_id=chunk_id, source_file=source_file, text=chunk_text, position=position, title=title)
        )
        if end == len(sentences):
            break
        start = end - overlap
        position += 1
    return chunks


def load_corpus(corpus_dir: str) -> list[Chunk]:
    all_chunks = []
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(corpus_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_document(text, source_file=fname))
    return all_chunks


def build_index(corpus_dir: str, index_dir: str, backend: str = "lsa"):
    os.makedirs(index_dir, exist_ok=True)

    chunks = load_corpus(corpus_dir)
    texts = [c.text for c in chunks]
    print(f"Loaded {len(texts)} chunks from {corpus_dir}")

    embedder = get_embedder(backend)
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, metadata=[asdict(c) for c in chunks])
    store.save(index_dir)
    embedder.save(os.path.join(index_dir, "embedder.joblib"))

    with open(os.path.join(index_dir, "config.json"), "w") as f:
        json.dump({"backend": backend, "n_chunks": len(chunks), "corpus_dir": corpus_dir}, f, indent=2)

    print(f"Built index with {len(chunks)} chunks -> saved to {index_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ingest a corpus and build a FAISS index")
    parser.add_argument("--corpus-dir", type=str, default="data/sample_papers")
    parser.add_argument("--index-dir", type=str, default="indexes")
    parser.add_argument("--backend", type=str, default="lsa", choices=["lsa"])
    args = parser.parse_args()

    build_index(args.corpus_dir, args.index_dir, backend=args.backend)


if __name__ == "__main__":
    main()
