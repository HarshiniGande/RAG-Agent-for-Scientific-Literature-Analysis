"""
corpus_audit.py
------------------
Responsible-AI data-quality and bias audit of the source corpus, run before
trusting it for retrieval. Flags issues that would otherwise silently skew
which papers the agent surfaces:

- Source/journal imbalance: if one journal or venue dominates the corpus,
  the agent's answers will systematically over-represent that venue's
  framing of a topic.
- Publication-year skew: a corpus weighted toward old papers risks giving
  stale answers on fast-moving topics without any explicit warning.
- Near-duplicate documents: duplicate or near-duplicate papers inflate
  their apparent evidentiary weight during retrieval.
- Length outliers: abstracts far longer/shorter than the corpus norm can
  dominate or be starved during chunking, distorting retrieval independent
  of actual relevance.
"""
import argparse
import os
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_paper_metadata(text: str) -> dict:
    """Extracts simple header fields (Authors/Year/Journal) if present,
    from the plain-text paper format used in data/sample_papers."""
    meta = {}
    for field in ["Authors", "Year", "Journal", "Title"]:
        match = re.search(rf"^{field}:\s*(.+)$", text, re.MULTILINE)
        meta[field.lower()] = match.group(1).strip() if match else None
    return meta


def load_papers(corpus_dir: str) -> list[dict]:
    papers = []
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()
        meta = parse_paper_metadata(text)
        meta["file"] = fname
        meta["text"] = text
        meta["length_words"] = len(text.split())
        papers.append(meta)
    return papers


def audit_source_imbalance(papers: list[dict], dominance_threshold: float = 0.5) -> dict:
    journals = [p["journal"] for p in papers if p["journal"]]
    counts = Counter(journals)
    total = sum(counts.values())
    shares = {j: c / total for j, c in counts.items()} if total else {}
    dominant = [j for j, s in shares.items() if s > dominance_threshold]
    return {"journal_counts": dict(counts), "journal_shares": shares, "flagged_dominant_journals": dominant}


def audit_recency(papers: list[dict], stale_threshold_years: int = 10, current_year: int = 2026) -> dict:
    years = [int(p["year"]) for p in papers if p["year"]]
    if not years:
        return {"note": "no year metadata found"}
    stale_count = sum(1 for y in years if current_year - y > stale_threshold_years)
    return {
        "min_year": min(years),
        "max_year": max(years),
        "median_year": int(np.median(years)),
        "stale_paper_count": stale_count,
        "stale_fraction": stale_count / len(years),
        "flagged": (stale_count / len(years)) > 0.4,
    }


def audit_near_duplicates(papers: list[dict], similarity_threshold: float = 0.85) -> list[tuple]:
    texts = [p["text"] for p in papers]
    if len(texts) < 2:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    sims = cosine_similarity(tfidf)

    duplicates = []
    n = len(papers)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] > similarity_threshold:
                duplicates.append((papers[i]["file"], papers[j]["file"], float(sims[i, j])))
    return duplicates


def audit_length_outliers(papers: list[dict], z_threshold: float = 2.0) -> list[dict]:
    lengths = np.array([p["length_words"] for p in papers])
    mean, std = lengths.mean(), lengths.std()
    if std == 0:
        return []
    outliers = []
    for p, l in zip(papers, lengths):
        z = (l - mean) / std
        if abs(z) > z_threshold:
            outliers.append({"file": p["file"], "length_words": int(l), "z_score": round(float(z), 2)})
    return outliers


def run_audit(corpus_dir: str) -> dict:
    papers = load_papers(corpus_dir)
    return {
        "n_papers": len(papers),
        "source_imbalance": audit_source_imbalance(papers),
        "recency": audit_recency(papers),
        "near_duplicates": audit_near_duplicates(papers),
        "length_outliers": audit_length_outliers(papers),
    }


def print_audit(results: dict):
    print(f"Corpus size: {results['n_papers']} papers\n")

    print("--- Source / journal imbalance ---")
    print(results["source_imbalance"]["journal_counts"])
    if results["source_imbalance"]["flagged_dominant_journals"]:
        print(f"⚠️  Dominant journal(s): {results['source_imbalance']['flagged_dominant_journals']}")
    print()

    print("--- Recency ---")
    print(results["recency"])
    if results["recency"].get("flagged"):
        print("⚠️  Corpus is recency-skewed toward older papers.")
    print()

    print("--- Near-duplicate documents ---")
    if results["near_duplicates"]:
        for a, b, sim in results["near_duplicates"]:
            print(f"⚠️  {a} <-> {b}: cosine similarity {sim:.3f}")
    else:
        print("None found above threshold.")
    print()

    print("--- Length outliers ---")
    if results["length_outliers"]:
        for o in results["length_outliers"]:
            print(f"⚠️  {o['file']}: {o['length_words']} words (z={o['z_score']})")
    else:
        print("None found.")


def main():
    parser = argparse.ArgumentParser(description="Audit a corpus for data quality / bias risks")
    parser.add_argument("--corpus-dir", type=str, default="data/sample_papers")
    args = parser.parse_args()

    results = run_audit(args.corpus_dir)
    print_audit(results)


if __name__ == "__main__":
    main()
