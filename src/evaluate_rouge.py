"""
evaluate_rouge.py
--------------------
Benchmarks the RAG pipeline's answer quality against a small set of
hand-written reference summaries using ROUGE-1/2/L, the standard metric
for scoring generated/extracted text against a gold-standard summary.

Since the extractive fallback mode doesn't require an API key, this script
runs fully offline by default and reports how well the *retrieval* step
alone surfaces the right content, which is often the metric of interest
when the LLM step is a fixed, swappable component.
"""
import argparse
import json
import os

from rouge_score import rouge_scorer

from agent import answer_question

# A handful of (question, reference_summary) pairs written against the
# sample corpus (real arXiv papers on nearest-neighbor search and
# retrieval), to demonstrate the benchmarking harness end-to-end.
EVAL_SET = [
    {
        "question": "What approaches exist for fast approximate nearest neighbor search at scale?",
        "reference": (
            "Approximate nearest neighbor search methods trade exact accuracy for speed using "
            "techniques such as hashing, kNN-graph-based algorithms like EFANNA, and GPU "
            "acceleration to handle large, high-dimensional datasets efficiently."
        ),
    },
    {
        "question": "How is nearest neighbor search applied to natural language processing tasks?",
        "reference": (
            "Many NLP tasks such as word analogy, document similarity, and question answering "
            "can be framed as nearest neighbor search problems, though noisy data and large "
            "search spaces make this challenging in practice."
        ),
    },
    {
        "question": "How does hashing help with efficient nearest neighbor search?",
        "reference": (
            "Hashing algorithms map high-dimensional vectors into compact binary codes so that "
            "approximate nearest neighbors can be found quickly, trading some accuracy for "
            "large gains in search speed and memory efficiency."
        ),
    },
    {
        "question": "How can retrieval augmentation improve neural network robustness?",
        "reference": (
            "Retrieval-augmented neural networks incorporate information retrieved from external "
            "examples during inference, which can improve robustness against adversarial "
            "perturbations compared to models relying only on learned parameters."
        ),
    },
]


def run_benchmark(index_dir: str, backend: str = "lsa", top_k: int = 5, use_llm: bool = False) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    per_question_scores = []
    for item in EVAL_SET:
        result = answer_question(item["question"], index_dir, backend=backend, top_k=top_k, use_llm=use_llm)
        generated = result["answer"]
        scores = scorer.score(item["reference"], generated)
        per_question_scores.append(
            {
                "question": item["question"],
                "sources_used": result["sources"],
                "rouge1_f": scores["rouge1"].fmeasure,
                "rouge2_f": scores["rouge2"].fmeasure,
                "rougeL_f": scores["rougeL"].fmeasure,
            }
        )

    avg = {
        metric: sum(s[metric] for s in per_question_scores) / len(per_question_scores)
        for metric in ["rouge1_f", "rouge2_f", "rougeL_f"]
    }
    return {"per_question": per_question_scores, "average": avg}


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG retrieval quality with ROUGE")
    parser.add_argument("--index-dir", type=str, default="indexes")
    parser.add_argument("--backend", type=str, default="lsa")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-llm", action="store_true", help="Requires ANTHROPIC_API_KEY")
    args = parser.parse_args()

    if args.use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: --use-llm set but ANTHROPIC_API_KEY is not set. Falling back to extractive mode.")
        args.use_llm = False

    results = run_benchmark(args.index_dir, backend=args.backend, top_k=args.top_k, use_llm=args.use_llm)

    print("Per-question ROUGE scores:")
    for r in results["per_question"]:
        print(
            f"  Q: {r['question'][:70]}...\n"
            f"     sources={r['sources_used']}  "
            f"ROUGE-1={r['rouge1_f']:.3f}  ROUGE-2={r['rouge2_f']:.3f}  ROUGE-L={r['rougeL_f']:.3f}"
        )

    print("\nAverage scores:")
    for k, v in results["average"].items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
