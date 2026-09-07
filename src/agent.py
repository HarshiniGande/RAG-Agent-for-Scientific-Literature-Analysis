"""
agent.py
---------
The RAG agent pipeline: retrieve relevant chunks -> build a grounded prompt
with inline source citations -> generate a structured answer.

Two generation modes:
  - Extractive fallback (default, no API key needed): returns the retrieved
    passages themselves, attributed to source, so the pipeline is fully
    runnable offline for demo/testing purposes.
  - LLM synthesis (--use-llm, requires ANTHROPIC_API_KEY): sends the
    retrieved context + question to Claude and asks it to synthesize a
    structured, cited answer, refusing to answer claims the context doesn't
    support (basic hallucination guardrail).
"""
import argparse
import os

from embeddings import load_embedder
from vector_store import VectorStore
from retriever import mmr_rerank


SYSTEM_PROMPT = """You are a research assistant answering questions strictly from the \
provided excerpts of scientific papers. Rules:
1. Only make claims that are directly supported by the excerpts below.
2. After each claim, cite the source in square brackets, e.g. [paper_001.txt].
3. If the excerpts do not contain enough information to answer, say so explicitly \
   rather than filling gaps from general knowledge.
4. Keep the answer concise (3-6 sentences) and structured as a short list of findings.
"""


def build_context_block(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c['source_file']}] {c['text']}")
    return "\n\n".join(lines)


def extractive_answer(question: str, chunks: list[dict], max_chunks: int = 3) -> str:
    """No-API-key fallback: presents the retrieved evidence directly,
    attributed to source, without LLM synthesis. Uses only the top
    `max_chunks` most relevant excerpts to keep the answer concise."""
    lines = [f"Question: {question}", "", "Top supporting excerpts (extractive, no LLM used):"]
    for i, c in enumerate(chunks[:max_chunks], 1):
        title = c.get("title", "")
        prefix = f"[{c['source_file']}" + (f" — {title}" if title else "") + "]"
        lines.append(f"{i}. {prefix} (score={c['score']:.3f}) {c['text']}")
    return "\n".join(lines)


def llm_answer(question: str, chunks: list[dict], model: str = "claude-sonnet-4-6") -> str:
    """Synthesizes a grounded answer using the Anthropic API. Requires
    ANTHROPIC_API_KEY to be set in the environment."""
    import anthropic

    client = anthropic.Anthropic()
    context_block = build_context_block(chunks)
    user_message = (
        f"Excerpts:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, with inline [source_file] citations."
    )
    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def answer_question(
    question: str,
    index_dir: str,
    backend: str = "lsa",
    top_k: int = 5,
    use_llm: bool = False,
) -> dict:
    store = VectorStore.load(index_dir)
    embedder = load_embedder(os.path.join(index_dir, "embedder.joblib"), backend=backend)

    retrieved = mmr_rerank(store, embedder, question, fetch_k=min(20, len(store.metadata)), top_k=top_k)

    if use_llm:
        answer = llm_answer(question, retrieved)
    else:
        answer = extractive_answer(question, retrieved)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "sources": sorted(set(c["source_file"] for c in retrieved)),
    }


def main():
    parser = argparse.ArgumentParser(description="Ask the RAG agent a question")
    parser.add_argument("--index-dir", type=str, default="indexes")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", type=str, default="lsa")
    parser.add_argument("--use-llm", action="store_true", help="Requires ANTHROPIC_API_KEY")
    args = parser.parse_args()

    if args.use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: --use-llm set but ANTHROPIC_API_KEY is not set. Falling back to extractive mode.")
        args.use_llm = False

    result = answer_question(
        args.question, args.index_dir, backend=args.backend, top_k=args.top_k, use_llm=args.use_llm
    )
    print(result["answer"])
    print(f"\nSources consulted: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
