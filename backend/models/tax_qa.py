"""
models/tax_qa.py
─────────────────────────────────────────────────────────────────────────────
Tax NLP Q&A Model — Retrieval-Augmented Generation (RAG)

Architecture:
  1. Sentence-BERT encodes the knowledge base into a FAISS vector store
  2. At query time, top-k semantically similar entries are retrieved
  3. Retrieved context is passed to the LLM (via LangChain) for generation

Can also be used in standalone retrieval mode (no LLM) for fast lookup.
"""

from __future__ import annotations
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any

VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "models/vector_store"))
INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
DOCS_PATH = VECTOR_STORE_DIR / "documents.pkl"


# ─── Build / load vector store ───────────────────────────────────────────────

def build_vector_store(qa_pairs: list[dict], save: bool = True) -> Any:
    """
    Encode Q&A pairs with Sentence-BERT and build a FAISS index.

    Args:
        qa_pairs: List of {"q": str, "a": str} dicts.
        save: Whether to persist the index.

    Returns:
        (index, documents) tuple.
    """
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Install dependencies: pip install faiss-cpu sentence-transformers"
        )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Combine question + answer for richer embeddings
    texts = [f"Q: {p['q']}\nA: {p['a']}" for p in qa_pairs]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
    index.add(embeddings.astype(np.float32))

    documents = [
        {"text": texts[i], "question": p["q"], "answer": p["a"], "idx": i}
        for i, p in enumerate(qa_pairs)
    ]

    if save:
        faiss.write_index(index, str(INDEX_PATH))
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)
        # Save model name for consistent loading
        (VECTOR_STORE_DIR / "model_name.txt").write_text("all-MiniLM-L6-v2")
        print(f"✅ Vector store saved → {VECTOR_STORE_DIR} ({len(documents)} docs)")

    return index, documents


def load_vector_store():
    """Load persisted FAISS index and documents."""
    try:
        import faiss
    except ImportError:
        raise ImportError("Install: pip install faiss-cpu")

    for p in [INDEX_PATH, DOCS_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Vector store not found at {p}. Run build_vector_store() first."
            )

    index = faiss.read_index(str(INDEX_PATH))
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    return index, documents


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top-k most relevant Q&A entries for a query.

    Args:
        query: User's tax question.
        top_k: Number of results to return.

    Returns:
        List of dicts with question, answer, and similarity score.
    """
    from sentence_transformers import SentenceTransformer

    index, documents = load_vector_store()
    model_name = (VECTOR_STORE_DIR / "model_name.txt").read_text().strip()
    model = SentenceTransformer(model_name)

    query_embedding = model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = documents[idx]
        results.append({
            "question": doc["question"],
            "answer": doc["answer"],
            "similarity_score": round(float(score), 4),
        })

    return results


# ─── Full RAG answer (requires OpenAI) ────────────────────────────────────────

def answer_with_rag(query: str, top_k: int = 3) -> dict:
    """
    Answer a tax question using RAG (retrieval + LLM generation).

    Requires OPENAI_API_KEY environment variable.

    Args:
        query: User's tax question.
        top_k: Number of context chunks to retrieve.

    Returns:
        dict with answer, sources, and retrieved context.
    """
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage

    retrieved = retrieve(query, top_k=top_k)
    context = "\n\n".join(
        f"[Source {i+1}]\nQ: {r['question']}\nA: {r['answer']}"
        for i, r in enumerate(retrieved)
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content=(
            "You are TaxMind, an expert AI tax assistant. "
            "Answer the user's tax question using the provided context. "
            "Be accurate, cite relevant tax codes or rules when known, "
            "and always recommend consulting a licensed CPA for specific tax advice. "
            "If the context doesn't fully answer the question, say so honestly."
        )),
        HumanMessage(content=(
            f"Context from knowledge base:\n{context}\n\n"
            f"User question: {query}"
        )),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "retrieved_sources": retrieved,
        "model": "gpt-4o-mini",
        "rag_enabled": True,
    }


# ─── Standalone retrieval mode (no LLM) ──────────────────────────────────────

def answer_retrieval_only(query: str, top_k: int = 1) -> dict:
    """
    Answer by returning the best matching Q&A pair (no LLM required).
    Good for exact FAQ-style questions.
    """
    results = retrieve(query, top_k=top_k)
    if not results:
        return {"answer": "I couldn't find a relevant answer in the knowledge base.", 
                "retrieved_sources": []}

    best = results[0]
    return {
        "answer": best["answer"],
        "matched_question": best["question"],
        "similarity_score": best["similarity_score"],
        "retrieved_sources": results,
        "rag_enabled": False,
    }


# ─── CLI build ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    KB_PATH = Path("data/synthetic/qa_knowledge_base.json")
    if not KB_PATH.exists():
        print("⚠️  Knowledge base not found. Run: python data/generate_synthetic.py")
    else:
        qa_pairs = json.loads(KB_PATH.read_text())
        build_vector_store(qa_pairs)
