from __future__ import annotations
import os, json, pickle
import numpy as np
from pathlib import Path
from typing import Any

VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "models/vector_store"))
INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
DOCS_PATH  = VECTOR_STORE_DIR / "documents.pkl"

def _embed(texts):
    from services.embedding_service import get_embeddings
    return np.array(get_embeddings(texts), dtype=np.float32)

def build_vector_store(qa_pairs, save=True):
    import faiss
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    texts = [f"Q: {p['q']}\nA: {p['a']}" for p in qa_pairs]
    embeddings = _embed(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    documents = [{"text": texts[i], "question": p["q"], "answer": p["a"], "idx": i} for i, p in enumerate(qa_pairs)]
    if save:
        faiss.write_index(index, str(INDEX_PATH))
        with open(DOCS_PATH, "wb") as f: pickle.dump(documents, f)
        (VECTOR_STORE_DIR / "model_name.txt").write_text("openai-text-embedding-3-small")
    return index, documents

def load_vector_store():
    import faiss
    for p in [INDEX_PATH, DOCS_PATH]:
        if not p.exists(): raise FileNotFoundError(f"Vector store not found at {p}")
    index = faiss.read_index(str(INDEX_PATH))
    with open(DOCS_PATH, "rb") as f: documents = pickle.load(f)
    return index, documents

def retrieve(query, top_k=3):
    index, documents = load_vector_store()
    query_embedding = _embed([query])
    scores, indices = index.search(query_embedding, top_k)
    return [{"question": documents[idx]["question"], "answer": documents[idx]["answer"],
             "similarity_score": round(float(score), 4)}
            for score, idx in zip(scores[0], indices[0])]

def answer_with_rag(query, top_k=3):
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    retrieved = retrieve(query, top_k=top_k)
    context = "\n\n".join(f"[Source {i+1}]\nQ: {r['question']}\nA: {r['answer']}" for i, r in enumerate(retrieved))
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content="You are TaxMind, an expert AI tax assistant. Answer using the provided context. Recommend consulting a CPA for specific advice."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "retrieved_sources": retrieved, "model": "gpt-4o-mini", "rag_enabled": True}

def answer_retrieval_only(query, top_k=1):
    results = retrieve(query, top_k=top_k)
    if not results: return {"answer": "No relevant answer found.", "retrieved_sources": []}
    best = results[0]
    return {"answer": best["answer"], "matched_question": best["question"],
            "similarity_score": best["similarity_score"], "retrieved_sources": results, "rag_enabled": False}

if __name__ == "__main__":
    KB_PATH = Path("data/synthetic/qa_knowledge_base.json")
    if KB_PATH.exists():
        qa_pairs = json.loads(KB_PATH.read_text())
        build_vector_store(qa_pairs)
