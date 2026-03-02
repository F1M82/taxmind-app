"""services/opensearch_service.py — Hybrid BM25 search using OpenSearch"""
import os
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

HOST       = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
INDEX      = "taxmind-knowledge"
MODEL_NAME = "all-MiniLM-L6-v2"

def _client(): return OpenSearch(HOST, verify_certs=False, ssl_show_warn=False)
def _model():  return SentenceTransformer(MODEL_NAME)

def create_index():
    c = _client()
    if c.indices.exists(index=INDEX): return
    c.indices.create(index=INDEX, body={
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {
            "question": {"type": "text", "analyzer": "english"},
            "answer":   {"type": "text", "analyzer": "english"},
        }}
    })
    print(f"✅ Created index '{INDEX}'")

def index_documents(qa_pairs):
    c = _client()
    create_index()
    embeddings = _model().encode([f"Q: {p['q']} A: {p['a']}" for p in qa_pairs], normalize_embeddings=True)
    for i, (pair, emb) in enumerate(zip(qa_pairs, embeddings)):
        c.index(index=INDEX, body={"question": pair["q"], "answer": pair["a"],
                "embedding": emb.tolist()}, id=str(i), params={"refresh": "true"})
    print(f"✅ Indexed {len(qa_pairs)} documents")

def hybrid_search(query, top_k=3):
    hits = _client().search(index=INDEX, body={
        "size": top_k,
        "query": {"multi_match": {"query": query, "fields": ["question^2", "answer"]}}
    })["hits"]["hits"]
    return [{"question": h["_source"]["question"], "answer": h["_source"]["answer"],
             "score": round(h["_score"], 4)} for h in hits]

def delete_index():
    _client().indices.delete(index=INDEX, params={"ignore_unavailable": "true"})
    print(f"✅ Deleted index '{INDEX}'")
