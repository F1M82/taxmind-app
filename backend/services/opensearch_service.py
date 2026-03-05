"""services/opensearch_service.py - Hybrid BM25 search using OpenSearch (Railway-safe)"""
import os
from opensearchpy import OpenSearch

HOST  = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX = os.getenv("OPENSEARCH_INDEX", "taxmind-knowledge")

def _client():
    if HOST.startswith("https"):
        from opensearchpy import RequestsHttpConnection
        return OpenSearch(HOST, verify_certs=False, ssl_show_warn=False, connection_class=RequestsHttpConnection, use_ssl=True)
    return OpenSearch(HOST, verify_certs=False, ssl_show_warn=False)

def _embed(texts):
    from services.embedding_service import get_embeddings
    return get_embeddings(texts)

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

def index_documents(qa_pairs):
    c = _client()
    create_index()
    texts = [f"Q: {p['q']} A: {p['a']}" for p in qa_pairs]
    embeddings = _embed(texts)
    for i, (pair, emb) in enumerate(zip(qa_pairs, embeddings)):
        c.index(index=INDEX, body={"question": pair["q"], "answer": pair["a"], "embedding": emb},
                id=str(i), params={"refresh": "true"})

def hybrid_search(query, top_k=3):
    try:
        hits = _client().search(index=INDEX, body={
            "size": top_k,
            "query": {"multi_match": {"query": query, "fields": ["question^2", "answer"]}}
        })["hits"]["hits"]
        return [{"question": h["_source"]["question"], "answer": h["_source"]["answer"],
                 "score": round(h["_score"], 4)} for h in hits]
    except Exception as e:
        return {"error": str(e)}

def delete_index():
    _client().indices.delete(index=INDEX, params={"ignore_unavailable": "true"})


