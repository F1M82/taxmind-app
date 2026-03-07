"""
services/qdrant_service.py
Replaces OpenSearch for vector search in TaxMind.
Uses Qdrant Cloud free tier for semantic search over KB.
"""
import os
from typing import List, Optional

QDRANT_URL        = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "taxmind_kb")
VECTOR_SIZE       = 1536  # OpenAI text-embedding-3-small dimensions


def _client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _embed(text: str) -> List[float]:
    from services.embedding_service import get_single_embedding
    return get_single_embedding(text)


def ensure_collection():
    from qdrant_client.models import Distance, VectorParams
    client = _client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'")
    return client


def index_documents(chunks: list):
    from qdrant_client.models import PointStruct
    client = ensure_collection()
    from services.embedding_service import get_embeddings

    texts  = [c["text"] for c in chunks]
    vectors = get_embeddings(texts)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        # Use integer ID derived from hash
        int_id = int(chunk["id"], 16) % (2**63)
        points.append(PointStruct(
            id=int_id,
            vector=vector,
            payload={
                "text":         chunk["text"],
                "source":       chunk["source"],
                "content_type": chunk["metadata"].get("content_type", "unknown"),
                "section":      chunk["metadata"].get("section_number") or chunk["metadata"].get("section", ""),
                "year":         chunk["metadata"].get("year", ""),
                "language":     chunk["metadata"].get("language", "en"),
                "metadata":     chunk["metadata"],
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} documents into Qdrant '{COLLECTION_NAME}'")
    return len(points)


def hybrid_search(query: str, top_k: int = 3) -> list:
    try:
        client = _client()
        query_vector = _embed(query)

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text":         r.payload.get("text", ""),
                "source":       r.payload.get("source", ""),
                "content_type": r.payload.get("content_type", ""),
                "section":      r.payload.get("section", ""),
                "score":        round(r.score, 4),
            }
            for r in results
        ]
    except Exception as e:
        return {"error": str(e)}


def collection_stats() -> dict:
    try:
        client = _client()
        info = client.get_collection(COLLECTION_NAME)
        return {
            "collection":    COLLECTION_NAME,
            "total_vectors": info.vectors_count,
            "status":        str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}


def delete_collection():
    client = _client()
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted collection '{COLLECTION_NAME}'")


JUDGMENTS_COLLECTION = os.getenv("QDRANT_JUDGMENTS_COLLECTION", "taxmind-judgments")


def search_judgments(query: str, top_k: int = 5, filters: dict = None) -> list:
    """Search case law corpus for litigation risk analysis."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = _client()

        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            if filters.get("section"):
                conditions.append(FieldCondition(key="sections", match=MatchValue(value=filters["section"])))
            if filters.get("outcome"):
                conditions.append(FieldCondition(key="outcome", match=MatchValue(value=filters["outcome"])))
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Scroll-based keyword search (no vectors needed)
        records, _ = client.scroll(
            collection_name=JUDGMENTS_COLLECTION,
            scroll_filter=qdrant_filter,
            limit=top_k * 3,
            with_payload=True,
            with_vectors=False,
        )

        # Score by keyword relevance
        query_words = set(query.lower().split())
        scored = []
        for r in records:
            p = r.payload
            if p.get("doc_type") == "risk_signal":
                continue
            text = " ".join([
                p.get("litigation_trigger", ""),
                p.get("key_facts", ""),
                p.get("ratio_decidendi", ""),
                " ".join(p.get("sections", [])),
            ]).lower()
            score = sum(1 for w in query_words if w in text)
            scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "judgment_id": p.get("judgment_id", ""),
                "court": p.get("court", ""),
                "bench": p.get("bench", ""),
                "outcome": p.get("outcome", ""),
                "risk_level": p.get("risk_level", ""),
                "litigation_trigger": p.get("litigation_trigger", ""),
                "ratio_decidendi": p.get("ratio_decidendi", ""),
                "winning_argument": p.get("winning_argument", ""),
                "mitigation_signals": p.get("mitigation_signals", []),
                "sections": p.get("sections", []),
                "source_url": p.get("source_url", ""),
                "score": s,
            }
            for s, p in scored[:top_k]
        ]
    except Exception as e:
        return [{"error": str(e)}]


GST_COLLECTION = os.getenv("QDRANT_GST_COLLECTION", "taxmind-gst")


def search_gst(query: str, top_k: int = 5) -> list:
    """Search GST corpus."""
    try:
        client = _client()
        records, _ = client.scroll(
            collection_name=GST_COLLECTION,
            limit=top_k * 3,
            with_payload=True,
            with_vectors=False,
        )
        query_words = set(query.lower().split())
        scored = []
        for r in records:
            p = r.payload
            text = " ".join([
                p.get("title", ""),
                p.get("ratio_decidendi", ""),
                p.get("key_facts", ""),
                " ".join(p.get("sections", [])),
            ]).lower()
            score = sum(1 for w in query_words if w in text)
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "judgment_id": p.get("judgment_id", ""),
                "content_type": p.get("content_type", ""),
                "title": p.get("title", ""),
                "court": p.get("court", ""),
                "bench": p.get("bench", ""),
                "ratio_decidendi": p.get("ratio_decidendi", ""),
                "sections": p.get("sections", []),
                "source_url": p.get("source_url", ""),
                "hsn": p.get("hsn", ""),
                "rate": p.get("rate", ""),
                "circular_number": p.get("circular_number", ""),
                "score": s,
            }
            for s, p in scored[:top_k]
        ]
    except Exception as e:
        return [{"error": str(e)}]