"""
services/qdrant_service.py
Semantic search over TaxMind Qdrant collections.
"""
import os
from typing import List, Optional

QDRANT_URL       = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME  = os.getenv("QDRANT_COLLECTION", "taxmind_kb")
VECTOR_SIZE      = 1536

JUDGMENTS_COLLECTION = os.getenv("QDRANT_JUDGMENTS_COLLECTION", "taxmind-judgments")
GST_COLLECTION       = os.getenv("QDRANT_GST_COLLECTION", "taxmind-gst")


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

    texts   = [c["text"] for c in chunks]
    vectors = get_embeddings(texts)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
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
    """Semantic search over taxmind_kb collection."""
    try:
        client = _client()
        query_vector = _embed(query)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
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


def search_judgments(query: str, top_k: int = 5, filters: dict = None) -> list:
    """Semantic search over IT judgment corpus."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = _client()
        query_vector = _embed(query)

        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            if filters.get("section"):
                conditions.append(FieldCondition(
                    key="sections",
                    match=MatchValue(value=filters["section"])
                ))
            if filters.get("outcome"):
                conditions.append(FieldCondition(
                    key="outcome",
                    match=MatchValue(value=filters["outcome"])
                ))
            if conditions:
                from qdrant_client.models import Filter
                qdrant_filter = Filter(must=conditions)

        results = client.query_points(
            collection_name=JUDGMENTS_COLLECTION,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {
                "judgment_id":       r.payload.get("judgment_id", ""),
                "court":             r.payload.get("court", ""),
                "bench":             r.payload.get("bench", ""),
                "outcome":           r.payload.get("outcome", ""),
                "risk_level":        r.payload.get("risk_level", ""),
                "litigation_trigger":r.payload.get("litigation_trigger", ""),
                "ratio_decidendi":   r.payload.get("ratio_decidendi", ""),
                "winning_argument":  r.payload.get("winning_argument", ""),
                "mitigation_signals":r.payload.get("mitigation_signals", []),
                "sections":          r.payload.get("sections", []),
                "source_url":        r.payload.get("source_url", ""),
                "score":             round(r.score, 4),
            }
            for r in results
            if r.payload.get("doc_type") != "risk_signal"
        ]
    except Exception as e:
        return [{"error": str(e)}]


def search_gst(query: str, top_k: int = 5) -> list:
    """Semantic search over GST corpus."""
    try:
        client = _client()
        query_vector = _embed(query)

        results = client.query_points(
            collection_name=GST_COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {
                "judgment_id":    r.payload.get("judgment_id", ""),
                "content_type":   r.payload.get("content_type", ""),
                "title":          r.payload.get("title", ""),
                "court":          r.payload.get("court", ""),
                "outcome":        r.payload.get("outcome", ""),
                "ratio_decidendi":r.payload.get("ratio_decidendi", ""),
                "sections":       r.payload.get("sections", []),
                "source_url":     r.payload.get("source_url", ""),
                "hsn":            r.payload.get("hsn", ""),
                "rate":           r.payload.get("rate", ""),
                "circular_number":r.payload.get("circular_number", ""),
                "score":          round(r.score, 4),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


def collection_stats() -> dict:
    try:
        client = _client()
        stats = {}
        for col in [COLLECTION_NAME, JUDGMENTS_COLLECTION, GST_COLLECTION]:
            try:
                info = client.get_collection(col)
                stats[col] = {
                    "points": info.points_count,
                    "status": str(info.status),
                }