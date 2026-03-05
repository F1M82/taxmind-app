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