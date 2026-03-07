import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

load_dotenv("config/.env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
SOURCE_COLLECTION = "taxmind-judgments"
GST_COLLECTION = "taxmind-gst"

def migrate():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Create taxmind-gst collection
    existing = [c.name for c in client.get_collections().collections]
    if GST_COLLECTION not in existing:
        client.create_collection(
            collection_name=GST_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
        logger.success(f"Created collection: {GST_COLLECTION}")
    else:
        logger.info(f"Collection exists: {GST_COLLECTION}")

    # Scroll ALL docs from taxmind-judgments
    all_records = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        all_records.extend(records)
        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"Total records in taxmind-judgments: {len(all_records)}")

    # Filter GST docs in Python
    gst_records = [r for r in all_records if r.payload.get("court") == "GST"]
    it_records = [r for r in all_records if r.payload.get("court") != "GST"]

    logger.info(f"GST docs to migrate: {len(gst_records)}")
    logger.info(f"IT docs to keep: {len(it_records)}")

    if not gst_records:
        logger.warning("No GST documents found")
        return

    # Copy GST docs to taxmind-gst
    gst_points = [PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in gst_records]
    client.upsert(collection_name=GST_COLLECTION, points=gst_points)
    logger.success(f"Migrated {len(gst_points)} GST docs to {GST_COLLECTION}")

    # Rebuild taxmind-judgments with only IT docs
    client.delete_collection(SOURCE_COLLECTION)
    client.create_collection(
        collection_name=SOURCE_COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    if it_records:
        it_points = [PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in it_records]
        client.upsert(collection_name=SOURCE_COLLECTION, points=it_points)
    logger.success(f"Rebuilt {SOURCE_COLLECTION} with {len(it_records)} IT docs")

    it_stats = client.get_collection(SOURCE_COLLECTION).points_count
    gst_stats = client.get_collection(GST_COLLECTION).points_count
    logger.success(f"Done! taxmind-judgments: {it_stats} | taxmind-gst: {gst_stats}")

if __name__ == "__main__":
    migrate()
