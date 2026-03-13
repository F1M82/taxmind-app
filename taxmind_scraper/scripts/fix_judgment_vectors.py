# taxmind_scraper/scripts/fix_judgment_vectors.py
"""
Scroll all judgments from taxmind-judgments, generate real embeddings,
recreate collection with dim=1536, re-upsert.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv("backend/.env")

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
from loguru import logger

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "taxmind-judgments")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai = OpenAI(api_key=OPENAI_API_KEY)

def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [r.embedding for r in resp.data]

def make_embed_text(payload: dict) -> str:
    parts = []
    if payload.get("ratio_decidendi"):
        parts.append(payload["ratio_decidendi"])
    if payload.get("litigation_trigger"):
        parts.append(payload["litigation_trigger"])
    if payload.get("transaction_type"):
        parts.append(payload["transaction_type"])
    if payload.get("sections"):
        parts.append(" ".join(payload["sections"]))
    return " | ".join(parts) if parts else payload.get("judgment_id", "unknown")

# Step 1: Scroll all points
logger.info("Scrolling all points from Qdrant...")
all_points = []
offset = None
while True:
    result, next_offset = qdrant.scroll(
        collection_name=COLLECTION,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    all_points.extend(result)
    if next_offset is None:
        break
    offset = next_offset

logger.info(f"Scrolled {len(all_points)} points total")

# Step 2: Filter IT judgments only (exclude GST/risk_signal leakage)
it_points = [
    p for p in all_points
    if p.payload.get("doc_type") != "risk_signal"
    and p.payload.get("court") in ("ITAT", "HIGH_COURT", None)
    and "CGST" not in str(p.payload.get("judgment_id", ""))
    and "IGST" not in str(p.payload.get("judgment_id", ""))
]
logger.info(f"IT judgments (after filtering): {len(it_points)}")

# Step 3: Recreate collection with dim=1536
logger.info("Recreating collection with dim=1536...")
qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
logger.success("Collection recreated")

# Step 4: Embed and upsert in batches of 50
BATCH = 50
total = 0
for i in range(0, len(it_points), BATCH):
    batch = it_points[i:i+BATCH]
    texts = [make_embed_text(p.payload) for p in batch]
    vectors = embed_batch(texts)
    
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=p.id, vector=v, payload=p.payload)
            for p, v in zip(batch, vectors)
        ]
    )
    total += len(batch)
    logger.info(f"Upserted {total}/{len(it_points)}")
    time.sleep(0.3)  # rate limit buffer

logger.success(f"Done! {total} IT judgments re-indexed with dim=1536 vectors")