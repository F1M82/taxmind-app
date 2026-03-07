from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from utils.models import JudgmentMetadata, JudgmentExtraction, RiskSignal
import os
from dotenv import load_dotenv

load_dotenv("config/.env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "taxmind-judgments")

class TaxMindIndexer:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection = QDRANT_COLLECTION

    def setup_indices(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
            logger.success(f"Created collection: {self.collection}")
        else:
            logger.info(f"Collection exists: {self.collection}")

    def index_judgment(self, metadata: JudgmentMetadata, extraction: Optional[JudgmentExtraction] = None) -> bool:
        try:
            import hashlib
            id_hash = int(hashlib.md5(metadata.judgment_id.encode()).hexdigest()[:8], 16)
            payload = {
                "judgment_id": metadata.judgment_id,
                "court": metadata.court.value if metadata.court else None,
                "bench": metadata.bench,
                "judgment_date": metadata.judgment_date.isoformat() if metadata.judgment_date else None,
                "assessment_years": metadata.assessment_years,
                "sections": metadata.sections,
                "source_url": metadata.source_url,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            if extraction:
                payload.update({
                    "outcome": extraction.outcome.value,
                    "risk_level": extraction.risk_level.value,
                    "transaction_type": extraction.transaction_type,
                    "ratio_decidendi": extraction.ratio_decidendi,
                    "litigation_trigger": extraction.litigation_trigger,
                    "key_facts": extraction.key_facts,
                    "winning_argument": extraction.winning_argument,
                    "mitigation_signals": extraction.mitigation_signals,
                    "risk_indicators": [{"trigger": ri.trigger, "section": ri.section, "notice_type": ri.notice_type, "outcome": ri.outcome.value, "mitigation_note": ri.mitigation_note} for ri in extraction.risk_indicators],
                    "extraction_model": extraction.extraction_model,
                })
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=id_hash, vector=[0.0], payload=payload)]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index {metadata.judgment_id}: {e}")
            return False

    def update_risk_signal(self, signal: RiskSignal) -> bool:
        try:
            import hashlib
            id_hash = int(hashlib.md5(signal.signal_id.encode()).hexdigest()[:8], 16)
            payload = signal.model_dump()
            payload["last_updated"] = datetime.now(timezone.utc).isoformat()
            payload["doc_type"] = "risk_signal"
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=id_hash, vector=[0.0], payload=payload)]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update risk signal: {e}")
            return False

    def get_stats(self) -> dict:
        try:
            info = self.client.get_collection(self.collection)
            return {self.collection: info.points_count}
        except Exception as e:
            return {self.collection: f"error: {e}"}
