import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from scrapers.gst_scraper import GSTScraper
from pipeline.indexer import TaxMindIndexer
from config.settings import settings

logger.add("data/logs/gst_pipeline.log", rotation="1 day", level="INFO")


def run_gst_pipeline():
    logger.info("=== GST PIPELINE STARTING ===")
    scraper = GSTScraper()
    indexer = TaxMindIndexer()
    indexer.setup_indices()

    total = 0
    errors = 0

    for doc in scraper.get_all():
        try:
            from qdrant_client.models import PointStruct
            import hashlib
            id_hash = int(hashlib.md5(doc["judgment_id"].encode()).hexdigest()[:8], 16)
            payload = {
                "judgment_id": doc["judgment_id"],
                "court": doc["court"],
                "bench": doc.get("bench", ""),
                "sections": doc.get("sections", []),
                "source_url": doc.get("source_url", ""),
                "source_site": doc.get("source_site", ""),
                "content_type": doc.get("content_type", ""),
                "title": doc.get("title", ""),
                "litigation_trigger": doc.get("title", ""),
                "ratio_decidendi": doc.get("content", ""),
                "key_facts": doc.get("content", ""),
                "winning_argument": "",
                "outcome": "statute",
                "risk_level": "medium",
                "mitigation_signals": [],
                "risk_indicators": [],
                "hsn": doc.get("hsn", ""),
                "rate": doc.get("rate", ""),
                "circular_number": doc.get("circular_number", ""),
            }
            indexer.client.upsert(
                collection_name=settings.qdrant_collection,
                points=[PointStruct(id=id_hash, vector=[0.0], payload=payload)]
            )
            total += 1
            logger.info(f"[{total}] Indexed: {doc['judgment_id']} - {doc.get('title', '')}")
        except Exception as e:
            errors += 1
            logger.error(f"Failed {doc['judgment_id']}: {e}")

    stats = indexer.get_stats()
    logger.success(f"=== GST DONE: {total} indexed, {errors} errors. Stats: {stats} ===")
    return total


if __name__ == "__main__":
    run_gst_pipeline()