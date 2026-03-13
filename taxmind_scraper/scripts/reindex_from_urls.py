# taxmind_scraper/scripts/reindex_from_urls.py
"""
Re-scrape and re-index all judgments from saved URLs in metadata.jsonl.
Skips already-indexed judgment_ids.
"""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv("backend/.env")

from pathlib import Path
from loguru import logger
from datetime import date
from bs4 import BeautifulSoup
from utils.http_client import EthicalHttpClient
from utils.models import JudgmentMetadata, Court
from extractors.claude_extractor import ClaudeExtractor
from pipeline.indexer import TaxMindIndexer
from config.settings import settings

logger.add(f"taxmind_scraper/data/logs/reindex_{date.today().isoformat()}.log", rotation="1 day")

METADATA_FILE = Path("taxmind_scraper/data/raw/itatonline/metadata.jsonl")

def get_unique_entries():
    seen = {}
    with open(METADATA_FILE) as f:
        for line in f:
            d = json.loads(line)
            jid = d["judgment_id"]
            if jid not in seen:
                seen[jid] = d
    return list(seen.values())

def get_indexed_ids(indexer: TaxMindIndexer) -> set:
    indexed = set()
    offset = None
    while True:
        results, next_offset = indexer.client.scroll(
            collection_name=indexer.collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for p in results:
            jid = p.payload.get("judgment_id")
            if jid:
                indexed.add(jid)
        if next_offset is None:
            break
        offset = next_offset
    return indexed

def fetch_editorial_text(client, url: str) -> str:
    response = client.get(url)
    if not response:
        return ""
    soup = BeautifulSoup(response.text, "lxml")
    for selector in ["article", "div.post", "div.entry", "div.content", "main"]:
        elem = soup.select_one(selector)
        if elem:
            text = elem.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return text
    return soup.get_text(separator=" ", strip=True)

def main():
    entries = get_unique_entries()
    logger.info(f"Total unique judgments in metadata.jsonl: {len(entries)}")

    indexer = TaxMindIndexer()
    indexer.setup_indices()

    indexed_ids = get_indexed_ids(indexer)
    logger.info(f"Already indexed: {len(indexed_ids)}")

    to_process = [e for e in entries if e["judgment_id"] not in indexed_ids]
    logger.info(f"To process: {len(to_process)}")

    extractor = ClaudeExtractor()
    http = EthicalHttpClient(
        user_agent=settings.user_agent,
        min_delay=settings.request_delay_seconds,
        max_per_hour=settings.max_requests_per_hour,
        respect_robots=settings.respect_robots_txt,
    )

    success = 0
    errors = 0

    for i, entry in enumerate(to_process):
        jid = entry["judgment_id"]
        url = entry.get("source_url")
        if not url:
            logger.warning(f"[{i+1}] No URL for {jid}, skipping")
            errors += 1
            continue

        try:
            text = fetch_editorial_text(http, url)
            if len(text) < 200:
                logger.warning(f"[{i+1}] Too short ({len(text)} chars): {jid}")
                errors += 1
                continue

            metadata = JudgmentMetadata(**{
                k: v for k, v in entry.items()
                if k in JudgmentMetadata.model_fields
            })

            extraction = extractor.extract(judgment_id=jid, judgment_text=text)
            if indexer.index_judgment(metadata, extraction):
                success += 1
                logger.info(f"[{i+1}/{len(to_process)}] ✓ {jid}")
            else:
                errors += 1
                logger.warning(f"[{i+1}] Index failed: {jid}")

        except Exception as e:
            errors += 1
            logger.error(f"[{i+1}] Error on {jid}: {e}")

    logger.success(f"Done: {success} indexed, {errors} errors")

if __name__ == "__main__":
    main()