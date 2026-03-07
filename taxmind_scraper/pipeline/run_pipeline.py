import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from datetime import date
from loguru import logger
from scrapers.itatonline_scraper import ITATOnlineScraper
from extractors.pdf_extractor import PDFExtractor
from extractors.claude_extractor import ClaudeExtractor
from pipeline.indexer import TaxMindIndexer
from pipeline.risk_aggregator import RiskAggregator
from utils.models import JudgmentMetadata
from config.settings import settings

logger.add(f"data/logs/pipeline_{date.today().isoformat()}.log", rotation="1 day", level="INFO")

class TaxMindPipeline:
    def __init__(self):
        self.itatonline_scraper = ITATOnlineScraper()
        self.pdf_extractor = PDFExtractor()
        self.claude_extractor = ClaudeExtractor()
        self.indexer = TaxMindIndexer()
        self.risk_aggregator = RiskAggregator(self.indexer)
        self.indexer.setup_indices()

    def process_single_judgment(self, metadata, editorial_context=None):
        extraction_text = editorial_context or ""
        if len(extraction_text) < 200:
            return False
        extraction = self.claude_extractor.extract(judgment_id=metadata.judgment_id, judgment_text=extraction_text)
        return self.indexer.index_judgment(metadata, extraction)

    def run_seed_corpus(self, max_pages=5):
        logger.info(f"=== SEED CORPUS: itatonline.org, {max_pages} pages ===")
        total = 0
        errors = 0
        for metadata, editorial_context in self.itatonline_scraper.scrape_paginated(max_pages=max_pages):
            if self.process_single_judgment(metadata, editorial_context):
                total += 1
                logger.info(f"[{total}] Indexed: {metadata.judgment_id}")
            else:
                errors += 1
        context_file = Path(settings.raw_data_dir) / "itatonline" / "editorial_context_EXTRACTION_ONLY.jsonl"
        if context_file.exists():
            context_file.unlink()
        if total > 0:
            self.risk_aggregator.rebuild_all()
        stats = self.indexer.get_stats()
        logger.success(f"=== DONE: {total} indexed, {errors} errors. Stats: {stats} ===")
        return total

    def test_connection(self):
        logger.info("Testing Qdrant connection...")
        try:
            stats = self.indexer.get_stats()
            logger.success(f"Connected! Stats: {stats}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "test"], default="test")
    parser.add_argument("--pages", type=int, default=5)
    args = parser.parse_args()
    pipeline = TaxMindPipeline()
    if args.mode == "test":
        pipeline.test_connection()
    elif args.mode == "seed":
        pipeline.run_seed_corpus(max_pages=args.pages)
