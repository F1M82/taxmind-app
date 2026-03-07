import re
import json
import hashlib
from pathlib import Path
from typing import Iterator, Optional
from loguru import logger
from bs4 import BeautifulSoup
from utils.http_client import EthicalHttpClient
from utils.models import JudgmentMetadata, Court
from config.settings import settings

ITATONLINE_DIGEST = "https://itatonline.org/digest/all-judgements/"
SECTION_PATTERN = re.compile(r"[Ss]\.?\s*(\d+[A-Za-z]*(?:\([a-z0-9]+\))*)", re.IGNORECASE)

class ITATOnlineScraper:
    def __init__(self):
        self.client = EthicalHttpClient(
            user_agent=settings.user_agent,
            min_delay=settings.request_delay_seconds,
            max_per_hour=settings.max_requests_per_hour,
            respect_robots=settings.respect_robots_txt,
        )
        self.raw_dir = Path(settings.raw_data_dir) / "itatonline"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.raw_dir / "metadata.jsonl"
        self.context_file = self.raw_dir / "editorial_context_EXTRACTION_ONLY.jsonl"

    def generate_judgment_id(self, citation: str) -> str:
        hash_val = hashlib.md5(citation.encode()).hexdigest()[:10].upper()
        return f"ITATONLINE-{hash_val}"

    def _extract_sections(self, text: str) -> list:
        matches = SECTION_PATTERN.findall(text)
        return list(set(f"section_{m.lower()}" for m in matches))

    def _detect_court(self, text: str) -> Court:
        text_lower = text.lower()
        if "high court" in text_lower or "(hc)" in text_lower or "(bom)" in text_lower or "(del)" in text_lower:
            return Court.HIGH_COURT
        if "supreme court" in text_lower or "(sc)" in text_lower:
            return Court.SUPREME_COURT
        return Court.ITAT

    def scrape_page(self, page_url: str) -> Iterator[tuple]:
        response = self.client.get(page_url)
        if not response:
            return
        soup = BeautifulSoup(response.text, "lxml")
        entries = soup.find_all("div", class_=re.compile(r"post|entry|judgment|case", re.I))
        if not entries:
            entries = soup.find_all("article")
        logger.info(f"Found {len(entries)} entries on {page_url}")
        for entry in entries:
            try:
                title_elem = entry.find(["h2", "h3", "strong", "b"])
                title = title_elem.get_text(strip=True) if title_elem else ""
                full_text = entry.get_text(separator=" ", strip=True)
                sections = self._extract_sections(full_text)
                court = self._detect_court(full_text)
                source_url = None
                for link in entry.find_all("a", href=True):
                    href = link["href"]
                    if "itatonline.org" in href or href.endswith(".pdf"):
                        source_url = href
                        break
                if not title and not source_url:
                    continue
                citation_key = title or source_url or full_text[:100]
                judgment_id = self.generate_judgment_id(citation_key)
                metadata = JudgmentMetadata(
                    judgment_id=judgment_id,
                    court=court,
                    sections=sections,
                    source_url=source_url,
                    source_site="itatonline",
                )
                with open(self.metadata_file, "a") as f:
                    f.write(metadata.model_dump_json() + "\n")
                context_record = {"judgment_id": judgment_id, "editorial_context": full_text, "NOTE": "EXTRACTION USE ONLY"}
                with open(self.context_file, "a") as f:
                    f.write(json.dumps(context_record) + "\n")
                yield metadata, full_text
            except Exception as e:
                logger.error(f"Error parsing entry: {e}")

    def scrape_paginated(self, max_pages: int = 50) -> Iterator[tuple]:
        for page_num in range(1, max_pages + 1):
            url = ITATONLINE_DIGEST if page_num == 1 else f"{ITATONLINE_DIGEST}page/{page_num}/"
            logger.info(f"Scraping page {page_num}: {url}")
            count = 0
            for result in self.scrape_page(url):
                count += 1
                yield result
            if count == 0:
                logger.info(f"No entries on page {page_num} - stopping")
                break
            logger.info(f"Page {page_num}: {count} judgments")
