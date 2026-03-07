import hashlib
from pathlib import Path
from datetime import date, timedelta
from typing import Iterator, Optional
from loguru import logger
from bs4 import BeautifulSoup
from utils.http_client import EthicalHttpClient
from utils.models import JudgmentMetadata, Court
from config.settings import settings

PRIORITY_BENCHES = ["Mumbai", "Delhi", "Ahmedabad", "Chennai", "Bangalore", "Pune", "Nagpur"]
ITAT_BASE_URL = "https://itat.nic.in"

class ITATScraper:
    def __init__(self):
        self.client = EthicalHttpClient(
            user_agent=settings.user_agent,
            min_delay=settings.request_delay_seconds,
            max_per_hour=settings.max_requests_per_hour,
            respect_robots=settings.respect_robots_txt,
        )
        self.raw_dir = Path(settings.raw_data_dir) / "itat_gov"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.raw_dir / "metadata.jsonl"

    def generate_judgment_id(self, bench: str, order_date: str, filename: str) -> str:
        key = f"ITAT-{bench.upper()}-{order_date}-{filename}"
        hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8].upper()
        return f"ITAT-{bench[:3].upper()}-{order_date.replace('-', '')}-{hash_suffix}"

    def scrape_date_range(self, bench: str, start_date: date, end_date: date) -> Iterator[JudgmentMetadata]:
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                date_str = current.strftime("%d%m%Y")
                url = f"{ITAT_BASE_URL}/Judicial/PronounceList/{bench}/{date_str}"
                response = self.client.get(url)
                if response:
                    soup = BeautifulSoup(response.text, "lxml")
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        if href.endswith(".pdf"):
                            pdf_url = href if href.startswith("http") else f"{ITAT_BASE_URL}{href}"
                            judgment_id = self.generate_judgment_id(bench, current.isoformat(), Path(href).name)
                            pdf_path = self.raw_dir / f"{judgment_id}.pdf"
                            if not pdf_path.exists():
                                self.client.get_pdf(pdf_url, str(pdf_path))
                            metadata = JudgmentMetadata(
                                judgment_id=judgment_id,
                                court=Court.ITAT,
                                bench=bench,
                                judgment_date=current,
                                source_url=pdf_url,
                                source_site="itat_gov",
                                pdf_path=str(pdf_path),
                            )
                            with open(self.metadata_file, "a") as f:
                                f.write(metadata.model_dump_json() + "\n")
                            yield metadata
            current += timedelta(days=1)

    def scrape_seed_corpus(self, days_back: int = 365) -> int:
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        total = 0
        for bench in PRIORITY_BENCHES:
            logger.info(f"Scraping bench: {bench}")
            for metadata in self.scrape_date_range(bench, start_date, end_date):
                total += 1
                logger.info(f"[{total}] Scraped: {metadata.judgment_id}")
        logger.success(f"Seed corpus complete: {total} judgments")
        return total
