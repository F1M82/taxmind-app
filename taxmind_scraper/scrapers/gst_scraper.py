"""
GST Act Scraper — CBIC Official PDFs
Downloads CGST Act 2017 and IGST Act 2017 PDFs from cbic-gst.gov.in,
extracts section text using pdfplumber, and outputs review queue JSON
consistent with the taxmind_scraper pipeline.

Usage:
    python -m scrapers.gst_scraper
    python -m scrapers.gst_scraper CGST       # single act
"""

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import pdfplumber
from loguru import logger

from config.settings import settings
from utils.http_client import EthicalHttpClient

# ─── Source PDFs ──────────────────────────────────────────────────────────────
ACT_SOURCES = [
    {
        "act": "CGST",
        "full_name": "Central Goods and Services Tax Act, 2017",
        "pdf_url": "https://cbic-gst.gov.in/pdf/CGST-Act-Updated-31082021.pdf",
        "year": "2017",
    },
    {
        "act": "IGST",
        "full_name": "Integrated Goods and Services Tax Act, 2017",
        "pdf_url": "https://cbic-gst.gov.in/pdf/IGST-Act-Updated-31082021.pdf",
        "year": "2017",
    },
]

# ─── Regex Patterns ───────────────────────────────────────────────────────────

# Matches: "2. Definitions." / "16A. Eligibility..." / "SECTION 9—Levy"
SECTION_START = re.compile(
    r"^(\d{1,3}[A-Z]?)\.\s{1,4}([A-Z][^\n]{3,})",
    re.MULTILINE,
)

# Cross-references like "section 16" or "section 17(5)"
SECTION_REF = re.compile(
    r"[Ss]ection\s+(\d+[A-Za-z]*(?:\([a-z0-9]+\))*)"
)

# Footnote noise to strip: superscript numbers, amendment notes
FOOTNOTE_NOISE = re.compile(
    r"\n\s*\d+\s+(Omitted|Substituted|Inserted|Amended)[^\n]{0,300}",
    re.IGNORECASE,
)

MAX_CHUNK_CHARS = 1500


class GSTScraper:
    """
    Downloads official CBIC PDFs for CGST and IGST Acts, parses section text,
    and emits review queue JSON for manual approval before Qdrant indexing.

    Follows ITATOnlineScraper conventions:
      - EthicalHttpClient for rate-limited downloads
      - Raw PDFs saved to raw_dir for reproducibility
      - Review queue JSON: data/raw/review_queue/gst_act_<act>_YYYYMMDD.json
    """

    def __init__(self):
        self.client = EthicalHttpClient(
            user_agent=settings.user_agent,
            min_delay=settings.request_delay_seconds,
            max_per_hour=settings.max_requests_per_hour,
            respect_robots=settings.respect_robots_txt,
        )
        self.raw_dir = Path(settings.raw_data_dir) / "gst_act"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.review_queue_dir = Path(settings.raw_data_dir) / "review_queue"
        self.review_queue_dir.mkdir(parents=True, exist_ok=True)

    # ─── ID & Metadata Helpers ────────────────────────────────────────────────

    def generate_id(self, act: str, section_number: str, chunk_index: int = 0) -> str:
        key = f"GST-ACT-{act}-S{section_number}-C{chunk_index}"
        hash_val = hashlib.md5(key.encode()).hexdigest()[:10].upper()
        return f"GST-ACT-{act}-{hash_val}"

    def _extract_section_refs(self, text: str) -> list[str]:
        matches = SECTION_REF.findall(text)
        return list(set(f"section_{m.lower()}" for m in matches))

    # ─── PDF Download ─────────────────────────────────────────────────────────

    def _download_pdf(self, act_meta: dict) -> Optional[Path]:
        """
        Download PDF from CBIC. Returns local path, or None on failure.
        Skips download if file already exists (cache).
        """
        act = act_meta["act"]
        pdf_path = self.raw_dir / f"{act.lower()}_act.pdf"

        if pdf_path.exists():
            logger.info(f"[{act}] Using cached PDF: {pdf_path}")
            return pdf_path

        logger.info(f"[{act}] Downloading PDF from {act_meta['pdf_url']}")
        response = self.client.get(act_meta["pdf_url"])
        if not response:
            logger.error(f"[{act}] PDF download failed")
            return None

        pdf_path.write_bytes(response.content)
        logger.info(f"[{act}] PDF saved → {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
        return pdf_path

    # ─── PDF Text Extraction ──────────────────────────────────────────────────

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract all text from PDF using pdfplumber."""
        logger.info(f"Extracting text from {pdf_path.name}")
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        full_text = "\n".join(pages)
        logger.info(f"Extracted {len(full_text):,} chars from {len(pages)} pages")
        return full_text

    # ─── Text Cleaning ────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Strip footnotes, fix hyphenated line breaks, normalize whitespace."""
        # Remove amendment footnotes
        text = FOOTNOTE_NOISE.sub("", text)
        # Fix hyphenated line-breaks (PDF artifact): "suppli-\ned" → "supplied"
        text = re.sub(r"-\n(\S)", r"\1", text)
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ─── Section Parsing ─────────────────────────────────────────────────────

    def _parse_sections(self, text: str, act_meta: dict) -> Iterator[dict]:
        """
        Split cleaned text into section dicts.
        Yields one dict per section (before chunking).
        """
        matches = list(SECTION_START.finditer(text))
        if not matches:
            logger.warning(f"[{act_meta['act']}] No sections found — check PDF quality")
            return

        logger.info(f"[{act_meta['act']}] Found {len(matches)} section boundaries")

        for i, match in enumerate(matches):
            section_number = match.group(1).strip()
            section_title = match.group(2).strip().rstrip(".")

            # Body text = everything until next section start
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body_text = text[body_start:body_end].strip()

            # Skip very short sections (likely parse artifacts)
            if len(body_text) < 30:
                logger.debug(f"[{act_meta['act']}] Skipping short section {section_number}")
                continue

            yield {
                "section_number": section_number,
                "section_title": section_title,
                "body_text": body_text,
            }

    # ─── Chunking ─────────────────────────────────────────────────────────────

    def _chunk_section(
        self,
        section: dict,
        act_meta: dict,
    ) -> list[dict]:
        """
        Split long sections at sub-section boundaries (1), (2), (a) etc.
        Short sections become a single chunk.
        """
        section_number = section["section_number"]
        section_title = section["section_title"]
        body_text = section["body_text"]
        act = act_meta["act"]
        source_url = act_meta["pdf_url"]

        # Split on sub-section markers
        parts = re.split(r"(?=\(\d+\)|\([a-z]\)\s)", body_text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            parts = [body_text]

        # Group parts into ≤ MAX_CHUNK_CHARS chunks
        chunks = []
        current = ""
        chunk_index = 0

        for part in parts:
            if len(current) + len(part) > MAX_CHUNK_CHARS and current:
                chunks.append(self._build_doc(
                    act, section_number, section_title,
                    current.strip(), chunk_index, source_url
                ))
                chunk_index += 1
                current = part
            else:
                current += " " + part

        if current.strip():
            chunks.append(self._build_doc(
                act, section_number, section_title,
                current.strip(), chunk_index, source_url
            ))

        # Patch total_chunks now we know the count
        for c in chunks:
            c["total_chunks"] = len(chunks)

        return chunks

    def _build_doc(
        self,
        act: str,
        section_number: str,
        section_title: str,
        body_text: str,
        chunk_index: int,
        source_url: str,
    ) -> dict:
        """
        Build a single document dict.
        Schema matches existing gst_scraper.py + Qdrant indexer expectations.
        """
        doc_id = self.generate_id(act, section_number, chunk_index)
        section_refs = self._extract_section_refs(body_text)
        own_tag = f"section_{section_number.lower()}"
        if own_tag not in section_refs:
            section_refs.insert(0, own_tag)

        full_content = (
            f"Section {section_number} {act} Act 2017 — {section_title}\n\n{body_text}"
        )

        return {
            # Core identity — matches Qdrant schema
            "judgment_id": doc_id,
            "court": "GST",
            "bench": act,
            "sections": section_refs,
            "source_url": source_url,
            "source_site": "cbic_gst",
            "content_type": "statute",
            # Content
            "title": f"Section {section_number} — {section_title}",
            "content": full_content,
            # GST-specific metadata
            "act": act,
            "section_number": section_number,
            "section_title": section_title,
            "chunk_index": chunk_index,
            "total_chunks": None,   # patched after all chunks known
            "scraped_date": datetime.utcnow().date().isoformat(),
        }

    # ─── Review Queue Output ──────────────────────────────────────────────────

    def _write_review_queue(self, chunks: list[dict], act: str) -> Path:
        date_str = datetime.utcnow().strftime("%Y%m%d")
        out_file = self.review_queue_dir / f"gst_act_{act.lower()}_{date_str}.json"

        payload = {
            "source": f"GSTScraper — {act} Act 2017 (CBIC PDF)",
            "scraped_at": datetime.utcnow().isoformat(),
            "total_chunks": len(chunks),
            "status": "pending_review",
            "chunks": chunks,
        }

        out_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Review queue → {out_file} ({len(chunks)} chunks)")
        return out_file

    # ─── Per-Act Pipeline ─────────────────────────────────────────────────────

    def scrape_act(self, act_meta: dict) -> list[dict]:
        act = act_meta["act"]

        # 1. Download PDF
        pdf_path = self._download_pdf(act_meta)
        if not pdf_path:
            return []

        # 2. Extract text
        raw_text = self._extract_text_from_pdf(pdf_path)

        # 3. Clean
        clean_text = self._clean_text(raw_text)

        # 4. Parse sections
        all_chunks = []
        section_count = 0
        for section in self._parse_sections(clean_text, act_meta):
            chunks = self._chunk_section(section, act_meta)
            all_chunks.extend(chunks)
            section_count += 1

        logger.info(f"[{act}] {section_count} sections → {len(all_chunks)} chunks")
        return all_chunks

    # ─── Compatibility: get_all() for existing pipeline callers ──────────────

    def get_all(self) -> Iterator[dict]:
        """
        Drop-in replacement for the old hardcoded get_all().
        Scrapes both acts and yields chunks one by one.
        Also writes review queue files as a side effect.
        """
        for act_meta in ACT_SOURCES:
            chunks = self.scrape_act(act_meta)
            if chunks:
                self._write_review_queue(chunks, act_meta["act"])
                yield from chunks

    # ─── Main Entry Point ─────────────────────────────────────────────────────

    def run(self, acts: Optional[list[str]] = None) -> dict[str, Path]:
        sources = ACT_SOURCES
        if acts:
            sources = [s for s in ACT_SOURCES if s["act"] in acts]

        results = {}
        for act_meta in sources:
            chunks = self.scrape_act(act_meta)
            if chunks:
                out_path = self._write_review_queue(chunks, act_meta["act"])
                results[act_meta["act"]] = out_path
            else:
                logger.warning(f"[{act_meta['act']}] No chunks — check PDF and logs")

        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    acts_filter = sys.argv[1:] if len(sys.argv) > 1 else None
    scraper = GSTScraper()
    output_files = scraper.run(acts=acts_filter)

    print("\n── Review Queue Files ──")
    for act, path in output_files.items():
        print(f"  {act}: {path}")
    print("\nReview chunks, then index to Qdrant.")