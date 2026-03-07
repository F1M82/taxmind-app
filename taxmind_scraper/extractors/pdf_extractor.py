import re
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

IT_ACT_SECTIONS = re.compile(r"[Ss]ection\s+(\d+[A-Za-z]*(?:\([a-z0-9]+\))*)", re.IGNORECASE)
AY_PATTERN = re.compile(r"(?:assessment\s+year|A\.?Y\.?)\s+(\d{4}[-]\d{2,4})", re.IGNORECASE)

class PDFExtractor:
    def extract_text(self, pdf_path: str) -> Optional[str]:
        path = Path(pdf_path)
        if not path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return None
        if PDFPLUMBER_AVAILABLE:
            try:
                text = self._extract_pdfplumber(pdf_path)
                if text and len(text) > 100:
                    return text
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        if PYPDF_AVAILABLE:
            try:
                text = self._extract_pypdf(pdf_path)
                if text and len(text) > 100:
                    return text
            except Exception as e:
                logger.warning(f"pypdf failed: {e}")
        return None

    def _extract_pdfplumber(self, pdf_path: str) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)

    def _extract_pypdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    def extract_quick_metadata(self, text: str) -> dict:
        if not text:
            return {}
        section_matches = IT_ACT_SECTIONS.findall(text)
        sections = list(set(f"section_{m.lower()}" for m in section_matches))
        ay_matches = AY_PATTERN.findall(text)
        assessment_years = list(set(ay_matches))
        return {
            "sections": sections,
            "assessment_years": assessment_years,
        }

    def get_extraction_chunk(self, text: str, max_chars: int = 8000) -> str:
        if len(text) <= max_chars:
            return text
        head = text[:5000]
        tail = text[-3000:]
        return f"{head}\n\n[... middle omitted ...]\n\n{tail}"
