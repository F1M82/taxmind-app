import json
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import anthropic
from utils.models import JudgmentExtraction, RiskIndicator, Outcome, RiskLevel
from config.settings import settings

SYSTEM_PROMPT = "You are a tax litigation intelligence analyst. Analyze Indian Income Tax judgments. Write ALL text in YOUR OWN WORDS. Return ONLY valid JSON, no markdown."

def to_str(val):
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val) if val else ""

class ClaudeExtractor:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.extraction_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def extract(self, judgment_id, judgment_text, editorial_context=None):
        if not judgment_text or len(judgment_text) < 200:
            return None
        prompt = "Analyze this Indian tax judgment and return JSON with these exact keys: outcome (assessee_favored or revenue_favored or mixed or remanded), ratio_decidendi (string, 2 sentences), litigation_trigger (string), key_facts (string), winning_argument (string), risk_level (high or medium or low), transaction_type (string), risk_indicators (list of objects with trigger/section/notice_type/outcome/mitigation_note), mitigation_signals (list of 3 strings)\n\n" + judgment_text[:6000]
        try:
            logger.info(f"Extracting: {judgment_id}")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            data = json.loads(raw)
            risk_indicators = []
            for ri in data.get("risk_indicators", []):
                try:
                    risk_indicators.append(RiskIndicator(
                        trigger=to_str(ri.get("trigger", "")),
                        section=to_str(ri.get("section", "")),
                        notice_type=ri.get("notice_type"),
                        outcome=Outcome(ri.get("outcome", "mixed")),
                        mitigation_note=to_str(ri.get("mitigation_note", ""))
                    ))
                except Exception:
                    pass
            extraction = JudgmentExtraction(
                judgment_id=judgment_id,
                outcome=Outcome(data.get("outcome", "mixed")),
                ratio_decidendi=to_str(data.get("ratio_decidendi", "")),
                litigation_trigger=to_str(data.get("litigation_trigger", "")),
                key_facts=to_str(data.get("key_facts", "")),
                winning_argument=to_str(data.get("winning_argument", "")),
                risk_level=RiskLevel(data.get("risk_level", "medium")),
                transaction_type=to_str(data.get("transaction_type", "")),
                risk_indicators=risk_indicators,
                mitigation_signals=[to_str(s) for s in data.get("mitigation_signals", [])],
                extraction_model=self.model
            )
            logger.success(f"Extracted: {judgment_id} -> {extraction.outcome}")
            return extraction
        except Exception as e:
            logger.error(f"Failed {judgment_id}: {e}")
            raise