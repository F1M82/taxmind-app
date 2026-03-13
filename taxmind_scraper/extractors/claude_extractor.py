import json
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import Groq
from utils.models import JudgmentExtraction, RiskIndicator, Outcome, RiskLevel
from config.settings import settings
import os

SYSTEM_PROMPT = "You are a tax litigation intelligence analyst. Analyze Indian Income Tax judgments. Write ALL text in YOUR OWN WORDS. Return ONLY valid JSON, no markdown."

VALID_OUTCOMES = {e.value for e in Outcome}
VALID_RISK_LEVELS = {e.value for e in RiskLevel}

def to_str(val):
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val) if val else ""

class ClaudeExtractor:
    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY") or settings.groq_api_key if hasattr(settings, 'groq_api_key') else os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=groq_key)
        self.model = "llama-3.3-70b-versatile"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def extract(self, judgment_id, judgment_text, editorial_context=None):
        if not judgment_text or len(judgment_text) < 200:
            return None

        prompt = (
            "Analyze this Indian tax judgment and return JSON with these exact keys: "
            "outcome (assessee_favored or revenue_favored or mixed or remanded), "
            "ratio_decidendi (string, 2 sentences), "
            "litigation_trigger (string), "
            "key_facts (string), "
            "winning_argument (string), "
            "risk_level (high or medium or low), "
            "transaction_type (string), "
            "risk_indicators (list of objects with trigger/section/notice_type/outcome/mitigation_note), "
            "mitigation_signals (list of 3 strings)\n\n"
            + judgment_text[:6000]
        )

        try:
            logger.info(f"Extracting: {judgment_id}")
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            raw = response.choices[0].message.content.strip()

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
                        outcome=Outcome(ri.get("outcome") if ri.get("outcome") in VALID_OUTCOMES else "mixed"),
                        mitigation_note=to_str(ri.get("mitigation_note", ""))
                    ))
                except Exception:
                    pass

            extraction = JudgmentExtraction(
                judgment_id=judgment_id,
                outcome=Outcome(data.get("outcome") if data.get("outcome") in VALID_OUTCOMES else "mixed"),
                ratio_decidendi=to_str(data.get("ratio_decidendi", "")),
                litigation_trigger=to_str(data.get("litigation_trigger", "")),
                key_facts=to_str(data.get("key_facts", "")),
                winning_argument=to_str(data.get("winning_argument", "")),
                risk_level=RiskLevel(data.get("risk_level") if data.get("risk_level") in VALID_RISK_LEVELS else "medium"),
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