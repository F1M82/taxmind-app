from datetime import datetime, timezone
from collections import defaultdict
from loguru import logger
from utils.models import RiskSignal
from config.settings import settings

class RiskAggregator:
    def __init__(self, indexer):
        self.indexer = indexer
        self.client = indexer.client
        self.collection = settings.qdrant_collection

    def rebuild_all(self):
        logger.info("Rebuilding risk signals...")
        try:
            signal_map = defaultdict(lambda: {"assessee_won": 0, "revenue_won": 0, "mixed": 0, "mitigation_notes": [], "judgment_ids": []})
            records, _ = self.client.scroll(collection_name=self.collection, limit=1000, with_payload=True, with_vectors=False)
            total = 0
            for record in records:
                doc = record.payload
                if doc.get("doc_type") == "risk_signal":
                    continue
                judgment_id = doc.get("judgment_id", "")
                outcome = doc.get("outcome", "mixed")
                for ri in doc.get("risk_indicators", []):
                    section = ri.get("section", "unknown")
                    trigger = ri.get("trigger", "unknown")
                    key = f"{section}__{trigger}"
                    signal_map[key]["judgment_ids"].append(judgment_id)
                    if outcome == "assessee_favored":
                        signal_map[key]["assessee_won"] += 1
                    elif outcome == "revenue_favored":
                        signal_map[key]["revenue_won"] += 1
                    else:
                        signal_map[key]["mixed"] += 1
                    if ri.get("mitigation_note"):
                        signal_map[key]["mitigation_notes"].append(ri["mitigation_note"])
                total += 1
            written = 0
            for key, data in signal_map.items():
                section, trigger = key.split("__", 1)
                total_cases = data["assessee_won"] + data["revenue_won"] + data["mixed"]
                if total_cases < 2:
                    continue
                signal = RiskSignal(
                    signal_id=key,
                    section=section,
                    trigger=trigger,
                    total_cases=total_cases,
                    assessee_won=data["assessee_won"],
                    revenue_won=data["revenue_won"],
                    mixed=data["mixed"],
                    notice_probability=round(data["revenue_won"] / total_cases, 3),
                    top_mitigation_strategies=list(set(data["mitigation_notes"]))[:3],
                    supporting_judgment_ids=list(set(data["judgment_ids"]))[:20],
                )
                self.indexer.update_risk_signal(signal)
                written += 1
            logger.success(f"Risk signals rebuilt: {written} signals from {total} judgments")
            return written
        except Exception as e:
            logger.error(f"Risk aggregation failed: {e}")
            return 0
