"""api/main.py - TaxMind FastAPI backend"""
import os
import traceback
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import tempfile

app = FastAPI(title="TaxMind API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def serve_frontend():
    path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    return FileResponse(path)

class TaxPredictRequest(BaseModel):
    gross_income: float = Field(..., example=85000.0)
    investment_income: float = 0.0
    business_income: float = 0.0
    total_deductions: float = 14600.0
    retirement_contributions: float = 0.0
    dependents: int = 0
    age: int = 35
    filing_status: str = "Single"
    state: str = "CA"

class DocTextRequest(BaseModel):
    text: str

class AnomalyRequest(BaseModel):
    gross_income: float
    total_deductions: float
    taxable_income: float
    effective_tax_rate: float
    tax_liability: float = 0.0
    investment_income: float = 0.0
    business_income: float = 0.0
    filing_status: str = "Single"
    state: str = "CA"
    dependents: int = 0
    retirement_contributions: float = 0.0

class QARequest(BaseModel):
    question: str
    use_rag: bool = False
    top_k: int = Field(3, ge=1, le=10)

class AgentRequest(BaseModel):
    message: str
    provider: str = Field("auto", description="openai|claude|gemini|gemini-lite|auto")
    task: Optional[str] = None
    history: Optional[list] = []
    use_langgraph: bool = True
    domain: str = Field("it", description="it|gst")

# ── SMART QUERY ROUTER ─────────────────────────────────────────────────────────
# Routes queries to the most cost-effective model based on task type.
#
#  simple     → Gemini Flash   (definitions, FAQs, basic lookups)
#  legal      → Claude Sonnet  (section interpretation, risk analysis, judgments)
#  drafting   → Claude Haiku   (notices, replies, letters, formats)
#  document   → GPT-4o-mini    (classification, extraction, OCR structured output)
#
# Provider string matches what langgraph_agent.run_langgraph_agent() expects.

_LEGAL_KEYWORDS = {
    "section","provision","interpret","applicable","liable","liability",
    "circular","notification","judgment","itat","tribunal",
    "analysis","legal","analyse","analyze","risk","penalty","appeal",
    "reassessment","addition","disallowance","exemption","deduction",
    "taxable","reversal","applicability","validity","quash","whether",
}
_DRAFT_KEYWORDS = {
    "draft","write","prepare","format","notice","reply","letter","response",
    "application","submission","ground","objection","appeal memo","written",
    "compose","create a letter","create a notice",
}
_DOC_KEYWORDS = {
    "classify","extract","identify","type of","what kind","document","ocr",
    "uploaded","this notice","this document","attached","pdf",
}

def classify_query(message: str, provider: str) -> dict:
    """
    Returns {provider, model, task_type} for a given message.
    If provider is not 'auto', it is passed through unchanged.
    """
    if provider != "auto":
        # Manual override — map provider string to a concrete model
        MODEL_MAP = {
            "claude":       ("claude", "claude-sonnet-4-6",        "manual"),
            "claude-haiku": ("claude", "claude-haiku-4-5-20251001","manual"),
            "openai":       ("openai", "gpt-4o-mini",              "manual"),
            "gemini":       ("gemini", "gemini-2.0-flash",         "manual"),
            "gemini-lite":  ("gemini", "gemini-2.0-flash",         "manual"),
        }
        p, m, t = MODEL_MAP.get(provider, ("claude","claude-sonnet-4-6","manual"))
        return {"provider": p, "model": m, "task_type": t}

    tokens = set(message.lower().split())
    msg_lower = message.lower()

    # Check document classification first (usually short + explicit)
    if any(kw in msg_lower for kw in _DOC_KEYWORDS):
        return {"provider": "openai", "model": "gpt-4o-mini", "task_type": "document"}

    # Check drafting
    if any(kw in msg_lower for kw in _DRAFT_KEYWORDS):
        return {"provider": "claude", "model": "claude-haiku-4-5-20251001", "task_type": "drafting"}

    # Check legal reasoning (multi-word phrases + single tokens)
    legal_phrases = ["high court","supreme court","capital gains","place of supply","input tax credit"]
    if any(ph in msg_lower for ph in legal_phrases) or \
       any(kw in tokens for kw in _LEGAL_KEYWORDS):
        return {"provider": "claude", "model": "claude-sonnet-4-6", "task_type": "legal"}

    # Default: simple query → Gemini Flash
    return {"provider": "gemini", "model": "gemini-2.0-flash", "task_type": "simple"}
# ──────────────────────────────────────────────────────────────────────────────

class LitigationRiskRequest(BaseModel):
    query: str
    section: Optional[str] = None
    top_k: int = Field(5, ge=1, le=10)

@app.get("/health")
def health():
    from services.cache_service import cache_stats
    return {"status": "ok", "version": "2.0.0", "cache": cache_stats()}

@app.post("/predict/tax")
def predict_tax(req: TaxPredictRequest):
    try:
        from models.tax_predictor import predict
        return {"success": True, "prediction": predict(req.model_dump())}
    except FileNotFoundError: raise HTTPException(503, "Model not trained.")
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/classify/document")
def classify_doc(req: DocTextRequest):
    try:
        from models.doc_classifier import classify
        return {"success": True, "classification": classify(req.text)}
    except FileNotFoundError: raise HTTPException(503, "Model not trained.")
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/classify/document/upload")
async def classify_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"): raise HTTPException(400, "Only PDF files supported.")
    try:
        from models.doc_classifier import classify, extract_text_from_pdf
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await file.read()); tmp_path = tmp.name
        text = extract_text_from_pdf(tmp_path)
        if not text.strip(): raise HTTPException(422, "No text extracted.")
        return {"success": True, "filename": file.filename, "classification": classify(text)}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/detect/anomaly")
def detect_anomaly(req: AnomalyRequest):
    try:
        from models.anomaly_detector import detect
        return {"success": True, "detection": detect(req.model_dump())}
    except FileNotFoundError: raise HTTPException(503, "Model not trained.")
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/qa/ask")
def ask(req: QARequest):
    try:
        if req.use_rag:
            from models.tax_qa import answer_with_rag
            return {"success": True, "response": answer_with_rag(req.question, req.top_k)}
        from models.tax_qa import answer_retrieval_only
        return {"success": True, "response": answer_retrieval_only(req.question, req.top_k)}
    except FileNotFoundError: raise HTTPException(503, "Vector store not built.")
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/agent/chat")
def agent_chat(req: AgentRequest):
    try:
        # ── Route to best model ──────────────────────────────────────────────
        routing = classify_query(req.message, req.provider)
        routed_provider = routing["provider"]
        routed_model    = routing["model"]
        task_type       = routing["task_type"]

        if req.domain == "gst":
            # GST domain: always use Qdrant context + Claude Haiku for drafting,
            # Claude Sonnet for legal reasoning, Gemini Flash for simple lookups.
            # Override: if task_type is legal → Sonnet, else Haiku (cost saving).
            from services.qdrant_service import search_gst
            import anthropic
            results = search_gst(req.message, top_k=5)
            context = ""
            for i, r in enumerate(results, 1):
                context += f"\n[{i}] {r.get('title','')} ({r.get('content_type','')})\n{r.get('ratio_decidendi','')}\n"

            gst_model = "claude-sonnet-4-6" if task_type == "legal" else "claude-haiku-4-5-20251001"
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            system = "You are TaxMind, an expert Indian GST assistant. Answer questions about the GST Act, rules, circulars and rates clearly and concisely. Always cite relevant sections. Use simple language suited for CA practitioners. Recommend consulting a CA for specific advice."
            prompt = f"GST Context:\n{context}\n\nQuestion: {req.message}"
            response = client.messages.create(
                model=gst_model,
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                "success": True,
                "response": {
                    "answer":    response.content[0].text,
                    "domain":    "gst",
                    "sources":   len(results),
                    "cached":    False,
                    "task_type": task_type,
                    "model":     gst_model,
                }
            }

        else:
            # IT domain: pass routed_provider to langgraph agent
            # If provider not supported by agent, fall back to Claude Haiku
            from agent.langgraph_agent import run_langgraph_agent
            SUPPORTED_PROVIDERS = {"claude", "openai"}
            actual_provider = routed_provider if routed_provider in SUPPORTED_PROVIDERS else "claude"
            if actual_provider != routed_provider:
                routed_model = "claude-haiku-4-5-20251001"
                task_type    = task_type + "_fallback"
            result = run_langgraph_agent(req.message, provider=actual_provider, history=req.history)
            from services.monitoring import log_agent_run
            log_agent_run(req.message, result, actual_provider)
            return {
                "success": True,
                "response": {
                    **result,
                    "domain":    "it",
                    "task_type": task_type,
                    "model":     routed_model,
                }
            }

    except Exception as e:
        raise HTTPException(500, detail=traceback.format_exc())

@app.get("/agent/route")
def debug_route(q: str, provider: str = "auto"):
    """Debug endpoint — shows which model would handle a given query."""
    return classify_query(q, provider)


def get_cache_stats():
    from services.cache_service import cache_stats
    return cache_stats()

@app.delete("/cache/flush")
def flush_cache():
    from services.cache_service import flush_cache
    return flush_cache()

@app.get("/search")
def search(q: str, top_k: int = 3):
    try:
        from services.qdrant_service import hybrid_search
        return {"success": True, "results": hybrid_search(q, top_k)}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/litigation/risk")
def litigation_risk(req: LitigationRiskRequest):
    try:
        from services.qdrant_service import search_judgments
        import anthropic
        judgments = search_judgments(req.query, top_k=req.top_k, filters={"section": req.section} if req.section else None)
        if not judgments or "error" in judgments[0]:
            return {"success": False, "error": "No judgments found"}
        cases_text = ""
        for i, j in enumerate(judgments, 1):
            cases_text += f"\nCase {i}: {j.get('judgment_id','')}\nOutcome: {j.get('outcome','')}\nTrigger: {j.get('litigation_trigger','')}\nRatio: {j.get('ratio_decidendi','')}\nWinning argument: {j.get('winning_argument','')}\nMitigation: {', '.join(j.get('mitigation_signals', []))}\n"
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        prompt = f"A CA asks: {req.query}\n\nRelevant case law:\n{cases_text}\n\nProvide a concise litigation risk assessment: (1) Risk level, (2) Why this triggers scrutiny, (3) Key precedents, (4) Top 3 mitigation strategies."
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "success": True,
            "risk_assessment": response.content[0].text,
            "supporting_cases": judgments,
            "cases_analyzed": len(judgments)
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/gst/search")
def gst_search(q: str, top_k: int = 5):
    try:
        from services.qdrant_service import search_gst
        return {"success": True, "results": search_gst(q, top_k)}
    except Exception as e: raise HTTPException(500, str(e))

# ── ANALYTICS: LITIGATION DASHBOARD ───────────────────────────────────────────
# Real schema from taxmind-judgments Qdrant collection (as of March 2026):
#   outcome:          assessee_favored | revenue_favored | mixed | remanded
#   risk_level:       high | medium | low
#   sections:         list[str]  — populated in ~50% of records
#   transaction_type: str        — populated in 100% of records
#   bench/court:      mostly empty — not used for aggregation yet
#   judgment_date:    empty       — not used
#
# "remanded" maps to our "Remanded / Sent Back" category (quashed / no proper
#  grounds cases will be a subset here once the scraper tags them explicitly).

@app.get("/analytics/litigation")
def litigation_analytics():
    """
    Aggregate judgment data from Qdrant for the litigation risk dashboard.
    Adapted to real itatonline scraper schema.
    Falls back to a structured empty response if Qdrant is unavailable.
    """
    try:
        from qdrant_client import QdrantClient
        from collections import Counter
        import os

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        collection = os.getenv("QDRANT_COLLECTION", "taxmind-judgments")

        # Scroll all points (paginate for collections > 1000)
        all_points = []
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=collection,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            return _empty_analytics()

        # ── Outcome normalisation
        # Maps raw outcome strings → display labels
        OUTCOME_MAP = {
            "assessee_favored": "Assessee Favored",
            "revenue_favored":  "Revenue Favored",
            "mixed":            "Mixed / Partial",
            "remanded":         "Remanded / Sent Back",
        }

        # ── Counters
        outcome_counts    = Counter()
        risk_counts       = Counter()   # high / medium / low
        section_disputes  = Counter()
        section_outcomes  = {}          # section → Counter of outcome labels
        tx_type_counts    = Counter()   # transaction_type → total

        for pt in all_points:
            p = pt.payload or {}

            raw_outcome  = (p.get("outcome") or "unknown").lower().strip()
            display_out  = OUTCOME_MAP.get(raw_outcome, raw_outcome.replace("_", " ").title())
            risk_level   = (p.get("risk_level") or "unknown").lower().strip()
            tx_type      = (p.get("transaction_type") or "").strip()

            outcome_counts[display_out] += 1
            risk_counts[risk_level]     += 1

            # Transaction type (truncate long strings)
            if tx_type:
                label = tx_type if len(tx_type) <= 55 else tx_type[:52] + "…"
                tx_type_counts[label] += 1

            # Sections
            raw_secs = p.get("sections") or []
            if isinstance(raw_secs, str):
                raw_secs = [s.strip() for s in raw_secs.split(",") if s.strip()]

            for sec in raw_secs:
                sec = sec.strip().upper()
                if not sec:
                    continue
                section_disputes[sec] += 1
                if sec not in section_outcomes:
                    section_outcomes[sec] = Counter()
                section_outcomes[sec][display_out] += 1

        total = len(all_points)

        # ── Section risk table (top 20 by volume)
        section_risk = []
        for sec, tot in section_disputes.most_common(20):
            oc = section_outcomes.get(sec, Counter())
            rev_w = oc.get("Revenue Favored", 0)
            # Risk score = % revenue-favored (higher = riskier for taxpayer)
            risk_pct = round(rev_w / tot * 100) if tot else 0
            section_risk.append({
                "section":          sec,
                "total":            tot,
                "assessee_favored": oc.get("Assessee Favored", 0),
                "revenue_favored":  rev_w,
                "mixed":            oc.get("Mixed / Partial", 0),
                "remanded":         oc.get("Remanded / Sent Back", 0),
                "risk_score":       risk_pct,
            })

        # ── Top transaction types (top 15)
        top_tx_types = [
            {"name": name, "count": count}
            for name, count in tx_type_counts.most_common(15)
        ]

        return {
            "ok":            True,
            "total_cases":   total,
            "outcome_split": [{"label": k, "count": v} for k, v in outcome_counts.items()],
            "risk_split":    [{"label": k, "count": v} for k, v in risk_counts.items()],
            "section_risk":  section_risk,
            "top_tx_types":  top_tx_types,
        }

    except Exception as e:
        return {**_empty_analytics(), "error": str(e)}


def _empty_analytics():
    return {
        "ok":            True,
        "total_cases":   0,
        "outcome_split": [],
        "risk_split":    [],
        "section_risk":  [],
        "top_tx_types":  [],
    }


@app.get("/analytics/gst")
def gst_analytics():
    """
    Aggregate GST knowledge base data from taxmind-gst Qdrant collection.
    Shows content_type breakdown, act breakdown (CGST/IGST), top sections,
    and rate schedule summary.
    """
    try:
        from qdrant_client import QdrantClient
        from collections import Counter
        import os

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        collection = "taxmind-gst"

        all_points = []
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=collection,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            return _empty_gst_analytics()

        content_type_counts = Counter()
        act_counts          = Counter()   # CGST / IGST / RATE_SCHEDULE
        section_counts      = Counter()
        rate_rows           = []          # {hsn, title, rate}

        for pt in all_points:
            p = pt.payload or {}

            ct    = (p.get("content_type") or "unknown").strip()
            bench = (p.get("bench") or "unknown").strip().upper()
            title = (p.get("title") or "").strip()
            hsn   = (p.get("hsn") or "").strip()
            rate  = (p.get("rate") or "").strip()

            content_type_counts[ct]    += 1
            act_counts[bench]          += 1

            # Sections
            raw_secs = p.get("sections") or []
            if isinstance(raw_secs, str):
                raw_secs = [s.strip() for s in raw_secs.split(",") if s.strip()]
            for sec in raw_secs:
                sec = sec.strip().upper()
                if sec:
                    section_counts[sec] += 1

            # Rate schedule rows
            if ct == "rate_schedule" and hsn and rate:
                rate_rows.append({"hsn": hsn, "title": title[:60], "rate": rate})

        total = len(all_points)

        # Top 20 sections
        top_sections = [
            {"section": sec, "count": cnt}
            for sec, cnt in section_counts.most_common(20)
        ]

        # Rate schedule: deduplicate by hsn, keep first occurrence, top 20
        seen_hsn = set()
        top_rates = []
        for r in rate_rows:
            if r["hsn"] not in seen_hsn:
                seen_hsn.add(r["hsn"])
                top_rates.append(r)
            if len(top_rates) >= 20:
                break

        return {
            "ok":                True,
            "total_entries":     total,
            "content_type_split":[{"label": k, "count": v} for k, v in content_type_counts.items()],
            "act_split":         [{"label": k, "count": v} for k, v in act_counts.items()],
            "top_sections":      top_sections,
            "top_rates":         top_rates,
        }

    except Exception as e:
        return {**_empty_gst_analytics(), "error": str(e)}


def _empty_gst_analytics():
    return {
        "ok":                True,
        "total_entries":     0,
        "content_type_split":[],
        "act_split":         [],
        "top_sections":      [],
        "top_rates":         [],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)),
                reload=os.getenv("APP_ENV") == "development")