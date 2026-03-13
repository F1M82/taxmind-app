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

# ── SMART QUERY ROUTER ────────────────────────────────────────────────────────
# Routes queries to the most cost-effective model based on task type.
#
#  simple     → Gemini Flash   (definitions, FAQs, basic lookups)
#  legal      → Claude Sonnet  (section interpretation, risk analysis, judgments)
#  drafting   → Claude Haiku   (notices, replies, letters, formats)
#  document   → GPT-4o-mini    (classification, extraction, OCR structured output)

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
    if provider != "auto":
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

    if any(kw in msg_lower for kw in _DOC_KEYWORDS):
        return {"provider": "openai", "model": "gpt-4o-mini", "task_type": "document"}

    if any(kw in msg_lower for kw in _DRAFT_KEYWORDS):
        return {"provider": "claude", "model": "claude-haiku-4-5-20251001", "task_type": "drafting"}

    legal_phrases = ["high court","supreme court","capital gains","place of supply","input tax credit"]
    if any(ph in msg_lower for ph in legal_phrases) or \
       any(kw in tokens for kw in _LEGAL_KEYWORDS):
        return {"provider": "claude", "model": "claude-sonnet-4-6", "task_type": "legal"}

    return {"provider": "gemini", "model": "gemini-2.0-flash", "task_type": "simple"}

# ─────────────────────────────────────────────────────────────────────────────

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
        routing = classify_query(req.message, req.provider)
        routed_provider = routing["provider"]
        routed_model    = routing["model"]
        task_type       = routing["task_type"]

        if req.domain == "gst":
            from services.qdrant_service import search_gst
            from agent.llm_router import get_llm
            from langchain_core.messages import HumanMessage, SystemMessage
            results = search_gst(req.message, top_k=5)
            context = ""
            for i, r in enumerate(results, 1):
                context += f"\n[{i}] {r.get('title','')} ({r.get('content_type','')})\n{r.get('ratio_decidendi','')}\n"

            system = "You are TaxMind, an expert Indian GST assistant. Answer questions about the GST Act, rules, circulars and rates clearly and concisely. Always cite relevant sections. Use simple language suited for CA practitioners. Recommend consulting a CA for specific advice."
            prompt = f"GST Context:\n{context}\n\nQuestion: {req.message}"

            llm = get_llm("claude", temperature=0)
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=prompt)
            ])
            return {
                "success": True,
                "response": {
                    "answer":    response.content,
                    "domain":    "gst",
                    "sources":   len(results),
                    "cached":    False,
                    "task_type": task_type,
                    "model":     routed_model,
                }
            }

        else:
            from agent.langgraph_agent import run_langgraph_agent
            SUPPORTED_PROVIDERS = {"claude", "openai", "gemini", "gemini-lite", "groq"}
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
    """Debug endpoint – shows which model would handle a given query."""
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
        from agent.llm_router import get_llm
        from langchain_core.messages import HumanMessage

        judgments = search_judgments(req.query, top_k=req.top_k, filters={"section": req.section} if req.section else None)
        if not judgments or "error" in judgments[0]:
            return {"success": False, "error": "No judgments found"}

        cases_text = ""
        for i, j in enumerate(judgments, 1):
            cases_text += f"\nCase {i}: {j.get('judgment_id','')}\nOutcome: {j.get('outcome','')}\nTrigger: {j.get('litigation_trigger','')}\nRatio: {j.get('ratio_decidendi','')}\nWinning argument: {j.get('winning_argument','')}\nMitigation: {', '.join(j.get('mitigation_signals', []))}\n"

        prompt = f"A CA asks: {req.query}\n\nRelevant case law:\n{cases_text}\n\nProvide a concise litigation risk assessment: (1) Risk level, (2) Why this triggers scrutiny, (3) Key precedents, (4) Top 3 mitigation strategies."

        llm = get_llm("claude", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])

        return {
            "success": True,
            "risk_assessment": response.content,
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

# ── CLAUDE PROXY (frontend compatibility) ─────────────────────────────────────
# The frontend calls /api/claude directly with {model, max_tokens, system, messages}.
# This endpoint proxies through FallbackLLM so Groq kicks in when Anthropic credits run out.

class ClaudeProxyRequest(BaseModel):
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 2000
    system: str = ""
    messages: list = []

@app.post("/api/claude")
async def claude_proxy(req: ClaudeProxyRequest):
    try:
        from agent.llm_router import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        llm = get_llm("claude", temperature=0)

        lc_messages = []
        if req.system:
            lc_messages.append(SystemMessage(content=req.system))
        for m in req.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        result = llm.invoke(lc_messages)
        # Return in Anthropic-compatible format so frontend works unchanged
        return {"content": [{"type": "text", "text": result.content}]}

    except Exception as e:
        raise HTTPException(500, {"error": {"type": "api_error", "message": str(e)}})

# ── ANALYTICS: LITIGATION DASHBOARD ──────────────────────────────────────────
# Real schema from taxmind-judgments Qdrant collection (as of March 2026):
#   outcome:          assessee_favored | revenue_favored | mixed | remanded
#   risk_level:       high | medium | low
#   sections:         list[str]  – populated in ~50% of records
#   transaction_type: str        – populated in 100% of records

@app.get("/analytics/litigation")
def litigation_analytics():
    try:
        from qdrant_client import QdrantClient
        from collections import Counter
        import os

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        collection = os.getenv("QDRANT_COLLECTION", "taxmind-judgments")

        all_points = []
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        total = len(all_points)
        if total == 0:
            return {**_empty_lit_analytics(), "ok": True}

        payloads = [p.payload for p in all_points]

        # Outcome split
        OUTCOME_LABELS = {
            "assessee_favored": "Assessee Favored",
            "revenue_favored":  "Revenue Favored",
            "mixed":            "Mixed / Partial",
            "remanded":         "Remanded / Sent Back",
        }
        outcome_counter = Counter()
        for p in payloads:
            o = (p.get("outcome") or "mixed").lower().strip()
            outcome_counter[OUTCOME_LABELS.get(o, "Mixed / Partial")] += 1
        outcome_split = [{"label": k, "count": v} for k, v in outcome_counter.most_common()]

        # Risk split
        risk_counter = Counter()
        for p in payloads:
            r = (p.get("risk_level") or "medium").lower().strip()
            if r not in ("high", "medium", "low"):
                r = "medium"
            risk_counter[r] += 1
        risk_split = [{"label": k, "count": v} for k, v in risk_counter.most_common()]

        # Section risk
        sec_data: dict[str, dict] = {}
        for p in payloads:
            secs = p.get("sections") or []
            if isinstance(secs, str):
                secs = [secs]
            outcome = (p.get("outcome") or "mixed").lower().strip()
            for sec in secs:
                if not sec:
                    continue
                sec = sec.upper().replace(" ", "_")
                if sec not in sec_data:
                    sec_data[sec] = {"total": 0, "assessee_favored": 0, "revenue_favored": 0}
                sec_data[sec]["total"] += 1
                if outcome == "assessee_favored":
                    sec_data[sec]["assessee_favored"] += 1
                elif outcome == "revenue_favored":
                    sec_data[sec]["revenue_favored"] += 1

        section_risk = []
        for sec, d in sorted(sec_data.items(), key=lambda x: -x[1]["total"]):
            rev = d["revenue_favored"]
            tot = d["total"]
            risk_score = round((rev / tot) * 10) if tot else 0
            section_risk.append({
                "section":          sec,
                "total":            tot,
                "assessee_favored": d["assessee_favored"],
                "revenue_favored":  rev,
                "risk_score":       risk_score,
            })

        # Top transaction types
        tx_counter = Counter()
        for p in payloads:
            tx = p.get("transaction_type") or ""
            if tx:
                tx_counter[tx] += 1
        top_tx_types = [{"label": k, "count": v} for k, v in tx_counter.most_common(10)]

        return {
            "ok":           True,
            "total_cases":  total,
            "outcome_split": outcome_split,
            "risk_split":    risk_split,
            "section_risk":  section_risk,
            "top_tx_types":  top_tx_types,
        }

    except Exception as e:
        return {**_empty_lit_analytics(), "error": str(e)}

def _empty_lit_analytics():
    return {
        "ok": True, "total_cases": 0,
        "outcome_split": [], "risk_split": [],
        "section_risk": [], "top_tx_types": [],
    }

@app.get("/analytics/gst")
def gst_analytics():
    try:
        from qdrant_client import QdrantClient
        from collections import Counter
        import os

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        collection = os.getenv("QDRANT_GST_COLLECTION", "taxmind-gst")

        all_points = []
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        total = len(all_points)
        if total == 0:
            return {**_empty_gst_analytics(), "ok": True}

        payloads = [p.payload for p in all_points]

        content_type_counter = Counter()
        for p in payloads:
            ct = p.get("content_type") or "statute"
            content_type_counter[ct] += 1
        content_type_split = [{"label": k, "count": v} for k, v in content_type_counter.most_common()]

        act_counter = Counter()
        for p in payloads:
            act = p.get("act") or ""
            if "IGST" in act.upper():
                act_counter["IGST"] += 1
            elif "CGST" in act.upper():
                act_counter["CGST"] += 1
            else:
                act_counter["Other"] += 1
        act_split = [{"label": k, "count": v} for k, v in act_counter.most_common()]

        sec_counter = Counter()
        for p in payloads:
            sec = p.get("section") or p.get("section_number") or ""
            if sec:
                sec_counter[str(sec).upper().replace(" ", "_")] += 1
        top_sections = [{"section": k, "count": v} for k, v in sec_counter.most_common(20)]

        rate_counter = Counter()
        for p in payloads:
            rate = p.get("rate") or p.get("gst_rate") or ""
            if rate:
                rate_counter[str(rate)] += 1
        top_rates = [{"rate": k, "count": v} for k, v in rate_counter.most_common(10)]

        return {
            "ok":                True,
            "total_entries":     total,
            "content_type_split": content_type_split,
            "act_split":         act_split,
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