"""api/main.py — TaxMind FastAPI backend"""
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import tempfile

app = FastAPI(title="TaxMind API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Request models ────────────────────────────────────────────────────────────

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
    provider: str = Field("claude", description="openai|claude|gemini|gemini-lite|auto")
    task: Optional[str] = None
    history: Optional[list] = []
    use_langgraph: bool = True

# ── Endpoints ─────────────────────────────────────────────────────────────────

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
    except Exception as e:    raise HTTPException(500, str(e))

@app.post("/classify/document")
def classify_doc(req: DocTextRequest):
    try:
        from models.doc_classifier import classify
        return {"success": True, "classification": classify(req.text)}
    except FileNotFoundError: raise HTTPException(503, "Model not trained.")
    except Exception as e:    raise HTTPException(500, str(e))

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
    except Exception as e:    raise HTTPException(500, str(e))

@app.post("/qa/ask")
def ask(req: QARequest):
    try:
        if req.use_rag:
            from models.tax_qa import answer_with_rag
            return {"success": True, "response": answer_with_rag(req.question, req.top_k)}
        from models.tax_qa import answer_retrieval_only
        return {"success": True, "response": answer_retrieval_only(req.question, req.top_k)}
    except FileNotFoundError: raise HTTPException(503, "Vector store not built.")
    except Exception as e:    raise HTTPException(500, str(e))

@app.post("/agent/chat")
def agent_chat(req: AgentRequest):
    try:
        if req.use_langgraph:
            from agent.langgraph_agent import run_langgraph_agent
            result = run_langgraph_agent(req.message, provider=req.provider, history=req.history)
        else:
            from agent.taxmind_agent import run_agent
            result = run_agent(req.message, req.history, req.provider, req.task)
        from services.monitoring import log_agent_run
        log_agent_run(req.message, result, req.provider)
        return {"success": True, "response": result}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/cache/stats")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)),
                reload=os.getenv("APP_ENV") == "development")
