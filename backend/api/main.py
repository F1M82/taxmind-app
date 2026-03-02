"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
TaxMind FastAPI Application

Endpoints:
  POST /predict/tax           — Tax liability prediction
  POST /classify/document     — Document type classification
  POST /detect/anomaly        — Fraud / anomaly detection
  POST /qa/ask                — NLP tax Q&A (RAG or retrieval-only)
  POST /agent/chat            — Hybrid agent (chat + tool use)
  GET  /health                — Health check
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import tempfile

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TaxMind API",
    description="ML-powered tax intelligence: prediction, classification, fraud detection, and Q&A.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class TaxPredictionRequest(BaseModel):
    gross_income: float = Field(..., example=85000.0)
    investment_income: float = Field(0.0, example=3000.0)
    business_income: float = Field(0.0, example=0.0)
    total_deductions: float = Field(14600.0, example=14600.0)
    retirement_contributions: float = Field(0.0, example=6000.0)
    dependents: int = Field(0, example=1)
    age: int = Field(35, example=35)
    filing_status: str = Field("Single", example="Single")
    state: str = Field("CA", example="CA")


class DocumentTextRequest(BaseModel):
    text: str = Field(..., example="Form W-2 Wage and Tax Statement...")


class AnomalyRequest(BaseModel):
    gross_income: float
    investment_income: float = 0.0
    business_income: float = 0.0
    total_deductions: float
    retirement_contributions: float = 0.0
    dependents: int = 0
    taxable_income: float
    effective_tax_rate: float
    tax_liability: float = 0.0
    filing_status: str = "Single"
    state: str = "CA"


class QARequest(BaseModel):
    question: str = Field(..., example="What is the standard deduction for 2024?")
    use_rag: bool = Field(False, example=False,
                          description="Set True to use LLM-powered RAG (requires OPENAI_API_KEY)")
    top_k: int = Field(3, ge=1, le=10)


class AgentChatRequest(BaseModel):
    message: str = Field(..., example="Analyze my tax situation: I earn $120k, single filer, $15k deductions.")
    provider: str = Field(
        "openai",
        example="openai",
        description=(
            "LLM provider to use. Options: "
            "openai, openai-powerful, claude, claude-lite, "
            "gemini, gemini-lite, auto"
        ),
    )
    task: Optional[str] = Field(
        None,
        example="long_document",
        description=(
            "When provider='auto', routes to the best model for this task. "
            "Options: long_document, multimodal, cheap, fast, tool_use, "
            "structured_output, reasoning, legal, tax_analysis, general"
        ),
    )
    session_id: Optional[str] = Field(None)
    history: Optional[list] = Field(default_factory=list)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok", "service": "TaxMind API", "version": "1.0.0"}


# ─── 1. Tax Liability Prediction ─────────────────────────────────────────────

@app.post("/predict/tax", tags=["ML Models"])
def predict_tax(req: TaxPredictionRequest):
    """
    Predict federal tax liability from income and financial details.
    Uses XGBoost regression trained on IRS-style tax data.
    """
    try:
        from models.tax_predictor import predict
        result = predict(req.model_dump())
        return {"success": True, "input": req.model_dump(), "prediction": result}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Tax predictor model not trained yet. Run: python models/train_all.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 2. Document Classification ───────────────────────────────────────────────

@app.post("/classify/document", tags=["ML Models"])
def classify_document(req: DocumentTextRequest):
    """
    Classify a tax document from its text content.
    Returns predicted class (W2, 1099, invoice, receipt, tax_return) and confidence.
    """
    try:
        from models.doc_classifier import classify
        result = classify(req.text)
        return {"success": True, "classification": result}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Document classifier not trained yet. Run: python models/train_all.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/document/upload", tags=["ML Models"])
async def classify_document_upload(file: UploadFile = File(...)):
    """
    Upload a PDF and classify it automatically.
    Extracts text from the PDF then runs classification.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        from models.doc_classifier import classify, extract_text_from_pdf
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        text = extract_text_from_pdf(tmp_path)
        if not text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from PDF.")

        result = classify(text)
        return {
            "success": True,
            "filename": file.filename,
            "extracted_text_preview": text[:300] + "..." if len(text) > 300 else text,
            "classification": result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 3. Anomaly Detection ─────────────────────────────────────────────────────

@app.post("/detect/anomaly", tags=["ML Models"])
def detect_anomaly(req: AnomalyRequest):
    """
    Detect tax fraud / anomalies in a tax record.
    Uses ensemble of Isolation Forest + XGBoost with rule-based explanations.
    """
    try:
        from models.anomaly_detector import detect
        result = detect(req.model_dump())
        return {"success": True, "input": req.model_dump(), "detection": result}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Anomaly detector not trained yet. Run: python models/train_all.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 4. Q&A ───────────────────────────────────────────────────────────────────

@app.post("/qa/ask", tags=["ML Models"])
def ask_question(req: QARequest):
    """
    Answer a tax question using the RAG knowledge base.
    Set use_rag=True for LLM-powered answers (requires OPENAI_API_KEY).
    Set use_rag=False for fast retrieval-only answers.
    """
    try:
        if req.use_rag:
            if not os.getenv("OPENAI_API_KEY"):
                raise HTTPException(
                    status_code=400,
                    detail="OPENAI_API_KEY not set. Either set it or use use_rag=false."
                )
            from models.tax_qa import answer_with_rag
            result = answer_with_rag(req.question, top_k=req.top_k)
        else:
            from models.tax_qa import answer_retrieval_only
            result = answer_retrieval_only(req.question, top_k=req.top_k)

        return {"success": True, "question": req.question, "response": result}

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Q&A vector store not built yet. Run: python models/train_all.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 5. Agent ─────────────────────────────────────────────────────────────────

@app.post("/agent/chat", tags=["Agent"])
def agent_chat(req: AgentChatRequest):
    """
    Chat with the TaxMind hybrid agent.
    The agent can answer questions AND autonomously invoke ML tools.
    """
    required_keys = {
        "openai": "OPENAI_API_KEY", "openai-powerful": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY", "claude-lite": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY", "gemini-lite": "GOOGLE_API_KEY",
    }
    needed_key = required_keys.get(req.provider)
    if needed_key and not os.getenv(needed_key):
        raise HTTPException(
            status_code=400,
            detail=f"{needed_key} is required for provider '{req.provider}'. Add it to your .env file."
        )
    try:
        from agent.taxmind_agent import run_agent
        result = run_agent(
            message=req.message,
            history=req.history or [],
            provider=req.provider,
            task=req.task,
        )
        return {"success": True, "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("APP_ENV", "development") == "development",
    )
