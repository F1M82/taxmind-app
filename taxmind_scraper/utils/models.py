from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date
from enum import Enum

class Court(str, Enum):
    ITAT = "ITAT"
    HIGH_COURT = "HC"
    SUPREME_COURT = "SC"

class Outcome(str, Enum):
    ASSESSEE_FAVORED = "assessee_favored"
    REVENUE_FAVORED = "revenue_favored"
    MIXED = "mixed"
    REMANDED = "remanded"

class TaxpayerType(str, Enum):
    INDIVIDUAL = "individual"
    COMPANY = "company"
    HUF = "huf"
    TRUST = "trust"
    FIRM = "firm"
    AOP = "aop"
    OTHER = "other"

class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RiskIndicator(BaseModel):
    trigger: str
    section: str
    notice_type: Optional[str] = None
    outcome: Outcome
    mitigation_note: Optional[str] = None

class JudgmentMetadata(BaseModel):
    judgment_id: str
    court: Court
    bench: Optional[str] = None
    judgment_date: Optional[date] = None
    assessment_years: List[str] = []
    sections: List[str] = []
    taxpayer_type: Optional[TaxpayerType] = None
    quantum_inr: Optional[int] = None
    source_url: Optional[str] = None
    source_site: Optional[str] = None
    pdf_path: Optional[str] = None

class JudgmentExtraction(BaseModel):
    judgment_id: str
    outcome: Outcome
    ratio_decidendi: str
    litigation_trigger: str
    key_facts: str
    winning_argument: str
    risk_level: RiskLevel
    risk_indicators: List[RiskIndicator] = []
    mitigation_signals: List[str] = []
    transaction_type: Optional[str] = None
    extraction_model: str = ""

class RiskSignal(BaseModel):
    signal_id: str
    section: str
    trigger: str
    total_cases: int = 0
    assessee_won: int = 0
    revenue_won: int = 0
    mixed: int = 0
    notice_probability: float = 0.0
    top_mitigation_strategies: List[str] = []
    supporting_judgment_ids: List[str] = []
    last_updated: Optional[str] = None
