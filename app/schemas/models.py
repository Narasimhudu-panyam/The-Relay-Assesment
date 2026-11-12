"""
Pydantic schemas for The Relay pipeline, API endpoints, model calls, and CSV telemetry.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class Citation(BaseModel):
    paper_id: str
    paper_title: str
    paper_version: str
    page_number: int
    section: str
    chunk_id: Optional[str] = None


class StageTiming(BaseModel):
    stage: str
    duration_ms: float
    details: Dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0


class ModelCallMetadata(BaseModel):
    model: str
    source: Literal["real", "stand-in"]
    http_status: Optional[int] = None
    first_token_ms: Optional[float] = None
    total_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    rate_limit_headers: Dict[str, str] = Field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retry_after_seconds: Optional[float] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    source: Literal["real", "stand-in"] = "stand-in"
    stream: bool = False
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_reranking: Optional[bool] = None
    retrieval_setting_name: Optional[str] = "default"
    scenario: Optional[str] = "interactive"
    deadline_seconds: Optional[float] = Field(default=None, gt=0.1, le=120.0)


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    refused: bool
    refusal_reason: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    source: Literal["real", "stand-in"]
    status: Literal["ok", "rate_limited", "timeout", "error"]
    retries: int = 0
    queue_wait_ms: float = 0.0
    first_token_ms: Optional[float] = None
    total_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    retrieval_setting: str = "default"
    stage_timings: List[StageTiming] = Field(default_factory=list)
    verification_passed: bool = False
    refinement_attempts: int = 0


class RawResultRow(BaseModel):
    """Exact 12-column schema required for results/results.csv."""
    request_id: str
    scenario: str
    source: Literal["real", "stand-in"]
    start_time: str
    first_token_ms: Optional[float] = None
    total_ms: Optional[float] = None
    status: Literal["ok", "rate_limited", "timeout", "error"]
    retries: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    retrieval_setting: str
    answer_grade: Optional[Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]] = None


class HealthResponse(BaseModel):
    status: str
    papers_loaded: int
    chunks_indexed: int
    groq_key_configured: bool
    circuit_breaker_state: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
