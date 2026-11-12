"""
Application configuration using python-dotenv and Pydantic.
Distinguishes provider-observed limits, application-enforced limits,
and experimental reduced limits. Never exposes or logs GROQ_API_KEY.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


class ProviderObservedLimits(BaseModel):
    """Dashboard-observed organization limits for openai/gpt-oss-120b (not hardcoded guarantees)."""
    rpm: int = Field(default_factory=lambda: int(os.getenv("PROVIDER_RPM", "30")))
    tpm: int = Field(default_factory=lambda: int(os.getenv("PROVIDER_TPM", "8000")))
    rpd: int = Field(default_factory=lambda: int(os.getenv("PROVIDER_RPD", "1000")))
    tpd: int = Field(default_factory=lambda: int(os.getenv("PROVIDER_TPD", "200000")))


class AppEnforcedLimits(BaseModel):
    """Conservative application-level pacing & concurrency protections."""
    rpm: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_RPM", "25")))
    tpm: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_TPM", "7000")))
    rpd: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_RPD", "900")))
    tpd: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_TPD", "180000")))
    max_concurrent_requests: int = Field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "3")))
    max_queue_depth: int = Field(default_factory=lambda: int(os.getenv("MAX_QUEUE_DEPTH", "20")))
    queue_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("QUEUE_TIMEOUT_SECONDS", "10.0")))
    request_deadline_seconds: float = Field(default_factory=lambda: float(os.getenv("REQUEST_DEADLINE_SECONDS", "30.0")))
    max_retries: int = Field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "2")))
    retry_budget_ratio: float = Field(default_factory=lambda: float(os.getenv("RETRY_BUDGET_RATIO", "0.25")))
    circuit_breaker_failure_threshold: int = Field(default_factory=lambda: int(os.getenv("CB_FAILURE_THRESHOLD", "5")))
    circuit_breaker_recovery_seconds: float = Field(default_factory=lambda: float(os.getenv("CB_RECOVERY_SECONDS", "15.0")))


class Settings(BaseModel):
    base_dir: Path = BASE_DIR
    papers_dir: Path = Field(default_factory=lambda: BASE_DIR / os.getenv("PAPERS_DIR", "papers"))
    indices_dir: Path = Field(default_factory=lambda: BASE_DIR / "papers" / "indices")
    results_csv_path: Path = Field(default_factory=lambda: BASE_DIR / os.getenv("RESULTS_CSV_PATH", "results/results.csv"))

    # Model & API config (secret never printed/serialized)
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    groq_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("GROQ_TIMEOUT_SECONDS", "20.0")))

    # Retrieval defaults
    retrieval_chunk_size: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_CHUNK_SIZE", "400")))
    retrieval_chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "50")))
    retrieval_top_k: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5")))
    use_reranker: bool = Field(default_factory=lambda: os.getenv("USE_RERANKER", "false").lower() == "true")
    min_similarity_threshold: float = Field(default_factory=lambda: float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.22")))

    # Stand-in service
    standin_host: str = Field(default_factory=lambda: os.getenv("STANDIN_HOST", "127.0.0.1"))
    standin_port: int = Field(default_factory=lambda: int(os.getenv("STANDIN_PORT", "8001")))
    standin_base_url: str = Field(default_factory=lambda: os.getenv("STANDIN_BASE_URL", "http://127.0.0.1:8001"))

    provider_limits: ProviderObservedLimits = Field(default_factory=ProviderObservedLimits)
    app_limits: AppEnforcedLimits = Field(default_factory=AppEnforcedLimits)

    @staticmethod
    def has_groq_api_key() -> bool:
        """Checks strictly whether GROQ_API_KEY is configured and non-empty without revealing it."""
        load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
        val = os.getenv("GROQ_API_KEY", "").strip()
        return bool(val) and val != "your_groq_api_key_here"

    @staticmethod
    def get_groq_api_key() -> Optional[str]:
        """Retrieves GROQ_API_KEY from environment safely."""
        load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
        val = os.getenv("GROQ_API_KEY", "").strip()
        if not val or val == "your_groq_api_key_here":
            return None
        return val


def get_settings() -> Settings:
    return Settings()
