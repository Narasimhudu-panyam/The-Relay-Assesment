"""
Phase 8: Controlled Real Groq Baseline Experiment.
Sends a modest number (5) of controlled requests to the real Groq Free Tier
to measure empirical TTFT, total latency, token throughput, and x-ratelimit-* headers.
NEVER prints or logs the API key.
"""
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config.settings import Settings, get_settings
from app.pipeline.orchestrator import RelayPipeline
from app.pipeline.retriever import LibraryRetriever
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import FAISSVectorStore
from app.schemas.models import AskRequest, RawResultRow
from app.services.groq_service import GroqService
from experiments.logger import ResultsCSVLogger

BASELINE_QUESTIONS = [
    ("Q01", "In Attention Is All You Need, what scaling factor is applied to the dot products before softmax in Scaled Dot-Product Attention, and why?", "answer"),
    ("Q04", "What is the name and architecture of the neural audio codec used by Moshi, and at what frame rate does it operate?", "answer"),
    ("Q07", "What problem does Dynamic Chunking (H-Net) aim to solve compared to fixed BPE tokenization?", "answer"),
    ("Q13", "What is the exact API pricing per million tokens of Groq's Llama 3.3 70B model in 2026?", "refuse"),
    ("Q02", "How many parallel attention heads and what per-head dimensions (d_k, d_v) does the base Transformer model use in Attention Is All You Need?", "answer"),
]


async def verify_model_availability(groq_svc: GroqService, target_model: str) -> bool:
    """Checks model availability via Groq client.models.list() without exposing secrets."""
    client = groq_svc._get_client()
    models_page = await client.models.list()
    available_ids = [m.id for m in models_page.data]
    is_avail = target_model in available_ids
    print(f"Model verification for '{target_model}': {'AVAILABLE' if is_avail else 'NOT FOUND'}")
    if not is_avail:
        print(f"Available sample models: {available_ids[:10]}")
    return is_avail


async def run_baseline():
    settings = get_settings()
    if not Settings.has_groq_api_key():
        print("ERROR: GROQ_API_KEY is not configured in local .env.")
        sys.exit(1)

    groq_svc = GroqService(settings)
    if not await verify_model_availability(groq_svc, settings.groq_model):
        print(f"ERROR: Target model {settings.groq_model} is not available to this account. Stopping.")
        sys.exit(1)

    print(f"Initializing RAG index and running {len(BASELINE_QUESTIONS)} controlled real Groq requests...")
    print(f"Target model: {settings.groq_model}")

    engine = EmbeddingEngine()
    store = FAISSVectorStore(embedding_engine=engine)
    retriever = LibraryRetriever(
        vector_store=store,
        index_dir=settings.indices_dir,
        chunk_size_words=400,
        chunk_overlap_words=50,
        use_reranking=False,
    )
    chunks_count = retriever.initialize(papers_dir=settings.papers_dir)
    print(f"Indexed {chunks_count} chunks.")

    pipeline = RelayPipeline(retriever=retriever, groq_service=groq_svc, settings=settings)

    csv_logger = ResultsCSVLogger(settings.results_csv_path)
    raw_records = []
    csv_rows = []

    for qid, question, expected in BASELINE_QUESTIONS:
        start_iso = datetime.now(timezone.utc).isoformat()
        # Also capture raw headers directly on first request if needed
        res = await pipeline.process(
            AskRequest(
                question=question,
                source="real",
                stream=True,
                retrieval_setting_name="S2_400w_k5_norerank",
                scenario="real_baseline",
            )
        )
        grade = None
        if res.status == "ok":
            if expected == "refuse":
                grade = "refused_correctly" if res.refused else "wrong"
            else:
                grade = "correct" if (not res.refused and len(res.citations) > 0) else "refused_wrongly"

        csv_rows.append(
            RawResultRow(
                request_id=res.request_id,
                scenario="real_baseline",
                source="real",
                start_time=start_iso,
                first_token_ms=res.first_token_ms,
                total_ms=res.total_ms,
                status=res.status,
                retries=res.retries,
                tokens_in=res.tokens_in if res.tokens_in > 0 else None,
                tokens_out=res.tokens_out if res.tokens_out > 0 else None,
                retrieval_setting="S2_400w_k5_norerank",
                answer_grade=grade,
            )
        )
        raw_records.append({
            "qid": qid,
            "request_id": res.request_id,
            "start_time": start_iso,
            "model": settings.groq_model,
            "status": res.status,
            "refused": res.refused,
            "first_token_ms": res.first_token_ms,
            "total_ms": res.total_ms,
            "tokens_in": res.tokens_in,
            "tokens_out": res.tokens_out,
            "retries": res.retries,
            "citations": [c.model_dump() for c in res.citations],
            "answer_preview": res.answer[:240],
            "stage_timings": [s.model_dump() for s in res.stage_timings],
            "rate_limiter_snapshot": await pipeline.rate_limiter.get_snapshot(),
        })
        print(
            f"[{qid}] status={res.status} refused={res.refused} "
            f"ttft={res.first_token_ms}ms total={res.total_ms}ms "
            f"tokens={res.tokens_in}/{res.tokens_out} grade={grade}"
        )
        # Polite pacing between baseline probes
        await asyncio.sleep(2.0)

    csv_logger.append_rows(csv_rows)
    out_json = BASE_DIR / "results" / "groq_baseline.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2)
    print(f"\nSaved raw baseline telemetry to {out_json} and appended to {settings.results_csv_path}")


if __name__ == "__main__":
    asyncio.run(run_baseline())
