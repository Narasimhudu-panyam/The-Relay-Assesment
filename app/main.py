"""
The Relay — FastAPI Gateway Application.
Exposes POST /ask, GET /health, and GET /metrics.
"""
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from app.config.settings import Settings, get_settings
from app.pipeline.orchestrator import RelayPipeline
from app.pipeline.retriever import LibraryRetriever
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import FAISSVectorStore
from app.schemas.models import AskRequest, AskResponse, HealthResponse

_pipeline: Optional[RelayPipeline] = None
_chunks_indexed: int = 0


def get_pipeline() -> RelayPipeline:
    global _pipeline, _chunks_indexed
    if _pipeline is None:
        settings = get_settings()
        engine = EmbeddingEngine()
        store = FAISSVectorStore(embedding_engine=engine)
        retriever = LibraryRetriever(
            vector_store=store,
            index_dir=settings.indices_dir,
            chunk_size_words=settings.retrieval_chunk_size,
            chunk_overlap_words=settings.retrieval_chunk_overlap,
            use_reranking=settings.use_reranker,
        )
        _chunks_indexed = retriever.initialize(papers_dir=settings.papers_dir)
        _pipeline = RelayPipeline(retriever=retriever, settings=settings)
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    yield


app = FastAPI(
    title="The Relay — Resilient Research Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse)
async def ask_question(
    req: AskRequest,
    sse: bool = Query(default=False, description="Set true to stream response over Server-Sent Events"),
):
    pipeline = get_pipeline()
    if not sse:
        return await pipeline.process(req)

    async def event_stream() -> AsyncGenerator[str, None]:
        res = await pipeline.process(req)
        words = res.answer.split(" ")
        for i, w in enumerate(words):
            token = w + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield f"data: {json.dumps({'type': 'final', 'response': res.model_dump()})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    pipeline = get_pipeline()
    settings = get_settings()
    cb_status = pipeline.retry_manager.circuit_breaker.get_status()
    return HealthResponse(
        status="ok",
        papers_loaded=3,
        chunks_indexed=_chunks_indexed,
        groq_key_configured=Settings.has_groq_api_key(),
        circuit_breaker_state=str(cb_status["state"]),
    )


@app.get("/metrics")
async def get_metrics():
    pipeline = get_pipeline()
    return {
        "chunks_indexed": _chunks_indexed,
        "rate_limiter": await pipeline.rate_limiter.get_snapshot(),
        "queue_manager": pipeline.queue_manager.get_stats(),
        "retry_manager": pipeline.retry_manager.get_stats(),
    }
