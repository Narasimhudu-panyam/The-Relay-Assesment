"""
End-to-End Integration Tests for the 5-stage pipeline, Groq error handling (mocked),
Stand-in integration, reliability protection, and FastAPI endpoints.
Covers all 10 required Phase 7 scenarios.
"""
import asyncio
from typing import Any, Dict, List, Optional
import httpx
import pytest

from app.config.settings import Settings
from app.pipeline.answer_verifier import AnswerVerifier
from app.pipeline.decision_engine import DecisionEngine
from app.pipeline.orchestrator import RelayPipeline
from app.pipeline.question_analyzer import QuestionAnalyzer
from app.schemas.models import AskRequest, ModelCallMetadata
from app.services.queue_manager import ConcurrencyAndQueueManager
from app.services.rate_limiter import RateLimiter
from app.services.retry_manager import CircuitBreaker, RetryManager, RetryableError
from app.services.standin_service import StandinClient, StandinConfig, state as standin_state


class StubRetriever:
    """Fast deterministic retriever stub for testing pipeline logic without loading PyTorch."""

    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None, score: float = 0.72):
        self.use_reranking = False
        self.score = score
        self.chunks = chunks if chunks is not None else [
            {
                "chunk_id": "1706.03762v7_p4_c1",
                "paper_id": "1706.03762v7",
                "paper_title": "Attention Is All You Need",
                "paper_version": "v7",
                "page_number": 4,
                "section": "3.2.1 Scaled Dot-Product Attention",
                "text": "We compute the dot products of the query with all keys, divide each by sqrt(dk), and apply a softmax function to obtain the weights on the values.",
                "similarity_score": score,
            }
        ]

    def search_library(self, query: str, top_k: int = 5, min_score: float = 0.0, filter_paper_id: Optional[str] = None):
        return {
            "tool": "search_library",
            "query": query,
            "duration_ms": 1.2,
            "count": len(self.chunks),
            "chunks": self.chunks[:top_k],
            "highest_score": self.score if self.chunks else 0.0,
        }


@pytest.fixture(autouse=True)
def reset_standin():
    standin_state.config = StandinConfig(
        base_ttft_ms=10.0,
        ttft_jitter_ms=2.0,
        inter_token_ms=0.5,
        concurrency_penalty_ms=10.0,
        target_output_tokens=30,
    )
    standin_state.reset()
    yield
    standin_state.reset()


def make_pipeline(retriever=None, max_concurrent=3, max_queue=10, max_retries=2, cb_threshold=3, cb_recovery=0.2) -> RelayPipeline:
    cb = CircuitBreaker(failure_threshold=cb_threshold, recovery_timeout_seconds=cb_recovery)
    rm = RetryManager(max_retries=max_retries, base_delay_seconds=0.02, max_delay_seconds=0.1, circuit_breaker=cb)
    qm = ConcurrencyAndQueueManager(max_concurrent=max_concurrent, max_queue_depth=max_queue, default_queue_timeout=1.0)
    rl = RateLimiter(rpm=100, tpm=100000, rpd=1000, tpd=500000)
    return RelayPipeline(
        retriever=retriever or StubRetriever(),
        rate_limiter=rl,
        queue_manager=qm,
        retry_manager=rm,
    )


async def test_scenario_1_and_4_supported_and_cited_question():
    pipeline = make_pipeline()
    res = await pipeline.process(AskRequest(question="How does scaled dot-product attention work?", source="stand-in", stream=True))
    assert res.status == "ok"
    assert not res.refused
    assert res.verification_passed
    assert len(res.citations) >= 1
    assert res.citations[0].paper_id == "1706.03762v7"
    assert res.citations[0].page_number == 4
    assert res.first_token_ms is not None
    assert len(res.stage_timings) >= 4


async def test_scenario_2_and_3_unsupported_and_insufficient_evidence():
    low_ret = StubRetriever(
        chunks=[{
            "chunk_id": "x1",
            "paper_id": "1706.03762v7",
            "paper_title": "Attention Is All You Need",
            "paper_version": "v7",
            "page_number": 1,
            "section": "Abstract",
            "text": "Sequence transduction models based on attention.",
            "similarity_score": 0.08,
        }],
        score=0.08,
    )
    pipeline = make_pipeline(retriever=low_ret)
    res = await pipeline.process(AskRequest(question="What is the boiling point of liquid nitrogen?", source="stand-in"))
    assert res.status == "ok"
    assert res.refused
    assert res.tokens_out == 0  # Refused in DecisionEngine without calling LLM


async def test_scenario_5_invalid_citation_caught_by_verifier():
    verifier = AnswerVerifier()
    chunks = [{
        "chunk_id": "c1",
        "paper_id": "1706.03762v7",
        "paper_title": "Attention Is All You Need",
        "paper_version": "v7",
        "page_number": 4,
        "section": "3.2.1",
        "text": "Scaled dot product",
    }]
    # Model cites Page 99 which was NOT in retrieved chunks
    bad_ans = "Attention divides by sqrt(dk) [Attention Is All You Need (1706.03762v7), Page 99, Section: 3.2.1]."
    v_res = verifier.verify(bad_ans, chunks)
    assert not v_res.passed
    assert "1706.03762v7 Page 99" in v_res.invalid_citations


async def test_scenario_6_groq_timeout_handled():
    pipeline = make_pipeline(max_retries=1)

    async def timeout_gen(*args, **kwargs):
        raise RetryableError("Simulated Groq timeout", error_type="timeout")

    pipeline.generator.generate = timeout_gen
    res = await pipeline.process(AskRequest(question="Explain scaled dot-product attention?", source="real"))
    assert res.status == "timeout"
    assert res.refused
    assert res.retries == 1


async def test_scenario_7_rate_limit_retry_and_recover():
    pipeline = make_pipeline(max_retries=2)
    calls = 0

    async def flaky_429(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableError("429 Too Many Requests", error_type="rate_limited", retry_after=0.02)
        return (
            "Scaled dot-product attention divides by sqrt(dk) [Attention Is All You Need (1706.03762v7), Page 4, Section: 3.2.1].",
            ModelCallMetadata(model="mock", source="real", http_status=200, first_token_ms=15.0, total_ms=30.0, tokens_in=50, tokens_out=20),
        )

    pipeline.generator.generate = flaky_429
    res = await pipeline.process(AskRequest(question="Explain scaled dot-product attention?", source="real"))
    assert res.status == "ok"
    assert res.retries == 1
    assert not res.refused


async def test_scenario_8_queue_full():
    pipeline = make_pipeline(max_concurrent=1, max_queue=1)
    standin_state.config.base_ttft_ms = 500.0

    t1 = asyncio.create_task(pipeline.process(AskRequest(question="Explain scaled dot-product attention?", source="stand-in")))
    t2 = asyncio.create_task(pipeline.process(AskRequest(question="Explain scaled dot-product attention?", source="stand-in")))
    await asyncio.sleep(0.05)
    res3 = await pipeline.process(AskRequest(question="Explain scaled dot-product attention?", source="stand-in"))
    assert res3.status == "rate_limited"
    assert "Queue full" in (res3.refusal_reason or "")
    await asyncio.gather(t1, t2)


async def test_scenario_9_and_10_circuit_breaker_open_and_recovery():
    pipeline = make_pipeline(max_retries=0, cb_threshold=2, cb_recovery=0.15)
    standin_state.config.deterministic_fail_mode = "always_500"

    # Trigger 2 failures -> opens circuit
    r1 = await pipeline.process(AskRequest(question="Explain scaled dot-product attention?", source="stand-in"))
    r2 = await pipeline.process(AskRequest(question="Explain scaled dot-product attention?", source="stand-in"))
    assert r1.status == "error"
    assert r2.status == "error"

    # 3rd request fails fast via CircuitOpenError
    r3 = await pipeline.process(AskRequest(question="Explain scaled dot-product attention?", source="stand-in"))
    assert r3.status == "error"
    assert "Circuit breaker is OPEN" in (r3.refusal_reason or "")

    # Recover provider and wait past recovery timeout -> HALF_OPEN -> CLOSED
    standin_state.config.deterministic_fail_mode = "none"
    await asyncio.sleep(0.2)
    r4 = await pipeline.process(AskRequest(question="Explain scaled dot-product attention?", source="stand-in"))
    assert r4.status == "ok"
    assert not r4.refused


async def test_fastapi_endpoints_with_stub_pipeline(monkeypatch):
    import app.main as main_mod
    stub_pipe = make_pipeline()
    monkeypatch.setattr(main_mod, "get_pipeline", lambda: stub_pipe)
    monkeypatch.setattr(main_mod, "_chunks_indexed", 42)

    transport = httpx.ASGITransport(app=main_mod.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        h = await client.get("/health")
        assert h.status_code == 200
        assert h.json()["chunks_indexed"] == 42

        m = await client.get("/metrics")
        assert m.status_code == 200
        assert "rate_limiter" in m.json()

        a = await client.post("/ask", json={"question": "Explain scaled dot-product attention?", "source": "stand-in"})
        assert a.status_code == 200
        assert a.json()["status"] == "ok"
        assert len(a.json()["citations"]) >= 1

        # Test SSE streaming endpoint
        s = await client.post("/ask?sse=true", json={"question": "Explain scaled dot-product attention?", "source": "stand-in", "stream": True})
        assert s.status_code == 200
        assert "data: [DONE]" in s.text
