"""
Unit tests for Stand-in FastAPI service and StandinClient.
"""
import asyncio
import pytest

from app.services.standin_service import StandinClient, StandinConfig, state
from app.services.retry_manager import PermanentError, RetryableError


@pytest.fixture(autouse=True)
def reset_standin():
    state.config = StandinConfig(
        base_ttft_ms=30.0,
        ttft_jitter_ms=5.0,
        inter_token_ms=1.0,
        concurrency_penalty_ms=25.0,
        target_output_tokens=20,
    )
    state.reset()
    yield
    state.reset()


async def test_standin_streaming_and_citation_synthesis():
    client = StandinClient(in_process=True)
    messages = [
        {
            "role": "user",
            "content": "[SOURCE: Attention Is All You Need (1706.03762v7), Page 3, Section: 3.2 Attention]\nExplain scaled dot-product attention.",
        }
    ]
    text, meta = await client.generate(messages, stream=True)
    assert meta.source == "stand-in"
    assert meta.http_status == 200
    assert meta.first_token_ms is not None
    assert meta.total_ms >= meta.first_token_ms
    assert meta.tokens_in > 0
    assert meta.tokens_out > 0
    assert "1706.03762v7" in text
    assert "Page 3" in text


async def test_standin_failure_modes():
    client = StandinClient(in_process=True, timeout=0.3)

    # 1. Always 429
    state.config.deterministic_fail_mode = "always_429"
    with pytest.raises(RetryableError) as exc_info:
        await client.generate([{"role": "user", "content": "hi"}], stream=False)
    assert exc_info.value.error_type == "rate_limited"
    assert exc_info.value.retry_after == 1.5

    # 2. Daily quota exhausted -> PermanentError
    state.config.deterministic_fail_mode = "daily_exhausted"
    with pytest.raises(PermanentError) as exc_info2:
        await client.generate([{"role": "user", "content": "hi"}], stream=False)
    assert exc_info2.value.error_type == "daily_quota_exhausted"

    # 3. Always 500
    state.config.deterministic_fail_mode = "always_500"
    with pytest.raises(RetryableError) as exc_info3:
        await client.generate([{"role": "user", "content": "hi"}], stream=False)
    assert exc_info3.value.error_type == "server_error"
