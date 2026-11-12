"""
Unit tests for RateLimiter, ConcurrencyAndQueueManager, RetryManager, and CircuitBreaker.
"""
import asyncio
import time
import pytest

from app.services.rate_limiter import RateLimiter, parse_reset_duration
from app.services.queue_manager import ConcurrencyAndQueueManager, QueueFullError, QueueTimeoutError
from app.services.retry_manager import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    PermanentError,
    RetryableError,
    RetryManager,
)


def test_parse_reset_duration():
    assert parse_reset_duration("14s") == 14.0
    assert abs(parse_reset_duration("2m30.5s") - 150.5) < 1e-6
    assert abs(parse_reset_duration("500ms") - 0.5) < 1e-6
    assert parse_reset_duration(None) is None


async def test_rate_limiter_rpm_and_daily_exhaustion():
    rl = RateLimiter(rpm=2, tpm=1000, rpd=3, tpd=2000, window_seconds=2.0)

    r1 = await rl.check_and_reserve(300)
    assert r1.allowed
    r2 = await rl.check_and_reserve(300)
    assert r2.allowed

    # 3rd request within window hits RPM
    r3 = await rl.check_and_reserve(300)
    assert not r3.allowed
    assert "RPM" in r3.reason
    assert r3.wait_seconds > 0

    # Wait for window to slide
    await asyncio.sleep(2.05)
    r4 = await rl.check_and_reserve(300)
    assert r4.allowed

    # 4th total request hits daily RPD=3
    r5 = await rl.check_and_reserve(100)
    assert not r5.allowed
    assert r5.is_daily_exhausted


async def test_rate_limiter_header_updates():
    rl = RateLimiter(rpm=30, tpm=8000, rpd=1000, tpd=200000)
    await rl.update_from_headers({
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "1s",
    })
    res = await rl.check_and_reserve(100)
    assert not res.allowed
    assert "Provider remaining RPM reached 0" in res.reason


async def test_queue_manager_concurrency_full_and_timeout():
    qm = ConcurrencyAndQueueManager(max_concurrent=1, max_queue_depth=1, default_queue_timeout=0.3)

    async def slow_task(wait_ms: float):
        await asyncio.sleep(0.5)
        return "done"

    t1 = asyncio.create_task(qm.execute("req-1", slow_task))
    await asyncio.sleep(0.05)
    t2 = asyncio.create_task(qm.execute("req-2", slow_task))
    await asyncio.sleep(0.05)

    # 3rd request should be rejected immediately with QueueFullError
    with pytest.raises(QueueFullError):
        await qm.execute("req-3", slow_task)

    # req-2 should time out waiting in queue (0.3s < 0.5s)
    with pytest.raises(QueueTimeoutError):
        await t2

    res1 = await t1
    assert res1 == "done"
    stats = qm.get_stats()
    assert stats["peak_queue_length"] == 1
    assert stats["total_rejected_full"] == 1
    assert stats["total_timed_out"] == 1


async def test_retry_manager_backoff_and_permanent_errors():
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=1.0)
    rm = RetryManager(max_retries=2, base_delay_seconds=0.05, max_delay_seconds=0.2, circuit_breaker=cb)

    attempts = 0

    async def flaky_op(attempt: int):
        nonlocal attempts
        attempts += 1
        if attempt < 2:
            raise RetryableError("429 rate limited", error_type="rate_limited", retry_after=0.05)
        return "ok"

    res, retries, wait_ms = await rm.run_with_retries(flaky_op)
    assert res == "ok"
    assert retries == 2
    assert attempts == 3
    assert wait_ms >= 40.0

    # Permanent error must not retry
    perm_attempts = 0

    async def perm_op(attempt: int):
        nonlocal perm_attempts
        perm_attempts += 1
        raise PermanentError("Invalid API key", error_type="auth_error")

    with pytest.raises(PermanentError):
        await rm.run_with_retries(perm_op)
    assert perm_attempts == 1


async def test_circuit_breaker_lifecycle():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.2)
    rm = RetryManager(max_retries=0, circuit_breaker=cb)

    async def fail_op(attempt: int):
        raise RetryableError("503 down", error_type="server_error")

    for _ in range(2):
        with pytest.raises(RetryableError):
            await rm.run_with_retries(fail_op)

    assert cb.state == CircuitState.OPEN

    # Next call should fail fast with CircuitOpenError
    with pytest.raises(CircuitOpenError):
        await rm.run_with_retries(fail_op)

    # Wait for recovery timeout -> HALF_OPEN -> success closes circuit
    await asyncio.sleep(0.25)

    async def ok_op(attempt: int):
        return "recovered"

    res, _, _ = await rm.run_with_retries(ok_op)
    assert res == "recovered"
    assert cb.state == CircuitState.CLOSED
