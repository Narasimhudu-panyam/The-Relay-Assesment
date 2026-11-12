"""
Retry Manager & Circuit Breaker.
Implements exponential backoff with full jitter, Retry-After honoring,
global retry budget (retry-storm prevention), permanent error classification,
and a 3-state Circuit Breaker (CLOSED, OPEN, HALF_OPEN).
"""
import asyncio
import random
import time
from enum import Enum
from typing import Awaitable, Callable, Dict, Optional, Tuple, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is OPEN and rejects calls fast."""


class RetryableError(Exception):
    """Transient error eligible for bounded retry."""

    def __init__(self, message: str, error_type: str = "transient", retry_after: Optional[float] = None):
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after


class PermanentError(Exception):
    """Non-retryable error (e.g., 401 auth, 400 bad request, daily quota exhausted)."""

    def __init__(self, message: str, error_type: str = "permanent"):
        super().__init__(message)
        self.error_type = error_type


class CircuitBreaker:
    """Three-state circuit breaker protecting the upstream LLM provider."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: float = 15.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.opened_at: Optional[float] = None
        self._half_open_probe_in_flight: bool = False
        self._lock = asyncio.Lock()

    async def allow_request(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if self.opened_at is not None and (now - self.opened_at) >= self.recovery_timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self._half_open_probe_in_flight = True
                    return True
                return False
            # HALF_OPEN: allow only one probe at a time
            if not self._half_open_probe_in_flight:
                self._half_open_probe_in_flight = True
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            self.consecutive_failures = 0
            self.state = CircuitState.CLOSED
            self.opened_at = None
            self._half_open_probe_in_flight = False

    async def record_failure(self) -> None:
        async with self._lock:
            self.consecutive_failures += 1
            self._half_open_probe_in_flight = False
            if self.state == CircuitState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()

    async def force_open(self) -> None:
        async with self._lock:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()

    async def reset(self) -> None:
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.opened_at = None
            self._half_open_probe_in_flight = False

    def get_status(self) -> Dict[str, str | int]:
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
        }


class RetryManager:
    """Bounded retry engine with exponential backoff, jitter, Retry-After, and retry budget."""

    def __init__(
        self,
        max_retries: int = 2,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 8.0,
        retry_budget_ratio: float = 0.25,
        min_budget_requests: int = 5,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.retry_budget_ratio = retry_budget_ratio
        self.min_budget_requests = min_budget_requests
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        self._lock = asyncio.Lock()
        self.total_primary_requests: int = 0
        self.total_retries_executed: int = 0
        self.total_retries_suppressed_by_budget: int = 0

    def compute_backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Calculates backoff delay with random jitter, respecting Retry-After if larger."""
        exp_delay = min(self.max_delay_seconds, self.base_delay_seconds * (2 ** attempt))
        jittered = exp_delay * random.uniform(0.5, 1.5)
        if retry_after is not None and retry_after > 0:
            return min(self.max_delay_seconds * 2, max(jittered, retry_after))
        return jittered

    async def _can_spend_retry_budget(self) -> bool:
        async with self._lock:
            if self.total_primary_requests < self.min_budget_requests:
                return True
            allowed_retries = self.total_primary_requests * self.retry_budget_ratio
            if self.total_retries_executed < allowed_retries:
                return True
            self.total_retries_suppressed_by_budget += 1
            return False

    async def run_with_retries(
        self,
        operation: Callable[[int], Awaitable[T]],
        deadline_at: Optional[float] = None,
    ) -> Tuple[T, int, float]:
        """
        Executes `operation(attempt)` with bounded retries and circuit breaker protection.
        Returns `(result, retries_used, retry_wait_ms)`.
        """
        async with self._lock:
            self.total_primary_requests += 1

        retries_used = 0
        retry_wait_ms = 0.0

        for attempt in range(self.max_retries + 1):
            if not await self.circuit_breaker.allow_request():
                raise CircuitOpenError("Circuit breaker is OPEN; failing fast without calling provider.")

            try:
                result = await operation(attempt)
                await self.circuit_breaker.record_success()
                return result, retries_used, round(retry_wait_ms, 2)

            except PermanentError:
                # Permanent errors (401, 400, daily quota exhausted) are NEVER retried
                raise

            except RetryableError as exc:
                await self.circuit_breaker.record_failure()
                if attempt >= self.max_retries:
                    raise

                if not await self._can_spend_retry_budget():
                    raise RetryableError(
                        f"Retry suppressed by global retry budget ({exc})",
                        error_type=exc.error_type,
                        retry_after=exc.retry_after,
                    ) from exc

                delay = self.compute_backoff(attempt, exc.retry_after)
                if deadline_at is not None:
                    remaining = deadline_at - time.monotonic()
                    if remaining <= delay or remaining <= 0.05:
                        raise RetryableError(
                            f"Insufficient deadline budget ({remaining:.2f}s) for retry backoff ({delay:.2f}s)",
                            error_type="timeout",
                        ) from exc

                async with self._lock:
                    self.total_retries_executed += 1

                retries_used += 1
                t_sleep_start = time.perf_counter()
                await asyncio.sleep(delay)
                retry_wait_ms += (time.perf_counter() - t_sleep_start) * 1000.0

        raise RuntimeError("Unreachable retry loop exit")

    def get_stats(self) -> Dict[str, int | float | dict]:
        return {
            "total_primary_requests": self.total_primary_requests,
            "total_retries_executed": self.total_retries_executed,
            "total_retries_suppressed_by_budget": self.total_retries_suppressed_by_budget,
            "circuit_breaker": self.circuit_breaker.get_status(),
        }

    def reset_stats(self) -> None:
        self.total_primary_requests = 0
        self.total_retries_executed = 0
        self.total_retries_suppressed_by_budget = 0
