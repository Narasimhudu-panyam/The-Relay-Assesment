"""
Bounded FIFO Queue & Concurrency Manager.
Controls max in-flight requests, enforces queue depth & per-request deadlines,
tracks peak queue depth, queue wait times, and cancellations.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, TypeVar

T = TypeVar("T")


class QueueFullError(Exception):
    """Raised when bounded FIFO queue is at capacity."""


class QueueTimeoutError(Exception):
    """Raised when a queued request exceeds its queue wait deadline."""


@dataclass
class _QueueItem:
    request_id: str
    enqueued_at: float
    deadline_at: float
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


class ConcurrencyAndQueueManager:
    """Manages bounded FIFO queuing and maximum concurrent execution slots."""

    def __init__(
        self,
        max_concurrent: int = 3,
        max_queue_depth: int = 20,
        default_queue_timeout: float = 10.0,
    ):
        self.max_concurrent = max_concurrent
        self.max_queue_depth = max_queue_depth
        self.default_queue_timeout = default_queue_timeout

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()

        self.active_requests: int = 0
        self.current_queue_depth: int = 0
        self.peak_queue_length: int = 0
        self.total_enqueued: int = 0
        self.total_rejected_full: int = 0
        self.total_timed_out: int = 0
        self.total_cancelled: int = 0

    async def execute(
        self,
        request_id: str,
        fn: Callable[[float], Awaitable[T]],
        queue_timeout: float | None = None,
    ) -> T:
        """
        Acquires an execution slot (waiting in bounded FIFO queue if saturated),
        measures wait time, and executes `fn(queue_wait_ms)`.
        """
        timeout = queue_timeout if queue_timeout is not None else self.default_queue_timeout
        t_enter = time.perf_counter()

        # Check fast-path or bounded queue capacity atomically under lock
        async with self._lock:
            if self.active_requests + self.current_queue_depth >= self.max_concurrent + self.max_queue_depth:
                self.total_rejected_full += 1
                raise QueueFullError(
                    f"Queue full (active={self.active_requests}/{self.max_concurrent}, "
                    f"queued={self.current_queue_depth}/{self.max_queue_depth})"
                )
            if self.active_requests >= self.max_concurrent:
                self.current_queue_depth += 1
                if self.current_queue_depth > self.peak_queue_length:
                    self.peak_queue_length = self.current_queue_depth
                was_queued = True
            else:
                self.active_requests += 1
                was_queued = False
            self.total_enqueued += 1

        acquired = False
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            acquired = True
        except asyncio.TimeoutError:
            async with self._lock:
                if was_queued:
                    if self.current_queue_depth > 0:
                        self.current_queue_depth -= 1
                else:
                    self.active_requests = max(0, self.active_requests - 1)
                self.total_timed_out += 1
            raise QueueTimeoutError(f"Request {request_id} expired in queue after {timeout:.2f}s")
        except asyncio.CancelledError:
            async with self._lock:
                if was_queued:
                    if self.current_queue_depth > 0:
                        self.current_queue_depth -= 1
                else:
                    self.active_requests = max(0, self.active_requests - 1)
                self.total_cancelled += 1
            raise

        if was_queued:
            async with self._lock:
                if self.current_queue_depth > 0:
                    self.current_queue_depth -= 1
                self.active_requests += 1

        queue_wait_ms = round((time.perf_counter() - t_enter) * 1000.0, 2)
        try:
            return await fn(queue_wait_ms)
        finally:
            async with self._lock:
                self.active_requests = max(0, self.active_requests - 1)
            if acquired:
                self._semaphore.release()

    def get_stats(self) -> Dict[str, int]:
        return {
            "max_concurrent": self.max_concurrent,
            "max_queue_depth": self.max_queue_depth,
            "active_requests": self.active_requests,
            "current_queue_depth": self.current_queue_depth,
            "peak_queue_length": self.peak_queue_length,
            "total_enqueued": self.total_enqueued,
            "total_rejected_full": self.total_rejected_full,
            "total_timed_out": self.total_timed_out,
            "total_cancelled": self.total_cancelled,
        }

    def reset_stats(self) -> None:
        self.peak_queue_length = 0
        self.total_enqueued = 0
        self.total_rejected_full = 0
        self.total_timed_out = 0
        self.total_cancelled = 0
