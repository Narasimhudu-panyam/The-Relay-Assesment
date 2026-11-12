"""
Sliding-window Rate Limiter & Daily Budget Tracker.
Tracks RPM, TPM (60s sliding window), RPD, and TPD (daily budget),
and dynamically incorporates provider-reported x-ratelimit-* headers.
"""
import asyncio
import time
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def parse_reset_duration(reset_str: Optional[str]) -> Optional[float]:
    """Parses Groq duration strings like '2m30.5s', '14s', '500ms' into seconds."""
    if not reset_str:
        return None
    s = reset_str.strip().lower()
    if s.isdigit():
        return float(s)
    total = 0.0
    for val, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", s):
        v = float(val)
        if unit == "ms":
            total += v / 1000.0
        elif unit == "s":
            total += v
        elif unit == "m":
            total += v * 60.0
        elif unit == "h":
            total += v * 3600.0
    return total if total > 0 else None


@dataclass
class RateLimitCheckResult:
    allowed: bool
    reason: str
    wait_seconds: float = 0.0
    is_daily_exhausted: bool = False


class RateLimiter:
    """Thread/async-safe sliding-window RPM/TPM + Daily RPD/TPD rate limiter."""

    def __init__(
        self,
        rpm: int = 25,
        tpm: int = 7000,
        rpd: int = 900,
        tpd: int = 180000,
        window_seconds: float = 60.0,
    ):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.tpd = tpd
        self.window_seconds = window_seconds

        self._lock = asyncio.Lock()
        self._request_timestamps: deque[float] = deque()
        self._token_events: deque[Tuple[float, int]] = deque()

        self.daily_requests: int = 0
        self.daily_tokens: int = 0
        self.daily_exhausted_flag: bool = False

        # Provider-reported overrides
        self.provider_remaining_requests: Optional[int] = None
        self.provider_remaining_tokens: Optional[int] = None
        self.provider_reset_requests_at: Optional[float] = None
        self.provider_reset_tokens_at: Optional[float] = None

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
        while self._token_events and self._token_events[0][0] < cutoff:
            self._token_events.popleft()

    async def check_and_reserve(self, estimated_tokens: int = 500) -> RateLimitCheckResult:
        """Checks daily and per-minute budgets; reserves capacity if allowed."""
        async with self._lock:
            now = time.monotonic()
            self._prune(now)

            # 1. Daily budget checks (hard stop — never retry)
            if self.daily_exhausted_flag or self.daily_requests >= self.rpd:
                return RateLimitCheckResult(
                    allowed=False,
                    reason=f"Daily request budget exhausted ({self.daily_requests}/{self.rpd} RPD)",
                    wait_seconds=0.0,
                    is_daily_exhausted=True,
                )
            if self.daily_tokens + estimated_tokens > self.tpd:
                return RateLimitCheckResult(
                    allowed=False,
                    reason=f"Daily token budget exhausted ({self.daily_tokens}/{self.tpd} TPD)",
                    wait_seconds=0.0,
                    is_daily_exhausted=True,
                )

            # 2. Provider-reported cooldown if remaining == 0
            if (
                self.provider_remaining_requests is not None
                and self.provider_remaining_requests <= 0
                and self.provider_reset_requests_at is not None
                and now < self.provider_reset_requests_at
            ):
                wait_s = max(0.05, self.provider_reset_requests_at - now)
                return RateLimitCheckResult(
                    allowed=False,
                    reason="Provider remaining RPM reached 0",
                    wait_seconds=wait_s,
                )

            # 3. Sliding-window RPM check
            if len(self._request_timestamps) >= self.rpm:
                oldest = self._request_timestamps[0]
                wait_s = max(0.05, (oldest + self.window_seconds) - now)
                return RateLimitCheckResult(
                    allowed=False,
                    reason=f"Local RPM limit reached ({len(self._request_timestamps)}/{self.rpm})",
                    wait_seconds=wait_s,
                )

            # 4. Sliding-window TPM check
            current_tpm = sum(t for _, t in self._token_events)
            if current_tpm + estimated_tokens > self.tpm and self._token_events:
                oldest_t = self._token_events[0][0]
                wait_s = max(0.05, (oldest_t + self.window_seconds) - now)
                return RateLimitCheckResult(
                    allowed=False,
                    reason=f"Local TPM limit reached ({current_tpm}/{self.tpm})",
                    wait_seconds=wait_s,
                )

            # Reserve slot
            self._request_timestamps.append(now)
            self._token_events.append((now, estimated_tokens))
            self.daily_requests += 1
            self.daily_tokens += estimated_tokens
            return RateLimitCheckResult(allowed=True, reason="ok")

    async def reconcile_tokens(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Adjusts token accounting after real usage is returned."""
        diff = actual_tokens - estimated_tokens
        if diff == 0:
            return
        async with self._lock:
            now = time.monotonic()
            self._token_events.append((now, diff))
            self.daily_tokens = max(0, self.daily_tokens + diff)

    async def update_from_headers(self, headers: Dict[str, str]) -> None:
        """Updates internal state from provider x-ratelimit-* headers."""
        if not headers:
            return
        lower = {k.lower(): v for k, v in headers.items()}
        async with self._lock:
            now = time.monotonic()
            if "x-ratelimit-remaining-requests" in lower:
                try:
                    self.provider_remaining_requests = int(lower["x-ratelimit-remaining-requests"])
                except ValueError:
                    pass
            if "x-ratelimit-remaining-tokens" in lower:
                try:
                    self.provider_remaining_tokens = int(lower["x-ratelimit-remaining-tokens"])
                except ValueError:
                    pass
            if "x-ratelimit-reset-requests" in lower:
                sec = parse_reset_duration(lower["x-ratelimit-reset-requests"])
                if sec is not None:
                    self.provider_reset_requests_at = now + sec
            if "x-ratelimit-reset-tokens" in lower:
                sec = parse_reset_duration(lower["x-ratelimit-reset-tokens"])
                if sec is not None:
                    self.provider_reset_tokens_at = now + sec

    async def mark_daily_exhausted(self) -> None:
        async with self._lock:
            self.daily_exhausted_flag = True

    async def reset_daily(self) -> None:
        async with self._lock:
            self.daily_requests = 0
            self.daily_tokens = 0
            self.daily_exhausted_flag = False

    async def get_snapshot(self) -> Dict[str, int | bool | None]:
        async with self._lock:
            now = time.monotonic()
            self._prune(now)
            return {
                "current_rpm": len(self._request_timestamps),
                "max_rpm": self.rpm,
                "current_tpm": sum(t for _, t in self._token_events),
                "max_tpm": self.tpm,
                "daily_requests": self.daily_requests,
                "max_rpd": self.rpd,
                "daily_tokens": self.daily_tokens,
                "max_tpd": self.tpd,
                "daily_exhausted": self.daily_exhausted_flag,
                "provider_remaining_requests": self.provider_remaining_requests,
                "provider_remaining_tokens": self.provider_remaining_tokens,
            }
