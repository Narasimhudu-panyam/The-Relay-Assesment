"""
Stand-in Model Service (FastAPI) + Async Client Adapter.
Simulates Groq's /v1/chat/completions API with calibrated TTFT,
inter-token streaming speed, concurrency slowdown, rate limits (429 + Retry-After),
daily budget exhaustion, random/deterministic failures, and timeouts.
"""
import asyncio
import json
import random
import re
import time
from collections import deque
from typing import AsyncGenerator, Dict, List, Literal, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.schemas.models import ModelCallMetadata
from app.services.retry_manager import PermanentError, RetryableError


class StandinConfig(BaseModel):
    base_ttft_ms: float = 180.0
    ttft_jitter_ms: float = 40.0
    inter_token_ms: float = 3.5
    concurrency_penalty_ms: float = 45.0
    failure_probability: float = 0.0
    rate_limit_probability: float = 0.0
    timeout_probability: float = 0.0
    rpm_limit: int = 30
    rpd_limit: int = 1000
    daily_exhausted: bool = False
    deterministic_fail_mode: Literal[
        "none", "always_429", "always_500", "always_timeout", "daily_exhausted"
    ] = "none"
    target_output_tokens: int = 80
    slow_response_extra_ms: float = 0.0
    seed: Optional[int] = 42


class StandinState:
    def __init__(self, config: Optional[StandinConfig] = None):
        self.config = config or StandinConfig()
        self.rng = random.Random(self.config.seed)
        self.active_requests = 0
        self.total_requests = 0
        self.daily_requests = 0
        self.rate_limited_count = 0
        self.failed_count = 0
        self.window_timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    def reset(self) -> None:
        self.active_requests = 0
        self.total_requests = 0
        self.daily_requests = 0
        self.rate_limited_count = 0
        self.failed_count = 0
        self.window_timestamps.clear()
        self.rng = random.Random(self.config.seed)


state = StandinState()
standin_app = FastAPI(title="The Relay Stand-in Model Service", version="1.0.0")


def _synthesize_grounded_reply(messages: List[Dict[str, str]], target_tokens: int) -> str:
    """
    Synthesizes a deterministic, context-aware reply for stand-in tests.
    If the prompt includes retrieved paper excerpts, cites the top passage cleanly.
    """
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    # Look for excerpt headers formatted as: [SOURCE: <title> (<id>), Page <p>, Section: <s>]
    matches = re.findall(
        r"\[SOURCE:\s*([^()]+?)\s*\(([^)]+)\),\s*Page\s*(\d+),\s*Section:\s*([^\]]+)\]",
        user_msg,
    )
    if matches:
        title, pid, page, section = matches[0]
        base = (
            f"Based on the retrieved research evidence, [{title.strip()} ({pid.strip()}), "
            f"Section: {section.strip()}, Page {page}] directly addresses the query. "
            f"The paper describes the relevant mechanism and experimental findings within that section."
        )
    else:
        base = (
            "The provided research papers do not contain sufficient evidence to answer this question."
        )

    words = base.split()
    if len(words) < target_tokens:
        pad = " Additional verified context is grounded strictly in the retrieved passages."
        while len(words) < target_tokens:
            words.extend(pad.split())
    return " ".join(words[: max(12, target_tokens)])


@standin_app.get("/health")
async def standin_health():
    return {"status": "ok", "service": "stand-in", "active_requests": state.active_requests}


@standin_app.post("/admin/config")
async def update_config(cfg: StandinConfig):
    state.config = cfg
    state.rng = random.Random(cfg.seed)
    return {"status": "updated", "config": state.config.model_dump()}


@standin_app.post("/admin/reset")
async def reset_state():
    state.reset()
    return {"status": "reset"}


@standin_app.get("/admin/stats")
async def get_stats():
    return {
        "active_requests": state.active_requests,
        "total_requests": state.total_requests,
        "daily_requests": state.daily_requests,
        "rate_limited_count": state.rate_limited_count,
        "failed_count": state.failed_count,
        "config": state.config.model_dump(),
    }


@standin_app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens", state.config.target_output_tokens))
    target_tokens = min(max_tokens, state.config.target_output_tokens)

    cfg = state.config

    async with state._lock:
        state.total_requests += 1
        now = time.monotonic()
        while state.window_timestamps and now - state.window_timestamps[0] > 60.0:
            state.window_timestamps.popleft()

        # Check daily exhaustion
        if cfg.daily_exhausted or cfg.deterministic_fail_mode == "daily_exhausted" or state.daily_requests >= cfg.rpd_limit:
            state.rate_limited_count += 1
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Daily request budget exhausted (insufficient_quota).",
                        "type": "insufficient_quota",
                        "code": "daily_limit_exceeded",
                    }
                },
                headers={"x-ratelimit-remaining-requests": "0"},
            )

        # Check deterministic or stochastic 429 RPM limit
        if (
            cfg.deterministic_fail_mode == "always_429"
            or len(state.window_timestamps) >= cfg.rpm_limit
            or state.rng.random() < cfg.rate_limit_probability
        ):
            state.rate_limited_count += 1
            retry_after = "1.5"
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit reached for requests per minute.",
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={
                    "retry-after": retry_after,
                    "x-ratelimit-limit-requests": str(cfg.rpm_limit),
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-reset-requests": f"{retry_after}s",
                },
            )

        # Deterministic / stochastic timeout simulation
        if cfg.deterministic_fail_mode == "always_timeout" or state.rng.random() < cfg.timeout_probability:
            await asyncio.sleep(30.0)

        # Deterministic / stochastic 500 failure
        if cfg.deterministic_fail_mode == "always_500" or state.rng.random() < cfg.failure_probability:
            state.failed_count += 1
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "Simulated upstream service unavailable", "type": "server_error"}},
            )

        state.window_timestamps.append(now)
        state.daily_requests += 1
        state.active_requests += 1
        in_flight = state.active_requests

    remaining_req = max(0, cfg.rpm_limit - len(state.window_timestamps))
    rl_headers = {
        "x-ratelimit-limit-requests": str(cfg.rpm_limit),
        "x-ratelimit-remaining-requests": str(remaining_req),
        "x-ratelimit-reset-requests": "60s",
    }

    reply_text = _synthesize_grounded_reply(messages, target_tokens)
    tokens_in = max(1, sum(len(m.get("content", "").split()) for m in messages) * 4 // 3)
    reply_pieces = [w + " " for w in reply_text.split()]
    tokens_out = len(reply_pieces)

    # Calibrated TTFT + concurrency drag
    ttft_ms = max(
        10.0,
        cfg.base_ttft_ms
        + state.rng.uniform(-cfg.ttft_jitter_ms, cfg.ttft_jitter_ms)
        + max(0, in_flight - 1) * cfg.concurrency_penalty_ms
        + cfg.slow_response_extra_ms,
    )

    if not stream:
        try:
            total_gen_ms = tokens_out * cfg.inter_token_ms
            await asyncio.sleep((ttft_ms + total_gen_ms) / 1000.0)
            return JSONResponse(
                status_code=200,
                headers=rl_headers,
                content={
                    "id": f"standin-{state.total_requests}",
                    "model": "stand-in/calibrated-gpt-oss-120b",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_text}}],
                    "usage": {
                        "prompt_tokens": tokens_in,
                        "completion_tokens": tokens_out,
                        "total_tokens": tokens_in + tokens_out,
                    },
                },
            )
        finally:
            async with state._lock:
                state.active_requests = max(0, state.active_requests - 1)

    async def sse_generator() -> AsyncGenerator[str, None]:
        try:
            await asyncio.sleep(ttft_ms / 1000.0)
            for idx, piece in enumerate(reply_pieces):
                if idx > 0:
                    await asyncio.sleep(cfg.inter_token_ms / 1000.0)
                chunk_payload = {
                    "id": f"standin-{state.total_requests}",
                    "choices": [{"index": 0, "delta": {"content": piece}}],
                }
                yield f"data: {json.dumps(chunk_payload)}\n\n"
            usage_payload = {
                "choices": [],
                "usage": {
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                },
            }
            yield f"data: {json.dumps(usage_payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            async with state._lock:
                state.active_requests = max(0, state.active_requests - 1)

    return StreamingResponse(sse_generator(), media_type="text/event-stream", headers=rl_headers)


class StandinClient:
    """Async client matching GroqService.generate() for the Stand-in service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001", in_process: bool = True, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.in_process = in_process
        self.timeout = timeout

    def _make_client(self) -> httpx.AsyncClient:
        if self.in_process:
            transport = httpx.ASGITransport(app=standin_app)
            return httpx.AsyncClient(transport=transport, base_url="http://standin.local", timeout=self.timeout)
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> Tuple[str, ModelCallMetadata]:
        t_start = time.perf_counter()
        first_token_ms: Optional[float] = None
        collected: List[str] = []
        tokens_in = 0
        tokens_out = 0
        headers: Dict[str, str] = {}

        payload = {
            "model": "stand-in/calibrated-gpt-oss-120b",
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with self._make_client() as client:
                if stream:
                    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                        headers = {k.lower(): v for k, v in resp.headers.items()}
                        if resp.status_code == 429:
                            body_bytes = await resp.aread()
                            body_text = body_bytes.decode("utf-8", errors="ignore")
                            if "insufficient_quota" in body_text or "daily" in body_text:
                                raise PermanentError("Stand-in daily budget exhausted", error_type="daily_quota_exhausted")
                            retry_after = float(headers["retry-after"]) if "retry-after" in headers else None
                            raise RetryableError("Stand-in 429 rate limited", error_type="rate_limited", retry_after=retry_after)
                        if resp.status_code >= 500:
                            raise RetryableError(f"Stand-in server error {resp.status_code}", error_type="server_error")

                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            obj = json.loads(data_str)
                            choices = obj.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}).get("content")
                                if delta:
                                    if first_token_ms is None:
                                        first_token_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                                    collected.append(delta)
                            if "usage" in obj:
                                tokens_in = obj["usage"].get("prompt_tokens", tokens_in)
                                tokens_out = obj["usage"].get("completion_tokens", tokens_out)

                        text = "".join(collected).strip()
                        http_status = resp.status_code
                else:
                    resp = await client.post("/v1/chat/completions", json=payload)
                    headers = {k.lower(): v for k, v in resp.headers.items()}
                    http_status = resp.status_code
                    if resp.status_code == 429:
                        if "insufficient_quota" in resp.text or "daily" in resp.text:
                            raise PermanentError("Stand-in daily budget exhausted", error_type="daily_quota_exhausted")
                        retry_after = float(headers["retry-after"]) if "retry-after" in headers else None
                        raise RetryableError("Stand-in 429 rate limited", error_type="rate_limited", retry_after=retry_after)
                    if resp.status_code >= 500:
                        raise RetryableError(f"Stand-in server error {resp.status_code}", error_type="server_error")

                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    first_token_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                    tokens_in = data.get("usage", {}).get("prompt_tokens", 0)
                    tokens_out = data.get("usage", {}).get("completion_tokens", 0)

            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            if first_token_ms is None:
                first_token_ms = total_ms

            meta = ModelCallMetadata(
                model="stand-in/calibrated-gpt-oss-120b",
                source="stand-in",
                http_status=http_status,
                first_token_ms=first_token_ms,
                total_ms=total_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                rate_limit_headers={
                    k: v for k, v in headers.items() if k.startswith("x-ratelimit") or k == "retry-after"
                },
            )
            return text, meta

        except httpx.TimeoutException as exc:
            raise RetryableError(f"Stand-in timeout: {exc}", error_type="timeout") from exc
        except httpx.ConnectError as exc:
            raise RetryableError(f"Stand-in connection error: {exc}", error_type="connection_error") from exc
