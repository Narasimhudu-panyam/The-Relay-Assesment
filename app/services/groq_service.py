"""
Groq Hosted LLM Service Adapter.
Uses official `groq` Python SDK with `max_retries=0` so our own RetryManager
can control and measure retry behavior. Captures HTTP status codes,
x-ratelimit-* headers, TTFT, total latency, and token usage.
"""
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple
import groq
from groq import AsyncGroq

from app.config.settings import Settings, get_settings
from app.schemas.models import ModelCallMetadata
from app.services.retry_manager import PermanentError, RetryableError


def _extract_headers(exc: Exception) -> Dict[str, str]:
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "headers"):
        return {k.lower(): str(v) for k, v in resp.headers.items()}
    return {}


def _parse_retry_after(headers: Dict[str, str]) -> Optional[float]:
    val = headers.get("retry-after")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


class GroqService:
    """Async client wrapper around Groq's official SDK with raw telemetry."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client: Optional[AsyncGroq] = None

    def _get_client(self) -> AsyncGroq:
        api_key = self.settings.get_groq_api_key()
        if not api_key:
            raise PermanentError(
                "GROQ_API_KEY is not configured in local .env.",
                error_type="auth_error",
            )
        if self._client is None:
            # Disable SDK internal retries so our RetryManager measures & controls retries
            self._client = AsyncGroq(
                api_key=api_key,
                max_retries=0,
                timeout=self.settings.groq_timeout_seconds,
            )
        return self._client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> Tuple[str, ModelCallMetadata]:
        """Executes a single chat completion call against Groq, returning (text, metadata)."""
        client = self._get_client()
        model = self.settings.groq_model
        t_start = time.perf_counter()
        first_token_ms: Optional[float] = None
        collected_chunks: List[str] = []
        tokens_in = 0
        tokens_out = 0
        headers: Dict[str, str] = {}
        http_status: Optional[int] = 200

        try:
            if stream:
                raw_resp = await client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                http_status = raw_resp.status_code
                headers = {k.lower(): str(v) for k, v in raw_resp.headers.items()}
                stream_obj = await raw_resp.parse()

                async for chunk in stream_obj:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        if first_token_ms is None:
                            first_token_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                        collected_chunks.append(chunk.choices[0].delta.content)
                    usage = getattr(chunk, "x_groq", None)
                    if usage and getattr(usage, "usage", None):
                        tokens_in = usage.usage.prompt_tokens or tokens_in
                        tokens_out = usage.usage.completion_tokens or tokens_out
                    elif getattr(chunk, "usage", None):
                        tokens_in = chunk.usage.prompt_tokens or tokens_in
                        tokens_out = chunk.usage.completion_tokens or tokens_out

                text = "".join(collected_chunks)
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            else:
                raw_resp = await client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                http_status = raw_resp.status_code
                headers = {k.lower(): str(v) for k, v in raw_resp.headers.items()}
                completion = await raw_resp.parse()
                total_elapsed = round((time.perf_counter() - t_start) * 1000.0, 2)
                first_token_ms = total_elapsed
                text = completion.choices[0].message.content or ""
                if completion.usage:
                    tokens_in = completion.usage.prompt_tokens or 0
                    tokens_out = completion.usage.completion_tokens or 0

            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            # Fallback token estimation if stream chunk omitted usage
            if tokens_in == 0:
                tokens_in = max(1, sum(len(m.get("content", "").split()) for m in messages) * 4 // 3)
            if tokens_out == 0 and text:
                tokens_out = max(1, len(text.split()) * 4 // 3)

            meta = ModelCallMetadata(
                model=model,
                source="real",
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

        except groq.AuthenticationError as exc:
            raise PermanentError(f"Groq authentication failed: {exc}", error_type="auth_error") from exc
        except groq.NotFoundError as exc:
            raise PermanentError(f"Groq model unavailable ({model}): {exc}", error_type="model_unavailable") from exc
        except groq.BadRequestError as exc:
            raise PermanentError(f"Groq bad request: {exc}", error_type="bad_request") from exc
        except groq.RateLimitError as exc:
            err_headers = _extract_headers(exc)
            retry_after = _parse_retry_after(err_headers)
            msg = str(exc).lower()
            if "insufficient_quota" in msg or "daily" in msg or "tokens per day" in msg or "requests per day" in msg:
                raise PermanentError(f"Groq daily quota exhausted: {exc}", error_type="daily_quota_exhausted") from exc
            raise RetryableError(
                f"Groq 429 rate limited: {exc}",
                error_type="rate_limited",
                retry_after=retry_after,
            ) from exc
        except groq.APITimeoutError as exc:
            raise RetryableError(f"Groq timeout: {exc}", error_type="timeout") from exc
        except groq.APIConnectionError as exc:
            raise RetryableError(f"Groq connection error: {exc}", error_type="connection_error") from exc
        except groq.APIStatusError as exc:
            status = getattr(exc, "status_code", 500)
            if status >= 500:
                raise RetryableError(f"Groq server error ({status}): {exc}", error_type="server_error") from exc
            raise PermanentError(f"Groq client error ({status}): {exc}", error_type="client_error") from exc

    async def generate_stream_tokens(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> AsyncGenerator[str, None]:
        """Yields text tokens as they arrive from Groq."""
        client = self._get_client()
        stream_obj = await client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream_obj:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
