"""
End-to-End Asynchronous Pipeline Orchestrator.
Integrates Stages 1-5 with ConcurrencyAndQueueManager, RateLimiter,
RetryManager, and CircuitBreaker. Records per-stage timings and token usage.
"""
import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from app.config.settings import Settings, get_settings
from app.pipeline.answer_generator import AnswerGenerator
from app.pipeline.answer_verifier import AnswerVerifier
from app.pipeline.decision_engine import DecisionEngine
from app.pipeline.question_analyzer import QuestionAnalyzer
from app.pipeline.retriever import LibraryRetriever
from app.schemas.models import AskRequest, AskResponse, StageTiming
from app.services.groq_service import GroqService
from app.services.queue_manager import ConcurrencyAndQueueManager, QueueFullError, QueueTimeoutError
from app.services.rate_limiter import RateLimiter
from app.services.retry_manager import CircuitBreaker, CircuitOpenError, PermanentError, RetryManager, RetryableError
from app.services.standin_service import StandinClient


class RelayPipeline:
    """Orchestrates the 5-stage AI pipeline with full reliability and observability."""

    def __init__(
        self,
        retriever: LibraryRetriever,
        groq_service: Optional[GroqService] = None,
        standin_client: Optional[StandinClient] = None,
        rate_limiter: Optional[RateLimiter] = None,
        queue_manager: Optional[ConcurrencyAndQueueManager] = None,
        retry_manager: Optional[RetryManager] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.groq_service = groq_service or GroqService(self.settings)
        self.standin_client = standin_client or StandinClient(
            base_url=self.settings.standin_base_url,
            in_process=True,
        )

        self.analyzer = QuestionAnalyzer()
        self.decision_engine = DecisionEngine(
            refuse_threshold=self.settings.min_similarity_threshold,
            proceed_threshold=0.30,
            max_refinements=1,
        )
        self.generator = AnswerGenerator(self.groq_service, self.standin_client)
        self.verifier = AnswerVerifier()

        self.rate_limiter = rate_limiter or RateLimiter(
            rpm=self.settings.app_limits.rpm,
            tpm=self.settings.app_limits.tpm,
            rpd=self.settings.app_limits.rpd,
            tpd=self.settings.app_limits.tpd,
        )
        self.queue_manager = queue_manager or ConcurrencyAndQueueManager(
            max_concurrent=self.settings.app_limits.max_concurrent_requests,
            max_queue_depth=self.settings.app_limits.max_queue_depth,
            default_queue_timeout=self.settings.app_limits.queue_timeout_seconds,
        )
        cb = CircuitBreaker(
            failure_threshold=self.settings.app_limits.circuit_breaker_failure_threshold,
            recovery_timeout_seconds=self.settings.app_limits.circuit_breaker_recovery_seconds,
        )
        self.retry_manager = retry_manager or RetryManager(
            max_retries=self.settings.app_limits.max_retries,
            retry_budget_ratio=self.settings.app_limits.retry_budget_ratio,
            circuit_breaker=cb,
        )

    async def process(self, req: AskRequest) -> AskResponse:
        request_id = str(uuid.uuid4())
        t_req_start = time.perf_counter()
        deadline_s = req.deadline_seconds or self.settings.app_limits.request_deadline_seconds
        deadline_at = time.monotonic() + deadline_s
        stage_timings: List[StageTiming] = []

        top_k = req.top_k or self.settings.retrieval_top_k
        if req.use_reranking is not None:
            self.retriever.use_reranking = req.use_reranking

        # --- Stage 1: Question Understanding ---
        t0 = time.perf_counter()
        try:
            analysis = self.analyzer.analyze(req.question)
        except ValueError as exc:
            return AskResponse(
                request_id=request_id,
                question=req.question,
                answer=f"Invalid question: {exc}",
                refused=True,
                refusal_reason=str(exc),
                source=req.source,
                status="error",
                total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                retrieval_setting=req.retrieval_setting_name or "default",
            )
        stage_timings.append(
            StageTiming(
                stage="question_analysis",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                details={"intent": analysis.intent, "keywords": analysis.keywords},
            )
        )

        # --- Stage 2 & 3: Retrieval Tool + Decision Engine (with bounded refinement loop) ---
        refinement_attempts = 0
        query = analysis.retrieval_query
        retrieval_out: Dict[str, Any] = {}
        decision = None

        for attempt in range(2):
            t_ret = time.perf_counter()
            retrieval_out = self.retriever.search_library(
                query=query,
                top_k=top_k,
                filter_paper_id=analysis.target_paper_id if attempt == 0 else None,
            )
            stage_timings.append(
                StageTiming(
                    stage=f"retrieval_tool_attempt_{attempt}",
                    duration_ms=round((time.perf_counter() - t_ret) * 1000.0, 2),
                    details={"count": retrieval_out["count"], "top_score": retrieval_out["highest_score"]},
                )
            )

            t_dec = time.perf_counter()
            decision = self.decision_engine.evaluate(analysis, retrieval_out, refinement_attempt=attempt)
            stage_timings.append(
                StageTiming(
                    stage=f"decision_engine_attempt_{attempt}",
                    duration_ms=round((time.perf_counter() - t_dec) * 1000.0, 2),
                    details={"action": decision.action, "top_score": decision.top_score},
                )
            )

            if decision.action == "REFINE":
                refinement_attempts += 1
                query = decision.refined_query
                continue
            break

        if decision is not None and decision.action == "REFUSE":
            return AskResponse(
                request_id=request_id,
                question=req.question,
                answer="The provided research papers do not contain sufficient evidence to answer this question.",
                refused=True,
                refusal_reason=decision.reason,
                source=req.source,
                status="ok",
                total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                retrieval_setting=req.retrieval_setting_name or "default",
                stage_timings=stage_timings,
                verification_passed=True,
                refinement_attempts=refinement_attempts,
            )

        chunks = retrieval_out.get("chunks", [])
        est_tokens = max(300, sum(len(c["text"].split()) for c in chunks) * 4 // 3 + 200)

        # --- Queue, Rate Limiter, Retry & Model Generation ---
        async def _execute_in_slot(queue_wait_ms: float) -> AskResponse:
            nonlocal refinement_attempts
            stage_timings.append(StageTiming(stage="queue_wait", duration_ms=queue_wait_ms))

            # Rate-limit pacing before calling provider
            t_rl = time.perf_counter()
            rl_wait_total = 0.0
            while True:
                rl_check = await self.rate_limiter.check_and_reserve(estimated_tokens=est_tokens)
                if rl_check.allowed:
                    break
                if rl_check.is_daily_exhausted:
                    return AskResponse(
                        request_id=request_id,
                        question=req.question,
                        answer=(
                            "Daily API budget is exhausted. The service has temporarily paused "
                            "external model calls to protect provider limits; please retry after daily reset."
                        ),
                        refused=True,
                        refusal_reason=rl_check.reason,
                        source=req.source,
                        status="rate_limited",
                        queue_wait_ms=queue_wait_ms,
                        total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                        retrieval_setting=req.retrieval_setting_name or "default",
                        stage_timings=stage_timings,
                    )
                remaining_s = deadline_at - time.monotonic()
                if rl_check.wait_seconds >= remaining_s or remaining_s <= 0.05:
                    return AskResponse(
                        request_id=request_id,
                        question=req.question,
                        answer="Request could not be served within deadline due to rate-limit pacing.",
                        refused=True,
                        refusal_reason=rl_check.reason,
                        source=req.source,
                        status="rate_limited",
                        queue_wait_ms=queue_wait_ms,
                        total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                        retrieval_setting=req.retrieval_setting_name or "default",
                        stage_timings=stage_timings,
                    )
                sleep_s = min(rl_check.wait_seconds, remaining_s - 0.05)
                await asyncio.sleep(sleep_s)
                rl_wait_total += sleep_s * 1000.0

            if rl_wait_total > 0:
                stage_timings.append(StageTiming(stage="rate_limit_wait", duration_ms=round(rl_wait_total, 2)))

            # Execute Stage 4 (Generation) + Stage 5 (Verification) with bounded 1x verification loop
            total_retries = 0
            total_tokens_in = 0
            total_tokens_out = 0
            first_token_ms: Optional[float] = None
            active_chunks = chunks

            for ver_attempt in range(2):
                async def _model_op(_attempt: int):
                    return await self.generator.generate(
                        question=req.question,
                        chunks=active_chunks,
                        source=req.source,
                        stream=req.stream,
                    )

                (answer_text, meta), retries_used, retry_wait_ms = await self.retry_manager.run_with_retries(
                    _model_op, deadline_at=deadline_at
                )
                total_retries += retries_used
                total_tokens_in += meta.tokens_in
                total_tokens_out += meta.tokens_out
                if first_token_ms is None:
                    first_token_ms = meta.first_token_ms

                await self.rate_limiter.update_from_headers(meta.rate_limit_headers)
                await self.rate_limiter.reconcile_tokens(est_tokens, meta.tokens_in + meta.tokens_out)

                stage_timings.append(
                    StageTiming(
                        stage=f"answer_generation_{ver_attempt}",
                        duration_ms=meta.total_ms,
                        details={
                            "http_status": meta.http_status,
                            "first_token_ms": meta.first_token_ms,
                            "retries": retries_used,
                            "retry_wait_ms": retry_wait_ms,
                            "tokens_in": meta.tokens_in,
                            "tokens_out": meta.tokens_out,
                            "rate_limit_headers": meta.rate_limit_headers,
                        },
                    )
                )

                # Stage 5: Verification
                t_ver = time.perf_counter()
                ver_res = self.verifier.verify(answer_text, active_chunks)
                stage_timings.append(
                    StageTiming(
                        stage=f"answer_verification_{ver_attempt}",
                        duration_ms=round((time.perf_counter() - t_ver) * 1000.0, 2),
                        details={"passed": ver_res.passed, "reason": ver_res.reason},
                    )
                )

                if ver_res.passed:
                    return AskResponse(
                        request_id=request_id,
                        question=req.question,
                        answer=answer_text,
                        refused=ver_res.is_refusal,
                        refusal_reason=ver_res.reason if ver_res.is_refusal else None,
                        citations=ver_res.citations,
                        source=req.source,
                        status="ok",
                        retries=total_retries,
                        queue_wait_ms=queue_wait_ms,
                        first_token_ms=first_token_ms,
                        total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                        tokens_in=total_tokens_in,
                        tokens_out=total_tokens_out,
                        retrieval_setting=req.retrieval_setting_name or "default",
                        stage_timings=stage_timings,
                        verification_passed=True,
                        refinement_attempts=refinement_attempts,
                    )

                # Verification failed -> bounded single retry with expanded retrieval
                if ver_attempt == 0:
                    refinement_attempts += 1
                    refined_out = self.retriever.search_library(
                        query=analysis.cleaned_question,
                        top_k=min(10, top_k + 2),
                    )
                    if refined_out["chunks"]:
                        active_chunks = refined_out["chunks"]
                    continue

            # Never send an unverified answer! Force safe refusal.
            return AskResponse(
                request_id=request_id,
                question=req.question,
                answer="The provided research papers do not contain sufficient verified evidence to answer this question reliably.",
                refused=True,
                refusal_reason="Answer verification failed after bounded refinement; refused to avoid unverified claims.",
                source=req.source,
                status="ok",
                retries=total_retries,
                queue_wait_ms=queue_wait_ms,
                first_token_ms=first_token_ms,
                total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                retrieval_setting=req.retrieval_setting_name or "default",
                stage_timings=stage_timings,
                verification_passed=False,
                refinement_attempts=refinement_attempts,
            )

        try:
            q_timeout = min(self.settings.app_limits.queue_timeout_seconds, deadline_s)
            return await self.queue_manager.execute(request_id, _execute_in_slot, queue_timeout=q_timeout)
        except QueueFullError as exc:
            return AskResponse(
                request_id=request_id,
                question=req.question,
                answer="System is currently at capacity; please retry shortly.",
                refused=True,
                refusal_reason=str(exc),
                source=req.source,
                status="rate_limited",
                total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                retrieval_setting=req.retrieval_setting_name or "default",
                stage_timings=stage_timings,
            )
        except QueueTimeoutError as exc:
            return AskResponse(
                request_id=request_id,
                question=req.question,
                answer="Request timed out waiting in queue.",
                refused=True,
                refusal_reason=str(exc),
                source=req.source,
                status="timeout",
                total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                retrieval_setting=req.retrieval_setting_name or "default",
                stage_timings=stage_timings,
            )
        except CircuitOpenError as exc:
            return AskResponse(
                request_id=request_id,
                question=req.question,
                answer="Upstream model provider circuit breaker is open; failing fast.",
                refused=True,
                refusal_reason=str(exc),
                source=req.source,
                status="error",
                total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                retrieval_setting=req.retrieval_setting_name or "default",
                stage_timings=stage_timings,
            )
        except PermanentError as exc:
            if exc.error_type == "daily_quota_exhausted":
                await self.rate_limiter.mark_daily_exhausted()
                status_val = "rate_limited"
            else:
                status_val = "error"
            return AskResponse(
                request_id=request_id,
                question=req.question,
                answer=f"Service unavailable: {exc.error_type}",
                refused=True,
                refusal_reason=str(exc),
                source=req.source,
                status=status_val,
                total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                retrieval_setting=req.retrieval_setting_name or "default",
                stage_timings=stage_timings,
            )
        except RetryableError as exc:
            status_val = "rate_limited" if exc.error_type == "rate_limited" else ("timeout" if exc.error_type == "timeout" else "error")
            return AskResponse(
                request_id=request_id,
                question=req.question,
                answer=f"Request failed after bounded retries ({exc.error_type}).",
                refused=True,
                refusal_reason=str(exc),
                source=req.source,
                status=status_val,
                retries=self.retry_manager.max_retries,
                total_ms=round((time.perf_counter() - t_req_start) * 1000.0, 2),
                retrieval_setting=req.retrieval_setting_name or "default",
                stage_timings=stage_timings,
            )
