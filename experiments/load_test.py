"""
Phase 10: Reproducible Load Testing Harness (Stand-in Model Service).
Runs Scenario A (Steady), Scenario B (Burst), Scenario C (Sustained Overload),
plus Scenario D (Per-Minute Limit Halved) and Scenario E (Daily Budget Exhausted).
Saves raw per-request CSVs and reproducible summary metrics.
"""
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config.settings import get_settings
from app.pipeline.orchestrator import RelayPipeline
from app.pipeline.retriever import LibraryRetriever
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import FAISSVectorStore
from app.schemas.models import AskRequest, RawResultRow
from app.services.queue_manager import ConcurrencyAndQueueManager
from app.services.rate_limiter import RateLimiter
from app.services.retry_manager import CircuitBreaker, RetryManager
from app.services.standin_service import StandinClient, StandinConfig, state as standin_state
from experiments.logger import ResultsCSVLogger

SAMPLE_QUESTIONS = [
    "In Attention Is All You Need, what scaling factor is applied to the dot products before softmax?",
    "What is the name and frame rate of the neural audio codec used by Moshi?",
    "What problem does Dynamic Chunking (H-Net) solve compared to fixed BPE tokenization?",
    "How many parallel attention heads does the base Transformer model use?",
    "What is the Inner Monologue mechanism in Moshi?",
]


def load_calibrated_standin_config() -> StandinConfig:
    cal_file = BASE_DIR / "results" / "calibration_config.json"
    if cal_file.exists():
        with open(cal_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StandinConfig(**data["standin_config"])
    return StandinConfig(seed=42)


def percentile_or_none(values: List[float], p: float) -> Any:
    if not values:
        return None
    return round(float(np.percentile(values, p)), 2)


async def run_scenario(
    scenario_name: str,
    retriever: LibraryRetriever,
    standin_cfg: StandinConfig,
    total_requests: int,
    inter_arrival_seconds: float,
    burst_mode: bool,
    rpm_limit: int,
    tpm_limit: int,
    rpd_limit: int,
    max_concurrent: int,
    max_queue_depth: int,
    queue_timeout: float,
    max_retries: int,
    cb_threshold: int,
    daily_exhausted_at_start: bool = False,
) -> Dict[str, Any]:
    standin_state.config = standin_cfg.model_copy()
    standin_state.config.rpm_limit = rpm_limit
    standin_state.config.rpd_limit = rpd_limit
    standin_state.config.daily_exhausted = daily_exhausted_at_start
    standin_state.reset()

    rl = RateLimiter(rpm=rpm_limit, tpm=tpm_limit, rpd=rpd_limit, tpd=200000)
    if daily_exhausted_at_start:
        await rl.mark_daily_exhausted()

    qm = ConcurrencyAndQueueManager(
        max_concurrent=max_concurrent,
        max_queue_depth=max_queue_depth,
        default_queue_timeout=queue_timeout,
    )
    cb = CircuitBreaker(failure_threshold=cb_threshold, recovery_timeout_seconds=5.0)
    rm = RetryManager(
        max_retries=max_retries,
        base_delay_seconds=0.25,
        max_delay_seconds=2.0,
        retry_budget_ratio=0.30,
        circuit_breaker=cb,
    )
    pipeline = RelayPipeline(
        retriever=retriever,
        standin_client=StandinClient(in_process=True, timeout=8.0),
        rate_limiter=rl,
        queue_manager=qm,
        retry_manager=rm,
    )

    csv_rows: List[RawResultRow] = []
    start_ts_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    async def _one_request(idx: int):
        q = SAMPLE_QUESTIONS[idx % len(SAMPLE_QUESTIONS)]
        req_start_iso = datetime.now(timezone.utc).isoformat()
        res = await pipeline.process(
            AskRequest(
                question=q,
                source="stand-in",
                stream=True,
                retrieval_setting_name="S2_400w_k5_norerank",
                scenario=scenario_name,
                deadline_seconds=queue_timeout + 8.0,
            )
        )
        grade = None
        if res.status == "ok":
            grade = "correct" if (not res.refused and len(res.citations) > 0) else "refused_wrongly"
        return RawResultRow(
            request_id=res.request_id,
            scenario=scenario_name,
            source="stand-in",
            start_time=req_start_iso,
            first_token_ms=res.first_token_ms,
            total_ms=res.total_ms,
            status=res.status,
            retries=res.retries,
            tokens_in=res.tokens_in if res.tokens_in > 0 else None,
            tokens_out=res.tokens_out if res.tokens_out > 0 else None,
            retrieval_setting="S2_400w_k5_norerank",
            answer_grade=grade,
        )

    tasks = []
    for i in range(total_requests):
        tasks.append(asyncio.create_task(_one_request(i)))
        if not burst_mode and inter_arrival_seconds > 0 and i < total_requests - 1:
            await asyncio.sleep(inter_arrival_seconds)

    csv_rows = await asyncio.gather(*tasks)
    duration_s = round(time.perf_counter() - t0, 2)
    end_ts_iso = datetime.now(timezone.utc).isoformat()

    # Save scenario-specific CSV
    scenario_csv = BASE_DIR / "results" / f"load_{scenario_name}.csv"
    s_logger = ResultsCSVLogger(scenario_csv)
    s_logger.reset_file()
    s_logger.append_rows(csv_rows)

    # Append to master results/results.csv
    master_logger = ResultsCSVLogger(BASE_DIR / "results" / "results.csv")
    master_logger.append_rows(csv_rows)

    ok_rows = [r for r in csv_rows if r.status == "ok"]
    rl_rows = [r for r in csv_rows if r.status == "rate_limited"]
    to_rows = [r for r in csv_rows if r.status == "timeout"]
    err_rows = [r for r in csv_rows if r.status == "error"]

    ttfts = [r.first_token_ms for r in ok_rows if r.first_token_ms is not None]
    totals = [r.total_ms for r in csv_rows if r.total_ms is not None]
    ins = [r.tokens_in for r in ok_rows if r.tokens_in is not None]
    outs = [r.tokens_out for r in ok_rows if r.tokens_out is not None]

    q_stats = qm.get_stats()
    offered_rpm = round((total_requests / max(duration_s, 0.01)) * 60.0, 1)
    completed_rpm = round((len(ok_rows) / max(duration_s, 0.01)) * 60.0, 1)

    avg_in = round(float(np.mean(ins)), 1) if ins else None
    avg_out = round(float(np.mean(outs)), 1) if outs else None
    avg_total_tokens = round(avg_in + avg_out, 1) if (avg_in is not None and avg_out is not None) else None

    summary = {
        "scenario": scenario_name,
        "source": "stand-in",
        "start_timestamp": start_ts_iso,
        "end_timestamp": end_ts_iso,
        "requests_sent": total_requests,
        "duration_s": duration_s,
        "offered_rate_rpm": offered_rpm,
        "completed_rate_rpm": completed_rpm,
        "ttft_p50_ms": percentile_or_none(ttfts, 50),
        "ttft_p95_ms": percentile_or_none(ttfts, 95),
        "ttft_p99_ms": percentile_or_none(ttfts, 99),
        "total_p50_ms": percentile_or_none(totals, 50),
        "total_p95_ms": percentile_or_none(totals, 95),
        "total_p99_ms": percentile_or_none(totals, 99),
        "success_pct": round(100.0 * len(ok_rows) / total_requests, 1),
        "rate_limited_pct": round(100.0 * len(rl_rows) / total_requests, 1),
        "timeout_pct": round(100.0 * len(to_rows) / total_requests, 1),
        "error_pct": round(100.0 * len(err_rows) / total_requests, 1),
        "avg_retries_per_request": round(sum(r.retries for r in csv_rows) / total_requests, 2),
        "peak_queue_length": q_stats["peak_queue_length"],
        "max_concurrent_configured": max_concurrent,
        "queue_expiry_count": q_stats["total_timed_out"],
        "rejected_queue_full_count": q_stats["total_rejected_full"],
        "circuit_breaker_state_end": cb.state.value,
        "avg_tokens_in": avg_in,
        "avg_tokens_out": avg_out,
        "tokens_per_answer": avg_total_tokens,
        "cost_per_1000_answers": "$0.00 (Free Tier; unmeasured paid equivalent)",
        "config": {
            "rpm_limit": rpm_limit,
            "tpm_limit": tpm_limit,
            "max_concurrent": max_concurrent,
            "max_queue_depth": max_queue_depth,
            "queue_timeout": queue_timeout,
            "max_retries": max_retries,
            "seed": standin_cfg.seed,
        },
    }
    print(
        f"[{scenario_name}] sent={total_requests} dur={duration_s}s offered={offered_rpm}/m "
        f"ok={summary['success_pct']}% rl={summary['rate_limited_pct']}% "
        f"to={summary['timeout_pct']}% peak_q={summary['peak_queue_length']} "
        f"ttft_p50={summary['ttft_p50_ms']}ms total_p50={summary['total_p50_ms']}ms"
    )
    return summary


async def main():
    settings = get_settings()
    engine = EmbeddingEngine()
    store = FAISSVectorStore(embedding_engine=engine)
    retriever = LibraryRetriever(
        vector_store=store,
        index_dir=settings.indices_dir,
        chunk_size_words=400,
        chunk_overlap_words=50,
    )
    retriever.initialize(papers_dir=settings.papers_dir)
    standin_cfg = load_calibrated_standin_config()

    summaries = []

    # Scenario A: Steady load (12 requests spaced 1.5s apart -> ~40 RPM? No, 2.5s -> ~24 RPM, within 30 RPM)
    summaries.append(
        await run_scenario(
            scenario_name="steady",
            retriever=retriever,
            standin_cfg=standin_cfg,
            total_requests=12,
            inter_arrival_seconds=2.5,
            burst_mode=False,
            rpm_limit=30,
            tpm_limit=50000,
            rpd_limit=1000,
            max_concurrent=3,
            max_queue_depth=15,
            queue_timeout=8.0,
            max_retries=2,
            cb_threshold=5,
        )
    )

    # Scenario B: Sudden burst (18 requests arriving simultaneously)
    summaries.append(
        await run_scenario(
            scenario_name="burst",
            retriever=retriever,
            standin_cfg=standin_cfg,
            total_requests=18,
            inter_arrival_seconds=0.0,
            burst_mode=True,
            rpm_limit=30,
            tpm_limit=24000,
            rpd_limit=1000,
            max_concurrent=3,
            max_queue_depth=10,
            queue_timeout=5.0,
            max_retries=2,
            cb_threshold=5,
        )
    )

    # Scenario C: Sustained overload (36 requests at ~180 RPM, above 30 RPM and queue depth 8)
    overload_cfg = standin_cfg.model_copy()
    overload_cfg.failure_probability = 0.15
    summaries.append(
        await run_scenario(
            scenario_name="overload",
            retriever=retriever,
            standin_cfg=overload_cfg,
            total_requests=36,
            inter_arrival_seconds=0.25,
            burst_mode=False,
            rpm_limit=30,
            tpm_limit=8000,
            rpd_limit=1000,
            max_concurrent=2,
            max_queue_depth=6,
            queue_timeout=2.5,
            max_retries=2,
            cb_threshold=4,
        )
    )

    # Scenario D: Per-minute limit halved (15 RPM instead of 30 RPM, 18 requests over ~18s)
    summaries.append(
        await run_scenario(
            scenario_name="halved_limit",
            retriever=retriever,
            standin_cfg=standin_cfg,
            total_requests=18,
            inter_arrival_seconds=1.0,
            burst_mode=False,
            rpm_limit=15,
            tpm_limit=4000,
            rpd_limit=1000,
            max_concurrent=3,
            max_queue_depth=10,
            queue_timeout=4.0,
            max_retries=1,
            cb_threshold=5,
        )
    )

    # Scenario E: Daily budget exhausted (10 requests arrive after daily budget is gone)
    summaries.append(
        await run_scenario(
            scenario_name="daily_exhausted",
            retriever=retriever,
            standin_cfg=standin_cfg,
            total_requests=10,
            inter_arrival_seconds=0.2,
            burst_mode=False,
            rpm_limit=30,
            tpm_limit=8000,
            rpd_limit=1000,
            max_concurrent=3,
            max_queue_depth=10,
            queue_timeout=4.0,
            max_retries=2,
            cb_threshold=5,
            daily_exhausted_at_start=True,
        )
    )

    out_summary = BASE_DIR / "results" / "load_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved load test summary to {out_summary}")


if __name__ == "__main__":
    asyncio.run(main())
