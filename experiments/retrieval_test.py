"""
Phase 11: Retrieval Quality vs. Cost Experiment.
Evaluates all 16 benchmark questions across 3 retrieval configurations,
varying chunk size, overlap, top-k passages, and re-ranking.
Uses deterministic ground-truth evidence checks against the original papers
rather than LLM-as-a-judge.
"""
import asyncio
import json
import sys
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

CONFIGS = [
    {
        "name": "S1_250w_k3_norerank",
        "chunk_size": 250,
        "overlap": 40,
        "top_k": 3,
        "rerank": False,
    },
    {
        "name": "S2_400w_k5_norerank",
        "chunk_size": 400,
        "overlap": 50,
        "top_k": 5,
        "rerank": False,
    },
    {
        "name": "S3_500w_k7_rerank",
        "chunk_size": 500,
        "overlap": 75,
        "top_k": 7,
        "rerank": True,
    },
]


def load_calibrated_standin() -> StandinConfig:
    cal_file = BASE_DIR / "results" / "calibration_config.json"
    if cal_file.exists():
        with open(cal_file, "r", encoding="utf-8") as f:
            return StandinConfig(**json.load(f)["standin_config"])
    return StandinConfig(seed=42)


def grade_against_paper_evidence(
    q_item: Dict[str, Any],
    res_refused: bool,
    res_citations: List[Any],
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    expected = q_item["expected_behavior"]
    target_paper = q_item["target_paper"]
    gt_keywords = [kw.lower() for kw in q_item.get("ground_truth_keywords", [])]

    if expected == "refuse":
        return "refused_correctly" if res_refused else "wrong"

    # Expected to answer
    if res_refused:
        return "refused_wrongly"

    if not res_citations:
        return "wrong"

    # Verify retrieved chunks actually came from the target paper(s) and contain ground truth terms
    retrieved_papers = {c["paper_id"] for c in retrieved_chunks}
    if target_paper == "cross":
        paper_match = len(retrieved_papers) >= 2
    else:
        paper_match = target_paper in retrieved_papers

    combined_text = " ".join(c["text"].lower() for c in retrieved_chunks)
    kw_hits = sum(1 for kw in gt_keywords if kw in combined_text)
    min_hits = 1 if len(gt_keywords) <= 2 else 2

    if paper_match and kw_hits >= min_hits:
        return "correct"
    return "wrong"


async def evaluate_config(
    cfg: Dict[str, Any],
    questions: List[Dict[str, Any]],
    engine: EmbeddingEngine,
    standin_cfg: StandinConfig,
) -> Dict[str, Any]:
    settings = get_settings()
    store = FAISSVectorStore(embedding_engine=engine)
    idx_dir = settings.indices_dir / cfg["name"]
    retriever = LibraryRetriever(
        vector_store=store,
        index_dir=idx_dir,
        chunk_size_words=cfg["chunk_size"],
        chunk_overlap_words=cfg["overlap"],
        use_reranking=cfg["rerank"],
    )
    chunks_count = retriever.initialize(papers_dir=settings.papers_dir)
    print(f"\n=== Evaluating {cfg['name']} (chunks={chunks_count}, top_k={cfg['top_k']}, rerank={cfg['rerank']}) ===")

    standin_state.config = standin_cfg.model_copy()
    standin_state.reset()

    rl = RateLimiter(rpm=120, tpm=500000, rpd=5000, tpd=1000000)
    qm = ConcurrencyAndQueueManager(max_concurrent=4, max_queue_depth=20, default_queue_timeout=15.0)
    rm = RetryManager(max_retries=1, circuit_breaker=CircuitBreaker())
    pipeline = RelayPipeline(
        retriever=retriever,
        standin_client=StandinClient(in_process=True, timeout=15.0),
        rate_limiter=rl,
        queue_manager=qm,
        retry_manager=rm,
    )

    rows: List[RawResultRow] = []
    details: List[Dict[str, Any]] = []

    for q_item in questions:
        start_iso = datetime.now(timezone.utc).isoformat()
        # Capture retrieved chunks directly for deterministic evidence verification
        ret_check = retriever.search_library(q_item["question"], top_k=cfg["top_k"])
        res = await pipeline.process(
            AskRequest(
                question=q_item["question"],
                source="stand-in",
                stream=True,
                top_k=cfg["top_k"],
                use_reranking=cfg["rerank"],
                retrieval_setting_name=cfg["name"],
                scenario="retrieval_eval",
            )
        )
        grade = grade_against_paper_evidence(
            q_item=q_item,
            res_refused=res.refused,
            res_citations=res.citations,
            retrieved_chunks=ret_check["chunks"],
        )
        rows.append(
            RawResultRow(
                request_id=res.request_id,
                scenario="retrieval_eval",
                source="stand-in",
                start_time=start_iso,
                first_token_ms=res.first_token_ms,
                total_ms=res.total_ms,
                status=res.status,
                retries=res.retries,
                tokens_in=res.tokens_in if res.tokens_in > 0 else None,
                tokens_out=res.tokens_out if res.tokens_out > 0 else None,
                retrieval_setting=cfg["name"],
                answer_grade=grade,
            )
        )
        details.append({
            "qid": q_item["id"],
            "expected": q_item["expected_behavior"],
            "refused": res.refused,
            "top_score": ret_check["highest_score"],
            "grade": grade,
            "tokens_in": res.tokens_in,
            "tokens_out": res.tokens_out,
            "ttft_ms": res.first_token_ms,
            "total_ms": res.total_ms,
        })
        print(f"  [{q_item['id']}] expected={q_item['expected_behavior']} refused={res.refused} score={ret_check['highest_score']:.3f} grade={grade} tokens={res.tokens_in}/{res.tokens_out}")

    # Append to master results.csv and retrieval-specific CSV
    ret_csv_logger = ResultsCSVLogger(BASE_DIR / "results" / "retrieval_experiments.csv")
    ret_csv_logger.append_rows(rows)
    master_logger = ResultsCSVLogger(BASE_DIR / "results" / "results.csv")
    master_logger.append_rows(rows)

    n = len(rows)
    correct_n = sum(1 for r in rows if r.answer_grade == "correct")
    wrong_n = sum(1 for r in rows if r.answer_grade == "wrong")
    refused_ok_n = sum(1 for r in rows if r.answer_grade == "refused_correctly")
    refused_wrong_n = sum(1 for r in rows if r.answer_grade == "refused_wrongly")

    answered = [r for r in rows if (r.tokens_in or 0) > 0]
    tok_per_ans = round(float(np.mean([(r.tokens_in or 0) + (r.tokens_out or 0) for r in answered])), 1) if answered else None
    ttfts = [r.first_token_ms for r in rows if r.first_token_ms is not None]
    totals = [r.total_ms for r in rows if r.total_ms is not None]

    return {
        "setting": cfg["name"],
        "chunk_size_words": cfg["chunk_size"],
        "overlap_words": cfg["overlap"],
        "pieces_fetched": cfg["top_k"],
        "reranking": cfg["rerank"],
        "total_chunks_indexed": chunks_count,
        "questions_asked": n,
        "correct_and_cited_pct": round(100.0 * correct_n / n, 1),
        "wrong_pct": round(100.0 * wrong_n / n, 1),
        "correctly_refused_pct": round(100.0 * refused_ok_n / n, 1),
        "refused_wrongly_pct": round(100.0 * refused_wrong_n / n, 1),
        "tokens_per_answer": tok_per_ans,
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else None,
        "ttft_p95_ms": round(float(np.percentile(ttfts, 95)), 1) if ttfts else None,
        "total_p50_ms": round(float(np.percentile(totals, 50)), 1) if totals else None,
        "total_p95_ms": round(float(np.percentile(totals, 95)), 1) if totals else None,
        "details": details,
    }


async def main():
    q_path = BASE_DIR / "experiments" / "benchmark_questions.json"
    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Reset retrieval_experiments.csv for clean run
    ResultsCSVLogger(BASE_DIR / "results" / "retrieval_experiments.csv").reset_file()

    engine = EmbeddingEngine()
    standin_cfg = load_calibrated_standin()
    summaries = []

    for cfg in CONFIGS:
        summaries.append(await evaluate_config(cfg, questions, engine, standin_cfg))

    out_json = BASE_DIR / "results" / "retrieval_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved retrieval summary to {out_json}")


if __name__ == "__main__":
    asyncio.run(main())
