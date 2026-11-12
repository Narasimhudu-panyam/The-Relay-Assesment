# FOUR-DAY IMPLEMENTATION PLAN & RISK REGISTER

## 1. Four-Day Phased Roadmap

| Day / Phase | Primary Milestone | Key Deliverables & Verifications |
|---|---|---|
| **Phase 1 (Day 1)** | **Environment Setup & Corpus Ingestion** | <ul><li>Python virtual environment creation & dependency pinning.</li><li>Download & hash-verify the 3 exact arXiv papers: `1706.03762v7`, `2410.00037v1`, `2507.07955v2`.</li><li>`document_loader.py` & `chunker.py` with section/page tracking.</li><li>Unit tests: text extraction fidelity, chunk metadata bounds.</li></ul> |
| **Phase 2 (Day 2)** | **Pipeline & Real Groq Integration** | <ul><li>5-stage async pipeline: Question Analyzer, Retriever, Decision Engine, Answer Generator, Answer Verifier.</li><li>Tool call (`search_library`) and decision branch/loop (refuse vs refine).</li><li>`groq_service.py` with zero SDK retries, header inspection (`x-ratelimit-*`).</li><li>Rate, queue & retry managers (`rate_limiter.py`, `queue_manager.py`, `retry_manager.py`).</li><li>Unit tests: retrieval accuracy, refusal on out-of-scope questions, citation verification.</li></ul> |
| **Phase 3 (Day 3)** | **Stand-in Service & Calibration** | <ul><li>FastAPI Stand-in service replicating Groq's `/openai/v1/chat/completions`.</li><li>Calibrate stand-in latency (TTFT, inter-token generation) against empirical Groq measurements.</li><li>Implement simulated failure modes: 429 rate limit, 503 overload, daily quota exhaustion, concurrency drag.</li><li>Unit tests: stand-in streaming, error generation, queue overflow.</li></ul> |
| **Phase 4 (Day 4)** | **Experiments, Results Generation & Final Report** | <ul><li>Run 3 retrieval configurations on 15 quality questions (`retrieval_test.py`).</li><li>Run 5 load scenarios on stand-in & modest real runs (`load_test.py`): steady, burst, sustained overload, halved limits, daily exhausted.</li><li>Produce populated `results/results.csv` (12 exact columns).</li><li>Generate Tables 1, 2a, 2b, 3 from raw CSV data.</li><li>Draft exact 2-page final report in `docs/REPORT.md`.</li></ul> |

---

## 2. Dependency & Installation Plan

- **Platform:** Windows 10, Python 3.10.
- **Environment:** Dedicated isolated virtual environment `d:\assessment\the-relay\.venv`.
- **Pre-built binary wheels:**
  - `pymupdf` provides native C-extension wheels for Windows x86_64.
  - `faiss-cpu` provides pre-compiled Windows wheels via pip.
  - `torch` (for `sentence-transformers`): CPU wheel to keep installation light and fast without requiring CUDA.
  - `uvicorn` and `fastapi`: asynchronous ASGI runtime.

---

## 3. Risk Register & Mitigations

| Risk | Severity | Mitigation Strategy |
|---|---|---|
| **Groq Free Tier Lockout** (Daily limit hit during test development) | **High** | Strictly decouple test harness: use Stand-in for heavy load tests, mock unit tests for CI, and reserve real Groq calls only for calibration and modest validation (15-20 requests max). |
| **arXiv 2507.07955v2 availability** (arXiv identifier formatting or connection drops) | **Medium** | Verify arXiv download response headers, content type (`application/pdf`), and inspect file size/page count with PyMuPDF immediately upon download. |
| **Windows Process & Event Loop Quirks** (Subprocess / ProactorEventLoop issues with asyncio) | **Low** | Use standard standard-library `asyncio` constructs compatible with Windows Proactor loop; ensure HTTP clients reuse persistent `httpx.AsyncClient`. |
| **Retriever Hallucinations / Naive Matching** | **Medium** | Pre-design 3 canonical failure modes where terminology collides across papers (e.g., "Moshi audio tokens" vs "Transformer BPE tokens", "Dynamic chunking" vs standard chunking). Implement strict verifier checks. |
