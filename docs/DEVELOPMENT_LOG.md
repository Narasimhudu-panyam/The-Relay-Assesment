# DEVELOPMENT LOG — THE RELAY PROJECT

## Session Start: 2026-11-10

### Initial Prompt:
User provided master prompt specifying complete implementation of `The_Relay_Assignment_Brief.pdf`.
Key mandate: Follow phased workflow, do not skip inspection and planning, verify every step, record all decisions and results, maintain strict honesty regarding real vs. stand-in data.

---

### Phase 0: Workspace & Requirements Inspection
- **Actions taken:**
  - Checked local environment: Windows 10, Git 2.54.0, Python 3.10.0 (active in PATH), Python 3.12 (available in AppData).
  - Located assignment brief at `C:\Users\new\Downloads\The_Relay_Assignment_Brief.pdf`.
  - Created project directory structure under `the-relay/`.
  - Produced `docs/ASSIGNMENT_REQUIREMENTS.md` with requirement matrix, traceability table, ambiguous requirements, and explicit assumptions.
  - Produced `docs/ARCHITECTURE.md` specifying asynchronous pipeline, decision engine branching, verification loop, stand-in architecture, rate/queue manager, and raw CSV logging.
- **Decisions & Observations:**
  - Active Python is 3.10.0. Python 3.10 is fully compatible with FastAPI, PyMuPDF, sentence-transformers, faiss-cpu, and groq SDK. We will set up a virtual environment `.venv` to keep dependencies isolated and reproducible.
  - The 3 papers need to be fetched directly from arXiv with their exact versions.

---

### Phase 1: Environment Setup & Paper Acquisition (In Progress)
- **Objectives:**
  1. Initialize virtual environment and install core pinned dependencies.
  2. Create `.env.example` and `.gitignore`.
  3. Fetch the three exact paper versions:
     - `1706.03762v7.pdf` (Attention Is All You Need)
     - `2410.00037v1.pdf` (Moshi)
     - `2507.07955v2.pdf` (Dynamic Chunking)
  4. Write automated test verifying file downloads, SHA-256 integrity, page count, and PyMuPDF text readability.

---

### Phases 6–10: Reliability, Integration, Real Groq Baseline, Calibration & Load Testing (2026-11-12)
- **Changes Implemented:**
  - `app/services/rate_limiter.py`: Sliding-window RPM/TPM + Daily RPD/TPD + `x-ratelimit-*` header ingestion.
  - `app/services/queue_manager.py`: Bounded FIFO queue, concurrency cap, deadlines, peak queue tracking.
  - `app/services/retry_manager.py`: Exponential backoff with jitter, `Retry-After` honoring, retry budget, 3-state `CircuitBreaker`.
  - `app/services/groq_service.py` & `app/services/standin_service.py`: Real Groq (`max_retries=0`) and FastAPI Stand-in service.
  - `app/pipeline/`: `question_analyzer.py`, `decision_engine.py`, `answer_generator.py`, `answer_verifier.py`, `orchestrator.py`, and `app/main.py`.
  - `experiments/`: `groq_baseline.py`, `calibrate_standin.py`, `load_test.py`, `validate_csv.py`, `logger.py`.
- **Errors Caught & Corrected:**
  1. `queue_manager.py`: Fixed async race condition where concurrent tasks could enter `wait_for(semaphore.acquire())` before `active_requests` incremented. Made slot reservation atomic under `self._lock`.
  2. `orchestrator.py`: Fixed `RetryableError` handler reporting `settings.app_limits.max_retries` instead of `self.retry_manager.max_retries`.
  3. `groq_service.py`: Fixed `raw_resp.parse()` in `groq==1.7.0` `AsyncGroq` needing `await raw_resp.parse()`.
- **Phase 8 Real Groq Baseline (`openai/gpt-oss-120b`):**
  - Model availability verified via `client.models.list()`: `AVAILABLE`.
  - `Q01` (`1029.33 ms` TTFT, `1756.97 ms` total, `2440/389` tokens) and `Q04` (`1235.20 ms` TTFT, `1567.97 ms` total, `3438/406` tokens) succeeded with verified citations.
  - `Q07`, `Q13`, `Q02` were proactively rate-limited locally (`rate_limited`) because `Q01` + `Q04` consumed `6,673` tokens in <6s against our `7,000 TPM` app cap (Groq's real `x-ratelimit-limit-tokens` header was `8000`, dropping to `1808` remaining).
- **Phase 9 Stand-in Calibration:**
  - Calibrated `base_ttft_ms=1132.26`, `ttft_jitter_ms=102.94`, `inter_token_ms=0.72` from the 2 real Groq calls (`seed=42`). Documented in `docs/STANDIN_CALIBRATION.md`.
- **Phase 10 Load Testing (Stand-in):**
  - Executed `steady` (100.0% ok), `burst` (55.6% ok, peak_q=10, 5 rejected full), `overload` (8.3% ok, 91.7% rl, peak_q=6, 5 rejected full), `halved_limit` (11.1% ok, 88.9% rl), `daily_exhausted` (0.0% ok, 100.0% rl).
  - Raw CSV schema validated (`99` rows, 12 exact columns).
