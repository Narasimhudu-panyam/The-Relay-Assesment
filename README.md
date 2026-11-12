# The Relay — Resilient Research Paper Assistant

An asynchronous, reliability-engineered AI research assistant answering engineering questions strictly from three arXiv papers (`1706.03762v7`, `2410.00037v1`, `2507.07955v2`), built with FastAPI, FAISS, `sentence-transformers/all-MiniLM-L6-v2`, and Groq (`openai/gpt-oss-120b`).

### Architecture & Technologies
- **5-Stage Async Pipeline (`app/pipeline/`)**: `QuestionAnalyzer` → `LibraryRetriever.search_library` (FAISS tool call) → `DecisionEngine` (evidence gate + bounded refinement) → `AnswerGenerator` → `AnswerVerifier` (citation/page check + bounded refinement).
- **Reliability Layer (`app/services/`)**: Sliding-window `RateLimiter` (RPM/TPM/RPD/TPD + `x-ratelimit-*` headers), bounded FIFO `ConcurrencyAndQueueManager`, `RetryManager` (exponential backoff + jitter + `Retry-After` + retry budget), 3-state `CircuitBreaker`, and calibrated FastAPI `standin_service.py`.
- **Stack**: Python 3.10, FastAPI, Uvicorn, HTTPX, Pydantic, Groq SDK (`max_retries=0`), PyMuPDF, Sentence Transformers, FAISS CPU, Pandas, pytest, Locust.
- **Final Report**: See [docs/REPORT.pdf](docs/REPORT.pdf) and [docs/REPORT.md](docs/REPORT.md).

---

## 1. Setup & Environment

```powershell
cd d:\assessment\the-relay
.\.venv\Scripts\Activate.ps1
```

Copy `.env.example` to `.env` and add your `GROQ_API_KEY` locally (`.env` is gitignored):
```env
GROQ_API_KEY=your_secret_key
GROQ_MODEL=openai/gpt-oss-120b
```

Verify the 3 exact research papers:
```powershell
python papers\verify_papers.py
```

---

## 2. Running the Services

Start the **Stand-in Model Service** (optional standalone mode, port `8001`):
```powershell
uvicorn app.services.standin_service:standin_app --host 127.0.0.1 --port 8001
```

Start the **Relay FastAPI Gateway** (port `8000`):
```powershell
$env:HF_HUB_OFFLINE="1"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Endpoints:
- `GET /health`
- `GET /metrics`
- `POST /ask` (supports `"source": "stand-in"` or `"real"`, and `?sse=true` for streaming)

---

## 3. Running Tests

```powershell
$env:HF_HUB_OFFLINE="1"
pytest -c pytest.ini -v
```

---

## 4. Reproducing Experiments

```powershell
$env:HF_HUB_OFFLINE="1"

# Phase 8: Controlled 5-question real Groq baseline (uses GROQ_API_KEY)
python experiments\groq_baseline.py

# Phase 9: Calibrate stand-in from real Groq measurements
python experiments\calibrate_standin.py

# Phase 10: Run 5 load scenarios (steady, burst, overload, halved_limit, daily_exhausted)
python experiments\load_test.py

# Phase 11: Evaluate 3 retrieval configurations across 16 benchmark questions
python experiments\retrieval_test.py

# Phase 12: Validate results/results.csv schema & recompute summary metrics
python experiments\validate_csv.py
```

---

## 5. Key Artifacts

- [docs/REPORT.md](docs/REPORT.md) — Final 2-page engineering report
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture & reliability design
- [docs/STANDIN_CALIBRATION.md](docs/STANDIN_CALIBRATION.md) — Stand-in calibration details
- [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md) — Audit trail of prompts, bugs caught, and decisions
- [docs/FINAL_AUDIT.md](docs/FINAL_AUDIT.md) — Final submission QA checklist
- [results/results.csv](results/results.csv) — Raw per-request telemetry (147 rows, 12 exact columns)
