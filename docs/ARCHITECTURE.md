# THE RELAY — SYSTEM ARCHITECTURE SPECIFICATION

## 1. High-Level Architecture Overview

The Relay is an asynchronous, resilient question-answering system designed for engineering teams querying a defined research paper corpus. The architecture separates concerns between **model execution**, **retrieval orchestration**, **provider reliability engineering**, and **observability**.

```
                           [ Engineer / Client ]
                                     │
                                     ▼
                        [ FastAPI Gateway: /query ]
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
            [ Real Groq Router ]            [ Stand-in Router ]
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                   [ Concurrency Limiter & Bounded Queue ]
                                     │
                   [ Rate Limiter (RPM / TPM / RPD) ]
                                     │
                   [ Circuit Breaker & Retry Manager ]
                                     │
                                     ▼
             ================ AI Pipeline ================
             Stage 1: Question Understanding (Validate & Extract)
                                     │
             Stage 2: Retrieval Tool (Dense FAISS Vector Search)
                                     │
             Stage 3: Decision Engine (Evidence Sufficiency Gate)
                     ├── Insufficient ──> [ Correct Refusal ]
                     └── Sufficient   ──> Stage 4
                                     │
             Stage 4: Answer Generator (Strict Grounded Prompting)
                                     │
             Stage 5: Answer Verifier (Citation & Grounding Check)
                     ├── Valid   ────────> [ Final Verified Answer ]
                     └── Invalid ────────> [ 1x Refinement Loop / Refusal ]
             =============================================
                                     │
                                     ▼
                   [ Raw Results Recorder: results.csv ]
```

---

## 2. Pipeline Components

### 2.1 Stage 1: Question Understanding (`app/pipeline/question_analyzer.py`)
- Sanitizes and validates user input (rejects empty or oversized queries).
- Extracts core technical entities, query intent, and whether the question is out-of-scope for the corpus.
- Emits structured metadata (`analyzed_query`, `intent`, `is_comparative`).

### 2.2 Stage 2: Retrieval (`app/pipeline/retriever.py` & `app/retrieval/vector_store.py`)
- Implemented as a callable tool interface (`search_library(query, top_k, filter_paper)`).
- Chunking strategy: PyMuPDF parses page-by-page text, preserving section headers, page numbers, and paper IDs.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (fast, deterministic, zero-external-API cost).
- Storage: FAISS vector index with metadata mapping for chunk title, page, section header, and character offset.
- Optional Re-ranker: Cross-encoder or BM25 reciprocal rank fusion for retrieval experiments.

### 2.3 Stage 3: Decision Engine (`app/pipeline/decision_engine.py`)
- **Explicit Branching Point**:
  - Calculates retrieval similarity confidence scores.
  - If top scores fall below confidence threshold or chunks lack relevant terms: branches immediately to **Refusal Mode** ("The papers do not contain sufficient evidence to answer...").
  - If evidence is sufficient: branches to **Answer Generation**.

### 2.4 Stage 4: Answer Generator (`app/pipeline/answer_generator.py`)
- Calls Groq API (or Stand-in) with temperature=0.0.
- System prompt enforces:
  - Answers must strictly originate from provided excerpts.
  - Explicit inline citations mandatory: `[Paper Title, Section X, Page Y]`.
  - Zero external hallucination.

### 2.5 Stage 5: Answer Verifier (`app/pipeline/answer_verifier.py`)
- **Explicit Loop / Gate**:
  - Parses generated citations against retrieved chunk metadata.
  - Verifies factual claim alignment (NLI verification or lexical cross-check).
  - If verification fails: executes a bounded single retry loop with stricter retrieval or safely downgrades to a correct refusal rather than returning an embarrassing unverified answer.

---

## 3. Reliability & Provider Protection (`app/services/`)

### 3.1 Rate Limiting (`services/rate_limiter.py`)
- Sliding-window tracking for:
  - **RPM** (Requests Per Minute)
  - **TPM** (Tokens Per Minute)
  - **RPD** (Requests Per Day)
- Transparently halts or queues requests when approaching provider saturation.

### 3.2 Bounded Queue & Concurrency Manager (`services/queue_manager.py`)
- Caps maximum concurrent in-flight requests (e.g., max 2-5 concurrent).
- Requests beyond capacity enter a FIFO queue with maximum depth (e.g. 20).
- Queue timeout (deadline enforcement): requests waiting > 10s exit with HTTP 504 / `status=timeout` and are accounted as given-up requests.

### 3.3 Retry Manager with Exponential Backoff & Jitter (`services/retry_manager.py`)
- Intercepts HTTP 429 and transient 5xx errors.
- Honors `Retry-After` HTTP headers from Groq.
- Applies truncated exponential backoff with full jitter:
  $$\Delta t = \min(t_{\max}, t_{\text{base}} \times 2^{\text{attempt}}) \times \text{Uniform}(0.5, 1.5)$$
- Enforces strict retry budget (max 2 retries) to prevent retry storms.

### 3.4 Circuit Breaker
- States: `CLOSED` (normal), `OPEN` (fail-fast without calling Groq when provider is down or daily budget is dead), `HALF_OPEN` (probe request).
- Protects the real provider and avoids wasting daily quota on doomed requests.

---

## 4. Stand-in Model Service (`app/services/standin_service.py`)

A fully controllable FastAPI mock server simulating Groq's `/openai/v1/chat/completions` endpoint:
- **Calibrated Latency:** Emulates measured TTFT (time-to-first-token) and inter-token generation speed using empirical parameters from real Groq runs.
- **Concurrency Penalty:** Simulates latency degradation as concurrent in-flight requests increase.
- **Controllable Failure Injections:**
  - Configurable 429 Rate Limit (with dynamic `retry-after` headers).
  - Daily Quota Exhaustion simulation (429 with specific error body: `insufficient_quota`).
  - Network timeouts & 500 internal server errors.
- **Streaming Support:** Server-Sent Events (`text/event-stream`) streaming tokens piece by piece.

---

## 5. Observability & Raw Metric Collection (`experiments/logger.py`)

Every request records an entry directly into `results/results.csv`:
- `request_id`: UUID
- `scenario`: name of load scenario
- `source`: `real` or `stand-in`
- `start_time`: ISO8601 timestamp
- `first_token_ms`: milliseconds to first streamed chunk
- `total_ms`: milliseconds to full response completion
- `status`: `ok`, `rate_limited`, `timeout`, `error`
- `retries`: integer count
- `tokens_in`: prompt token count
- `tokens_out`: completion token count
- `retrieval_setting`: setting identifier (e.g., `S1_300w_k3_norerank`)
- `answer_grade`: `correct`, `wrong`, `refused_correctly`, `refused_wrongly`
