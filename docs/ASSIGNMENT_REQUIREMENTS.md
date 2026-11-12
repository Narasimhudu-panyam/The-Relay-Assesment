# THE RELAY — ASSIGNMENT REQUIREMENTS & TRACEABILITY

## 1. Overview and Core Premise
The system is an asynchronous AI assistant answering engineers' questions about a small research paper library (3 specific arXiv papers).
It is **not** a generic RAG chatbot: it is an end-to-end model orchestration, retrieval, reliability, and performance engineering system.

---

## 2. Requirements Matrix: Mandatory vs. Optional

| ID | Category | Requirement Description | Type | Source in Brief |
|---|---|---|---|---|
| **REQ-01** | Pipeline | Implement an asynchronous pipeline with at least 4 steps | **Mandatory** | Page 1 ("at least four steps") |
| **REQ-02** | Pipeline | Implement at least one actual tool call | **Mandatory** | Page 1 ("at least one tool call") |
| **REQ-03** | Pipeline | Implement at least one explicit branch or loop | **Mandatory** | Page 1 ("at least one branch or loop") |
| **REQ-04** | Pipeline | Step must retrieve passages from the 3 specified papers | **Mandatory** | Page 1 ("One step must retrieve passages...") |
| **REQ-05** | Library | Paper 1: "Attention Is All You Need", arXiv:1706.03762v7 | **Mandatory** | Page 1 |
| **REQ-06** | Library | Paper 2: "Moshi: a speech-text foundation model for real-time dialogue", arXiv:2410.00037v1 | **Mandatory** | Page 1 |
| **REQ-07** | Library | Paper 3: "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", arXiv:2507.07955v2 | **Mandatory** | Page 1 |
| **REQ-08** | Citations | Every answer must cite where in the papers it came from (Title, Page, Section) | **Mandatory** | Page 1 & Prompt Section 4 |
| **REQ-09** | Refusal | If papers do not say, assistant must say so instead of guessing (correct refusal) | **Mandatory** | Page 1 & Prompt Section 4 |
| **REQ-10** | Provider | Use a real hosted model on Groq's free tier with own API key | **Mandatory** | Page 1 |
| **REQ-11** | Provider | Discover Groq limits from documentation & empirical 429 response headers | **Mandatory** | Page 1, Page 3 (Table 1) |
| **REQ-12** | Stand-in | Implement a separate FastAPI stand-in model service you control | **Mandatory** | Page 1 & Prompt Section 6 |
| **REQ-13** | Stand-in | Stand-in must support: slow response, random/configured failure, 429 simulation, daily budget exhaustion, latency increase under concurrent load, token streaming piece-by-piece, configurable response sizes | **Mandatory** | Page 1 & Prompt Section 6 |
| **REQ-14** | Stand-in | Calibrate stand-in from real Groq measurements; document what differs | **Mandatory** | Page 1, Page 2 |
| **REQ-15** | Experiments | Run real service with modest requests to learn latency spread and limits | **Mandatory** | Page 1 |
| **REQ-16** | Experiments | Run pipeline under at least 3 load shapes (steady, sudden burst, sustained overload) + 2 extra (limit halved, daily budget exhausted) | **Mandatory** | Page 1, Page 3 (Table 2a/2b) |
| **REQ-17** | Experiments | Test at least 3 retrieval configurations (varying chunk size, top_k, re-ranking) | **Mandatory** | Page 2, Page 3 (Table 3) |
| **REQ-18** | Evaluation | Evaluate at least 15 self-written quality questions across configurations | **Mandatory** | Page 2, Prompt Section 9 |
| **REQ-19** | Reliability | Implement rate limiting (RPM/TPM/RPD), bounded queues, timeouts, concurrency limits, deadlines, retry budget, exponential backoff with jitter, retry-after, circuit breaker | **Mandatory** | Prompt Section 7 |
| **REQ-20** | Deliverable | `results/results.csv` with exact 12 columns | **Mandatory** | Page 4 |
| **REQ-21** | Deliverable | Maximum 2-page engineering report using exact headings and order | **Mandatory** | Page 2 |
| **REQ-22** | Deliverable | Fully populated Results Tables 1, 2a, 2b, 3 matching `results.csv` exactly | **Mandatory** | Page 2, 3 |
| **REQ-23** | Deliverable | Raw conversation history with assistant and evaluation section | **Mandatory** | Page 2 |
| **REQ-24** | Deliverable | Feedback on Vaani (https://www.vaani.dev/) based on real review | **Mandatory** | Page 2 |
| **OPT-01** | Architecture | Advanced hybrid dense-sparse retrieval (BM25 + FAISS) | Optional / Enhanced | Enhances retrieval robustness |
| **OPT-02** | UI | Web UI / Frontend | Explicitly Discouraged | Prompt Section 15 ("Avoid unnecessary UI") |

---

## 3. Requirement-to-Implementation Traceability Matrix

| Requirement | Design / Implementation Module | Test / Verification Method | Output / Artifact |
|---|---|---|---|
| **REQ-01 (4 Steps)** | `app/pipeline/`: `question_analyzer.py`, `retriever.py`, `decision_engine.py`, `answer_generator.py`, `answer_verifier.py` (5 steps) | `tests/test_pipeline.py` | Pipeline execution log & timing |
| **REQ-02 (Tool Call)** | `retrieval/vector_store.py` invoked as an explicit retrieval tool call + metadata inspector tool | `tests/test_pipeline.py` | Tool invocation traces in response metadata |
| **REQ-03 (Branch/Loop)** | `decision_engine.py` (branch to answer vs refuse) & `answer_verifier.py` (loop back to refine retrieval or refuse) | `tests/test_decision_engine.py` | Verification branch traces |
| **REQ-04 to 07 (Papers)** | `papers/` downloaded via arXiv direct URL; `retrieval/document_loader.py` with PyMuPDF | `tests/test_document_loader.py` | Verified PDF checksums and extracted text blocks |
| **REQ-08, 09 (Cite & Refuse)** | `answer_generator.py` prompt template enforcing strict citation format; `answer_verifier.py` checking claim grounding | `tests/test_verification.py` | `answer_grade` in `results.csv` |
| **REQ-10, 11 (Groq API)** | `services/groq_service.py` using official `groq` SDK with zero default SDK retries, header inspection | `tests/test_groq_service.py` | Table 1 in report |
| **REQ-12 to 14 (Stand-in)** | `services/standin_service.py` FastAPI app emulating Groq endpoint with calibrated latency distribution & failure modes | `tests/test_standin_service.py` | Calibration section in report |
| **REQ-16 (Load Scenarios)** | `experiments/load_test.py` running 5 scenarios (steady, burst, overload, halved limit, exhausted budget) | Automated run script | Table 2a, 2b in report |
| **REQ-17 (Retrieval Tuning)** | `experiments/retrieval_test.py` testing 3 settings: (S1: 300 words, k=3, no re-rank), (S2: 600 words, k=5, no re-rank), (S3: 600 words, k=8, with re-rank) | Automated evaluation script | Table 3 in report |
| **REQ-18 (15 Quality Qs)** | `experiments/benchmark_questions.json` containing 15 targeted questions (grounded, out-of-scope, misleading, adversarial) | `experiments/quality_test.py` | Grade breakdown in report |
| **REQ-19 (Rate & Retries)** | `services/rate_limiter.py`, `services/queue_manager.py`, `services/retry_manager.py` | `tests/test_rate_limiter.py` | Retries and queue metrics in `results.csv` |
| **REQ-20 (CSV Export)** | `experiments/logger.py` appending to `results/results.csv` with exact 12 columns | `tests/test_csv_format.py` | Validated `results/results.csv` |
| **REQ-21 to 24 (Report)** | `docs/REPORT.md` following exact 8 headings + 2 additional + Vaani + Assistant reflection | Word count & rubric check | 2-page report |

---

## 4. Ambiguous Requirements & Documented Assumptions

1. **"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", arXiv 2507.07955v2**:
   - *Ambiguity:* arXiv identifiers with "2507" typically indicate year 2025 month 07. We will fetch exactly `https://arxiv.org/pdf/2507.07955v2` or inspect the arXiv endpoint to ensure exact version matching.
   - *Assumption:* We will download the exact version specified and record its SHA256 checksum and title.
2. **"At least one tool call and at least one branch or loop"**:
   - *Decision:* The pipeline will structure `retriever.search_library` as a formal callable tool within the orchestration flow. The decision engine introduces an explicit branch (sufficient evidence -> generate, insufficient -> immediate refusal), and the answer verifier introduces a bounded loop (if ungrounded claims detected, execute one query reformulation loop; if still failing, produce a safe refusal).
3. **Groq Model Selection**:
   - *Decision:* We will use `llama-3.1-8b-instant` (or current active equivalent such as `llama-3.3-70b-versatile` if available on free tier). `llama-3.1-8b-instant` is ideal because it has high free-tier RPM/TPM allowances allowing controlled rate limit calibration without premature daily lockout.
4. **Pricing Assumptions for Cost per 1,000 Answers**:
   - *Decision:* Even though Groq free tier is free of charge, to compute "Cost per 1,000 answers" realistically for Table 2b, we will document Groq's standard pay-as-you-go pricing for the chosen model (e.g., \$0.05 / 1M input tokens, \$0.08 / 1M output tokens for Llama 3.1 8B).
5. **Report Length Constraint**:
   - *Constraint:* Maximum 2 pages of writing.
   - *Decision:* Keep `docs/REPORT.md` dense, rigorous, and within ~1,000 to 1,200 words (standard 2-page single-spaced format), referencing appendices in `docs/` for extensive raw logs.
