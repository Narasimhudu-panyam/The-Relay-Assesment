# STAND-IN MODEL SERVICE CALIBRATION REPORT

**Calibration Date:** 2026-11-12  
**Target Real Model:** `openai/gpt-oss-120b` (Groq Free Tier)  
**Random Seed:** `42` (fixed for reproducible load tests)  
**Calibration Artifact:** `results/calibration_config.json`

---

## 1. Real Groq Baseline Observations (Sample Size & Findings)

Five controlled requests (`Q01`, `Q04`, `Q07`, `Q13`, `Q02`) were dispatched sequentially with 2.0s spacing via `experiments/groq_baseline.py` (`max_retries=0`, streaming enabled):
- **Provider calls completed (`n = 2`):**
  - `Q01`: HTTP 200, TTFT = `1029.33 ms`, total pipeline = `1756.97 ms` (`answer_generation` = `1288.65 ms`), tokens = `2440 in / 389 out`.
  - `Q04`: HTTP 200, TTFT = `1235.20 ms`, total pipeline = `1567.97 ms` (`answer_generation` = `1551.81 ms`), tokens = `3438 in / 406 out`.
- **Local rate-limit pacing rejections (`n = 3`):**
  - Because `Q01` + `Q04` consumed **6,673 tokens** in under 6 seconds, our sliding-window `RateLimiter` (`max_tpm=7000`, `request_deadline=30.0s`) proactively rejected `Q07`, `Q13`, and `Q02` locally (`status=rate_limited`) rather than sending doomed requests that would hit Groq's `8,000 TPM` ceiling.
- **Empirical `x-ratelimit-*` Headers Observed (from `Q01` and `Q04`):**
  - `x-ratelimit-limit-tokens`: `8000` (per-minute)
  - `x-ratelimit-remaining-tokens`: dropped from `5207` (after `Q01`) to `1808` (after `Q04`)
  - `x-ratelimit-reset-tokens`: `20.947s` (after `Q01`)
  - `x-ratelimit-limit-requests`: `1000` (daily RPD window)
  - `x-ratelimit-remaining-requests`: `998` -> `997`

---

## 2. Calibrated Stand-in Parameters

| Parameter | Value | Basis (Measured vs. Assumed) |
|---|---|---|
| `base_ttft_ms` | `1132.26 ms` | **Measured:** Median TTFT of `Q01` (`1029.33 ms`) and `Q04` (`1235.20 ms`). |
| `ttft_jitter_ms` | `102.94 ms` | **Measured:** Standard deviation across the 2 real TTFT samples. |
| `inter_token_ms` | `0.72 ms/tok` | **Measured:** Median `(gen_total_ms - ttft_ms) / (tokens_out - 1)` (~1,389 tok/s). |
| `concurrency_penalty_ms` | `60.0 ms` | **Assumed:** Synthetic per-in-flight-request queuing drag under load. |
| `rpm_limit` | `30` | **Observed (Dashboard):** Inherited org limit; not stressed to 429 on real Groq. |
| `rpd_limit` | `1000` | **Observed (Header/Dashboard):** Matches `x-ratelimit-limit-requests: 1000`. |
| `failure_probability` | `0.0` (baseline) / `0.15` (overload) | **Artificial Injection:** Real Groq had 0% 5xx in `n=2`; injected in overload scenario to test retries/CB. |
| `target_output_tokens` | `150` | **Configured:** Capped for fast load-test turnarounds while preserving streaming behavior. |

---

## 3. Differences & Limitations (Why Stand-in != Real Groq)

1. **Tiny Sample Size (`n = 2`):** Two requests cannot capture Groq's true p95/p99 tail latency, multi-tenant noisy-neighbor spikes, or diurnal variance. Calibration is explicitly **provisional**.
2. **TPM vs. RPM Dominance:** On real Groq (`8,000 TPM`), large RAG prompts (~3,000 tokens each) hit the **TPM** wall after just 2 requests/minute—long before `30 RPM` is reached.
3. **Linear Concurrency Drag:** Real hardware batching (LPU/GPU continuous batching) exhibits non-linear step-function latency jumps at saturation rather than a smooth `60 ms` linear penalty per request.
4. **Synthetic Grounded Text:** The stand-in synthesizes deterministic cited replies from `[SOURCE: ...]` headers in the prompt rather than performing actual 120B-parameter reasoning.
