"""
Phase 9: Stand-in Service Calibration.
Reads real Groq measurements from results/groq_baseline.json,
computes empirical TTFT and inter-token generation speed,
saves results/calibration_config.json, and generates docs/STANDIN_CALIBRATION.md.
"""
import json
import sys
from pathlib import Path
from typing import List
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def main():
    baseline_path = BASE_DIR / "results" / "groq_baseline.json"
    if not baseline_path.exists():
        print(f"ERROR: {baseline_path} not found. Run Phase 8 first.")
        sys.exit(1)

    with open(baseline_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Filter strictly to actual provider calls (exclude local DecisionEngine refusals where tokens_in == 0)
    provider_calls = [r for r in records if r.get("tokens_in", 0) > 0 and r.get("status") == "ok"]
    local_refusals = [r for r in records if r.get("tokens_in", 0) == 0]

    ttfts: List[float] = []
    gen_speeds_ms_per_tok: List[float] = []
    totals: List[float] = []
    tokens_in_list: List[int] = []
    tokens_out_list: List[int] = []

    for r in provider_calls:
        # Extract model-level generation stage timing
        gen_stages = [s for s in r.get("stage_timings", []) if s["stage"].startswith("answer_generation")]
        if gen_stages:
            g = gen_stages[0]
            g_ttft = g["details"].get("first_token_ms")
            g_total = g["duration_ms"]
            t_out = g["details"].get("tokens_out", r.get("tokens_out", 0))
            t_in = g["details"].get("tokens_in", r.get("tokens_in", 0))
            if g_ttft is not None:
                ttfts.append(float(g_ttft))
                if t_out > 1 and g_total > g_ttft:
                    gen_speeds_ms_per_tok.append((g_total - g_ttft) / max(1, t_out - 1))
        if r.get("total_ms") is not None:
            totals.append(float(r["total_ms"]))
        tokens_in_list.append(int(r.get("tokens_in", 0)))
        tokens_out_list.append(int(r.get("tokens_out", 0)))

    if ttfts:
        base_ttft = round(float(np.median(ttfts)), 2)
        ttft_jitter = round(float(max(50.0, np.std(ttfts))), 2)
    else:
        base_ttft = 400.0
        ttft_jitter = 100.0

    if gen_speeds_ms_per_tok:
        inter_token_ms = round(float(np.median(gen_speeds_ms_per_tok)), 2)
    else:
        inter_token_ms = 3.0

    # Cap target_output_tokens reasonably for stand-in load tests while preserving realistic latency
    avg_out = int(round(np.mean(tokens_out_list))) if tokens_out_list else 120

    standin_cfg = {
        "base_ttft_ms": base_ttft,
        "ttft_jitter_ms": ttft_jitter,
        "inter_token_ms": inter_token_ms,
        "concurrency_penalty_ms": 60.0,
        "failure_probability": 0.0,
        "rate_limit_probability": 0.0,
        "timeout_probability": 0.0,
        "rpm_limit": 30,
        "rpd_limit": 1000,
        "daily_exhausted": False,
        "deterministic_fail_mode": "none",
        "target_output_tokens": min(150, max(60, avg_out)),
        "slow_response_extra_ms": 0.0,
        "seed": 42,
    }

    cal_summary = {
        "sample_size_provider_calls": len(provider_calls),
        "sample_size_local_refusals": len(local_refusals),
        "measured_ttft_ms": ttfts,
        "measured_ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 2) if ttfts else None,
        "measured_ttft_p95_ms": round(float(np.percentile(ttfts, 95)), 2) if ttfts else None,
        "measured_total_p50_ms": round(float(np.percentile(totals, 50)), 2) if totals else None,
        "measured_total_p95_ms": round(float(np.percentile(totals, 95)), 2) if totals else None,
        "measured_inter_token_ms_median": inter_token_ms,
        "avg_tokens_in": round(float(np.mean(tokens_in_list)), 1) if tokens_in_list else None,
        "avg_tokens_out": round(float(np.mean(tokens_out_list)), 1) if tokens_out_list else None,
        "standin_config": standin_cfg,
    }

    cal_file = BASE_DIR / "results" / "calibration_config.json"
    with open(cal_file, "w", encoding="utf-8") as f:
        json.dump(cal_summary, f, indent=2)
    print(f"Saved calibration config to {cal_file}")
    print(json.dumps(cal_summary, indent=2))


if __name__ == "__main__":
    main()
