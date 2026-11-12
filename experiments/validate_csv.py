"""
Validates results/results.csv against the strict 12-column schema
and recomputes summary metrics per scenario directly from raw CSV rows.
"""
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "results.csv"

EXPECTED_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]
ALLOWED_SOURCES = {"real", "stand-in"}
ALLOWED_STATUSES = {"ok", "rate_limited", "timeout", "error"}
ALLOWED_GRADES = {"correct", "wrong", "refused_correctly", "refused_wrongly"}


def validate_and_summarize():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} does not exist.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS, f"Column mismatch: {list(df.columns)}"
    assert len(df) > 0, "results.csv is empty"
    assert df["request_id"].is_unique, "Duplicate request_id found in results.csv"

    bad_sources = set(df["source"].dropna()) - ALLOWED_SOURCES
    assert not bad_sources, f"Invalid source values: {bad_sources}"

    bad_statuses = set(df["status"].dropna()) - ALLOWED_STATUSES
    assert not bad_statuses, f"Invalid status values: {bad_statuses}"

    bad_grades = set(df["answer_grade"].dropna()) - ALLOWED_GRADES
    assert not bad_grades, f"Invalid answer_grade values: {bad_grades}"

    print(f"CSV Validation PASSED: {len(df)} total rows, 12 exact columns, all enums valid.\n")
    print("Recomputed Per-Scenario Summary from results.csv:")
    for (scenario, source), grp in df.groupby(["scenario", "source"], sort=False):
        n = len(grp)
        ok_grp = grp[grp["status"] == "ok"]
        rl_pct = 100.0 * (grp["status"] == "rate_limited").sum() / n
        to_pct = 100.0 * (grp["status"] == "timeout").sum() / n
        ok_pct = 100.0 * len(ok_grp) / n
        ttfts = ok_grp["first_token_ms"].dropna()
        totals = grp["total_ms"].dropna()
        ttft_p50 = f"{ttfts.quantile(0.5):.1f}" if len(ttfts) else "not measured"
        ttft_p95 = f"{ttfts.quantile(0.95):.1f}" if len(ttfts) else "not measured"
        tot_p50 = f"{totals.quantile(0.5):.1f}" if len(totals) else "not measured"
        tot_p95 = f"{totals.quantile(0.95):.1f}" if len(totals) else "not measured"
        print(
            f"  - {scenario} ({source}): n={n}, ok={ok_pct:.1f}%, rl={rl_pct:.1f}%, "
            f"to={to_pct:.1f}%, ttft_p50/p95={ttft_p50}/{ttft_p95}ms, total_p50/p95={tot_p50}/{tot_p95}ms"
        )


if __name__ == "__main__":
    validate_and_summarize()
