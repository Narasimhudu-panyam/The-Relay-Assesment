"""
Raw CSV Result Logger & Schema Validator for results/results.csv.
Enforces the exact 12 required columns and allowed enum values.
"""
import csv
from pathlib import Path
from typing import List, Optional

from app.schemas.models import RawResultRow

CSV_COLUMNS = [
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


class ResultsCSVLogger:
    """Appends validated RawResultRow records to results/results.csv."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def append_rows(self, rows: List[RawResultRow]) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            for row in rows:
                d = row.model_dump()
                # Represent None as empty string in CSV
                for k, v in d.items():
                    if v is None:
                        d[k] = ""
                writer.writerow(d)

    def reset_file(self) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
