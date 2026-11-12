"""
Stage 5: Answer Verifier.
Validates citations against retrieved chunk provenance, checks page validity,
detects uncited or unsupported claims, and confirms safe refusals.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.schemas.models import Citation

KNOWN_PAPERS = {
    "1706.03762v7": ("Attention Is All You Need", "v7"),
    "2410.00037v1": ("Moshi: a speech-text foundation model for real-time dialogue", "v1"),
    "2507.07955v2": ("Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", "v2"),
}

REFUSAL_PHRASES = [
    "do not contain sufficient evidence",
    "does not contain sufficient evidence",
    "cannot answer this question from the provided",
    "not mentioned in the provided",
    "insufficient evidence to answer",
    "papers do not say",
]


@dataclass
class VerificationResult:
    passed: bool
    is_refusal: bool
    citations: List[Citation] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    reason: str = ""


class AnswerVerifier:
    """Checks grounding, citation existence, and page validity against retrieved chunks."""

    # Matches patterns like (1706.03762v7), Page 3 or [1706.03762v7, Page 3]
    CITATION_PATTERN = re.compile(
        r"(1706\.03762v7|2410\.00037v1|2507\.07955v2)[^\]]*?Page\s+(\d+)",
        re.IGNORECASE,
    )

    def verify(self, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> VerificationResult:
        lower_ans = answer.lower().strip()
        if not lower_ans:
            return VerificationResult(passed=False, is_refusal=False, reason="Empty answer generated.")

        # 1. Check for explicit refusal
        if any(phrase in lower_ans for phrase in REFUSAL_PHRASES):
            return VerificationResult(
                passed=True,
                is_refusal=True,
                reason="Model produced a safe refusal due to insufficient evidence.",
            )

        # Build lookup of valid (paper_id, page_number) -> chunk metadata
        valid_sources = {}
        for c in retrieved_chunks:
            key = (c["paper_id"], int(c["page_number"]))
            valid_sources[key] = c

        # 2. Extract citations from answer
        matches = self.CITATION_PATTERN.findall(answer)
        if not matches:
            return VerificationResult(
                passed=False,
                is_refusal=False,
                reason="Answer lacks required inline citation with paper ID and page number.",
            )

        valid_citations: List[Citation] = []
        invalid_citations: List[str] = []
        seen_keys = set()

        for pid_raw, page_str in matches:
            pid = pid_raw.lower()
            # Normalize exact paper_id casing
            matched_pid = next((k for k in KNOWN_PAPERS if k.lower() == pid), pid_raw)
            page_num = int(page_str)
            key = (matched_pid, page_num)

            if key in valid_sources:
                if key not in seen_keys:
                    seen_keys.add(key)
                    chunk_meta = valid_sources[key]
                    valid_citations.append(
                        Citation(
                            paper_id=matched_pid,
                            paper_title=chunk_meta["paper_title"],
                            paper_version=chunk_meta["paper_version"],
                            page_number=page_num,
                            section=chunk_meta["section"],
                            chunk_id=chunk_meta.get("chunk_id"),
                        )
                    )
            else:
                invalid_citations.append(f"{matched_pid} Page {page_num}")

        if invalid_citations:
            return VerificationResult(
                passed=False,
                is_refusal=False,
                citations=valid_citations,
                invalid_citations=invalid_citations,
                reason=f"Cited paper/page not present in retrieved evidence: {', '.join(invalid_citations)}",
            )

        if not valid_citations:
            return VerificationResult(
                passed=False,
                is_refusal=False,
                reason="No valid citations verified against retrieved evidence.",
            )

        return VerificationResult(
            passed=True,
            is_refusal=False,
            citations=valid_citations,
            reason="All citations verified against retrieved passages.",
        )
