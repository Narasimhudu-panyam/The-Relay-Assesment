"""
Stage 1: Question Understanding & Analysis.
Validates incoming engineering questions, extracts technical keywords,
detects paper targets, and prepares optimized retrieval queries.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

STOPWORDS = {
    "what", "is", "the", "a", "an", "in", "of", "for", "to", "how", "why",
    "does", "do", "did", "are", "was", "were", "with", "on", "by", "from",
    "and", "or", "that", "this", "which", "when", "where", "can", "explain",
    "describe", "between", "compare", "paper", "papers", "according",
}

PAPER_HINTS = {
    "1706.03762v7": ["attention is all you need", "transformer", "vaswani", "multi-head", "bleu", "wmt", "positional encoding"],
    "2410.00037v1": ["moshi", "mimi", "kyutai", "speech-text", "full-duplex", "inner monologue", "helium", "audio codec", "real-time dialogue"],
    "2507.07955v2": ["dynamic chunking", "h-net", "hierarchical sequence", "routing module", "chunking", "byte-level", "bpe-free"],
}


@dataclass
class QuestionAnalysis:
    original_question: str
    cleaned_question: str
    intent: str
    keywords: List[str]
    retrieval_query: str
    target_paper_id: Optional[str] = None
    is_comparative: bool = False


class QuestionAnalyzer:
    """Validates and decomposes engineer questions into structured retrieval intents."""

    def analyze(self, question: str) -> QuestionAnalysis:
        cleaned = " ".join(question.strip().split())
        if len(cleaned) < 3:
            raise ValueError("Question is too short to analyze.")
        if len(cleaned) > 2000:
            raise ValueError("Question exceeds maximum allowed length (2000 chars).")

        tokens = re.findall(r"[A-Za-z0-9\-\_]+", cleaned.lower())
        keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

        # Detect paper focus if unambiguously targeted
        matched_papers = []
        lower_q = cleaned.lower()
        for pid, hints in PAPER_HINTS.items():
            if any(h in lower_q for h in hints):
                matched_papers.append(pid)

        is_comparative = any(w in lower_q for w in ["compare", "difference", "versus", "vs", "both", "across"]) or len(matched_papers) > 1
        target_paper = matched_papers[0] if len(matched_papers) == 1 and not is_comparative else None

        if is_comparative:
            intent = "comparative"
        elif any(w in lower_q for w in ["how", "mechanism", "algorithm", "architecture", "work"]):
            intent = "mechanistic"
        else:
            intent = "factual"

        retrieval_query = " ".join(keywords) if keywords else cleaned
        return QuestionAnalysis(
            original_question=question,
            cleaned_question=cleaned,
            intent=intent,
            keywords=keywords,
            retrieval_query=retrieval_query,
            target_paper_id=target_paper,
            is_comparative=is_comparative,
        )
