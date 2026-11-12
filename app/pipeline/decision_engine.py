"""
Stage 3: Decision Engine.
Evaluates retrieved evidence sufficiency and branches explicitly into:
- PROCEED: sufficient evidence to generate a cited answer.
- REFINE: borderline evidence -> refine query & re-retrieve (bounded loop).
- REFUSE: insufficient evidence -> safe refusal without hallucination.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from app.pipeline.question_analyzer import QuestionAnalysis


@dataclass
class DecisionResult:
    action: Literal["PROCEED", "REFINE", "REFUSE"]
    reason: str
    refined_query: str = ""
    top_score: float = 0.0
    keyword_coverage: float = 0.0


class DecisionEngine:
    """Evidence sufficiency gate preventing hallucinated answers."""

    def __init__(
        self,
        refuse_threshold: float = 0.22,
        proceed_threshold: float = 0.30,
        max_refinements: int = 1,
    ):
        self.refuse_threshold = refuse_threshold
        self.proceed_threshold = proceed_threshold
        self.max_refinements = max_refinements

    def evaluate(
        self,
        analysis: QuestionAnalysis,
        retrieval_output: Dict[str, Any],
        refinement_attempt: int = 0,
    ) -> DecisionResult:
        chunks: List[Dict[str, Any]] = retrieval_output.get("chunks", [])
        if not chunks:
            return DecisionResult(
                action="REFUSE",
                reason="No passages retrieved from the research paper library.",
                top_score=0.0,
                keyword_coverage=0.0,
            )

        top_score = float(retrieval_output.get("highest_score", 0.0))
        combined_text = " ".join(c.get("text", "").lower() for c in chunks)

        # Compute keyword coverage of the question's key technical terms
        if analysis.keywords:
            matched_kw = sum(1 for kw in analysis.keywords if kw in combined_text)
            keyword_coverage = matched_kw / len(analysis.keywords)
        else:
            keyword_coverage = 1.0

        # Hard refusal if similarity is below minimum evidence floor
        if top_score < self.refuse_threshold and keyword_coverage < 0.35:
            return DecisionResult(
                action="REFUSE",
                reason=(
                    f"Retrieved evidence below confidence threshold "
                    f"(top_score={top_score:.3f} < {self.refuse_threshold:.2f}, "
                    f"keyword_coverage={keyword_coverage:.2f})."
                ),
                top_score=top_score,
                keyword_coverage=round(keyword_coverage, 3),
            )

        # Bounded refinement loop if borderline score or comparative query only hit 1 paper
        distinct_papers = {c.get("paper_id") for c in chunks}
        needs_multi_paper = analysis.is_comparative and len(distinct_papers) < 2
        borderline = top_score < self.proceed_threshold or keyword_coverage < 0.30

        if (borderline or needs_multi_paper) and refinement_attempt < self.max_refinements:
            refined = analysis.cleaned_question
            if needs_multi_paper:
                refined = f"{analysis.retrieval_query} comparison architecture"
            return DecisionResult(
                action="REFINE",
                reason=(
                    f"Borderline retrieval evidence (top_score={top_score:.3f}, "
                    f"papers={len(distinct_papers)}). Refining query."
                ),
                refined_query=refined,
                top_score=top_score,
                keyword_coverage=round(keyword_coverage, 3),
            )

        # After refinement, if still below minimum threshold with zero keyword match, refuse
        if top_score < self.refuse_threshold:
            return DecisionResult(
                action="REFUSE",
                reason=f"Even after refinement, top_score={top_score:.3f} remains below {self.refuse_threshold:.2f}.",
                top_score=top_score,
                keyword_coverage=round(keyword_coverage, 3),
            )

        return DecisionResult(
            action="PROCEED",
            reason=f"Sufficient evidence (top_score={top_score:.3f}, keyword_coverage={keyword_coverage:.2f}).",
            top_score=top_score,
            keyword_coverage=round(keyword_coverage, 3),
        )
