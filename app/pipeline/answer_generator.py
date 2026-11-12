"""
Stage 4: Answer Generator.
Builds strictly grounded prompts from retrieved paper excerpts and invokes
either the real Groq service or the Stand-in service.
"""
from typing import Any, Dict, List, Literal, Tuple

from app.schemas.models import ModelCallMetadata
from app.services.groq_service import GroqService
from app.services.standin_service import StandinClient

SYSTEM_PROMPT = (
    "You are a strict research assistant answering engineering questions ONLY from the supplied excerpts "
    "from three arXiv papers:\n"
    "1. Attention Is All You Need (1706.03762v7)\n"
    "2. Moshi: a speech-text foundation model for real-time dialogue (2410.00037v1)\n"
    "3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (2507.07955v2)\n\n"
    "RULES:\n"
    "- Answer ONLY using facts explicitly stated in the provided excerpts.\n"
    "- NEVER use outside or pre-trained knowledge to fill gaps.\n"
    "- Cite every factual claim inline using the exact format: "
    "[<Paper Title> (<Paper ID>), Page <N>, Section: <Section Name>].\n"
    "- If the provided excerpts do not contain enough information to answer the question, "
    "respond ONLY with: \"The provided research papers do not contain sufficient evidence to answer this question.\""
)


class AnswerGenerator:
    """Formats grounded evidence prompts and dispatches generation to the selected model service."""

    def __init__(self, groq_service: GroqService, standin_client: StandinClient):
        self.groq_service = groq_service
        self.standin_client = standin_client

    def build_messages(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        excerpts_blocks = []
        for c in chunks:
            header = (
                f"[SOURCE: {c['paper_title']} ({c['paper_id']}), "
                f"Page {c['page_number']}, Section: {c['section']}]"
            )
            excerpts_blocks.append(f"{header}\n{c['text']}")

        context_str = "\n\n---\n\n".join(excerpts_blocks)
        user_content = (
            f"RETRIEVED EXCERPTS:\n{context_str}\n\n"
            f"QUESTION: {question}\n\n"
            "Provide a concise, accurate answer with inline citations, or refuse if evidence is missing."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def generate(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        source: Literal["real", "stand-in"] = "stand-in",
        stream: bool = True,
    ) -> Tuple[str, ModelCallMetadata]:
        messages = self.build_messages(question, chunks)
        if source == "real":
            return await self.groq_service.generate(messages=messages, stream=stream)
        return await self.standin_client.generate(messages=messages, stream=stream)
