"""
Retriever tool module for the AI pipeline.
Implements the formal callable tool `search_library` adhering to REQ-02 and REQ-04.
"""
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

from app.retrieval.vector_store import FAISSVectorStore, RetrievedChunk
from app.retrieval.document_loader import DocumentLoader
from app.retrieval.chunker import DocumentChunker

class LibraryRetriever:
    """Formal tool interface for retrieving research paper evidence."""

    TOOL_SCHEMA = {
        "name": "search_library",
        "description": "Searches the 3 research papers (Attention, Moshi, Dynamic Chunking) for factual evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query or concept to find evidence for"},
                "top_k": {"type": "integer", "description": "Number of passages to retrieve", "default": 5},
                "filter_paper_id": {"type": "string", "description": "Optional paper ID constraint (1706.03762v7, 2410.00037v1, 2507.07955v2)"}
            },
            "required": ["query"]
        }
    }

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        index_dir: Path,
        chunk_size_words: int = 400,
        chunk_overlap_words: int = 50,
        use_reranking: bool = False
    ):
        self.vector_store = vector_store
        self.index_dir = Path(index_dir)
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.use_reranking = use_reranking
        self.index_file = self.index_dir / "faiss_index.bin"
        self.metadata_file = self.index_dir / "chunks_metadata.json"

    def initialize(self, papers_dir: Path, force_rebuild: bool = False) -> int:
        """Initializes or builds the vector index from papers directory."""
        if not force_rebuild and self.index_file.exists() and self.metadata_file.exists():
            return self.vector_store.load(self.index_file, self.metadata_file)

        loader = DocumentLoader(papers_dir=papers_dir)
        papers = loader.load_all_library_papers()
        chunker = DocumentChunker(
            chunk_size_words=self.chunk_size_words,
            overlap_words=self.chunk_overlap_words
        )
        chunks = chunker.chunk_all(papers)
        count = self.vector_store.build_from_chunks(chunks)
        self.vector_store.save(self.index_file, self.metadata_file)
        return count

    def search_library(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_paper_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Callable tool: executes vector search and returns structured results with timing."""
        start_t = time.perf_counter()
        chunks = self.vector_store.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filter_paper_id=filter_paper_id,
            use_reranking=self.use_reranking
        )
        duration_ms = round((time.perf_counter() - start_t) * 1000, 2)

        return {
            "tool": "search_library",
            "query": query,
            "duration_ms": duration_ms,
            "count": len(chunks),
            "chunks": [c.to_dict() for c in chunks],
            "highest_score": chunks[0].similarity_score if chunks else 0.0
        }
