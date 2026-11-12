"""
FAISS Vector Store module for indexing and retrieving research paper chunks.
Saves and reloads indices to disk, preserves complete citation metadata,
and supports configurable similarity thresholds and optional re-ranking.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np
import faiss

from app.retrieval.chunker import Chunk
from app.retrieval.embeddings import EmbeddingEngine

class RetrievedChunk:
    def __init__(
        self,
        chunk_id: str,
        paper_id: str,
        paper_title: str,
        paper_version: str,
        page_number: int,
        section: str,
        text: str,
        similarity_score: float
    ):
        self.chunk_id = chunk_id
        self.paper_id = paper_id
        self.paper_title = paper_title
        self.paper_version = paper_version
        self.page_number = page_number
        self.section = section
        self.text = text
        self.similarity_score = round(float(similarity_score), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "paper_version": self.paper_version,
            "page_number": self.page_number,
            "section": self.section,
            "text": self.text,
            "similarity_score": self.similarity_score
        }

class FAISSVectorStore:
    """Manages dense FAISS index and chunk metadata mapping."""

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks_metadata: List[Dict[str, Any]] = []

    def build_from_chunks(self, chunks: List[Chunk]) -> int:
        """Embeds all chunks and builds an in-memory FAISS IndexFlatIP."""
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        self.chunks_metadata = [c.to_dict() for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed_texts(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        return self.index.ntotal

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Serializes FAISS binary index and JSON metadata to disk."""
        if self.index is None or not self.chunks_metadata:
            raise ValueError("No index or metadata available to save")

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, indent=2)

    def load(self, index_path: Path, metadata_path: Path) -> int:
        """Loads FAISS binary index and metadata from disk."""
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing index ({index_path}) or metadata ({metadata_path})")

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.chunks_metadata = json.load(f)

        return self.index.ntotal

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_paper_id: Optional[str] = None,
        use_reranking: bool = False
    ) -> List[RetrievedChunk]:
        """Searches index for passages similar to query.
        Applies optional paper filtering and re-ranking."""
        if self.index is None or not self.chunks_metadata:
            raise RuntimeError("Vector store has not been initialized with an index")

        query_vec = self.embedding_engine.embed_query(query)
        
        # Over-fetch if filtering or re-ranking is applied
        fetch_k = min(len(self.chunks_metadata), top_k * 3 if (filter_paper_id or use_reranking) else top_k)
        scores, indices = self.index.search(query_vec, fetch_k)

        results: List[RetrievedChunk] = []
        scores_arr = scores[0]
        indices_arr = indices[0]

        for score, idx in zip(scores_arr, indices_arr):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            if score < min_score:
                continue

            meta = self.chunks_metadata[idx]
            if filter_paper_id and meta["paper_id"] != filter_paper_id:
                continue

            results.append(RetrievedChunk(
                chunk_id=meta["chunk_id"],
                paper_id=meta["paper_id"],
                paper_title=meta["paper_title"],
                paper_version=meta["paper_version"],
                page_number=meta["page_number"],
                section=meta["section"],
                text=meta["text"],
                similarity_score=float(score)
            ))

        if use_reranking and results:
            # Re-rank based on combined dense cosine score + lexical keyword density
            results = self._rerank_chunks(query, results)

        return results[:top_k]

    def _rerank_chunks(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Lexical and semantic hybrid re-ranker.
        Boosts chunks that contain exact query keyword tokens."""
        query_tokens = set(query.lower().split())
        scored: List[tuple] = []
        
        for c in chunks:
            chunk_tokens = set(c.text.lower().split())
            intersection = query_tokens.intersection(chunk_tokens)
            lexical_overlap = len(intersection) / max(len(query_tokens), 1)
            # 70% dense vector similarity + 30% lexical overlap score
            hybrid_score = (c.similarity_score * 0.70) + (lexical_overlap * 0.30)
            scored.append((hybrid_score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored]
