"""
Unit tests for embeddings, FAISS vector store, and retriever tool.
"""
from pathlib import Path
import pytest
import numpy as np

from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.chunker import Chunk
from app.pipeline.retriever import LibraryRetriever

PAPERS_DIR = Path(__file__).resolve().parent.parent / "papers"
TEST_INDEX_DIR = Path(__file__).resolve().parent / "test_indices"

@pytest.fixture(scope="session")
def embedding_engine():
    return EmbeddingEngine()

def test_embeddings_normalization(embedding_engine):
    texts = ["Transformer model for machine translation.", "Speech tokenization with Moshi."]
    embs = embedding_engine.embed_texts(texts)
    
    assert embs.shape[0] == 2
    assert embs.shape[1] == embedding_engine.dimension
    # Verify L2 normalization (norm should be ~1.0)
    for i in range(2):
        norm = np.linalg.norm(embs[i])
        assert abs(norm - 1.0) < 1e-4

def test_vector_store_build_and_search(embedding_engine, tmp_path):
    store = FAISSVectorStore(embedding_engine=embedding_engine)
    chunks = [
        Chunk("p1_c1", "1706.03762v7", "Attention Is All You Need", "v7", 1, "Abstract", "The Transformer is the first transduction model relying entirely on self-attention.", 12),
        Chunk("p2_c1", "2410.00037v1", "Moshi", "v1", 1, "Abstract", "Moshi is a speech-text foundation model for real-time dialogue using Mimi neural audio codec.", 14),
        Chunk("p3_c1", "2507.07955v2", "Dynamic Chunking", "v2", 1, "Abstract", "Dynamic chunking produces end-to-end hierarchical sequence modeling for long context.", 12)
    ]
    total = store.build_from_chunks(chunks)
    assert total == 3

    # Test search
    res = store.search("What is Moshi and Mimi?", top_k=2)
    assert len(res) > 0
    assert res[0].paper_id == "2410.00037v1"
    assert "Moshi" in res[0].text

    # Test persistence (save and reload)
    idx_path = tmp_path / "idx.bin"
    meta_path = tmp_path / "meta.json"
    store.save(idx_path, meta_path)
    assert idx_path.exists()
    assert meta_path.exists()

    store2 = FAISSVectorStore(embedding_engine=embedding_engine)
    loaded_total = store2.load(idx_path, meta_path)
    assert loaded_total == 3
    res2 = store2.search("self-attention transduction model", top_k=1)
    assert res2[0].paper_id == "1706.03762v7"

def test_retriever_tool_integration(embedding_engine, tmp_path):
    store = FAISSVectorStore(embedding_engine=embedding_engine)
    retriever = LibraryRetriever(
        vector_store=store,
        index_dir=tmp_path,
        chunk_size_words=300,
        chunk_overlap_words=50
    )
    indexed_count = retriever.initialize(papers_dir=PAPERS_DIR, force_rebuild=True)
    assert indexed_count > 50

    # Call tool
    output = retriever.search_library(query="scaled dot-product attention formula", top_k=3)
    assert output["tool"] == "search_library"
    assert output["count"] == 3
    assert output["duration_ms"] >= 0
    assert len(output["chunks"]) == 3
    
    top_chunk = output["chunks"][0]
    assert top_chunk["paper_id"] == "1706.03762v7"
    assert top_chunk["page_number"] >= 1
    assert "attention" in top_chunk["text"].lower()
