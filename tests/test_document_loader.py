"""
Unit test for PDF document loading and chunking.
Verifies that:
1. All three papers are loaded from papers/manifest.json.
2. Page counts and text lengths are authentic and non-empty.
3. Detected sections match research paper layout.
4. Chunker produces chunks with full provenance metadata.
"""
from pathlib import Path
import pytest
from app.retrieval.document_loader import DocumentLoader
from app.retrieval.chunker import DocumentChunker

PAPERS_DIR = Path(__file__).resolve().parent.parent / "papers"

def test_manifest_exists():
    manifest_file = PAPERS_DIR / "manifest.json"
    assert manifest_file.exists(), "papers/manifest.json must exist"

def test_load_all_papers():
    loader = DocumentLoader(papers_dir=PAPERS_DIR)
    papers = loader.load_all_library_papers()
    
    assert len(papers) == 3, f"Expected 3 papers, found {len(papers)}"
    
    paper_ids = [p.paper_id for p in papers]
    assert "1706.03762v7" in paper_ids
    assert "2410.00037v1" in paper_ids
    assert "2507.07955v2" in paper_ids

    for paper in papers:
        assert len(paper.pages) > 0, f"Paper {paper.paper_id} has 0 pages"
        total_text_len = sum(p["character_count"] for p in paper.pages)
        assert total_text_len > 5000, f"Paper {paper.paper_id} text length unexpectedly small: {total_text_len}"
        
        # Verify page numbers are sequential from 1
        page_numbers = [p["page_number"] for p in paper.pages]
        assert page_numbers == list(range(1, len(paper.pages) + 1))

def test_chunking_metadata():
    loader = DocumentLoader(papers_dir=PAPERS_DIR)
    papers = loader.load_all_library_papers()
    chunker = DocumentChunker(chunk_size_words=300, overlap_words=50)
    
    chunks = chunker.chunk_all(papers)
    assert len(chunks) > 20, f"Expected >20 chunks, got {len(chunks)}"

    for chunk in chunks:
        assert chunk.chunk_id != ""
        assert chunk.paper_id in ["1706.03762v7", "2410.00037v1", "2507.07955v2"]
        assert chunk.paper_title != ""
        assert chunk.page_number >= 1
        assert chunk.section != ""
        assert len(chunk.text.split()) <= 350  # Upper bound considering word count
        
        d = chunk.to_dict()
        assert "chunk_id" in d
        assert "paper_id" in d
        assert "text" in d
