"""
Text chunking module with metadata preservation (title, version, page, section).
Supports configurable chunk size and overlap for retrieval experiments.
"""
from typing import List, Dict, Any
from app.retrieval.document_loader import PaperDocument

class Chunk:
    def __init__(
        self,
        chunk_id: str,
        paper_id: str,
        paper_title: str,
        paper_version: str,
        page_number: int,
        section: str,
        text: str,
        word_count: int
    ):
        self.chunk_id = chunk_id
        self.paper_id = paper_id
        self.paper_title = paper_title
        self.paper_version = paper_version
        self.page_number = page_number
        self.section = section
        self.text = text
        self.word_count = word_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "paper_version": self.paper_version,
            "page_number": self.page_number,
            "section": self.section,
            "text": self.text,
            "word_count": self.word_count
        }

class DocumentChunker:
    """Splits documents into overlapping chunks while strictly preserving page and section provenance."""

    def __init__(self, chunk_size_words: int = 400, overlap_words: int = 50):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk_paper(self, paper: PaperDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_counter = 0

        for page in paper.pages:
            page_text = page["text"].strip()
            if not page_text:
                continue

            words = page_text.split()
            if not words:
                continue

            # If page text is within chunk size, keep as single chunk
            if len(words) <= self.chunk_size_words:
                chunk_counter += 1
                chunks.append(Chunk(
                    chunk_id=f"{paper.paper_id}_p{page['page_number']}_c{chunk_counter}",
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    paper_version=paper.version,
                    page_number=page["page_number"],
                    section=page["detected_section"],
                    text=page_text,
                    word_count=len(words)
                ))
            else:
                # Sliding window chunking within page boundaries
                start = 0
                step = self.chunk_size_words - self.overlap_words
                while start < len(words):
                    end = min(start + self.chunk_size_words, len(words))
                    sub_words = words[start:end]
                    chunk_counter += 1
                    chunks.append(Chunk(
                        chunk_id=f"{paper.paper_id}_p{page['page_number']}_c{chunk_counter}",
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        paper_version=paper.version,
                        page_number=page["page_number"],
                        section=page["detected_section"],
                        text=" ".join(sub_words),
                        word_count=len(sub_words)
                    ))
                    if end == len(words):
                        break
                    start += step

        return chunks

    def chunk_all(self, papers: List[PaperDocument]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for p in papers:
            all_chunks.extend(self.chunk_paper(p))
        return all_chunks
