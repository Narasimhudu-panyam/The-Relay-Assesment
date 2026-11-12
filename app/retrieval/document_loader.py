"""
Document loader module using PyMuPDF (fitz) to accurately extract text,
pages, and section headers from the research papers.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import fitz  # PyMuPDF

class PaperDocument:
    def __init__(self, paper_id: str, title: str, version: str, path: Path):
        self.paper_id = paper_id
        self.title = title
        self.version = version
        self.path = path
        self.pages: List[Dict[str, Any]] = []

class DocumentLoader:
    """Loads PDF documents and extracts structured text with page and section metadata."""
    
    SECTION_REGEX = re.compile(
        r'^(?:(?:\d+\.?\s+|[I|V|X]+\.?\s+|Abstract|Introduction|Conclusion|References|Related Work|Architecture|Background|Results|Experiments|Discussion))([A-Z][A-Za-z0-9\s,\-–\(\)]{2,80})$',
        re.MULTILINE
    )

    def __init__(self, papers_dir: Path):
        self.papers_dir = Path(papers_dir)

    def load_pdf(self, filename: str, paper_id: str, title: str, version: str) -> PaperDocument:
        pdf_path = self.papers_dir / filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        doc = fitz.open(pdf_path)
        paper_doc = PaperDocument(paper_id=paper_id, title=title, version=version, path=pdf_path)
        
        current_section = "Header / Abstract"
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            text = page.get_text("text")
            
            # Simple heuristic detection of sections to enrich metadata
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines[:10]:  # check top 10 lines of page for section headers
                if len(line) < 80 and any(line.startswith(prefix) for prefix in [
                    "1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 ",
                    "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ",
                    "Abstract", "Introduction", "Related Work", "Background",
                    "Model Architecture", "Experiments", "Results", "Conclusion"
                ]):
                    current_section = line
                    break

            paper_doc.pages.append({
                "page_number": page_num,
                "text": text,
                "detected_section": current_section,
                "character_count": len(text),
                "word_count": len(text.split())
            })
            
        doc.close()
        return paper_doc

    def load_all_library_papers(self, manifest_path: Optional[Path] = None) -> List[PaperDocument]:
        if manifest_path is None:
            manifest_path = self.papers_dir / "manifest.json"
            
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        loaded_papers = []
        for item in manifest:
            paper = self.load_pdf(
                filename=item["filename"],
                paper_id=item["id"],
                title=item["title"],
                version=item["version"]
            )
            loaded_papers.append(paper)
        return loaded_papers
