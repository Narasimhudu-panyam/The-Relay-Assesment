"""
Verify the downloaded papers and generate metadata manifest.
"""
import hashlib
import json
from pathlib import Path

PAPERS = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "version": "v7",
        "url": "https://arxiv.org/pdf/1706.03762v7",
        "filename": "1706.03762v7.pdf"
    },
    {
        "id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "version": "v1",
        "url": "https://arxiv.org/pdf/2410.00037v1",
        "filename": "2410.00037v1.pdf"
    },
    {
        "id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "version": "v2",
        "url": "https://arxiv.org/pdf/2507.07955v2",
        "filename": "2507.07955v2.pdf"
    }
]

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    papers_dir = Path(__file__).resolve().parent
    manifest = []
    
    for paper in PAPERS:
        path = papers_dir / paper["filename"]
        if not path.exists():
            raise FileNotFoundError(f"Missing paper: {path}")
        
        # Check PDF magic header %PDF
        with open(path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                raise ValueError(f"File {path} is not a valid PDF! Header: {header}")
                
        sha256 = compute_sha256(path)
        item = {
            "id": paper["id"],
            "title": paper["title"],
            "version": paper["version"],
            "filename": paper["filename"],
            "size_bytes": path.stat().st_size,
            "sha256": sha256
        }
        manifest.append(item)
        print(f"Verified: {paper['title']}")
        print(f"  File: {paper['filename']} ({path.stat().st_size:,} bytes)")
        print(f"  SHA-256: {sha256}")

    manifest_file = papers_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_file}")

if __name__ == "__main__":
    main()
