"""
Script to download and verify the three exact research papers specified in The Relay assignment brief:
1. "Attention Is All You Need", arXiv 1706.03762v7
   URL: https://arxiv.org/pdf/1706.03762v7
2. "Moshi: a speech-text foundation model for real-time dialogue", arXiv 2410.00037v1
   URL: https://arxiv.org/pdf/2410.00037v1
3. "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", arXiv 2507.07955v2
   URL: https://arxiv.org/pdf/2507.07955v2
"""

import os
import hashlib
import urllib.request
import sys
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

def download_paper(paper: dict, target_dir: Path) -> Path:
    target_path = target_dir / paper["filename"]
    if target_path.exists() and target_path.stat().st_size > 10000:
        print(f"[{paper['id']}] Already exists ({target_path.stat().st_size:,} bytes). Skipping download.")
        return target_path

    print(f"[{paper['id']}] Downloading from {paper['url']} ...")
    req = urllib.request.Request(
        paper["url"],
        headers={"User-Agent": "TheRelay-ResearchPaperFetcher/1.0 (academic research evaluation)"}
    )
    with urllib.request.urlopen(req, timeout=60) as response, open(target_path, "wb") as out_file:
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" not in content_type and "octet-stream" not in content_type:
            print(f"WARNING: Unexpected Content-Type for {paper['id']}: {content_type}")
        data = response.read()
        out_file.write(data)
    
    size = target_path.stat().st_size
    print(f"[{paper['id']}] Saved {target_path.name} ({size:,} bytes)")
    return target_path

def main():
    script_dir = Path(__file__).resolve().parent
    papers_dir = script_dir if script_dir.name == "papers" else script_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Target papers directory: {papers_dir}")
    manifest = []
    
    for paper in PAPERS:
        try:
            path = download_paper(paper, papers_dir)
            sha = compute_sha256(path)
            manifest.append({
                "id": paper["id"],
                "title": paper["title"],
                "version": paper["version"],
                "filename": paper["filename"],
                "size_bytes": path.stat().st_size,
                "sha256": sha
            })
            print(f"  SHA-256: {sha}")
        except Exception as e:
            print(f"ERROR downloading {paper['id']}: {e}", file=sys.stderr)
            raise e

    print("\n--- ALL PAPERS DOWNLOADED AND VERIFIED ---")
    for m in manifest:
        print(f"* {m['title']} ({m['id']}) - {m['size_bytes']:,} bytes - SHA256: {m['sha256'][:16]}...")

if __name__ == "__main__":
    main()
