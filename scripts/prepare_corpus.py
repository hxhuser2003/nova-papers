"""Download versioned papers and build a reproducible, page-aware local text corpus."""

import hashlib
import json
import re
from pathlib import Path

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def split_text(text: str, size: int = 1100, overlap: int = 180):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start + size // 2, end)
            end = boundary + 1 if boundary != -1 else text.rfind(" ", start, end)
        chunk = text[start:end].strip()
        if len(chunk) >= 70:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
        # Start at a word boundary, preserving the extracted text verbatim thereafter.
        space = text.find(" ", start)
        if space != -1:
            start = space + 1


def main():
    papers = json.loads((CORPUS / "papers.json").read_text())
    pdf_dir = CORPUS / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    chunks, inventory = [], []
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for paper in papers:
            path = pdf_dir / paper["filename"]
            if not path.exists():
                print(f"Downloading {paper['short_title']}…", flush=True)
                response = client.get(paper["pdf_url"])
                response.raise_for_status()
                if not response.content.startswith(b"%PDF"):
                    raise ValueError(f"Expected a PDF for {paper['id']}")
                temp = path.with_suffix(".tmp")
                temp.write_bytes(response.content)
                temp.replace(path)
            with pymupdf.open(path) as document:
                start_count = len(chunks)
                for page_index, page in enumerate(document):
                    for chunk_index, quote in enumerate(split_text(page.get_text(sort=True))):
                        chunks.append({
                            "id": f"{paper['id']}-p{page_index + 1}-{chunk_index}",
                            "paper_id": paper["id"], "page": page_index + 1,
                            "quote": quote,
                        })
                inventory.append({
                    **paper, "pages": len(document), "chunks": len(chunks) - start_count,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
            print(f"Indexed {paper['short_title']}: {inventory[-1]['chunks']} passages", flush=True)
    temp = CORPUS / "index.tmp"
    temp.write_text(json.dumps({"papers": inventory, "chunks": chunks}, ensure_ascii=False, indent=2))
    temp.replace(CORPUS / "index.json")
    print(f"Ready: {len(papers)} papers, {len(chunks)} passages")


if __name__ == "__main__":
    main()
