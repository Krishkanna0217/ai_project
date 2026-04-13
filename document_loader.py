"""
document_loader.py
Loads and chunks PDF, TXT, DOCX files and web URLs.
"""

import re
import os
import requests
from typing import List, Dict
from bs4 import BeautifulSoup


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def load_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def load_document(source: str) -> Dict:
    """Auto-detect source type and load it. Returns dict with text and metadata."""
    if source.startswith("http://") or source.startswith("https://"):
        return {"text": load_url(source), "source": source, "type": "url"}
    elif source.endswith(".pdf"):
        return {"text": load_pdf(source), "source": os.path.basename(source), "type": "pdf"}
    elif source.endswith(".docx"):
        return {"text": load_docx(source), "source": os.path.basename(source), "type": "docx"}
    else:
        return {"text": load_txt(source), "source": os.path.basename(source), "type": "txt"}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping word-based chunks."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start: start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_and_chunk(source: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """Load a source and return a list of chunk dicts with text + metadata."""
    doc = load_document(source)
    chunks = chunk_text(doc["text"], chunk_size, overlap)
    return [
        {"text": chunk, "source": doc["source"], "type": doc["type"], "chunk_id": i}
        for i, chunk in enumerate(chunks)
    ]