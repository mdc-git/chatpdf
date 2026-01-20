#!/usr/bin/env python3
"""Build FAISS index from PDF files."""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pymupdf
from fastembed import TextEmbedding


def extract_text(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into chunks on paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # Split long paragraphs by sentences
                words = para.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= chunk_size:
                        current = f"{current} {word}" if current else word
                    else:
                        if current:
                            chunks.append(current)
                        current = word
            else:
                current = para

    if current:
        chunks.append(current)

    # Add overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_tail = chunks[i - 1][-overlap:]
                chunk = prev_tail + " " + chunk
            overlapped.append(chunk)
        return overlapped

    return chunks


def build_index(pdf_dir: Path, out_dir: Path):
    """Build FAISS index from PDFs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize multilingual embedding model
    print("Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
    embedder = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    all_chunks = []
    metadata = []

    # Process PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "source": pdf_path.name,
                "chunk_id": i,
                "text": chunk
            })

    if not all_chunks:
        print("No text extracted from PDFs!")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Generating embeddings...")

    embeddings = list(embedder.embed(all_chunks))
    embeddings = np.array(embeddings, dtype=np.float32)

    # L2 normalize for cosine similarity via dot product
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim after normalization
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"\nSaved index ({index.ntotal} vectors) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=Path, default=Path("./pdfs"))
    parser.add_argument("--out_dir", type=Path, default=Path("./index"))
    args = parser.parse_args()

    build_index(args.pdf_dir, args.out_dir)
