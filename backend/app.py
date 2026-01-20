#!/usr/bin/env python3
"""FastAPI backend for PDF chat."""
import json
import re
from pathlib import Path
from contextlib import asynccontextmanager

import faiss
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastembed import TextEmbedding

# Config
INDEX_DIR = Path(__file__).parent.parent / "index"
LLAMA_SERVER = "http://127.0.0.1:8080"
TOP_K = 12
KEYWORD_BOOST = 0.5

SOURCE_PRIORITY = {
    "cv": 3.0,
    "skills": 2.0,
    "projekte": 1.0,
    "default": 0.5
}


def get_source_boost(source: str) -> float:
    src = source.lower()
    if "cv" in src or "lebenslauf" in src:
        return SOURCE_PRIORITY["cv"]
    if "skill" in src:
        return SOURCE_PRIORITY["skills"]
    if "projekt" in src:
        return SOURCE_PRIORITY["projekte"]
    return SOURCE_PRIORITY["default"]

# Globals
index = None
metadata = []
embedder = None
http_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, metadata, embedder, http_client

    # Load FAISS index
    index_path = INDEX_DIR / "index.faiss"
    meta_path = INDEX_DIR / "metadata.jsonl"

    if not index_path.exists():
        raise RuntimeError(f"Index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        metadata = [json.loads(line) for line in f]

    # Load embedding model
    embedder = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # HTTP client with connection pooling
    http_client = httpx.AsyncClient(timeout=120.0)

    print(f"Loaded index with {index.ntotal} vectors")
    yield

    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": index.ntotal if index else 0}


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful words from query."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Stopwords + ubiquitous terms in this corpus
    stopwords = {'the', 'and', 'for', 'with', 'what', 'does', 'has', 'have', 'how', 'who', 'where', 'when', 'which', 'about', 'know', 'use', 'work', 'worked', 'conrad', 'emde'}
    return {w for w in words if w not in stopwords}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # Extract keywords for boosting
    keywords = extract_keywords(req.question)

    # Embed query
    query_vec = list(embedder.embed([req.question]))
    query_vec = np.array(query_vec, dtype=np.float32)
    faiss.normalize_L2(query_vec)

    # Search all chunks for reranking (small index)
    scores, indices = index.search(query_vec, index.ntotal)

    # Score with keyword boost
    candidates = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            chunk = metadata[idx]
            text_lower = chunk["text"].lower()

            # Keyword boost: add score for each keyword found
            matched_kws = [kw for kw in keywords if kw in text_lower]
            keyword_score = len(matched_kws) * KEYWORD_BOOST
            source_boost = get_source_boost(chunk["source"])
            combined_score = float(scores[0][i]) + keyword_score + source_boost

            candidates.append((combined_score, chunk))

    # Sort by combined score and take top K
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:TOP_K]

    # Reorder: put chunks with keyword matches first (helps small LLMs)
    def has_keyword(chunk):
        text_lower = chunk["text"].lower()
        return any(kw in text_lower for kw in keywords)

    with_keywords = [(s, c) for s, c in candidates if has_keyword(c)]
    without_keywords = [(s, c) for s, c in candidates if not has_keyword(c)]
    candidates = with_keywords + without_keywords

    contexts = []
    sources = []
    for score, chunk in candidates:
        contexts.append(chunk["text"])
        sources.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "score": score
        })

    if not contexts:
        return ChatResponse(answer="No relevant context found.", sources=[])

    # Build prompt
    context_str = "\n\n---\n\n".join(contexts)
    prompt = f"""Based on Conrad Emde's CV and skills documentation below, answer the question.

Context:
{context_str}

Question: {req.question}

Answer:"""

    # Call llama-server
    try:
        resp = await http_client.post(
            f"{LLAMA_SERVER}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 768,
                "repeat_penalty": 1.2
            }
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
    except httpx.RequestError as e:
        raise HTTPException(503, f"LLM server error: {e}")

    return ChatResponse(answer=answer, sources=sources)


# Mount static files AFTER routes
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web", html=True), name="static")
