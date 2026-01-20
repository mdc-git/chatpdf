# PDF Chat Agent

Local CPU-only RAG (Retrieval-Augmented Generation) system for chatting with PDF documents. Extracts text from PDFs, creates embeddings, stores in FAISS vector index, retrieves relevant chunks, and generates answers using a local LLM.

## Architecture

```
PDFs → PyMuPDF → Chunks → Embeddings → FAISS Index
                                           ↓
User Query → Embed → Search → Top K Chunks → LLM → Answer
```

| Component | Choice | Notes |
|-----------|--------|-------|
| LLM | Qwen2.5-1.5B-Instruct (Q4_K_M) | ~1GB, good quality/size ratio |
| LLM Server | llama.cpp llama-server | OpenAI-compatible API, CPU optimized |
| PDF Extraction | PyMuPDF | Fast, minimal dependencies |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Multilingual support (German/English) |
| Vector Search | FAISS IndexFlatIP | Cosine similarity via inner product |
| API | FastAPI | Async, serves both API and web UI |

## Setup

### Prerequisites

- Python 3.10+
- Git
- C++ compiler (for llama.cpp)
- ~2GB disk space

### Installation

```bash
./bootstrap.sh
```

This script:
1. Creates directory structure (`pdfs/`, `index/`, `models/`, `backend/`, `web/`)
2. Sets up Python virtual environment
3. Installs dependencies (FastAPI, PyMuPDF, FAISS, FastEmbed, etc.)
4. Clones and builds llama.cpp
5. Downloads the Qwen2.5-1.5B-Instruct GGUF model (~1GB)
6. Builds FAISS index from PDFs in `pdfs/` directory

### Adding Documents

Place PDF files in the `pdfs/` directory before running bootstrap, or add them later and rebuild the index:

```bash
cp /path/to/document.pdf pdfs/
.venv/bin/python backend/build_index.py --pdf_dir ./pdfs --out_dir ./index
```

## Running

```bash
./run.sh
```

This starts:
- **llama-server** on port 8080 (LLM inference)
- **FastAPI** on port 8000 (API + web UI)

Press `Ctrl+C` to stop both servers.

## Usage

### Web Interface

Open http://127.0.0.1:8000 in your browser. Type a question and click "Ask".

### Command Line

```bash
./query.sh "What skills does Conrad have?"
./query.sh "What companies did he work for?"
./query.sh "Does Conrad know Kubernetes?"
```

### API

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

Response:
```json
{
  "answer": "...",
  "sources": [
    {"source": "document.pdf", "chunk_id": 0, "score": 0.75},
    ...
  ]
}
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok", "chunks": 21}
```

## How It Works

### Indexing (`build_index.py`)

1. **PDF Extraction**: PyMuPDF extracts text from each PDF
2. **Chunking**: Text split into ~1000 char chunks with 200 char overlap, respecting paragraph boundaries
3. **Embedding**: FastEmbed generates 384-dim vectors using multilingual MiniLM
4. **Normalization**: L2 normalize vectors for cosine similarity
5. **Storage**: FAISS IndexFlatIP + metadata JSONL

### Retrieval (`app.py`)

1. **Query Embedding**: Same model embeds the user question
2. **Keyword Extraction**: Extract meaningful words from query (excluding stopwords)
3. **Hybrid Search**:
   - Semantic: FAISS finds similar vectors
   - Keyword boost: +0.5 score for chunks containing query keywords
4. **Reranking**: Sort by combined score, keyword-matched chunks prioritized
5. **Top-K Selection**: Return top 5 most relevant chunks

### Generation

1. **Context Assembly**: Concatenate retrieved chunks
2. **Prompt Construction**: Simple prompt with context + question
3. **LLM Inference**: llama-server generates answer
4. **Response**: Return answer with source attribution

## Configuration

Edit `backend/app.py`:

```python
TOP_K = 5           # Number of chunks to retrieve
KEYWORD_BOOST = 0.5 # Score boost for keyword matches
LLAMA_SERVER = "http://127.0.0.1:8080"
```

LLM parameters:
```python
"temperature": 0.3,      # Creativity (0.0-1.0)
"max_tokens": 300,       # Max response length
"repeat_penalty": 1.2    # Prevents repetition loops
```

## File Structure

```
chatpdf/
├── bootstrap.sh          # One-time setup
├── run.sh                # Start servers
├── query.sh              # CLI query tool
├── pdfs/                 # Source PDF documents
├── index/
│   ├── index.faiss       # Vector index
│   └── metadata.jsonl    # Chunk text + source info
├── models/
│   └── Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
├── backend/
│   ├── build_index.py    # Indexing script
│   └── app.py            # FastAPI server
├── web/
│   └── index.html        # Chat UI
└── llama.cpp/            # LLM inference engine
    └── build/bin/llama-server
```

## Retrieval Optimizations

The system uses hybrid search to handle both semantic and keyword queries:

1. **Keyword Boosting**: Technical terms (e.g., "Kubernetes", "Docker") get boosted even if embeddings don't match well
2. **Corpus-Specific Stopwords**: Common terms like person names excluded from boosting
3. **Full Index Search**: Small corpus searched entirely, then reranked
4. **Keyword Priority**: Chunks with keyword matches placed first in context (helps small LLMs)

This addresses the "lost in the middle" problem where small LLMs miss information in mid-context positions.

## Limitations

- **1.5B Model**: Small model may miss nuances or hallucinate occasionally
- **CPU Only**: Inference is slower than GPU (~5-15 seconds per query)
- **Chunk Boundaries**: Information split across chunks may be incomplete
- **Vague Queries**: Work best with specific terms; "tell me everything" performs poorly

## Troubleshooting

**Server won't start:**
```bash
# Check if ports are in use
lsof -i :8000
lsof -i :8080
```

**No results / wrong answers:**
- Ensure PDFs have extractable text (not scanned images)
- Rebuild index after adding new PDFs
- Use specific keywords in queries

**Slow responses:**
- Normal for CPU inference
- Reduce `max_tokens` for shorter answers
- Use smaller context with lower `TOP_K`

## Dependencies

Python packages (installed by bootstrap.sh):
- fastapi[standard]
- pymupdf
- faiss-cpu
- fastembed
- httpx
- numpy
- cmake, ninja (for building llama.cpp)
