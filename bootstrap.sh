#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== PDF Chat Agent Setup ==="

# Check/install system dependencies
if command -v apt-get &>/dev/null; then
    MISSING=""
    command -v python3 &>/dev/null || MISSING="$MISSING python3"
    command -v git &>/dev/null || MISSING="$MISSING git"
    command -v g++ &>/dev/null || MISSING="$MISSING build-essential"
    python3 -c "import venv" 2>/dev/null || MISSING="$MISSING python3-venv"

    if [ -n "$MISSING" ]; then
        echo "Installing system dependencies:$MISSING"
        sudo apt-get update && sudo apt-get install -y $MISSING
    fi
fi

# Create directories
echo "Creating directories..."
mkdir -p pdfs index models backend web

# Python venv
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "Installing Python dependencies..."
.venv/bin/pip install -q "fastapi[standard]" pymupdf faiss-cpu fastembed httpx numpy cmake ninja

# Build llama.cpp
if [ ! -f "llama.cpp/build/bin/llama-server" ]; then
    echo "Cloning llama.cpp..."
    if [ ! -d "llama.cpp" ]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp
    fi
    echo "Building llama.cpp..."
    cd llama.cpp
    ./../.venv/bin/cmake -B build
    ./../.venv/bin/cmake --build build -j$(nproc) --target llama-server
    cd ..
fi

# Download model
if [ ! -f "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" ]; then
    echo "Downloading Qwen2.5-1.5B model..."
    .venv/bin/hf download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
        --include "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" \
        --local-dir ./models
fi

# Build index if PDFs exist
if ls pdfs/*.pdf 1>/dev/null 2>&1; then
    echo "Building FAISS index..."
    .venv/bin/python backend/build_index.py --pdf_dir ./pdfs --out_dir ./index
else
    echo "No PDFs found in pdfs/ - add PDFs and run: .venv/bin/python backend/build_index.py"
fi

echo ""
echo "=== Setup complete ==="
echo "Run ./run.sh to start the servers"
