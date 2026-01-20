#!/bin/bash
set -e

cd "$(dirname "$0")"

# Kill any existing servers
pkill -f "llama-server" 2>/dev/null || true
pkill -f "uvicorn.*app" 2>/dev/null || true
sleep 1

# Check prerequisites
if [ ! -f "llama.cpp/build/bin/llama-server" ]; then
    echo "Error: llama-server not built. Run ./bootstrap.sh first."
    exit 1
fi

if [ ! -f "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" ]; then
    echo "Error: Model not found. Run ./bootstrap.sh first."
    exit 1
fi

if [ ! -f "index/index.faiss" ]; then
    echo "Error: Index not built. Add PDFs to pdfs/ and run ./bootstrap.sh"
    exit 1
fi

cleanup() {
    echo "Shutting down..."
    kill $LLAMA_PID 2>/dev/null || true
    kill $UVICORN_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "Starting llama-server on port 8080..."
./llama.cpp/build/bin/llama-server \
    -m ./models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
    -t $(nproc) -c 4096 \
    --host 127.0.0.1 --port 8080 &
LLAMA_PID=$!

# Wait for llama-server
echo "Waiting for llama-server..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo "Starting FastAPI on port 8000..."
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!

sleep 2
echo ""
echo "=== Running ==="
echo "Open http://127.0.0.1:8000"
echo "Press Ctrl+C to stop"
echo ""

wait
