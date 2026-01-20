#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./query.sh \"your question here\""
    exit 1
fi

curl -s -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$1\"}" | jq -r '.answer'
