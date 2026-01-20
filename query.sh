#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./query.sh \"your question here\""
    exit 1
fi

# Capture output to variable first to allow debugging
RESPONSE=$(curl -s http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$1\"}")

# Check if response is empty
if [ -z "$RESPONSE" ]; then
    echo "Error: Empty response from server"
    exit 1
fi

# Try to parse with jq, show raw response if it fails
if ! echo "$RESPONSE" | jq -r '.answer' 2>/dev/null; then
    echo "Error parsing JSON response:"
    echo "$RESPONSE"
    exit 1
fi
