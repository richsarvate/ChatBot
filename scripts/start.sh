#!/bin/bash
# Start the Email QA web application

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if .env exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "ERROR: .env file not found. Please create one with OPENAI_API_KEY=sk-..."
    exit 1
fi

# Check if venv exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "ERROR: Virtual environment not found at $PROJECT_ROOT/.venv"
    exit 1
fi

PIDFILE="$PROJECT_ROOT/.uvicorn.pid"

# Check if already running
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Server is already running (PID: $PID)"
        exit 0
    else
        echo "Removing stale PID file"
        rm "$PIDFILE"
    fi
fi

echo "Starting Email QA server..."

# Check for SSL certificates
CERT_FILE="$PROJECT_ROOT/certs/cert.pem"
KEY_FILE="$PROJECT_ROOT/certs/key.pem"

if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "Starting with HTTPS (SSL enabled)..."
    nohup "$PROJECT_ROOT/.venv/bin/uvicorn" app.server:app \
        --host 0.0.0.0 \
        --port 8000 \
        --ssl-keyfile "$KEY_FILE" \
        --ssl-certfile "$CERT_FILE" \
        > "$PROJECT_ROOT/logs/server.log" 2>&1 &
    PROTOCOL="https"
else
    echo "Starting with HTTP (no SSL certificates found)..."
    nohup "$PROJECT_ROOT/.venv/bin/uvicorn" app.server:app \
        --host 0.0.0.0 \
        --port 8000 \
        > "$PROJECT_ROOT/logs/server.log" 2>&1 &
    PROTOCOL="http"
fi

PID=$!
echo $PID > "$PIDFILE"

echo "Server started (PID: $PID)"
echo "Logs: $PROJECT_ROOT/logs/server.log"
echo "Access at: $PROTOCOL://0.0.0.0:8000"
