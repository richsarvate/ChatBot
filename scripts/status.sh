#!/bin/bash
# Check the status of the Email QA web application

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PIDFILE="$PROJECT_ROOT/.uvicorn.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "✓ Server is running (PID: $PID)"
        echo "  Access at: http://0.0.0.0:8000"
        echo "  Logs: $PROJECT_ROOT/logs/server.log"
        exit 0
    else
        echo "✗ Server not running (stale PID file)"
        exit 1
    fi
else
    echo "✗ Server not running (no PID file)"
    exit 1
fi
