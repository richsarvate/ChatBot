#!/bin/bash
# Stop the Email QA web application

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PIDFILE="$PROJECT_ROOT/.uvicorn.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "No PID file found. Server may not be running."
    # Try to kill by process name as fallback
    pkill -f "uvicorn app.server:app" && echo "Killed server by process name" || echo "No server process found"
    exit 0
fi

PID=$(cat "$PIDFILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping server (PID: $PID)..."
    kill "$PID"
    
    # Wait for graceful shutdown
    for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "Server stopped successfully"
            rm "$PIDFILE"
            exit 0
        fi
        sleep 1
    done
    
    # Force kill if still running
    echo "Force killing server..."
    kill -9 "$PID" 2>/dev/null || true
    rm "$PIDFILE"
    echo "Server stopped (forced)"
else
    echo "Server not running (stale PID file)"
    rm "$PIDFILE"
fi
