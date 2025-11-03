#!/bin/bash
# Tail the server logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGFILE="$PROJECT_ROOT/logs/server.log"

if [ ! -f "$LOGFILE" ]; then
    echo "No log file found at $LOGFILE"
    exit 1
fi

tail -f "$LOGFILE"
