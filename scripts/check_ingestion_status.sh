#!/bin/bash
# Check the status of the continuous ingestion process

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/ingestion_progress.log"

echo "========================================="
echo "Ingestion Status Check"
echo "========================================="
echo ""

# Check if process is running
if pgrep -f "ingest_all.sh" > /dev/null; then
    PID=$(pgrep -f "ingest_all.sh")
    echo "✅ Ingestion process is RUNNING (PID: $PID)"
else
    echo "❌ Ingestion process is NOT running"
fi

echo ""
echo "Latest progress:"
echo "----------------------------------------"
tail -20 "$LOG_FILE" 2>/dev/null || echo "No log file found yet"

echo ""
echo "========================================="
echo "To view live progress, run:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop ingestion, run:"
echo "  pkill -f ingest_all.sh"
echo "========================================="
