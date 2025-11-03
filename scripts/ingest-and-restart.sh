#!/bin/bash
# Ingest emails and rebuild the index, then restart server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running ingestion..."
"$PROJECT_ROOT/.venv/bin/python" manage.py ingest --rebuild "$@"

echo ""
echo "Restarting server to load new index..."
"$SCRIPT_DIR/restart.sh"

echo ""
echo "Ingestion complete and server restarted!"
