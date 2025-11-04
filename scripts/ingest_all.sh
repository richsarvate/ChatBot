#!/bin/bash
# Continuous ingestion script - processes all remaining emails in chunks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/ingestion_progress.log"
CHUNK_SIZE=10000
START_EMAIL=20000  # Already processed 0-20000

cd "$PROJECT_ROOT"

# Create archive directory if it doesn't exist
mkdir -p data/archive

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to get current email count in full corpus
get_total_emails() {
    # Count from the main corpus file (you'll need to adjust path)
    # For now, we'll process in chunks until extraction fails
    echo "200000"  # Approximate total based on your corpus size
}

log "========================================="
log "Starting continuous ingestion process"
log "Chunk size: $CHUNK_SIZE emails"
log "Starting from email: $START_EMAIL"
log "========================================="

CURRENT_POS=$START_EMAIL
CHUNK_NUM=3

while true; do
    END_POS=$((CURRENT_POS + CHUNK_SIZE))
    CHUNK_FILE="data/raw/chunk_$(printf "%03d" $CHUNK_NUM).mbox"
    
    log "----------------------------------------"
    log "Processing chunk $CHUNK_NUM: emails $CURRENT_POS-$((END_POS-1))"
    
    # Extract chunk
    log "Extracting emails $CURRENT_POS-$((END_POS-1)) to $CHUNK_FILE..."
    if ! .venv/bin/python scripts/extract_chunk.py $CURRENT_POS $END_POS "$CHUNK_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        log "Extraction failed or reached end of corpus. Stopping."
        break
    fi
    
    # Check if file was created and has content
    if [ ! -f "$CHUNK_FILE" ] || [ ! -s "$CHUNK_FILE" ]; then
        log "No more emails to extract. Ingestion complete!"
        break
    fi
    
    FILE_SIZE=$(du -h "$CHUNK_FILE" | cut -f1)
    log "Chunk file size: $FILE_SIZE"
    
    # Archive previous chunk if it exists
    PREV_CHUNK="data/raw/chunk_$(printf "%03d" $((CHUNK_NUM-1))).mbox"
    if [ -f "$PREV_CHUNK" ]; then
        log "Archiving previous chunk: $PREV_CHUNK"
        mv "$PREV_CHUNK" data/archive/
    fi
    
    # Ingest the new chunk
    log "Starting ingestion of chunk $CHUNK_NUM..."
    INGEST_START=$(date +%s)
    
    if .venv/bin/python manage.py ingest 2>&1 | tee -a "$LOG_FILE"; then
        INGEST_END=$(date +%s)
        DURATION=$((INGEST_END - INGEST_START))
        log "Chunk $CHUNK_NUM ingestion completed in ${DURATION}s"
        
        # Extract stats from ingestion output
        KEPT=$(tail -50 "$LOG_FILE" | grep "Emails indexed:" | tail -1 | awk '{print $3}')
        CHUNKS=$(tail -50 "$LOG_FILE" | grep "Chunks created:" | tail -1 | awk '{print $3}')
        FILTERED=$(tail -50 "$LOG_FILE" | grep "Emails filtered:" | tail -1 | awk '{print $3}')
        
        log "Stats: Kept=$KEPT, Chunks=$CHUNKS, Filtered=$FILTERED"
    else
        log "ERROR: Ingestion failed for chunk $CHUNK_NUM"
        log "Check the logs above for details"
        break
    fi
    
    # Move to next chunk
    CURRENT_POS=$END_POS
    CHUNK_NUM=$((CHUNK_NUM + 1))
    
    log "Sleeping 5 seconds before next chunk..."
    sleep 5
done

log "========================================="
log "Ingestion process finished"
log "Final position: $CURRENT_POS"
log "Total chunks processed: $((CHUNK_NUM - 3))"
log "========================================="

# Get final database stats
log "Final database statistics:"
.venv/bin/python -c "
import chromadb
client = chromadb.PersistentClient(path='data/index/chroma')
collection = client.get_collection(name='emails')
print(f'Total chunks in database: {collection.count()}')
" 2>&1 | tee -a "$LOG_FILE"
