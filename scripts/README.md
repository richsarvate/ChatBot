# Operations Scripts

This directory contains operational scripts for managing the Email QA application.

## Scripts

### Server Management

- **`start.sh`** - Start the web server in background
- **`stop.sh`** - Stop the running web server
- **`restart.sh`** - Restart the web server
- **`status.sh`** - Check if server is running
- **`logs.sh`** - Tail the server logs

### Data Management

- **`ingest-and-restart.sh`** - Run ingestion and restart server to load new index

## Usage Examples

```bash
# Start the server
./scripts/start.sh

# Check status
./scripts/status.sh

# View logs
./scripts/logs.sh

# Stop the server
./scripts/stop.sh

# Restart after making changes
./scripts/restart.sh

# Ingest new emails and restart
./scripts/ingest-and-restart.sh --limit 100
```

## Notes

- Server runs on `http://0.0.0.0:8000`
- Logs are written to `logs/server.log`
- PID file is stored at `.uvicorn.pid`
- Requires `.env` file with `OPENAI_API_KEY`
