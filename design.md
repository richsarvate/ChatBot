# Email Chatbot Design
## Objective
- Provide question-answering over Gmail history with minimal code and zero SaaS fees.

## End-to-End Flow
1. Export mailbox via Google Takeout (single MBOX bundle).
2. Parse the MBOX into structured JSON using a Python `mailparser` script.
3. Generate embeddings with the OpenAI `text-embedding-3-large` API and persist vectors in Chroma.
4. Retrieve relevant chunks per query and answer with OpenAI `gpt-4.1` citing sources.

## Data Handling
- Store raw archives under `/data/raw/` (plain MBOX for the first pass; add encryption later once workflow is stable).
- Keep parsed JSON per message in `/data/processed/` named `<messageId>.json`.
- Persist embeddings and metadata under `/data/index/chroma/` using Chroma's persistent client.

## Parsing Notes
- Use the `mailparser` Python package to extract subject, participants, dates, labels, and plain-text body.
- Normalize timestamps to UTC ISO8601 with `pendulum`.
- Strip HTML, signatures, and quoted replies via `mailparser` options while preserving attachment metadata.
- Retain sensitive details (passwords, contact info) verbatim so they remain searchable later.

## Embeddings
- Use OpenAI `text-embedding-3-large` via the official Python SDK.
- Chunk bodies at 500 tokens with 50-token overlap using `langchain.text_splitter.RecursiveCharacterTextSplitter`.
- Store each chunk's cosine similarity scores, chunk order, and token counts inside the Chroma metadata payload.

## Index & Filters
- Run a local Chroma server (`chroma run --path ./data/index/chroma`).
- Support filters by date, participants, and labels through Chroma metadata queries.
- Snapshot the index after each refresh by copying the `./data/index/chroma` folder to `/data/backups/`.

## Query Orchestration
- Embed each user question with `text-embedding-3-large` and fetch top 6 chunks from Chroma.
- Re-rank the retrieved chunks by combining cosine score and keyword overlap implemented with `rapidfuzz`.
- Build context blocks containing subject, date, sender, snippet, and message ID for each chunk.
- Append every query, retrieved IDs, and response metadata to `/logs/interactions.jsonl`.

## LLM Layer
- Use OpenAI `gpt-4.1` with a system prompt instructing citation by subject and date and refusal when context is insufficient.
- Pass the retrieved snippets verbatim to allow answers that include stored passwords or contact details when requested.

## Interface
- Provide a FastAPI web app with a single-page front end (FastAPI + Jinja2) that submits questions, shows sourced answers, and links cited emails.
- Persist interaction logs locally for debugging; keep the machine offline except during OpenAI API calls.

## TODO (NOT PART OF FIRST PASS)
- Re-run Takeout on the first day of each month, parse only the new archive, and append fresh embeddings.
- Keep encrypted backups of raw, parsed, and index data by running a nightly `rsync` to an external drive.
- Maintain a regression suite of 20 representative questions in `tests/regression_cases.json` and review outputs quarterly.
- Plan a future upgrade to Gmail API incremental sync implemented with the same `mailparser` pipeline.
