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

## Known Issues - Retrieval Quality

### Problem: Duplicate Thread Chunks Dominating Results
**Symptom:** When querying "What comedy venues have we worked with?", the system returns only Comedy Bar despite having emails about multiple venues (Citizen Public Market, Rabbit Box, DoStuff venues, etc.).

**Root Cause:** The retrieval system (top_k=6) returns 5 chunks from the same email thread ("Re: TICKET LINK: The Setup / Rabbit Box - July 5th, 2025") plus 1 chunk from a different thread. This happens because:
1. Long email threads get chunked into multiple pieces that all match the query similarly
2. No thread deduplication is applied during retrieval
3. The re-ranker treats each chunk independently, allowing thread dominance

**Impact:** 
- Answers are technically correct but incomplete ("only Comedy Bar is mentioned in the excerpts")
- User sees 6 citations but 5 are from the same conversation
- Reduces information diversity and answer quality

**Potential Solutions:**
1. **Thread Deduplication** - Increase top_k to 20-30, then deduplicate by `thread_id` to keep max 1-2 chunks per thread
2. **MMR-style Diversification** - Penalize chunks from already-selected threads during re-ranking
3. **Metadata Filtering** - Prefer chunks from different senders/dates/subjects
4. **Semantic Deduplication** - Cluster similar chunks and take representatives

**Priority:** High - directly impacts answer completeness for factual queries

---

### Problem: Poor Ranking for Specific Queries (Dates, Names, Exact Phrases)
**Symptom:** Queries with specific details fail to retrieve the correct emails:
- "What's the password for the nudge portal?" → No results
- "What's the password for the nudge ticketing dashboard?" → Correct result
- "What did we discuss in the June 17, 2024 email with Janesh?" → Returns different emails
- Follow-up questions like "what did we talk about?" after system mentions a specific email → Cannot find the referenced email

**Root Cause:** Current retrieval pipeline relies heavily on semantic search (70% weight) with the following limitations:

1. **Vocabulary Mismatch** - Embeddings don't bridge terminology gaps well:
   - Email says "ticketing dashboard" but user searches "portal"
   - Email says "Setup Dates" but user searches "show scheduling"
   - No amount of query expansion (5→20 terms) helps if GPT doesn't generate the exact synonym

2. **Fixed Scoring Weights** - Uses `(semantic * 0.7) + (keyword * 0.3)` for all queries:
   - Conceptual queries ("what does our company do?") → Should favor semantic
   - Specific queries ("June 17, 2024 email") → Should favor exact matches
   - Current approach treats all queries identically

3. **No Metadata Filtering** - Database stores date, sender, recipient, subject but doesn't use them:
   - Query mentions "June 2024" → Could filter to that date range
   - Query mentions "Janesh Rahlan" → Could filter by sender/recipient
   - Query mentions "Setup" → Could filter by subject contains

4. **Limited Chunks Returned** - Only 6 chunks (10 after recent increase) reach the LLM:
   - With 200K emails, the right email might rank 7th → User never sees it
   - Small chunk limit means ranking quality is critical
   - No second chances if ranking fails

5. **Semantic-Only Multi-Query** - Expands query to 5 variations, all searched semantically:
   - Gets different semantic angles but all suffer same vocabulary gap
   - No keyword/exact-match fallback strategy

**Current Workarounds Implemented:**
- Query expansion with conversation history (helps with follow-ups)
- Increased chunks from 6→10 (more chances to surface right email)

**Potential Solutions (Priority Order):**

**Phase 1: Quick Wins (1-2 hours)**
1. **Metadata Filtering** - Detect dates/names in query and filter database:
   ```
   Query: "June 2024 email with Janesh"
   → Filter: date >= 2024-06-01 AND date < 2024-07-01 
   → Filter: from/to contains "janesh" OR "rahlan"
   → Then run semantic search on filtered subset
   ```

2. **Dynamic Scoring Weights** - Adapt based on query type:
   ```python
   if has_specific_date or has_person_name:
       score = (semantic * 0.3) + (keyword * 0.4) + (metadata_match * 0.3)
   else:
       score = (semantic * 0.7) + (keyword * 0.3)
   ```

**Phase 2: Hybrid Search (4-6 hours)**
3. **Add BM25 Keyword Search** - Run both semantic and keyword search, merge results:
   - Semantic search: Finds conceptual matches ("password" ≈ "credentials")
   - BM25 keyword: Finds exact word matches ("nudge" + "ticketing" + "dashboard")
   - Handles vocabulary gaps that semantic search misses

4. **Query Classification** - Route queries to appropriate search strategy:
   ```
   "what does our company do?" → CONCEPTUAL → Pure semantic
   "June 17 email with Janesh" → SPECIFIC → Metadata + keyword heavy
   "recent conversations" → TEMPORAL → Date-sorted semantic
   ```

**Phase 3: Advanced (8+ hours)**
5. **Cross-Encoder Reranking** - After retrieving 30 candidates, rerank with deeper model:
   - Embeddings are fast but approximate (separate query/doc vectors)
   - Cross-encoder is slow but accurate (processes query + doc together)
   - Adds 1-2 seconds but dramatically improves ranking

6. **Multi-Stage Retrieval** - Industry standard approach:
   - Stage 1: Fast, broad search (retrieve 100 candidates for recall)
   - Stage 2: Hybrid scoring (narrow to 30 for precision)
   - Stage 3: LLM reranking (final 10 for relevance)

**Impact:** Critical - Current semantic-only approach is the main bottleneck for:
- Specific date/name queries
- Follow-up questions referencing prior context
- Queries using different terminology than email content
- Any query where ranking accuracy matters (which is most queries)

**Priority:** High - More indexed data (200K emails) will amplify this problem without better ranking

```
