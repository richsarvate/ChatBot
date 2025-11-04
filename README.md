# Architecture

## Ingest Flow

1. **Parse emails**: Read .txt files → extract subject, from, to, date, body → save as JSON
2. **Chunk text**: Split body into 500-token chunks with 50-token overlap
3. **Generate embeddings**: Call OpenAI `text-embedding-3-large` (3072 dims) for each chunk
4. **Index in ChromaDB**: Store embeddings + metadata (message_id, thread_id, subject, from, date, chunk_index)
5. **Build BM25 index**: Tokenize all chunks (lowercase + whitespace split) → create BM25Okapi index
6. **Save to disk**: Write `bm25_index.pkl` (532 MB, 164,391 chunks)

Result: 26,700 emails → 164,391 searchable chunks

## Retrieval Flow

**Input**: User query + conversation history (last 10 messages)

### 1. Query Expansion
- Send query + history to `gpt-4o-mini`
- Get 5 query variations (synonyms, resolved pronouns, related terms)
- Example: "their email" → ["Maddox Ian Ford email", "contact info", "Bourbon Room email", ...]

### 2. Hybrid Search (run for each of 5 expansions)
- **BM25**: Tokenize query → score all 164k chunks → top 100
- **Semantic**: Embed query (`text-embedding-3-large`) → ChromaDB cosine search → top 100
- **Merge with RRF**: `score = 1/(bm25_rank + 60) + 1/(semantic_rank + 60)` → top 200

### 3. Aggregate Results
- Collect all chunk IDs from 5 expansions
- Batch retrieve from ChromaDB (100 chunks/batch)
- Keep highest RRF score per unique chunk

### 4. Rerank with Metadata
- Base score: RRF from step 2
- +0.1 if subject matches query keywords
- +0.15 if sender matches person name in query
- +0.15 if date matches query year
- -0.3 if spam (order confirmations, auto-replies, noreply@)

### 5. Deduplicate
- Sort by final score
- Take top 10 chunks, max 1 per thread_id

### 6. Generate Answer
- Send top 10 chunks + query + history to `gpt-4o`
- Return answer + citations

**Performance**: 5 expansions × ~200 chunks = ~1000 candidates → 10 results in 2-3 sec
