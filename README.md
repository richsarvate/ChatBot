# Email QA Chatbot

Personal email assistant that lets me ask questions about my 26,700+ emails and get actual answers.

## How it works

### The basic flow

1. User asks a question
2. System finds relevant emails using hybrid search
3. GPT reads those emails and answers the question
4. Shows sources so I can verify

### The interesting part: Retrieval

Getting the right emails is the hard part. I use a hybrid approach that combines three things:

**1. Query expansion**
- GPT rewrites my question 5 different ways
- Example: "who is Janesh" becomes:
  - "Janesh"
  - "Janesh Rahlan" 
  - "Janesh comedy"
  - "Janesh events"
  - etc.
- Also uses conversation history to resolve pronouns ("their email" → "Maddox and Ian Ford's email")

**2. BM25 keyword search**
- Classic information retrieval algorithm
- Finds exact keyword matches: "Janesh", "Rahlan", "Bourbon Room"
- Good for: names, places, specific terms
- Returns top 100 matches per query expansion

**3. Semantic search**
- OpenAI embeddings + ChromaDB vector search
- Understands concepts: "who does the setup talk to" finds contact emails even without word "contact"
- Good for: conceptual questions, synonyms, context
- Returns top 100 matches per query expansion

**4. Reciprocal Rank Fusion (RRF)**
- Combines BM25 and semantic results
- Formula: `score = 1/(rank + 60)` for each method
- Add the scores together
- Result: emails that match keywords AND concepts rank highest

**5. Metadata boosting**
- Boost if subject line matches query keywords (+0.1)
- Boost if sender matches person names (+0.15)
- Boost if date matches query year (+0.15)
- Penalize spam (order confirmations, auto-replies, etc.) (-0.3)

**6. Thread deduplication**
- Take top 10 emails after all that
- Max 1 email per thread to show diversity

### Why this works better than pure semantic search

Semantic search fails on:
- **Entity recognition**: "who is Janesh Rahlan" doesn't find emails with that name
- **Vocabulary gaps**: "portal" vs "ticketing dashboard" vs "nudge"
- **Exact matches**: dates, order numbers, specific phrases

BM25 handles all of these perfectly because it just looks for the exact words.

The hybrid approach gets the best of both worlds.

### Chat history

Conversation context flows through the whole pipeline:

1. Frontend sends `session_id` with each request
2. Server stores question + answer in memory per session
3. When you ask a follow-up, query expansion sees the history
4. GPT resolves pronouns: "their email" → looks at previous answer → sees "Maddox and Ian Ford" → expands to "Maddox Ian Ford email"
5. Retrieval finds the right emails using those expanded keywords

This is why follow-up questions mostly work.

## Tech stack

- **Python 3.12** - backend
- **FastAPI** - web server
- **ChromaDB** - vector database for embeddings
- **OpenAI API** - embeddings (text-embedding-3-large) and chat (gpt-4o)
- **rank-bm25** - BM25 keyword search
- **No frontend framework** - just vanilla HTML/JS

## File structure

```
app/
  ├── chat.py         # GPT prompting and response generation
  ├── retrieval.py    # Hybrid search (BM25 + semantic)
  ├── embedding.py    # OpenAI embeddings
  ├── index.py        # ChromaDB interface
  ├── ingest.py       # Email processing and indexing
  └── config.py       # Settings (including spam patterns)

data/
  ├── raw/            # Original .txt emails
  ├── processed/      # Parsed JSON emails
  └── index/chroma/   # ChromaDB + BM25 index

scripts/
  └── build_bm25_index.py  # Build BM25 from existing ChromaDB

templates/
  └── index.html      # Chat UI
```

## Performance notes

- **164,391 chunks** indexed from 26,700 emails
- **BM25 index**: 532 MB in memory
- **EC2 instance**: t3.xlarge (8GB RAM + 4GB swap)
- **Query time**: ~2-3 seconds for hybrid search + GPT response
- **Batch processing**: all chunk loading uses batches (100 at a time) to prevent OOM

## Known limitations

- Conversation history sometimes doesn't resolve pronouns correctly
- Spam filtering patterns need tuning (currently 10 patterns)
- No user authentication (single user)
- BM25 uses simple whitespace tokenization (no stemming/lemmatization)

## Future improvements

- Better pronoun resolution in query expansion
- Add more spam patterns as I find them
- Reranking model (like Cohere) after hybrid search
- Support for attachments (PDFs, images)
- Multi-user support with auth
