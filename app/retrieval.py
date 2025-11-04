from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from openai import OpenAI
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

from .config import Settings
from .embedding import Embedder
from .index import EmailIndex, IndexedChunk


@dataclass
class RetrievedChunk:
    chunk: IndexedChunk
    score: float


class Retriever:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedder = Embedder(settings)
        self._index = EmailIndex(settings)
        self._openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Load BM25 index for keyword search
        self._bm25_index = None
        self._bm25_chunk_ids = []
        self._bm25_metadatas = []
        self._bm25_corpus = []
        self._load_bm25_index()

    def _load_bm25_index(self) -> None:
        """Load BM25 index from disk. Fails gracefully if not found."""
        bm25_path = self._settings.index_dir / "bm25_index.pkl"
        if not bm25_path.exists():
            print("⚠️  BM25 index not found. Run ingestion to build it. Using semantic-only search.")
            return
        
        try:
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            
            self._bm25_index = bm25_data["index"]
            self._bm25_chunk_ids = bm25_data["chunk_ids"]
            self._bm25_metadatas = bm25_data["metadatas"]
            self._bm25_corpus = bm25_data["corpus"]
            print(f"✓ Loaded BM25 index: {len(self._bm25_chunk_ids)} chunks")
        except Exception as e:
            print(f"⚠️  Failed to load BM25 index: {e}. Using semantic-only search.")

    def _extract_keywords(self, query: str) -> Set[str]:
        """
        Extract important keywords from query for metadata matching.
        Returns normalized keywords (lowercase, stripped).
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'as', 'is', 'was', 'are', 'were',
            'what', 'when', 'where', 'who', 'how', 'did', 'do', 'does', 'tell', 'me',
            'my', 'your', 'our', 'their', 'have', 'has', 'had', 'be', 'been'
        }
        
        # Split and clean
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Keep meaningful words (3+ chars, not stop words)
        keywords = {w for w in words if len(w) >= 3 and w not in stop_words}
        
        # Also extract potential dates (various formats)
        dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}', query.lower())
        keywords.update(dates)
        
        # Extract years
        years = re.findall(r'\b20\d{2}\b', query)
        keywords.update(years)
        
        return keywords

    def _expand_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """
        Use GPT to expand the query with synonyms, abbreviations, and variations.
        This helps handle cases where semantic search misses specific terms like "PW" for "password".
        Also uses conversation history to resolve references like "that", "they", etc.
        """
        # Build context from recent conversation
        context = ""
        if conversation_history and len(conversation_history) >= 2:
            # Get last 2 exchanges (4 messages)
            recent = conversation_history[-4:]
            context = "\n\nRecent conversation context:\n"
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content'][:200]}...\n"
        
        prompt = f"""Given this search query{' and conversation context' if context else ''}, generate 3-5 related search terms including:
- Synonyms
- Common abbreviations
- Alternative phrasings
- Related terms
- Specific names, dates, or entities mentioned in context

{context}

Current query: "{query}"

If the query contains vague references like "that conversation", "what did we talk about", "they", etc., 
use the conversation context to identify the specific subject, person, or topic being referenced.

Return ONLY a comma-separated list of search terms, nothing else.
Example: "password, PW, pw:, credentials, login info"
"""
        
        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for simple task
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )
            
            expansions_text = response.choices[0].message.content.strip()
            # Parse comma-separated list
            expansions = [term.strip() for term in expansions_text.split(',') if term.strip()]
            
            # Always include original query first
            if query not in expansions:
                expansions.insert(0, query)
            
            return expansions[:5]  # Limit to 5 total
            
        except Exception as e:
            print(f"Query expansion failed: {e}, using original query")
            return [query]

    def _hybrid_search(self, query: str, top_k: int = 100) -> Dict[str, float]:
        """
        Perform hybrid search combining BM25 (keyword) and semantic search.
        Returns dict mapping chunk_id to combined score using reciprocal rank fusion.
        """
        # 1. BM25 keyword search
        bm25_scores = {}
        if self._bm25_index is not None:
            # Tokenize query (same way as corpus)
            tokenized_query = query.lower().split()
            bm25_raw_scores = self._bm25_index.get_scores(tokenized_query)
            
            # Get top BM25 results
            bm25_top_indices = sorted(range(len(bm25_raw_scores)), key=lambda i: bm25_raw_scores[i], reverse=True)[:top_k]
            for rank, idx in enumerate(bm25_top_indices, 1):
                chunk_id = self._bm25_chunk_ids[idx]
                # Reciprocal rank fusion: 1 / (rank + 60)
                bm25_scores[chunk_id] = 1.0 / (rank + 60)
        
        # 2. Semantic search
        semantic_scores = {}
        query_embedding = self._embedder.embed([query])
        if query_embedding:
            raw_hits = self._index.query(
                query_embedding=query_embedding[0],
                n_results=top_k,
                where=None,
            )
            
            for rank, chunk in enumerate(raw_hits, 1):
                chunk_id = chunk.chunk_id
                # Reciprocal rank fusion: 1 / (rank + 60)
                semantic_scores[chunk_id] = 1.0 / (rank + 60)
        
        # 3. Combine using reciprocal rank fusion
        combined_scores = {}
        all_chunk_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())
        
        for chunk_id in all_chunk_ids:
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            semantic_score = semantic_scores.get(chunk_id, 0.0)
            # Equal weight to both methods
            combined_scores[chunk_id] = bm25_score + semantic_score
        
        print(f"🔍 Hybrid search: {len(bm25_scores)} BM25 + {len(semantic_scores)} semantic = {len(combined_scores)} combined")
        
        return combined_scores

    def search(self, query: str, *, where: Optional[dict] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> List[RetrievedChunk]:
        # Extract keywords from query for metadata matching
        query_keywords = self._extract_keywords(query)
        
        # Detect if query has specific entities (dates, years, or proper nouns with caps)
        has_date = bool(re.search(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', query.lower()))
        has_proper_noun = bool(re.search(r'\b[A-Z][a-z]+\b', query))
        is_specific_query = has_date or has_proper_noun
        
        # Expand query to catch variations
        query_expansions = self._expand_query(query, conversation_history=conversation_history)
        print(f"🔍 Query expansions: {query_expansions}")
        print(f"🔑 Extracted keywords: {query_keywords}")
        print(f"📍 Query type: {'SPECIFIC' if is_specific_query else 'CONCEPTUAL'}")
        
        # Use hybrid search (BM25 + semantic) to get candidates
        all_chunks = {}  # Use dict to deduplicate by chunk ID
        
        # Perform hybrid search for each query expansion
        for expanded_query in query_expansions:
            hybrid_scores = self._hybrid_search(expanded_query, top_k=self._settings.top_k)
            
            # Get all chunk IDs from hybrid search (both BM25 and semantic)
            # Sort by score and take top candidates to limit memory usage
            sorted_chunk_ids = sorted(hybrid_scores.keys(), key=lambda cid: hybrid_scores[cid], reverse=True)
            top_chunk_ids = sorted_chunk_ids[:200]  # Limit to top 200 per expansion to prevent OOM
            
            # Retrieve full chunk data for ALL hybrid search results (not just semantic)
            # This ensures BM25-only hits aren't discarded
            if top_chunk_ids:
                chunks_by_id = self._index.get_chunks_by_ids(top_chunk_ids)
                
                # Map chunks and combine with hybrid scores
                for chunk_id, chunk in chunks_by_id.items():
                    rrf_score = hybrid_scores.get(chunk_id, 0.0)
                    
                    if chunk_id in all_chunks:
                        # Keep best RRF score across expansions
                        if rrf_score > all_chunks[chunk_id]['rrf_score']:
                            all_chunks[chunk_id] = {'chunk': chunk, 'rrf_score': rrf_score}
                        continue
                    
                    all_chunks[chunk_id] = {'chunk': chunk, 'rrf_score': rrf_score}
        
        # Now apply metadata boosting and reranking on top of RRF scores
        scored_chunks = {}
        for chunk_id, data in all_chunks.items():
            chunk = data['chunk']
            rrf_score = data['rrf_score']
            
            # Use RRF score as base (already combines semantic + keyword)
            base_score = rrf_score
            
            # Add metadata boost on top of RRF
            metadata_boost = 0.0
            
            # Boost if subject contains query keywords
            subject_lower = chunk.subject.lower()
            subject_matches = sum(1 for kw in query_keywords if kw in subject_lower)
            if subject_matches > 0:
                metadata_boost += 0.1 * (subject_matches / max(len(query_keywords), 1))
            
            # Boost if sender/recipient matches person names (capitalized words)
            potential_names = [w for w in query.split() if w and w[0].isupper() and len(w) > 2]
            from_lower = chunk.from_address.lower()
            for name in potential_names:
                if name.lower() in from_lower:
                    metadata_boost += 0.15
                    break
            
            # Boost if date matches query year
            for keyword in query_keywords:
                if keyword.isdigit() and len(keyword) == 4:  # Year
                    if keyword in chunk.date:
                        metadata_boost += 0.15
                        break
            
            # Penalize spammy/automated emails
            if self._is_spammy_email(chunk):
                metadata_boost -= self._settings.spam_penalty
            
            # Combined score: RRF base + metadata boost/penalties
            # RRF already balanced semantic and keyword, so just add metadata
            combined = base_score + metadata_boost
            
            scored_chunks[chunk_id] = RetrievedChunk(chunk=chunk, score=combined)
        
        # Sort by score
        reranked = sorted(scored_chunks.values(), key=lambda item: item.score, reverse=True)
        
        # Deduplicate by thread_id to improve diversity
        return self._deduplicate_by_thread(reranked, max_per_thread=1)
    
    def _is_spammy_email(self, chunk: IndexedChunk) -> bool:
        """
        Detect automated/promotional emails with low information value.
        Patterns are configurable in settings.
        """
        subject = chunk.subject.lower()
        from_addr = chunk.from_address.lower()
        
        # Check subject patterns
        for pattern in self._settings.spam_subject_patterns:
            if pattern in subject:
                return True
        
        # Check sender patterns
        for pattern in self._settings.spam_sender_patterns:
            if pattern in from_addr:
                return True
        
        return False
    
    def _deduplicate_by_thread(
        self, 
        chunks: List[RetrievedChunk], 
        max_per_thread: int = 2
    ) -> List[RetrievedChunk]:
        """
        Deduplicate chunks to ensure diverse results across email threads.
        Keeps at most max_per_thread chunks from each thread_id.
        Returns top_k_final chunks after deduplication.
        """
        seen_threads = {}
        deduplicated = []
        
        for item in chunks:
            thread_id = item.chunk.thread_id
            
            # Count how many chunks we've already taken from this thread
            thread_count = seen_threads.get(thread_id, 0)
            
            if thread_count < max_per_thread:
                deduplicated.append(item)
                seen_threads[thread_id] = thread_count + 1
                
                # Stop once we have enough diverse chunks
                if len(deduplicated) >= self._settings.top_k_final:
                    break
        
        return deduplicated
