from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI
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

    def _expand_query(self, query: str) -> List[str]:
        """
        Use GPT to expand the query with synonyms, abbreviations, and variations.
        This helps handle cases where semantic search misses specific terms like "PW" for "password".
        """
        prompt = f"""Given this search query, generate 3-5 related search terms including:
- Synonyms
- Common abbreviations
- Alternative phrasings
- Related terms

Query: "{query}"

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

    def search(self, query: str, *, where: Optional[dict] = None) -> List[RetrievedChunk]:
        # Expand query to catch variations
        query_expansions = self._expand_query(query)
        print(f"🔍 Query expansions: {query_expansions}")
        
        # Search with all query variations
        all_chunks = {}  # Use dict to deduplicate by chunk ID
        
        for expanded_query in query_expansions:
            query_embedding = self._embedder.embed([expanded_query])
            if not query_embedding:
                continue
                
            raw_hits = self._index.query(
                query_embedding=query_embedding[0],
                n_results=self._settings.top_k,
                where=where,
            )
            
            # Score and add to combined results
            for chunk in raw_hits:
                # Use the chunk_id field directly (already unique per chunk)
                chunk_id = chunk.chunk_id
                
                # Skip if we've already seen this chunk (take best score)
                if chunk_id in all_chunks:
                    continue
                
                keyword_score = fuzz.partial_ratio(query.lower(), chunk.snippet.lower()) / 100.0
                similarity_score = 1 - chunk.distance if chunk.distance else 0.0
                combined = (similarity_score * 0.7) + (keyword_score * 0.3)
                
                all_chunks[chunk_id] = RetrievedChunk(chunk=chunk, score=combined)
        
        # Sort by score
        reranked = sorted(all_chunks.values(), key=lambda item: item.score, reverse=True)
        
        # Deduplicate by thread_id to improve diversity
        return self._deduplicate_by_thread(reranked, max_per_thread=1)
    
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
