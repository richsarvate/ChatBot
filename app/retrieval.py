from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

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
                
                # Semantic similarity score
                similarity_score = 1 - chunk.distance if chunk.distance else 0.0
                
                # Improved keyword scoring - check if ANY query keywords appear
                keyword_matches = sum(1 for kw in query_keywords if kw in chunk.snippet.lower())
                keyword_score = min(keyword_matches / max(len(query_keywords), 1), 1.0)
                
                # Metadata boost - check subject, sender, date
                metadata_boost = 0.0
                
                # Boost if subject contains query keywords
                subject_lower = chunk.subject.lower()
                subject_matches = sum(1 for kw in query_keywords if kw in subject_lower)
                if subject_matches > 0:
                    metadata_boost += 0.2 * (subject_matches / len(query_keywords))
                
                # Boost if sender/recipient matches person names (capitalized words)
                potential_names = [w for w in query.split() if w and w[0].isupper() and len(w) > 2]
                from_lower = chunk.from_address.lower()
                for name in potential_names:
                    if name.lower() in from_lower:
                        metadata_boost += 0.3
                        break
                
                # Boost if date matches query year
                for keyword in query_keywords:
                    if keyword.isdigit() and len(keyword) == 4:  # Year
                        if keyword in chunk.date:
                            metadata_boost += 0.3
                            break
                
                # Cap metadata boost at 0.5
                metadata_boost = min(metadata_boost, 0.5)
                
                # Dynamic scoring weights based on query type
                if is_specific_query:
                    # Specific queries: favor exact matches and metadata
                    combined = (similarity_score * 0.3) + (keyword_score * 0.4) + (metadata_boost * 0.3)
                else:
                    # Conceptual queries: favor semantic similarity
                    combined = (similarity_score * 0.7) + (keyword_score * 0.2) + (metadata_boost * 0.1)
                
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
