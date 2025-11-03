from __future__ import annotations

import time
from typing import Iterable, List

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from .config import Settings


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=60.0,  # 60 second timeout
            max_retries=3
        )
        self._model = settings.embedding_model
        self._max_batch_size = 2048  # OpenAI limit for embeddings API

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        payload = list(texts)
        if not payload:
            return []
        
        # If batch is too large, split it
        if len(payload) > self._max_batch_size:
            all_embeddings = []
            for i in range(0, len(payload), self._max_batch_size):
                batch = payload[i:i + self._max_batch_size]
                batch_embeddings = self._embed_with_retry(batch, batch_num=i // self._max_batch_size + 1)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        else:
            return self._embed_with_retry(payload)
    
    def _embed_with_retry(self, payload: List[str], batch_num: int | None = None) -> List[List[float]]:
        """Embed with exponential backoff retry logic."""
        max_retries = 5
        base_delay = 1.0
        
        batch_label = f"batch {batch_num}" if batch_num else "texts"
        
        for attempt in range(max_retries):
            try:
                response = self._client.embeddings.create(model=self._model, input=payload)
                return [item.embedding for item in response.data]
            
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit hit for {batch_label} ({len(payload)} items), retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"Rate limit exceeded after {max_retries} attempts for {batch_label}")
                    raise
            
            except (APITimeoutError, APIConnectionError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"API timeout/connection error for {batch_label}, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"API timeout/connection failed after {max_retries} attempts for {batch_label}")
                    raise
            
            except Exception as e:
                print(f"Unexpected error embedding {batch_label}: {type(e).__name__}: {e}")
                raise
        
        # Should never reach here
        raise RuntimeError(f"Failed to embed {batch_label} after {max_retries} attempts")
