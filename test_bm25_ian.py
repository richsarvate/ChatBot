#!/usr/bin/env python3
"""Test BM25 search for Ian Ford to see what chunks it returns."""

import pickle
from pathlib import Path

# Load BM25 index
bm25_path = Path("/home/ubuntu/GitHubProjects/Assistant/data/index/chroma/bm25_index.pkl")
with open(bm25_path, "rb") as f:
    bm25_data = pickle.load(f)

bm25_index = bm25_data["index"]
chunk_ids = bm25_data["chunk_ids"]
metadatas = bm25_data["metadatas"]
corpus = bm25_data["corpus"]

# Test query
queries = [
    "Ian Ford",
    "Ian Ford email",
    "what is Ian's email",
    "Bourbon Room Ian"
]

for query in queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Get top 10 results
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    
    for rank, idx in enumerate(top_indices, 1):
        score = bm25_scores[idx]
        chunk_id = chunk_ids[idx]
        metadata = metadatas[idx]
        text_preview = corpus[idx][:200]
        
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   Subject: {metadata.get('subject', 'N/A')}")
        print(f"   From: {metadata.get('from_address', 'N/A')}")
        print(f"   Date: {metadata.get('date', 'N/A')}")
        print(f"   Text: {text_preview}...")
