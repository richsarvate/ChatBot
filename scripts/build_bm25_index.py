#!/usr/bin/env python3
"""
Build BM25 index from existing ChromaDB without re-embedding.
This is much faster than full ingestion (minutes vs hours).
"""

import pickle
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi
from app.config import get_settings
from app.index import EmailIndex


def build_bm25_from_chromadb():
    """Build BM25 index by reading all chunks from ChromaDB in batches."""
    print("=" * 80)
    print("Building BM25 Index from Existing ChromaDB")
    print("=" * 80)
    
    settings = get_settings()
    index = EmailIndex(settings)
    
    print("\n📚 Reading chunks from ChromaDB in batches (memory-efficient)...")
    
    # Process in smaller batches to avoid memory exhaustion
    batch_size = 10000
    offset = 0
    
    all_chunk_ids = []
    all_documents = []
    all_metadatas = []
    
    try:
        while True:
            print(f"  Batch starting at offset {offset}...")
            
            # Get batch from ChromaDB
            batch_data = index._collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset
            )
            
            batch_ids = batch_data["ids"]
            if not batch_ids:
                break  # No more data
            
            all_chunk_ids.extend(batch_ids)
            all_documents.extend(batch_data["documents"])
            all_metadatas.extend(batch_data["metadatas"])
            
            print(f"    ✓ Got {len(batch_ids)} chunks (total: {len(all_chunk_ids)})")
            
            offset += batch_size
            
            # Safety limit to prevent infinite loop
            if offset > 300000:
                print("    ⚠️  Reached safety limit, stopping")
                break
        
        print(f"\n  ✓ Total chunks collected: {len(all_chunk_ids)}")
        
        if not all_chunk_ids:
            print("\n❌ No chunks found in ChromaDB. Run ingestion first.")
            return False
        
    except Exception as e:
        print(f"\n❌ Error reading from ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🔨 Building BM25 index...")
    print("  (Tokenizing corpus in batches to save memory...)")
    
    # Tokenize in batches and build index incrementally
    # Note: BM25Okapi still needs full corpus, but we tokenize in chunks
    tokenized_corpus = []
    for i in range(0, len(all_documents), 10000):
        batch = all_documents[i:i+10000]
        tokenized_batch = [doc.lower().split() for doc in batch]
        tokenized_corpus.extend(tokenized_batch)
        print(f"    Tokenized {min(i+10000, len(all_documents))}/{len(all_documents)} chunks")
    
    print("  (Creating BM25 index...)")
    bm25_index = BM25Okapi(tokenized_corpus)
    
    print("  ✓ BM25 index built")
    
    print("\n💾 Saving BM25 index...")
    
    # Package data
    bm25_data = {
        "index": bm25_index,
        "chunk_ids": all_chunk_ids,
        "metadatas": all_metadatas,
        "corpus": all_documents,
    }
    
    # Save to disk
    bm25_path = settings.index_dir / "bm25_index.pkl"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
    
    print(f"  ✓ Saved to: {bm25_path}")
    print(f"  📊 Index size: {bm25_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\n" + "=" * 80)
    print("✅ BM25 Index Built Successfully!")
    print("=" * 80)
    print("\nYou can now:")
    print("  1. Restart the server: ./scripts/restart.sh")
    print("  2. Test hybrid search with queries like:")
    print("     - 'who is Janesh Rahlan'")
    print("     - 'what cities does the setup operate in'")
    print()
    
    return True


if __name__ == "__main__":
    success = build_bm25_from_chromadb()
    sys.exit(0 if success else 1)
