#!/usr/bin/env python3
"""
Extract a chunk of emails from the full MBOX archive.
Usage: python extract_chunk.py <start> <end> <output_file>
Example: python extract_chunk.py 1000 3000 chunk_002.mbox
"""

import sys
from pathlib import Path

def stream_mbox(path):
    """Stream MBOX file without building full index."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        msg_lines = []
        for line in f:
            if line.startswith('From '):
                if msg_lines:
                    yield ''.join(msg_lines)
                    msg_lines = []
            msg_lines.append(line)
        if msg_lines:
            yield ''.join(msg_lines)

def extract_chunk(source_path, start_idx, end_idx, output_path):
    """Extract emails from start_idx to end_idx (exclusive)."""
    print(f"Extracting emails {start_idx} to {end_idx-1} from {source_path}")
    print(f"Output: {output_path}")
    
    extracted = 0
    skipped = 0
    
    with open(output_path, 'w', encoding='utf-8', errors='ignore') as out:
        for idx, raw_msg in enumerate(stream_mbox(source_path)):
            if idx < start_idx:
                skipped += 1
                if skipped % 100 == 0:
                    print(f"Skipping... {skipped} emails processed", end='\r')
                continue
            
            if idx >= end_idx:
                break
            
            out.write(raw_msg)
            if not raw_msg.endswith('\n'):
                out.write('\n')
            
            extracted += 1
            if extracted % 100 == 0:
                print(f"Extracted: {extracted} emails (total processed: {idx+1})", end='\r')
    
    print(f"\n✓ Extracted {extracted} emails to {output_path}")
    print(f"  Total emails processed: {skipped + extracted}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python extract_chunk.py <start> <end> <output_file>")
        print("Example: python extract_chunk.py 1000 3000 chunk_002.mbox")
        sys.exit(1)
    
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    output = sys.argv[3]
    
    # Find the source mbox
    source = Path('/home/ubuntu/GitHubProjects/Assistant/data/All mail Including Spam and Trash.mbox')
    
    if not source.exists():
        print(f"Error: Source mbox not found at {source}")
        sys.exit(1)
    
    extract_chunk(str(source), start, end, output)
