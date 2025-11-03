#!/usr/bin/env python3
"""
Analyze ingested emails to detect new spam patterns.

This script analyzes the emails that were KEPT (not filtered) during ingestion
to identify high-frequency senders that might be spam but weren't caught by filters.

Usage:
    python scripts/analyze_spam_patterns.py [--chunk-file chunk_002.mbox] [--threshold 10]
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings


def analyze_kept_emails(data_dir: Path, threshold: int = 10) -> None:
    """Analyze kept emails to find potential spam patterns."""
    
    # Read processed emails
    processed_dir = data_dir / "processed"
    if not processed_dir.exists():
        print(f"Error: Processed directory not found at {processed_dir}")
        return
    
    import json
    
    sender_counts = Counter()
    sender_domains = Counter()
    sender_subjects = defaultdict(list)
    
    print("=" * 100)
    print("SPAM PATTERN DETECTION - Analyzing Kept Emails")
    print("=" * 100)
    print(f"\nScanning {processed_dir}...")
    
    json_files = list(processed_dir.glob("*.json"))
    print(f"Found {len(json_files)} processed emails")
    
    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            from_addr = data.get("from_address", "").lower()
            subject = data.get("subject", "")
            
            # Extract clean email
            email_match = re.search(r'[\w\.-]+@[\w\.-]+', from_addr)
            if email_match:
                clean_email = email_match.group(0)
                sender_counts[clean_email] += 1
                sender_subjects[clean_email].append(subject)
                
                # Extract domain
                domain = clean_email.split('@')[-1]
                sender_domains[domain] += 1
        except Exception as e:
            continue
    
    # Find high-frequency senders
    high_freq_senders = [(sender, count) for sender, count in sender_counts.most_common() if count >= threshold]
    
    if not high_freq_senders:
        print(f"\n✅ No high-frequency senders found (threshold: {threshold} emails)")
        print("   All kept emails appear to be legitimate!")
        return
    
    print(f"\n⚠️  Found {len(high_freq_senders)} high-frequency senders (>= {threshold} emails)")
    print("\n" + "=" * 100)
    print("HIGH-FREQUENCY SENDERS (Potential Spam)")
    print("=" * 100)
    print(f"{'Sender':<50s} {'Count':>8s}  Sample Subjects")
    print("-" * 100)
    
    # Analyze each high-frequency sender
    potential_spam = []
    for sender, count in high_freq_senders:
        subjects = sender_subjects[sender]
        
        # Sample up to 3 subjects
        sample_subjects = subjects[:3]
        
        # Check for spam indicators
        is_automated = any(pattern in sender for pattern in ['noreply', 'no-reply', 'notification', 'automated'])
        has_transactional_subjects = any(
            keyword in " ".join(subjects).lower() 
            for keyword in ['receipt', 'invoice', 'payment', 'order', 'confirmation', 'ticket']
        )
        
        # Determine if likely spam
        likely_spam = is_automated or (has_transactional_subjects and count > threshold * 2)
        
        if likely_spam:
            potential_spam.append((sender, count, is_automated, has_transactional_subjects))
        
        # Print with indicators
        indicator = "🚨" if likely_spam else "⚠️ "
        print(f"{indicator} {sender:<47s} {count:>8d}  {sample_subjects[0][:50]}")
        for i in range(1, min(3, len(sample_subjects))):
            print(f"   {'':<47s} {'':>8s}  {sample_subjects[i][:50]}")
    
    # Print recommendations
    if potential_spam:
        print("\n" + "=" * 100)
        print("RECOMMENDATIONS - Add to config/email_filters.yaml")
        print("=" * 100)
        print("\nblocked_senders:")
        for sender, count, is_automated, has_transactional in potential_spam:
            reason = "automated sender" if is_automated else "high-frequency transactional"
            print(f"  - {sender}  # {count} emails, {reason}")
    
    # Domain analysis
    print("\n" + "=" * 100)
    print("TOP DOMAINS IN KEPT EMAILS")
    print("=" * 100)
    print(f"{'Domain':<50s} {'Count':>8s}")
    print("-" * 100)
    for domain, count in sender_domains.most_common(20):
        print(f"{domain:<50s} {count:>8d}")
    
    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print("1. Review the high-frequency senders above")
    print("2. Add confirmed spam to config/email_filters.yaml")
    print("3. Re-run ingestion with --rebuild to apply new filters")
    print("4. Or continue ingesting new chunks - filters apply automatically")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Analyze kept emails for spam patterns")
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Minimum email count to flag as high-frequency (default: 10)"
    )
    
    args = parser.parse_args()
    
    settings = get_settings()
    analyze_kept_emails(settings.data_dir, threshold=args.threshold)


if __name__ == "__main__":
    main()
