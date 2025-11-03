from __future__ import annotations

import json
import mailbox
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import mailparser
import yaml
from email import message_from_string
from tqdm import tqdm

from .chunking import make_chunks
from .config import Settings, get_settings
from .email_parser import ParsedEmail, parse_email_file
from .embedding import Embedder, InsufficientFundsError
from .index import EmailIndex


@dataclass
class IngestionStats:
    processed_messages: int
    processed_chunks: int
    filtered_messages: int = 0


@dataclass
class IngestionCheckpoint:
    """Tracks ingestion progress for resumability."""
    current_file: str
    emails_processed_in_file: int
    total_emails_indexed: int
    total_chunks_created: int
    total_emails_filtered: int
    timestamp: str
    
    def save(self, path: Path) -> None:
        """Save checkpoint to JSON file."""
        data = {
            "current_file": self.current_file,
            "emails_processed_in_file": self.emails_processed_in_file,
            "total_emails_indexed": self.total_emails_indexed,
            "total_chunks_created": self.total_chunks_created,
            "total_emails_filtered": self.total_emails_filtered,
            "timestamp": self.timestamp,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    @classmethod
    def load(cls, path: Path) -> Optional["IngestionCheckpoint"]:
        """Load checkpoint from JSON file."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return None


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _persist_email_json(email: ParsedEmail, processed_dir: Path) -> Path:
    destination = processed_dir / f"{email.message_id}.json"
    destination.write_text(email.to_json(), encoding="utf-8")
    return destination


def _load_email_filters(settings: Settings) -> dict:
    """Load email filtering configuration from YAML file."""
    config_path = Path(settings.config_dir) / "email_filters.yaml"
    if not config_path.exists():
        print(f"Warning: Email filter config not found at {config_path}, filtering disabled")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _normalize_subject(subject: str) -> str:
    """Normalize email subject by removing Re:/Fwd: prefixes and whitespace."""
    normalized = re.sub(r'^(Re|RE|Fwd|FWD|Fw):\s*', '', subject, flags=re.IGNORECASE).strip()
    return normalized.lower()  # Case-insensitive comparison


def _build_thread_counts(files: List[Path]) -> dict[str, int]:
    """
    Build a map of normalized subject -> email count by streaming mbox files.
    
    This is memory efficient - only stores subject strings, not full email content.
    For 200K emails, this uses ~40-50MB of memory.
    
    Args:
        files: List of file paths to scan (mbox files and individual emails)
    
    Returns:
        Dictionary mapping normalized subject to count of emails with that subject
    """
    from collections import Counter
    
    thread_counts = Counter()
    total_scanned = 0
    
    print("\n" + "=" * 100)
    print("PASS 1: Building thread map (scanning subjects only)")
    print("=" * 100)
    
    for file_path in files:
        if not file_path.is_file():
            continue
        
        # Handle MBOX files
        if file_path.suffix == ".mbox" or "mbox" in file_path.name.lower():
            print(f"\nScanning: {file_path.name}")
            
            try:
                mbox = mailbox.mbox(str(file_path))
                for idx, message in enumerate(mbox):
                    if idx % 1000 == 0 and idx > 0:
                        print(f"  Scanned {idx} emails, found {len(thread_counts)} unique subjects...", end="\r")
                    
                    try:
                        # Extract only the subject (minimal parsing)
                        subject = message.get('subject', '(no subject)')
                        normalized = _normalize_subject(subject)
                        thread_counts[normalized] += 1
                        total_scanned += 1
                    except Exception:
                        continue
                
                print(f"  Scanned {idx + 1} emails from {file_path.name}                    ")
            except Exception as e:
                print(f"  Error scanning {file_path.name}: {e}")
                continue
        else:
            # Handle individual email files (rare, but supported)
            try:
                email = parse_email_file(file_path)
                normalized = _normalize_subject(email.subject)
                thread_counts[normalized] += 1
                total_scanned += 1
            except Exception:
                continue
    
    # Count threads (subjects with 2+ emails)
    multi_email_threads = sum(1 for count in thread_counts.values() if count >= 2)
    single_email_threads = sum(1 for count in thread_counts.values() if count == 1)
    
    print(f"\n✓ Thread map built!")
    print(f"  Total emails scanned:     {total_scanned:,}")
    print(f"  Unique subjects:          {len(thread_counts):,}")
    print(f"  Multi-email threads:      {multi_email_threads:,} subjects ({sum(count for count in thread_counts.values() if count >= 2):,} emails)")
    print(f"  Single-email threads:     {single_email_threads:,} subjects ({single_email_threads:,} emails)")
    print(f"  Potential filter savings: {single_email_threads:,} emails ({single_email_threads/total_scanned*100:.1f}%)")
    print("=" * 100)
    
    return dict(thread_counts)


def _should_filter_email(from_addr: str, subject: str, filters: dict, thread_counts: Optional[dict] = None) -> bool:
    """
    Return True if email should be FILTERED OUT (blocked).
    
    Filtering logic order:
    1. Whitelist check (always keep)
    2. Conversation prefix check (always keep Re:/Fwd:)
    3. Thread count check (if conversations_only enabled, filter single-email threads)
    4. Blocked senders (exact match)
    5. Blocked sender patterns (regex)
    6. Blocked domains (full block)
    7. Semi-trusted domains + subject keywords
    """
    if not filters:
        return False
    
    from_addr = from_addr.lower()
    
    # Extract email address from "Name <email@domain.com>" format
    email_match = re.search(r'<([^>]+)>', from_addr)
    if email_match:
        from_addr = email_match.group(1).lower()
    else:
        # Try to find email without brackets
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', from_addr)
        if email_match:
            from_addr = email_match.group(0).lower()
    
    # 1. Check whitelist first (always keep)
    whitelisted = filters.get('whitelisted_senders') or []
    if from_addr in [s.lower() for s in whitelisted]:
        return False
    
    # 2. Check if subject starts with conversation prefix (always keep)
    prefixes = filters.get('preserve_conversation_prefixes') or []
    for prefix in prefixes:
        if subject.strip().startswith(prefix):
            return False
    
    # 3. Check thread count (if conversations_only mode enabled)
    conversations_only = filters.get('conversations_only', False)
    if conversations_only and thread_counts is not None:
        normalized = _normalize_subject(subject)
        count = thread_counts.get(normalized, 0)
        if count < 2:
            # Single-email thread, filter it out
            return True
    
    # 4. Check blocked senders (exact match)
    blocked_senders = filters.get('blocked_senders') or []
    if from_addr in [s.lower() for s in blocked_senders]:
        return True
    
    # 5. Check blocked sender patterns (regex)
    blocked_patterns = filters.get('blocked_sender_patterns') or []
    for pattern in blocked_patterns:
        if re.search(pattern, from_addr, re.IGNORECASE):
            return True
    
    # 6. Check blocked domains (full block)
    domain = from_addr.split('@')[-1] if '@' in from_addr else ''
    blocked_domains = filters.get('blocked_domains') or []
    if domain in [d.lower() for d in blocked_domains]:
        return True
    
    # 7. Check semi-trusted domains with subject filtering
    semi_trusted = filters.get('semi_trusted_domains') or []
    if domain in [d.lower() for d in semi_trusted]:
        subject_lower = subject.lower()
        keywords = filters.get('transactional_subject_keywords') or []
        for keyword in keywords:
            if keyword.lower() in subject_lower:
                return True
    
    return False


def _parse_mbox_file(path: Path, limit: int | None = None, skip: int = 0) -> List[ParsedEmail]:
    """Parse an MBOX file and return list of ParsedEmail objects.
    
    Args:
        path: Path to MBOX file
        limit: Maximum number of emails to parse (None = all)
        skip: Number of emails to skip from start (for resuming)
    """
    print(f"Parsing MBOX file: {path}")
    if skip > 0:
        print(f"Skipping first {skip} emails (resuming from checkpoint)")
    if limit:
        print(f"Will process up to {limit} emails")
    mbox = mailbox.mbox(str(path))
    
    emails = []
    for idx, message in enumerate(mbox):
        # Skip emails we've already processed
        if idx < skip:
            if idx % 100 == 0 and idx > 0:
                print(f"Skipping to checkpoint... {idx}/{skip}", end="\r")
            continue
        
        if limit and idx >= (skip + limit):
            print(f"\nReached limit of {limit} emails")
            break
        
        if idx % 10 == 0:
            print(f"Parsing email {idx}...", end="\r")
            
        try:
            parsed = mailparser.parse_from_bytes(bytes(message))
            
            # Extract fields similar to parse_email_file
            message_id = parsed.message_id or f"mbox-{idx}"
            subject = parsed.subject or "(no subject)"
            
            # Use normalized subject as thread_id for better grouping
            # Remove Re:, Fwd:, FWD:, etc. and strip whitespace
            normalized_subject = re.sub(r'^(Re|RE|Fwd|FWD|Fw):\s*', '', subject, flags=re.IGNORECASE).strip()
            thread_id = normalized_subject or message_id
            
            from_address = "unknown"
            if parsed.from_:
                for _, addr in parsed.from_:
                    if addr:
                        from_address = addr
                        break
            
            to = [addr for _, addr in parsed.to] if parsed.to else []
            to = [addr for addr in to if addr]
            cc = [addr for _, addr in parsed.cc] if parsed.cc else []
            cc = [addr for addr in cc if addr]
            
            import pendulum
            date_str = parsed.date if isinstance(parsed.date, str) else parsed.date.isoformat() if parsed.date else None
            if not date_str:
                date_str = pendulum.now("UTC").to_iso8601_string()
            else:
                date_str = pendulum.parse(str(date_str)).to_iso8601_string()
            
            body_text = parsed.text_plain[0] if parsed.text_plain else parsed.body
            body_text = body_text or ""
            
            labels = parsed.headers.get("X-Gmail-Labels", "").split(",") if parsed.headers else []
            labels = [label.strip() for label in labels if label.strip()]
            
            attachments = []
            
            email = ParsedEmail(
                message_id=message_id,
                thread_id=thread_id,
                subject=subject,
                from_address=from_address,
                to=to,
                cc=cc,
                date=date_str,
                body_text=body_text,
                labels=labels,
                attachments=attachments,
                raw_path=path,
            )
            emails.append(email)
            
        except Exception as e:
            print(f"Error parsing email {idx}: {e}")
            continue
    
    print(f"Successfully parsed {len(emails)} emails from MBOX")
    return emails


def ingest_emails(*, rebuild: bool = False, resume: bool = False, limit: int | None = None, settings: Settings | None = None) -> IngestionStats:
    settings = settings or get_settings()
    _ensure_directory(settings.raw_dir)
    _ensure_directory(settings.processed_dir)
    _ensure_directory(settings.index_dir)

    # Checkpoint file location
    checkpoint_path = settings.data_dir / ".ingestion_checkpoint.json"
    
    # Load checkpoint if resuming
    checkpoint: Optional[IngestionCheckpoint] = None
    if resume:
        checkpoint = IngestionCheckpoint.load(checkpoint_path)
        if checkpoint:
            print(f"📍 Resuming from checkpoint:")
            print(f"   File: {checkpoint.current_file}")
            print(f"   Emails processed in file: {checkpoint.emails_processed_in_file}")
            print(f"   Total emails indexed: {checkpoint.total_emails_indexed}")
            print(f"   Total chunks created: {checkpoint.total_chunks_created}")
            print(f"   Last checkpoint: {checkpoint.timestamp}")
        else:
            print("No checkpoint found, starting from beginning")

    # Load email filters
    filters = _load_email_filters(settings)
    if filters:
        print("Email filtering enabled")
    else:
        print("Email filtering disabled")

    # Find files to process
    all_files = sorted(settings.raw_dir.glob("**/*"))
    
    # Build thread counts if conversations_only mode is enabled
    thread_counts: Optional[dict] = None
    conversations_only = filters.get('conversations_only', False) if filters else False
    
    if conversations_only:
        print("\n🔍 Conversations-only mode ENABLED")
        print("   Only emails that are part of multi-email threads will be indexed")
        thread_counts = _build_thread_counts(all_files)
    else:
        print("\n🔍 Conversations-only mode DISABLED")
        print("   All non-spam emails will be indexed (including single emails)")

    parser_summary: List[ParsedEmail] = []
    filtered_count = checkpoint.total_emails_filtered if checkpoint else 0
    skip_count = checkpoint.emails_processed_in_file if checkpoint else 0
    
    # Keep track of cumulative stats
    total_indexed = checkpoint.total_emails_indexed if checkpoint else 0
    total_chunks = checkpoint.total_chunks_created if checkpoint else 0
    start_processing = not resume  # If not resuming, start immediately
    
    for path in all_files:
        if path.is_file():
            # If resuming, skip until we find the checkpoint file
            if resume and checkpoint and not start_processing:
                if str(path) != checkpoint.current_file:
                    continue
                else:
                    start_processing = True
                    print(f"\n🔄 Resuming file: {path.name}")
            
            # Handle MBOX files
            if path.suffix == ".mbox" or "mbox" in path.name.lower():
                parsed_emails = _parse_mbox_file(path, limit=limit, skip=skip_count if start_processing and checkpoint else 0)
                
                # Reset skip count after first file
                skip_count = 0
                
                # Apply filtering to MBOX emails
                if filters:
                    for email in parsed_emails:
                        if _should_filter_email(email.from_address, email.subject, filters, thread_counts):
                            filtered_count += 1
                        else:
                            parser_summary.append(email)
                            if limit and len(parser_summary) >= limit:
                                break
                else:
                    parser_summary.extend(parsed_emails)
                    
                if limit and len(parser_summary) >= limit:
                    break
            else:
                # Handle individual email files
                email = parse_email_file(path)
                
                # Apply filtering to individual emails
                if filters and _should_filter_email(email.from_address, email.subject, filters, thread_counts):
                    filtered_count += 1
                else:
                    parser_summary.append(email)
                    
                if limit and len(parser_summary) >= limit:
                    break

    print(f"\nFiltering summary:")
    print(f"  Kept:     {len(parser_summary):5d} emails")
    print(f"  Filtered: {filtered_count - (checkpoint.total_emails_filtered if checkpoint else 0):5d} emails (this session)")
    if checkpoint:
        print(f"  Previously filtered: {checkpoint.total_emails_filtered:5d} emails")
        print(f"  Total filtered: {filtered_count:5d} emails")
    print(f"  Total:    {len(parser_summary) + filtered_count - (checkpoint.total_emails_indexed if checkpoint else 0):5d} emails processed (this session)")
    if len(parser_summary) + filtered_count > 0:
        session_total = len(parser_summary) + filtered_count - (checkpoint.total_emails_indexed if checkpoint else 0) - (checkpoint.total_emails_filtered if checkpoint else 0)
        if session_total > 0:
            print(f"  Filter rate: {(filtered_count - (checkpoint.total_emails_filtered if checkpoint else 0))/session_total*100:.1f}%")

    embedder = Embedder(settings)
    index = EmailIndex(settings)
    if rebuild:
        index.reset()

    print(f"\nEmbedding and indexing {len(parser_summary)} emails...")
    print("(Batching chunks to optimize API calls)")
    
    batch_documents: List[str] = []
    batch_metadatas: List[dict] = []
    batch_ids: List[str] = []
    batch_size_limit = 100  # Process 100 chunks at a time
    
    emails_processed = 0
    current_file_path = str(all_files[0]) if all_files else ""
    
    try:
        for email_idx, email in enumerate(parser_summary, 1):
            current_file_path = str(email.raw_path)
            _persist_email_json(email, settings.processed_dir)
            chunks = make_chunks(
                email.body_text,
                chunk_size=settings.chunk_size_tokens * 4,
                chunk_overlap=settings.chunk_overlap_tokens * 4,
            )
            if not chunks:
                continue
            
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{email.message_id}-{idx:04d}"
                batch_documents.append(chunk)
                batch_metadatas.append(
                    {
                        "message_id": email.message_id,
                        "thread_id": email.thread_id,
                        "subject": email.subject,
                        "from_address": email.from_address,
                        "to": ", ".join(email.to),
                        "date": email.date,
                        "chunk_index": idx,
                        "raw_path": str(email.raw_path),
                        "token_estimate": len(chunk.split()),
                    }
                )
                batch_ids.append(chunk_id)
            
            total_chunks += len(chunks)
            
            # Process batch when it reaches size limit or at the end
            should_process_batch = (
                len(batch_documents) >= batch_size_limit or 
                email_idx == len(parser_summary)
            )
            
            if should_process_batch and batch_documents:
                print(f"  Processing email {email_idx}/{len(parser_summary)}: embedding {len(batch_documents)} chunks (total: {total_chunks} chunks)")
                
                embeddings = embedder.embed(batch_documents)
                index.add_chunks(
                    documents=batch_documents, 
                    embeddings=embeddings, 
                    metadatas=batch_metadatas, 
                    ids=batch_ids
                )
                
                # Clear batch
                batch_documents = []
                batch_metadatas = []
                batch_ids = []
                
                # Update counters
                emails_processed = email_idx
                total_indexed += 1
                
                # Save checkpoint after each successful batch
                checkpoint = IngestionCheckpoint(
                    current_file=current_file_path,
                    emails_processed_in_file=emails_processed,
                    total_emails_indexed=total_indexed,
                    total_chunks_created=total_chunks,
                    total_emails_filtered=filtered_count,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                )
                checkpoint.save(checkpoint_path)
    
    except InsufficientFundsError as e:
        # Save checkpoint before exiting
        checkpoint = IngestionCheckpoint(
            current_file=current_file_path,
            emails_processed_in_file=emails_processed,
            total_emails_indexed=total_indexed,
            total_chunks_created=total_chunks,
            total_emails_filtered=filtered_count,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        checkpoint.save(checkpoint_path)
        
        print(f"\n" + "=" * 100)
        print("⚠️  OPENAI API CREDITS EXHAUSTED")
        print("=" * 100)
        print(f"\n📊 Progress saved:")
        print(f"   Emails indexed: {total_indexed}")
        print(f"   Chunks created: {total_chunks}")
        print(f"   Checkpoint saved to: {checkpoint_path}")
        print(f"\n💡 Next steps:")
        print(f"   1. Add funds to your OpenAI account")
        print(f"   2. Resume ingestion with: python manage.py ingest --resume")
        print("\n" + "=" * 100)
        return IngestionStats(
            processed_messages=total_indexed,
            processed_chunks=total_chunks,
            filtered_messages=filtered_count
        )
    
    print(f"\n✓ Ingestion complete!")
    print(f"  Emails indexed:  {len(parser_summary)}")
    print(f"  Chunks created:  {total_chunks}")
    print(f"  Emails filtered: {filtered_count}")
    
    # Delete checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Checkpoint cleared")
    
    return IngestionStats(
        processed_messages=len(parser_summary),
        processed_chunks=total_chunks,
        filtered_messages=filtered_count
    )
