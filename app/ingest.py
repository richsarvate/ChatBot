from __future__ import annotations

import mailbox
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mailparser
import yaml
from email import message_from_string
from tqdm import tqdm

from .chunking import make_chunks
from .config import Settings, get_settings
from .email_parser import ParsedEmail, parse_email_file
from .embedding import Embedder
from .index import EmailIndex


@dataclass
class IngestionStats:
    processed_messages: int
    processed_chunks: int
    filtered_messages: int = 0


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


def _should_filter_email(from_addr: str, subject: str, filters: dict) -> bool:
    """
    Return True if email should be FILTERED OUT (blocked).
    
    Filtering logic order:
    1. Whitelist check (always keep)
    2. Conversation prefix check (always keep Re:/Fwd:)
    3. Blocked senders (exact match)
    4. Blocked sender patterns (regex)
    5. Blocked domains (full block)
    6. Semi-trusted domains + subject keywords
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
    
    # 3. Check blocked senders (exact match)
    blocked_senders = filters.get('blocked_senders') or []
    if from_addr in [s.lower() for s in blocked_senders]:
        return True
    
    # 4. Check blocked sender patterns (regex)
    blocked_patterns = filters.get('blocked_sender_patterns') or []
    for pattern in blocked_patterns:
        if re.search(pattern, from_addr, re.IGNORECASE):
            return True
    
    # 5. Check blocked domains (full block)
    domain = from_addr.split('@')[-1] if '@' in from_addr else ''
    blocked_domains = filters.get('blocked_domains') or []
    if domain in [d.lower() for d in blocked_domains]:
        return True
    
    # 6. Check semi-trusted domains with subject filtering
    semi_trusted = filters.get('semi_trusted_domains') or []
    if domain in [d.lower() for d in semi_trusted]:
        subject_lower = subject.lower()
        keywords = filters.get('transactional_subject_keywords') or []
        for keyword in keywords:
            if keyword.lower() in subject_lower:
                return True
    
    return False


def _parse_mbox_file(path: Path, limit: int | None = None) -> List[ParsedEmail]:
    """Parse an MBOX file and return list of ParsedEmail objects."""
    print(f"Parsing MBOX file: {path}")
    if limit:
        print(f"Will process first {limit} emails")
    mbox = mailbox.mbox(str(path))
    
    emails = []
    for idx, message in enumerate(mbox):
        if limit and idx >= limit:
            print(f"\nReached limit of {limit} emails")
            break
        
        if idx % 10 == 0:
            print(f"Parsing email {idx}...", end="\r")
            
        try:
            parsed = mailparser.parse_from_bytes(bytes(message))
            
            # Extract fields similar to parse_email_file
            message_id = parsed.message_id or f"mbox-{idx}"
            thread_id = parsed.headers.get("Thread-Index") or message_id if parsed.headers else message_id
            subject = parsed.subject or "(no subject)"
            
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


def ingest_emails(*, rebuild: bool = False, limit: int | None = None, settings: Settings | None = None) -> IngestionStats:
    settings = settings or get_settings()
    _ensure_directory(settings.raw_dir)
    _ensure_directory(settings.processed_dir)
    _ensure_directory(settings.index_dir)

    # Load email filters
    filters = _load_email_filters(settings)
    if filters:
        print("Email filtering enabled")
    else:
        print("Email filtering disabled")

    parser_summary: List[ParsedEmail] = []
    filtered_count = 0
    
    for path in sorted(settings.raw_dir.glob("**/*")):
        if path.is_file():
            # Handle MBOX files
            if path.suffix == ".mbox" or "mbox" in path.name.lower():
                parsed_emails = _parse_mbox_file(path, limit=limit)
                
                # Apply filtering to MBOX emails
                if filters:
                    for email in parsed_emails:
                        if _should_filter_email(email.from_address, email.subject, filters):
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
                if filters and _should_filter_email(email.from_address, email.subject, filters):
                    filtered_count += 1
                else:
                    parser_summary.append(email)
                    
                if limit and len(parser_summary) >= limit:
                    break

    print(f"\nFiltering summary:")
    print(f"  Kept:     {len(parser_summary):5d} emails")
    print(f"  Filtered: {filtered_count:5d} emails")
    print(f"  Total:    {len(parser_summary) + filtered_count:5d} emails processed")
    if len(parser_summary) + filtered_count > 0:
        print(f"  Filter rate: {filtered_count/(len(parser_summary) + filtered_count)*100:.1f}%")

    embedder = Embedder(settings)
    index = EmailIndex(settings)
    if rebuild:
        index.reset()

    print(f"\nEmbedding and indexing {len(parser_summary)} emails...")
    print("(Batching chunks to optimize API calls)")
    
    total_chunks = 0
    batch_documents: List[str] = []
    batch_metadatas: List[dict] = []
    batch_ids: List[str] = []
    batch_size_limit = 100  # Process 100 chunks at a time
    
    for email_idx, email in enumerate(parser_summary, 1):
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
    
    print(f"\n✓ Ingestion complete!")
    print(f"  Emails indexed:  {len(parser_summary)}")
    print(f"  Chunks created:  {total_chunks}")
    print(f"  Emails filtered: {filtered_count}")
    
    return IngestionStats(
        processed_messages=len(parser_summary),
        processed_chunks=total_chunks,
        filtered_messages=filtered_count
    )
