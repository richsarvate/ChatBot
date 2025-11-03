from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import mailparser
import pendulum


@dataclass
class AttachmentMeta:
    filename: Optional[str]
    content_type: Optional[str]
    size: Optional[int]


@dataclass
class ParsedEmail:
    message_id: str
    thread_id: str
    subject: str
    from_address: str
    to: List[str]
    cc: List[str]
    date: str
    body_text: str
    labels: List[str]
    attachments: List[AttachmentMeta]
    raw_path: Path

    def to_json(self) -> str:
        payload = asdict(self)
        payload["raw_path"] = str(self.raw_path)
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _safe_message_id(candidate: Optional[str]) -> str:
    if candidate:
        return candidate.strip()
    return f"generated-{uuid.uuid4()}"


def parse_email_file(path: Path) -> ParsedEmail:
    parsed = mailparser.parse_from_file(str(path))
    message_id = _safe_message_id(parsed.message_id or path.stem)
    thread_id = parsed.headers.get("Thread-Index") or message_id
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
    date_str = parsed.date if isinstance(parsed.date, str) else parsed.date.isoformat() if parsed.date else None
    if not date_str:
        date_str = pendulum.now("UTC").to_iso8601_string()
    else:
        date_str = pendulum.parse(str(date_str)).to_iso8601_string()
    body_text = parsed.text_plain[0] if parsed.text_plain else parsed.body
    body_text = body_text or ""
    if not body_text.strip():
        body_text = path.read_text(encoding="utf-8")
    labels = parsed.headers.get("X-Gmail-Labels", "").split(",") if parsed.headers else []
    labels = [label.strip() for label in labels if label.strip()]
    attachments = [
        AttachmentMeta(
            filename=att.get("filename"),
            content_type=att.get("mail_content_type"),
            size=att.get("size")
        )
        for att in parsed.attachments
    ]
    return ParsedEmail(
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
