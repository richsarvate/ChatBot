from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from openai import OpenAI

from .config import Settings
from .retrieval import RetrievedChunk


SYSTEM_PROMPT = """You are an email research assistant. Answer using only the supplied email excerpts. Cite each supporting email by the label provided (e.g., [1]) and refrain from fabricating details.

Format your responses using Markdown for better readability:
- Use bullet points (- or *) for lists
- Use **bold** for emphasis on key information
- Use section headers (## or **Section:**) to organize longer answers
- Add line breaks between distinct sections
- Place citations inline or at the end of each section
- When listing items, present them as a proper bulleted list, not as a paragraph with dashes"""


class ChatService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._logs_path = settings.logs_dir / "interactions.jsonl"
        self._logs_path.parent.mkdir(parents=True, exist_ok=True)

    def answer(self, question: str, context_chunks: List[RetrievedChunk]) -> dict:
        context_blocks = self._format_context(context_chunks)
        if not context_blocks:
            return {"answer": "No relevant emails were found.", "citations": []}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_user_prompt(question, context_blocks),
            },
        ]
        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        answer_text = response.choices[0].message.content.strip()
        payload = {
            "answer": answer_text,
            "citations": self._build_citations(context_chunks),
        }
        self._log_interaction(question=question, payload=payload, context=context_blocks)
        return payload

    def _format_context(self, chunks: Iterable[RetrievedChunk]) -> List[str]:
        blocks: List[str] = []
        for idx, item in enumerate(chunks, start=1):
            chunk = item.chunk
            header = f"[{idx}] Subject: {chunk.subject} | Date: {chunk.date} | From: {chunk.from_address}"
            block = f"{header}\nSnippet:\n{chunk.snippet.strip()}"
            blocks.append(block)
        return blocks

    def _build_user_prompt(self, question: str, context_blocks: List[str]) -> str:
        context_text = "\n\n".join(context_blocks)
        instructions = "Use only the context provided below. If the answer is not contained, reply that you cannot find it."
        return f"{instructions}\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    def _build_citations(self, chunks: List[RetrievedChunk]) -> List[dict]:
        citations: List[dict] = []
        for idx, item in enumerate(chunks, start=1):
            chunk = item.chunk
            citations.append(
                {
                    "label": f"[{idx}] {chunk.subject}",
                    "date": chunk.date,
                    "from": chunk.from_address,
                    "message_id": chunk.message_id,
                    "score": round(item.score, 3),
                }
            )
        return citations

    def _log_interaction(self, *, question: str, payload: dict, context: List[str]) -> None:
        record = {
            "question": question,
            "response": payload,
            "context": context,
        }
        with self._logs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
