from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from openai import OpenAI

from .config import Settings
from .retrieval import RetrievedChunk


SYSTEM_PROMPT = """You are an email research assistant helping the user understand their email history. Answer questions using the supplied email excerpts and the conversation context.

**Guidelines:**
- Use the conversation history to understand references like "that call", "the first one", "they", etc.
- When asked to make inferences or guesses, analyze the available email data and provide reasonable conclusions
- If asked about frequency or patterns, examine the dates in the citations and provide analysis
- When in doubt make an educated guess. Being wrong is fine.
- If information is truly not available, explain what you searched for and suggest related information you do have
- Be conversational and helpful, not overly rigid

**Format responses with Markdown:**
- Use bullet points for lists
- Use **bold** for key information
- Use headers to organize longer answers
- Keep responses clear and concise"""


class ChatService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._logs_path = settings.logs_dir / "interactions.jsonl"
        self._logs_path.parent.mkdir(parents=True, exist_ok=True)

    def answer(self, question: str, context_chunks: List[RetrievedChunk], conversation_history: List[dict] = None) -> dict:
        context_blocks = self._format_context(context_chunks)
        if not context_blocks:
            return {"answer": "No relevant emails were found.", "citations": []}
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add recent conversation history (last 10 messages = 5 exchanges)
        if conversation_history:
            recent = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            messages.extend(recent)
        
        # Add current question with email context
        messages.append({
            "role": "user",
            "content": self._build_user_prompt(question, context_blocks),
        })
        
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
        instructions = """Answer the question using the email context below. Make reasonable inferences from the available information.

For questions like "who is X?":
- If emails mention X in a clear role/context, describe what you can infer
- Don't hedge with "I cannot find specific information" if you have relevant details
- Be direct and confident when the context clearly indicates who someone is

Only say you cannot find information if the context is truly unrelated or empty."""
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
