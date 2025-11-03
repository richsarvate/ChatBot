from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .chat import ChatService
from .config import Settings, get_settings
from .retrieval import RetrievedChunk, Retriever

# In-memory conversation storage: {session_id: [messages]}
conversation_store: Dict[str, List[Dict[str, str]]] = defaultdict(list)


class QueryPayload(BaseModel):
    question: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


def create_app() -> FastAPI:
    settings = get_settings()
    templates = Jinja2Templates(directory=str(settings.project_root / "templates"))
    retriever = Retriever(settings)
    chat = ChatService(settings)

    app = FastAPI(title="Email QA")

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    def get_retriever() -> Retriever:
        return retriever

    def get_chat() -> ChatService:
        return chat

    @app.post("/api/query", response_model=QueryResponse)
    async def run_query(
        payload: QueryPayload,
        retriever: Retriever = Depends(get_retriever),
        chat_service: ChatService = Depends(get_chat),
    ) -> QueryResponse:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
        # Get conversation history for this session
        conversation_history = []
        if payload.session_id:
            conversation_history = conversation_store[payload.session_id]
        
        # Pass conversation history to search for better query expansion
        hits = retriever.search(question, conversation_history=conversation_history)
        response = chat_service.answer(question, hits, conversation_history=conversation_history)
        
        # Save this exchange to conversation history
        if payload.session_id:
            conversation_store[payload.session_id].append({"role": "user", "content": question})
            conversation_store[payload.session_id].append({"role": "assistant", "content": response["answer"]})
            # Keep only last 20 messages (10 exchanges)
            if len(conversation_store[payload.session_id]) > 20:
                conversation_store[payload.session_id] = conversation_store[payload.session_id][-20:]
        
        return QueryResponse(**response)

    @app.post("/api/clear_conversation")
    async def clear_conversation(payload: dict) -> dict:
        session_id = payload.get("session_id")
        if session_id and session_id in conversation_store:
            del conversation_store[session_id]
        return {"status": "cleared"}

    return app


app = create_app()
