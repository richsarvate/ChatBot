from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .chat import ChatService
from .config import Settings, get_settings
from .retrieval import RetrievedChunk, Retriever


class QueryPayload(BaseModel):
    question: str


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
        hits = retriever.search(question)
        response = chat_service.answer(question, hits)
        return QueryResponse(**response)

    return app


app = create_app()
