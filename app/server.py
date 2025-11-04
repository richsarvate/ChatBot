from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from .chat import ChatService
from .config import Settings, get_settings
from .retrieval import RetrievedChunk, Retriever

# In-memory conversation storage: {session_id: [messages]}
conversation_store: Dict[str, List[Dict[str, str]]] = defaultdict(list)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response


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

    app = FastAPI(
        title="Email QA",
        docs_url=None,  # Disable docs in production for security
        redoc_url=None,  # Disable redoc in production for security
    )
    
    # Add security middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add CORS middleware with strict settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://localhost:8000", "https://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/favicon.svg")
    async def favicon() -> FileResponse:
        favicon_path = settings.project_root / "templates" / "favicon.svg"
        return FileResponse(favicon_path, media_type="image/svg+xml")

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
