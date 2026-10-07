"""FastAPI backend: POST /chat sends a user message to the LLM and returns the reply."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .llm import LLMError
from .service import ChatService, get_service

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="Pip - LLM Chatbot", version="1.0.0")


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    conversation_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    turns: int
    context_messages: int
    has_summary: bool
    trimmed: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Sync endpoints on purpose: FastAPI runs them in a threadpool, so the blocking SDK
# calls don't stall the event loop.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, service: ChatService = Depends(get_service)):
    try:
        result = service.chat(req.message, req.conversation_id)
    except LLMError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)
    except Exception:
        logging.exception("Unexpected error in /chat")
        raise HTTPException(status_code=500, detail="Something went wrong on the server. Please try again.")
    return ChatResponse(**result.__dict__)


@app.get("/conversations/{conversation_id}")
def get_history(conversation_id: str, service: ChatService = Depends(get_service)):
    data = service.history(conversation_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return data


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, service: ChatService = Depends(get_service)):
    if not service.reset(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"deleted": True}


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
