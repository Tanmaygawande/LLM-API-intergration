"""Per-conversation history + context window management.

Two strategies (set CONTEXT_STRATEGY):
  * truncate  - keep only the last MAX_HISTORY_MESSAGES messages.
  * summarize - when history grows past the limit, compress the older part into a
                running summary that is prepended to the system prompt, and keep the
                last KEEP_RECENT_MESSAGES verbatim. If summarisation fails, we fall
                back to truncation so the chat never breaks.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import Settings


@dataclass
class Conversation:
    id: str
    messages: list[dict] = field(default_factory=list)  # {"role", "content"}
    summary: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def total_chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages) + len(self.summary)


class ConversationStore:
    """In-memory store keyed by conversation id (swap for Redis/DB in production)."""

    def __init__(self) -> None:
        self._items: dict[str, Conversation] = {}
        self._guard = threading.Lock()

    def get(self, conversation_id: str) -> Optional[Conversation]:
        with self._guard:
            return self._items.get(conversation_id)

    def get_or_create(self, conversation_id: Optional[str] = None) -> Conversation:
        with self._guard:
            cid = conversation_id or uuid.uuid4().hex
            conv = self._items.get(cid)
            if conv is None:
                conv = Conversation(id=cid)
                self._items[cid] = conv
            return conv

    def delete(self, conversation_id: str) -> bool:
        with self._guard:
            return self._items.pop(conversation_id, None) is not None


Summarizer = Callable[[str, list[dict]], str]


def needs_compaction(conv: Conversation, s: Settings) -> bool:
    return len(conv.messages) > s.max_history_messages or conv.total_chars() > s.max_context_chars


def _drop_leading_assistant(messages: list[dict]) -> list[dict]:
    """Providers expect the history to start with a user turn."""
    i = 0
    while i < len(messages) and messages[i]["role"] != "user":
        i += 1
    return messages[i:]


def compact(conv: Conversation, s: Settings, summarize: Optional[Summarizer] = None) -> bool:
    """Shrink conv.messages in place if needed. Returns True if anything was trimmed."""
    if not needs_compaction(conv, s):
        return False

    if s.strategy == "summarize" and summarize is not None:
        keep = min(s.keep_recent_messages, s.max_history_messages)
        if len(conv.messages) <= keep:
            return False
        old, recent = conv.messages[:-keep], conv.messages[-keep:]
        try:
            conv.summary = summarize(conv.summary, old)
        except Exception:
            # Summarisation failed -> still trim so we stay inside the window.
            pass
        conv.messages = _drop_leading_assistant(recent)
        return True

    # truncate strategy (also the fallback)
    keep = s.max_history_messages
    if len(conv.messages) <= keep:
        # Few but very long messages: trim to the recent half so we do shrink.
        keep = max(2, len(conv.messages) // 2)
    conv.messages = _drop_leading_assistant(conv.messages[-keep:])
    return True


def build_system_prompt(base: str, summary: str) -> str:
    if not summary:
        return base
    return f"{base}\n\nSummary of the earlier part of this conversation (for your memory):\n{summary}"
