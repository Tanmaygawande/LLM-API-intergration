"""ChatService ties together: store + context management + LLM client."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import Settings, get_settings
from .llm import LLMClient, LLMError, build_client
from .memory import ConversationStore, build_system_prompt, compact
from .prompts import SUMMARY_SYSTEM_PROMPT, SYSTEM_PROMPT


@dataclass
class ChatResult:
    conversation_id: str
    reply: str
    turns: int
    context_messages: int
    has_summary: bool
    trimmed: bool


class ChatService:
    def __init__(self, client: LLMClient, settings: Optional[Settings] = None,
                 store: Optional[ConversationStore] = None):
        self.client = client
        self.settings = settings or get_settings()
        self.store = store or ConversationStore()

    # -- summarisation helper used by the context manager ------------------
    def _summarize(self, previous: str, old_messages: list[dict]) -> str:
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in old_messages)
        prompt = (
            f"Existing summary (may be empty):\n{previous or '(none)'}\n\n"
            f"New messages to fold in:\n{transcript}\n\nWrite the updated summary."
        )
        return self.client.generate(
            SUMMARY_SYSTEM_PROMPT, [{"role": "user", "content": prompt}],
            retries=self.settings.max_retries,
        )

    # -- main entry point ---------------------------------------------------
    def chat(self, message: str, conversation_id: Optional[str] = None) -> ChatResult:
        conv = self.store.get_or_create(conversation_id)
        with conv.lock:  # one request per conversation at a time keeps history ordered
            snapshot = (list(conv.messages), conv.summary)
            conv.messages.append({"role": "user", "content": message})
            try:
                trimmed = compact(conv, self.settings, self._summarize)
                system = build_system_prompt(SYSTEM_PROMPT, conv.summary)
                reply = self.client.generate(system, conv.messages, retries=self.settings.max_retries)
            except LLMError:
                # Roll back so a failed call never leaves a dangling user turn.
                conv.messages, conv.summary = snapshot
                raise
            conv.messages.append({"role": "assistant", "content": reply})
            return ChatResult(
                conversation_id=conv.id,
                reply=reply,
                turns=sum(1 for m in conv.messages if m["role"] == "user"),
                context_messages=len(conv.messages),
                has_summary=bool(conv.summary),
                trimmed=trimmed,
            )

    def history(self, conversation_id: str) -> Optional[dict]:
        conv = self.store.get(conversation_id)
        if conv is None:
            return None
        return {"conversation_id": conv.id, "summary": conv.summary, "messages": list(conv.messages)}

    def reset(self, conversation_id: str) -> bool:
        return self.store.delete(conversation_id)


_service: Optional[ChatService] = None


def get_service() -> ChatService:
    """Lazily build the singleton so importing the app never needs an API key."""
    global _service
    if _service is None:
        _service = ChatService(build_client())
    return _service
