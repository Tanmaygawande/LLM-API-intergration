"""Central configuration. Everything comes from environment variables
(optionally loaded from a local .env file) so no secret is ever in code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:  # .env support is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    provider: str
    gemini_api_key: str
    gemini_model: str
    openai_api_key: str
    openai_model: str
    strategy: str
    max_history_messages: int
    keep_recent_messages: int
    max_context_chars: int
    timeout_seconds: int
    max_retries: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        strategy=os.getenv("CONTEXT_STRATEGY", "summarize").strip().lower(),
        max_history_messages=max(2, _int("MAX_HISTORY_MESSAGES", 12)),
        keep_recent_messages=max(1, _int("KEEP_RECENT_MESSAGES", 6)),
        max_context_chars=max(500, _int("MAX_CONTEXT_CHARS", 12000)),
        timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 30),
        max_retries=_int("LLM_MAX_RETRIES", 2),
    )
