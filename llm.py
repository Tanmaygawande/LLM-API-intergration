"""Thin provider layer: OpenAI, Google Gemini, and an offline Mock for tests/demos.

Every provider takes (system_prompt, messages) where messages is a list of
{"role": "user" | "assistant", "content": str} and returns the reply text.
All provider failures are converted into LLMError with an HTTP-friendly status
and a user-safe message, so the API never leaks a stack trace or a broken UI.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from .config import Settings, get_settings

log = logging.getLogger("chatbot.llm")


class LLMError(Exception):
    """User-presentable LLM failure."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _classify(exc: Exception) -> LLMError:
    """Map any SDK exception to a clear, safe error."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)

    if "timeout" in name or "timed out" in text or code == 504:
        return LLMError("The AI service took too long to respond. Please try again.", 504)
    if "ratelimit" in name or code == 429 or "quota" in text or "rate limit" in text:
        return LLMError("Rate limit reached. Please wait a few seconds and try again.", 429)
    if "authentication" in name or code in (401, 403) or "api key" in text or "permission" in name:
        return LLMError("The server's API key is missing or invalid. Check the .env configuration.", 500)
    if "connection" in name or "network" in text or "connect" in text:
        return LLMError("Could not reach the AI service. Check your internet connection.", 503)
    return LLMError("The AI service returned an error. Please try again.", 502)


def _is_retryable(err: LLMError) -> bool:
    return err.status_code in (429, 502, 503, 504)


class LLMClient(ABC):
    @abstractmethod
    def _generate(self, system: str, messages: list[dict]) -> str: ...

    def generate(self, system: str, messages: list[dict], retries: int = 0) -> str:
        """Call the provider with retry + exponential backoff on transient errors."""
        attempt = 0
        while True:
            try:
                text = self._generate(system, messages)
            except LLMError as err:
                last = err
            except Exception as exc:  # SDK-specific errors
                log.warning("LLM call failed: %s: %s", type(exc).__name__, exc)
                last = _classify(exc)
            else:
                text = (text or "").strip()
                if not text:
                    last = LLMError("The AI returned an empty response. Please rephrase and try again.", 502)
                else:
                    return text

            if attempt >= retries or not _is_retryable(last):
                raise last
            time.sleep(min(2**attempt, 8))
            attempt += 1


class OpenAIClient(LLMClient):
    def __init__(self, s: Settings):
        if not s.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.", 500)
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=s.openai_api_key, timeout=s.timeout_seconds, max_retries=0)
        self._model = s.openai_model

    def _generate(self, system: str, messages: list[dict]) -> str:
        payload = [{"role": "system", "content": system}, *messages]
        resp = self._client.chat.completions.create(model=self._model, messages=payload)
        return resp.choices[0].message.content


class GeminiClient(LLMClient):
    def __init__(self, s: Settings):
        if not s.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.", 500)
        from google import genai  # lazy import
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=s.gemini_api_key,
            http_options=types.HttpOptions(timeout=s.timeout_seconds * 1000),
        )
        self._model = s.gemini_model

    def _generate(self, system: str, messages: list[dict]) -> str:
        t = self._types
        contents = [
            t.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[t.Part(text=m["content"])],
            )
            for m in messages
        ]
        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=t.GenerateContentConfig(system_instruction=system),
        )
        return resp.text


class MockClient(LLMClient):
    """Offline provider: lets you run the whole app and tests with no API key.
    It proves memory works by quoting earlier turns it can see."""

    def _generate(self, system: str, messages: list[dict]) -> str:
        if "compress chat history" in system:  # summarisation request
            return "Summary: " + messages[-1]["content"][-300:].replace("\n", " ")
        users = [m["content"] for m in messages if m["role"] == "user"]
        earlier = f" Earlier you said: '{users[0][:60]}'." if len(users) > 1 else ""
        return f"[mock] You said: '{users[-1][:80]}'. I can see {len(messages)} message(s) of context.{earlier}"


def build_client(settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    if s.provider == "openai":
        return OpenAIClient(s)
    if s.provider == "gemini":
        return GeminiClient(s)
    if s.provider == "mock":
        return MockClient()
    raise LLMError(f"Unknown LLM_PROVIDER '{s.provider}'. Use gemini, openai or mock.", 500)
