# Pip: LLM Chatbot with Multi-Turn Memory

A FastAPI chatbot backed by a real LLM API (**Google Gemini** or **OpenAI**, switchable with one env var),
with per-conversation history and context window management.

## Features (task checklist)

| Requirement | Where |
|---|---|
| Backend endpoint that sends messages to an LLM and returns the reply | `POST /chat` in `app/main.py` |
| Multi-turn history per conversation | `ConversationStore` in `app/memory.py` (keyed by `conversation_id`) |
| Context window management | `compact()` in `app/memory.py`: **truncation** and **running summary** (both implemented) |
| System prompt with a defined role | `app/prompts.py` ("Pip", tech mentor) |
| Error handling | `app/llm.py`: timeouts, rate limits, auth, network, empty replies, retries with backoff; failed turns are rolled back; UI shows a clear message |
| API keys in env vars, never committed | `.env` is git-ignored; see `.env.example` |
| Simple frontend / CLI | `static/index.html` (served at `/`) and `cli.py` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit .env and add your key
```

Get a key: [Google AI Studio](https://aistudio.google.com/) (Gemini) or the OpenAI platform.
Set `LLM_PROVIDER=gemini` or `openai` in `.env`. No key yet? Use `LLM_PROVIDER=mock` to try everything offline.

## Run

```bash
uvicorn app.main:app --reload        # web UI at http://127.0.0.1:8000
python cli.py                        # or chat in the terminal
```

## API

```
POST   /chat                      {"message": "...", "conversation_id": "optional"}
GET    /conversations/{id}        history + running summary
DELETE /conversations/{id}        reset a conversation
GET    /health
```

Example:

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"Hi, I am Tanmay"}'
# reuse the returned conversation_id on the next call so Pip remembers you
```

## How context management works

* Every request sends: system prompt (+ running summary, if any) + the stored messages for that conversation.
* When history exceeds `MAX_HISTORY_MESSAGES` (or `MAX_CONTEXT_CHARS`):
  * `CONTEXT_STRATEGY=summarize` (default): older messages are compressed by the LLM into a running summary that is
    prepended to the system prompt; the last `KEEP_RECENT_MESSAGES` stay verbatim. If summarising fails, it falls back to truncation.
  * `CONTEXT_STRATEGY=truncate`: only the last N messages are kept.
* Conversations are stored in memory (lost on restart). Swap `ConversationStore` for Redis/a DB for production.

## Tests

```bash
python -m pytest            # or: python -m unittest discover -s tests -t .
```

Tests run offline using the mock provider (memory, truncation, summarisation, rollback, retries, API routes).

## Security

Never commit real keys. `.env` is in `.gitignore`; only `.env.example` (placeholders) is tracked.
