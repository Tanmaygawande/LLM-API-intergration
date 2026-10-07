"""Terminal chat: python cli.py   (commands: /new  /history  /quit)"""
from app.llm import LLMError
from app.service import get_service


def main() -> None:
    try:
        service = get_service()
    except LLMError as err:
        raise SystemExit(f"Setup error: {err.message}")

    print("Pip - tech mentor bot. Type /new to reset, /history to view memory, /quit to exit.\n")
    cid = None
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            if cid:
                service.reset(cid)
            cid = None
            print("-- new conversation --")
            continue
        if text == "/history":
            data = service.history(cid) if cid else None
            if not data:
                print("(empty)")
                continue
            if data["summary"]:
                print(f"[summary] {data['summary']}")
            for m in data["messages"]:
                print(f"  {m['role']}: {m['content'][:100]}")
            continue
        try:
            result = service.chat(text, cid)
        except LLMError as err:
            print(f"! {err.message}\n")
            continue
        cid = result.conversation_id
        print(f"Pip: {result.reply}\n")


if __name__ == "__main__":
    main()
