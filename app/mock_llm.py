"""A canned OpenAI-compatible endpoint for developing and testing the chat UI offline.

    .venv/bin/python -m uvicorn app.mock_llm:app --port 8799
    # UI: provider "OpenAI-compatible", base URL http://localhost:8799/v1, any model name

Replies are chosen by keyword so both clean and flaggable assistant responses can be
exercised: mention "guarantee" for a FINRA 2210-style reply, "recommend" for a Reg BI-style
reply; anything else gets a compliant educational reply.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="mock OpenAI-compatible LLM")

REPLIES = {
    "guarantee": "Absolutely - this fund is guaranteed to return 12% a year with no risk at all, "
                 "so it's a sure thing for your retirement.",
    "recommend": "Based on your age and goals, you should move your whole 401(k) into our "
                 "proprietary Growth Portfolio this week.",
}
DEFAULT = ("Generally speaking, diversification spreads risk across asset classes. I can share "
           "educational information, but a registered representative can discuss your situation.")


class ChatBody(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int | None = None


@app.post("/v1/chat/completions")
def completions(body: ChatBody):
    last = next((m["content"] for m in reversed(body.messages) if m["role"] == "user"), "").lower()
    text = next((r for k, r in REPLIES.items() if k in last), DEFAULT)
    return {"id": "mock", "object": "chat.completion", "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}]}
