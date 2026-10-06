"""Supervised chat: any LLM, every turn screened by the CLM regime classifier.

    .venv/bin/python -m uvicorn app.server:app --port 8710      # from the repo root
    open http://localhost:8710

Each turn: screen the user's message (client-side regimes) -> call the selected LLM ->
screen the reply (assistant-side regimes) -> log the turn. Users can flag any verdict
as a false positive or false negative; flags are appended to the feedback file.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "regime_clf"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import LLMError, ProviderConfig, chat  # noqa: E402

CONFIG = yaml.safe_load(open(os.environ.get("CHAT_CONFIG", os.path.join(ROOT, "app", "config.yaml"))))
TURNS = os.path.join(ROOT, CONFIG["storage"]["turns"])
FEEDBACK = os.path.join(ROOT, CONFIG["storage"]["feedback"])
_write_lock = threading.Lock()
_clf = None
_clf_lock = threading.Lock()


def classifier():
    """Load lazily: Qwen3-8B takes ~20 s and ~16 GB, so the UI can come up first."""
    global _clf
    with _clf_lock:
        if _clf is None:
            from inference import RegimeClassifier
            c = CONFIG["classifier"]
            _clf = RegimeClassifier(os.path.join(ROOT, c["checkpoint"]), emb_url=c.get("emb_url") or None)
        return _clf


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _write_lock, open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def screen(text: str, unit: str) -> dict:
    use_unit = unit if CONFIG["screening"].get("unit_masking", True) else None
    return asdict(classifier().classify([text], unit=use_unit)[0])


class Provider(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 4096


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    provider: Provider
    history: list[Message] = Field(default_factory=list)
    message: str
    system: str | None = None


class ClassifyRequest(BaseModel):
    texts: list[str]
    unit: str | None = None


class FeedbackRequest(BaseModel):
    turn_id: str
    target: str                          # "user" | "assistant"
    kind: str                            # "false_positive" | "false_negative"
    text: str
    predicted_label: str
    correct_label: str | None = None
    note: str | None = None


app = FastAPI(title="CLM regime screening chat")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "app", "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "app", "static", "index.html"))


@app.get("/api/config")
def get_config():
    return {"presets": CONFIG["presets"], "system_prompt": CONFIG.get("system_prompt", ""),
            "on_flagged_response": CONFIG["screening"].get("on_flagged_response", "warn"),
            "labels": _labels()}


def _labels():
    from inference import RULE_NAMES
    return RULE_NAMES


@app.get("/api/health")
def health():
    return {"classifier_loaded": _clf is not None}


@app.post("/api/warmup")
def warmup():
    t0 = time.perf_counter()
    classifier()
    return {"classifier_loaded": True, "seconds": round(time.perf_counter() - t0, 1)}


@app.post("/api/classify")
def classify(req: ClassifyRequest):
    return [asdict(p) for p in classifier().classify(req.texts, unit=req.unit)]


@app.post("/api/chat")
def chat_turn(req: ChatRequest):
    turn_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    user_verdict = screen(req.message, "client_message")
    t_user = time.perf_counter() - t0
    cfg = ProviderConfig(provider=req.provider.provider, model=req.provider.model, base_url=req.provider.base_url,
                         api_key=req.provider.api_key, api_key_env=req.provider.api_key_env,
                         max_tokens=req.provider.max_tokens,
                         system=req.system if req.system is not None else CONFIG.get("system_prompt"))
    msgs = [m.model_dump() for m in req.history] + [{"role": "user", "content": req.message}]
    try:
        t1 = time.perf_counter()
        reply = chat(cfg, msgs)
        t_llm = time.perf_counter() - t1
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    t2 = time.perf_counter()
    reply_verdict = screen(reply, "assistant_response")
    t_reply = time.perf_counter() - t2
    turn = {"turn_id": turn_id, "session_id": req.session_id, "ts": time.time(),
            "provider": cfg.provider, "model": cfg.model, "base_url": cfg.base_url,
            "user": {"text": req.message, "verdict": user_verdict},
            "assistant": {"text": reply, "verdict": reply_verdict},
            "latency_s": {"screen_user": round(t_user, 3), "llm": round(t_llm, 3),
                          "screen_reply": round(t_reply, 3)}}
    append_jsonl(TURNS, turn)
    return turn


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if req.kind not in ("false_positive", "false_negative"):
        raise HTTPException(400, "kind must be false_positive or false_negative")
    if req.target not in ("user", "assistant"):
        raise HTTPException(400, "target must be user or assistant")
    row = {"ts": time.time(), **req.model_dump()}
    append_jsonl(FEEDBACK, row)
    return {"ok": True}
