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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "regime_clf"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import LLMError, ProviderConfig, chat  # noqa: E402
import training  # noqa: E402
import curation  # noqa: E402

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
            entry = training.registry_entry_for(c["checkpoint"]) or {}
            _clf = RegimeClassifier(os.path.join(ROOT, c["checkpoint"]), emb_url=c.get("emb_url") or None,
                                    unit_classes=entry.get("unit_classes"))
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
    scripted_reply: str | None = None    # provider "scripted": the transcript's assistant turn, no LLM call
    expected: dict | None = None         # optional {"user": label|"none", "assistant": label|"none"} from the transcript


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
app.mount("/docs", StaticFiles(directory=os.path.join(ROOT, "docs")), name="docs")      # linked from the "The classes" tab


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "app", "static", "index.html"))


@app.get("/api/config")
def get_config():
    labels, unit_classes, descriptions = _label_space()
    return {"presets": CONFIG["presets"], "system_prompt": CONFIG.get("system_prompt", ""),
            "on_flagged_response": CONFIG["screening"].get("on_flagged_response", "warn"),
            "labels": labels, "unit_classes": unit_classes, "descriptions": descriptions,
            "active": CONFIG["classifier"]["checkpoint"]}


def _label_space():
    """{label: display name} and {unit: [labels]} for the model the UI will use: the loaded
    classifier's if it is up, else the checkpoint's own class list (cheap: heads only)."""
    import torch
    from inference import RULE_NAMES, UNIT_CLASSES
    if _clf is not None:
        classes = dict(_clf.classes)
        unit = getattr(_clf, "unit_classes", None)
    else:
        ck = torch.load(os.path.join(ROOT, CONFIG["classifier"]["checkpoint"]), map_location="cpu")
        classes = dict(ck["classes"]); unit = None
    labels = list(classes); neg = labels[0]
    # user-trained models carry their own display names and speaker masks in the registry
    entry = training.registry_entry_for(CONFIG["classifier"]["checkpoint"]) or {}
    names = entry.get("names") or {}
    if entry.get("unit_classes"):
        unit = {u: [neg] + [l for l in ls if l in classes] for u, ls in entry["unit_classes"].items()}
    if unit is None:
        unit = {u: [neg] + [l for l in ls if l in classes] for u, ls in UNIT_CLASSES.items()}
    return ({l: names.get(l) or RULE_NAMES.get(l, l) for l in labels}, unit, classes)


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
        if req.provider.provider == "scripted":
            # Replay mode: the assistant turn comes from the transcript. An empty reply means the
            # conversation ended on a user turn; only the user message is screened.
            reply = req.scripted_reply or ""
        else:
            reply = chat(cfg, msgs)
        t_llm = time.perf_counter() - t1
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:   # anything else from a provider: still a readable error in the UI
        raise HTTPException(status_code=502, detail=f"LLM call failed ({type(e).__name__}): {e}")
    t2 = time.perf_counter()
    reply_verdict = screen(reply, "assistant_response") if reply else None
    t_reply = time.perf_counter() - t2
    turn = {"turn_id": turn_id, "session_id": req.session_id, "ts": time.time(),
            "provider": cfg.provider, "model": cfg.model, "base_url": cfg.base_url,
            "user": {"text": req.message, "verdict": user_verdict, "expected": (req.expected or {}).get("user")},
            "assistant": {"text": reply, "verdict": reply_verdict, "expected": (req.expected or {}).get("assistant")},
            "latency_s": {"screen_user": round(t_user, 3), "llm": round(t_llm, 3),
                          "screen_reply": round(t_reply, 3)}}
    append_jsonl(TURNS, turn)
    return turn


# ------------------------------------------------------------------ data curation tab (Claude Code)
class CurateRequest(BaseModel):
    prompt: str
    model: str | None = None
    max_turns: int = 40
    budget_usd: float = 5.0
    rows_per_class: int = 50
    resume_of: str | None = None


@app.get("/api/curate/status")
def curate_status():
    return curation.status(CONFIG)


@app.post("/api/curate/start")
def curate_start(req: CurateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "write a prompt first")
    try:
        return curation.start(req.prompt, CONFIG, model=req.model or None, max_turns=req.max_turns,
                              budget_usd=req.budget_usd, rows_per_class=req.rows_per_class, resume_of=req.resume_of)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/api/curate/jobs")
def curate_jobs():
    return curation.jobs()


@app.get("/api/curate/jobs/{job_id}")
def curate_job(job_id: str):
    j = curation.job(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return j


@app.post("/api/curate/jobs/{job_id}/cancel")
def curate_cancel(job_id: str):
    return {"cancelled": curation.cancel(job_id)}


@app.post("/api/curate/jobs/{job_id}/to_train")
def curate_to_train(job_id: str):
    try:
        return curation.to_upload(job_id)
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/curate/jobs/{job_id}/download")
def curate_download(job_id: str):
    j = curation.job(job_id)
    p = os.path.join(ROOT, j["workdir"], "dataset.jsonl") if j else None
    if not p or not os.path.exists(p):
        raise HTTPException(404, "no dataset yet")
    return FileResponse(p, filename=f"curated_{job_id}.jsonl", media_type="application/json")


# ------------------------------------------------------------------ training tabs
class ValidateRequest(BaseModel):
    upload_id: str
    spec: dict
    expert: bool = False     # post-training corrections: no per-class minimums


class StartRequest(BaseModel):
    name: str | None = None
    epochs: int = 30
    lr: float = 1e-3
    init: str = "reference"              # reference | active | <checkpoint path>
    upload_id: str | None = None         # train tab
    post_train: bool = False             # post-training tab
    use_feedback: bool = True
    extra_upload_id: str | None = None
    include_base_data: bool = True
    oversample: int = 5


class ActivateRequest(BaseModel):
    checkpoint: str
    reason: str | None = None


def _active_ckpt() -> str:
    return CONFIG["classifier"]["checkpoint"]


@app.post("/api/train/upload")
async def train_upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 50_000_000:
        raise HTTPException(413, "file larger than 50 MB")
    try:
        return training.parse_upload(file.filename or "upload.jsonl", data)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/files")
def project_files():
    """Data files inside the project, for the 'Project data' pickers."""
    return training.project_files()


class ProjectFileRequest(BaseModel):
    path: str


@app.post("/api/train/use_project_file")
def train_use_project_file(req: ProjectFileRequest):
    try:
        return training.use_project_file(req.path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/files/transcript")
def project_transcript(path: str):
    try:
        return training.read_project_transcript(path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e))


@app.get("/api/train/examples")
def train_examples():
    return training.examples()


class ExampleRequest(BaseModel):
    name: str


@app.post("/api/train/use_example")
def train_use_example(req: ExampleRequest):
    try:
        return training.use_example(req.name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/train/validate")
def train_validate(req: ValidateRequest):
    try:
        return training.validate(req.upload_id, req.spec, expert=req.expert)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/train/start")
def train_start(req: StartRequest):
    if not req.post_train and not req.upload_id:
        raise HTTPException(400, "upload_id required")
    clf = classifier()   # the job embeds with the screener's encoder
    try:
        return training.start_job(req.model_dump(), clf.embedder, _active_ckpt())
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/api/train/jobs")
def train_jobs():
    return training.jobs()


@app.get("/api/train/jobs/{job_id}")
def train_job(job_id: str):
    j = training.job(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return j


@app.get("/api/train/models")
def train_models():
    return {"active": _active_ckpt(), "models": training.registry(), "running": training._running.is_set()}


@app.get("/api/train/feedback")
def train_feedback():
    import torch
    ck = torch.load(os.path.join(ROOT, _active_ckpt()), map_location="cpu")
    labels = list(ck["classes"])
    fb = training.feedback_rows(labels, labels[0])
    return {"labels": labels, "n": fb["n"], "skipped": fb["skipped"], "by_label": fb["by_label"],
            "sample": fb["rows"][:5]}


@app.post("/api/train/activate")
def train_activate(req: ActivateRequest):
    """Hot-swap the screener to another checkpoint; the encoder stays loaded."""
    global _clf
    path = os.path.join(ROOT, req.checkpoint)
    if not os.path.exists(path):
        raise HTTPException(404, f"{req.checkpoint} not found")
    from inference import RegimeClassifier
    entry = training.registry_entry_for(req.checkpoint) or {}
    with _clf_lock:
        emb = _clf.embedder if _clf is not None else None
        _clf = RegimeClassifier(path, emb_url=CONFIG["classifier"].get("emb_url") or None, embedder=emb,
                                unit_classes=entry.get("unit_classes"))
    CONFIG["classifier"]["checkpoint"] = req.checkpoint
    append_jsonl(os.path.join(ROOT, "app", "data", "promotions.jsonl"),
                 {"ts": time.time(), "checkpoint": req.checkpoint, "reason": req.reason})
    return {"active": req.checkpoint, "labels": _clf.labels}


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if req.kind not in ("false_positive", "false_negative"):
        raise HTTPException(400, "kind must be false_positive or false_negative")
    if req.target not in ("user", "assistant"):
        raise HTTPException(400, "target must be user or assistant")
    row = {"ts": time.time(), **req.model_dump()}
    append_jsonl(FEEDBACK, row)
    return {"ok": True}
