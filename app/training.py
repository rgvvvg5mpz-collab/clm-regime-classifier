"""Training from the chat UI: uploads, validation, background jobs, the promotion gate and
the model registry.

Everything heavy reuses the existing pipeline: ``mps_embedder.embed_cached`` (shared
embedding store), ``train_api.train_heads`` (same recipe as the shipped models) and
``Tests/make_reports.py`` (same reports). The encoder is the one the chat screener already
holds, so training never loads a second Qwen3-8B.

Data flow
  upload  -> app/data/uploads/<upload_id>/{raw.<ext>, rows.jsonl, summary.json}
  validate-> app/data/uploads/<upload_id>/clean.jsonl (scrubbed, deduped, split assigned)
  train   -> regime_clf/checkpoints/<name>.pt + Tests/<date>_ui_<name>/ + registry entry
"""
from __future__ import annotations

import collections
import csv
import io
import json
import os
import random
import re
import sys
import threading
import time
import traceback
import uuid

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "regime_clf"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from classes import INSTRUCTIONS, LABEL_SETS  # noqa: E402
from clm.schema import state_text  # noqa: E402
from mps_embedder import embed_cached  # noqa: E402
from name_check import scrub  # noqa: E402

UPLOADS = os.path.join(ROOT, "app", "data", "uploads")
EXAMPLES = os.path.join(ROOT, "app", "examples")
REGISTRY = os.path.join(ROOT, "app", "data", "models.json")
FEEDBACK = os.path.join(ROOT, "app", "data", "feedback.jsonl")
CKPT_DIR = os.path.join(ROOT, "regime_clf", "checkpoints")
TESTS = os.path.join(ROOT, "Tests")
OOD_FILES = {"ood_hard": os.path.join(ROOT, "regime_clf", "ood", "ood_hard_v1.jsonl"),
             "ood_low_medium": os.path.join(ROOT, "regime_clf", "ood", "ood_low_medium_v1.jsonl")}
NEGATIVE_ALIASES = {"compliant", "no_flag", "none", "no_issue", "ok", "negative", "clean", "not_flagged", "nothing"}
SPEAKER_ALIASES = {"client": "client_message", "user": "client_message", "customer": "client_message",
                   "client_message": "client_message", "assistant": "assistant_response", "bot": "assistant_response",
                   "assistant_response": "assistant_response", "model": "assistant_response"}
MIN_ROWS_ERROR, MIN_ROWS_WARN = 10, 50
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()
_running = threading.Event()


# ------------------------------------------------------------------ uploads
def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_").lower()[:40] or "model"


def parse_upload(filename: str, data: bytes) -> dict:
    """CSV or JSONL with columns text, label, optional speaker/unit, optional split."""
    text = data.decode("utf-8-sig", errors="replace")
    rows = []
    if filename.lower().endswith(".csv"):
        rdr = csv.DictReader(io.StringIO(text))
        if not rdr.fieldnames:
            raise ValueError("CSV has no header row")
        cols = {c.strip().lower(): c for c in rdr.fieldnames}
        if "text" not in cols or "label" not in cols:
            raise ValueError(f"CSV needs 'text' and 'label' columns; found {rdr.fieldnames}")
        for r in rdr:
            rows.append({k.strip().lower(): (v or "").strip() for k, v in r.items() if k})
    else:
        for i, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {i + 1}: not valid JSON ({e.msg})") from e
            if not isinstance(r, dict) or "text" not in r or "label" not in r:
                raise ValueError(f"line {i + 1}: each row needs 'text' and 'label'")
            rows.append({k.lower(): v for k, v in r.items()})
    if not rows:
        raise ValueError("no rows found")
    norm = []
    for r in rows:
        sp = r.get("speaker", r.get("unit", "")) or ""
        norm.append({"text": str(r["text"]).strip(), "label": str(r["label"]).strip(),
                     "unit": SPEAKER_ALIASES.get(str(sp).strip().lower()) if sp else None,
                     "split": str(r.get("split", "")).strip().lower() or None})
    uid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    d = os.path.join(UPLOADS, uid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "raw" + os.path.splitext(filename)[1].lower()), "wb") as f:
        f.write(data)
    with open(os.path.join(d, "rows.jsonl"), "w") as f:
        for r in norm:
            f.write(json.dumps(r) + "\n")
    labels = collections.Counter(r["label"] for r in norm)
    summary = {"upload_id": uid, "filename": filename, "n_rows": len(norm),
               "labels": dict(labels.most_common()),
               "has_speaker": any(r["unit"] for r in norm),
               "speakers": dict(collections.Counter(r["unit"] or "unknown" for r in norm)),
               "has_split": any(r["split"] for r in norm),
               "splits": dict(collections.Counter(r["split"] or "none" for r in norm)),
               "suggested_negative": next((l for l in labels if l.lower() in NEGATIVE_ALIASES), None),
               "sample": norm[:5]}
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=2)
    return summary


# Folders the "Project data" pickers list, in display order: (relative path, group label, recursive)
PROJECT_DATA = [("data", "Your data (data/)", True), ("app/examples", "Examples (app/examples/)", False),
                ("app/static/transcripts", "Chat transcripts (app/static/transcripts/)", False),
                ("app/data/curation", "Curated by Claude Code (app/data/curation/)", True),
                ("regime_clf/data_5way", "Regulatory example, 5-way (regime_clf/data_5way/)", False),
                ("regime_clf/data", "Regulatory example, 7-way (regime_clf/data/)", False),
                ("regime_clf/ood", "Regulatory example, OOD sets (regime_clf/ood/)", False)]
DATA_EXT = (".csv", ".jsonl", ".json")


def project_files() -> list[dict]:
    """Data files inside the project, grouped for the pickers. .json = transcript, .csv/.jsonl = rows."""
    out = []
    for rel, group, recursive in PROJECT_DATA:
        base = os.path.join(ROOT, rel)
        if not os.path.isdir(base):
            continue
        walker = os.walk(base) if recursive else [(base, [], os.listdir(base))]
        for d, _, files in walker:
            for f in sorted(files):
                if not f.endswith(DATA_EXT) or f.endswith(".spec.json") or f in ("classes.json", "summary.json", "spec.json"):
                    continue
                if "raw_" in f or f.startswith("prompts"):
                    continue
                p = os.path.join(d, f)
                kind = "transcript" if f.endswith(".json") else "rows"
                try:
                    n = sum(1 for _ in open(p)) - (1 if f.endswith(".csv") else 0) if kind == "rows" else len(json.load(open(p)).get("conversations", []))
                except (OSError, ValueError, AttributeError):
                    continue
                out.append({"path": os.path.relpath(p, ROOT), "name": os.path.relpath(p, base), "group": group, "kind": kind,
                            "count": n, "expert": "expert" in f or "correction" in f,
                            "has_spec": os.path.exists(os.path.splitext(p)[0] + ".spec.json")})
    return out


def _safe_project_path(rel: str) -> str:
    p = os.path.realpath(os.path.join(ROOT, rel))
    if not any(p.startswith(os.path.realpath(os.path.join(ROOT, r)) + os.sep) for r, _, _ in PROJECT_DATA):
        raise FileNotFoundError(f"{rel} is not in a project data folder")
    if not os.path.isfile(p) or not p.endswith(DATA_EXT):
        raise FileNotFoundError(f"{rel} not found")
    return p


def use_project_file(rel: str) -> dict:
    """Stage a project data file as an upload (rows), prefilling the class table from a sibling .spec.json."""
    p = _safe_project_path(rel)
    summary = parse_upload(os.path.basename(p), open(p, "rb").read())
    spec_p = os.path.splitext(p)[0] + ".spec.json"
    if os.path.exists(spec_p):
        summary["spec"] = json.load(open(spec_p))["spec"]
    summary["example"] = os.path.basename(p)
    return summary


def read_project_transcript(rel: str) -> dict:
    p = _safe_project_path(rel)
    return json.load(open(p))


def examples() -> list[dict]:
    """Datasets shipped in app/examples/, each optionally with <name>.spec.json (prefilled classes)."""
    out = []
    for f in sorted(os.listdir(EXAMPLES)) if os.path.isdir(EXAMPLES) else []:
        if not f.endswith((".csv", ".jsonl")):
            continue
        path = os.path.join(EXAMPLES, f)
        n = sum(1 for _ in open(path)) - (1 if f.endswith(".csv") else 0)
        spec_p = os.path.join(EXAMPLES, os.path.splitext(f)[0] + ".spec.json")
        out.append({"name": f, "rows": n, "has_spec": os.path.exists(spec_p),
                    "kind": "expert" if "expert" in f or "correction" in f else "train"})
    return out


def use_example(name: str) -> dict:
    """Stage a shipped example as an upload; returns the upload summary plus any prefilled spec."""
    if "/" in name or name not in {e["name"] for e in examples()}:
        raise FileNotFoundError(f"unknown example {name}")
    summary = parse_upload(name, open(os.path.join(EXAMPLES, name), "rb").read())
    spec_p = os.path.join(EXAMPLES, os.path.splitext(name)[0] + ".spec.json")
    if os.path.exists(spec_p):
        summary["spec"] = json.load(open(spec_p))["spec"]
    summary["example"] = name
    return summary


def _rows(upload_id: str, name: str = "rows.jsonl") -> list[dict]:
    p = os.path.join(UPLOADS, upload_id, name)
    if not os.path.exists(p):
        if name == "clean.jsonl" and os.path.exists(os.path.join(UPLOADS, upload_id, "rows.jsonl")):
            raise FileNotFoundError(f"upload {upload_id} has not passed validation yet")
        raise FileNotFoundError(f"unknown upload {upload_id}")
    return [json.loads(l) for l in open(p)]


# ------------------------------------------------------------------ class spec + validation
def normalise_spec(spec: dict) -> tuple[dict[str, str], dict[str, str], str, dict[str, list[str]]]:
    """UI class spec -> (classes {label: description}, display names, negative label, unit_classes).

    spec = {label: {"name", "description", "speaker": client|assistant|either, "negative": bool}}
    The negative class is placed first (the checkpoint convention).
    """
    negs = [l for l, c in spec.items() if c.get("negative")]
    if len(negs) != 1:
        raise ValueError("exactly one class must be marked as the negative / nothing-to-flag class")
    neg = negs[0]
    order = [neg] + [l for l in spec if l != neg]
    classes, names, unit_classes = {}, {}, {"client_message": [], "assistant_response": []}
    for l in order:
        c = spec[l]
        desc = (c.get("description") or "").strip()
        if not desc:
            raise ValueError(f"class '{l}' needs a description: it is the text the model scores messages against")
        classes[l] = desc
        names[l] = (c.get("name") or l).strip()
        sp = (c.get("speaker") or "either").lower()
        if l != neg:
            if sp in ("client", "either"):
                unit_classes["client_message"].append(l)
            if sp in ("assistant", "either"):
                unit_classes["assistant_response"].append(l)
    return classes, names, neg, unit_classes


def validate(upload_id: str, spec: dict, seed: int = 20933, expert: bool = False) -> dict:
    """``expert=True`` validates a small set of corrections for post-training: no per-class
    minimums, no negative-share warning, and no held-out split (the job splits them itself)."""
    rows = _rows(upload_id)
    errors, warnings = [], []
    try:
        classes, names, neg, unit_classes = normalise_spec(spec)
    except ValueError as e:
        return {"ok": False, "errors": [str(e)], "warnings": [], "counts": {}}
    unknown = sorted({r["label"] for r in rows} - set(classes))
    if unknown:
        errors.append(f"labels in the data with no class definition: {unknown}")
    # scrub + basic hygiene
    scrubbed = 0; empty = 0; long_ = 0
    for r in rows:
        t = scrub(r["text"])
        if t != r["text"]:
            scrubbed += 1; r["text"] = t
        if not r["text"]:
            empty += 1
        if len(r["text"].split()) > 400:
            long_ += 1
    rows = [r for r in rows if r["text"]]
    if empty:
        warnings.append(f"{empty} empty texts dropped")
    if scrubbed:
        warnings.append(f"{scrubbed} rows contained a blocked real-firm name; replaced with a fictional one")
    if long_:
        warnings.append(f"{long_} texts longer than 400 words (the encoder keeps the last 2,048 tokens)")
    # duplicates / conflicts
    by_text = collections.defaultdict(list)
    for r in rows:
        by_text[r["text"]].append(r)
    conflicts = sum(1 for g in by_text.values() if len({r["label"] for r in g}) > 1)
    dups = sum(len(g) - 1 for g in by_text.values())
    if conflicts:
        warnings.append(f"{conflicts} texts appear with different labels; dropped as ambiguous")
    if dups - conflicts > 0:
        warnings.append(f"{dups} duplicate texts collapsed")
    rows = [g[0] for g in by_text.values() if len({r["label"] for r in g}) == 1]
    # speaker consistency
    spk_mismatch = 0
    for r in rows:
        if r["unit"] and r["label"] != neg and r["label"] in classes and r["label"] not in unit_classes.get(r["unit"], []):
            spk_mismatch += 1
    if spk_mismatch:
        warnings.append(f"{spk_mismatch} rows whose label is not allowed for the row's speaker (check the speaker settings)")
    if not any(r["unit"] for r in rows):
        warnings.append("no speaker column: speaker masking will be unavailable for this model")
    # counts
    counts = collections.Counter(r["label"] for r in rows)
    for l in classes:
        n = counts.get(l, 0)
        if expert:
            continue
        if n < MIN_ROWS_ERROR:
            errors.append(f"class '{l}' has only {n} rows (minimum {MIN_ROWS_ERROR})")
        elif n < MIN_ROWS_WARN:
            warnings.append(f"class '{l}' has only {n} rows; expect noisy results below {MIN_ROWS_WARN}")
    if not expert and counts and max(counts.values()) > 10 * max(1, min(counts.values())):
        warnings.append("class imbalance above 10:1; macro-F1 is reported so small classes still count")
    if not expert and counts.get(neg, 0) < 0.1 * len(rows):
        warnings.append(f"the negative class '{neg}' is under 10% of the data; the model will rarely say 'nothing to flag'")
    if not rows:
        errors.append("no usable rows")
    # split
    rng = random.Random(seed)
    if expert:
        src = "expert rows (the post-training job splits them 80/20 train/val)"
        for r in rows:
            r["split"] = "train"
    elif all(r["split"] in ("train", "val", "test") for r in rows):
        src = "from file"
    else:
        src = "stratified 80/10/10"
        by_label = collections.defaultdict(list)
        for r in rows:
            by_label[r["label"]].append(r)
        for g in by_label.values():
            rng.shuffle(g)
            n = len(g); nv, nt = max(1, int(0.1 * n)), max(1, int(0.1 * n))
            for i, r in enumerate(g):
                r["split"] = "test" if i < nt else "val" if i < nt + nv else "train"
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "counts": dict(counts)}
    with open(os.path.join(UPLOADS, upload_id, "clean.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    split_counts = {s: dict(collections.Counter(r["label"] for r in rows if r["split"] == s)) for s in ("train", "val", "test")}
    json.dump({"spec": spec, "classes": classes, "names": names, "negative": neg, "unit_classes": unit_classes},
              open(os.path.join(UPLOADS, upload_id, "spec.json"), "w"), indent=2)
    return {"ok": True, "errors": [], "warnings": warnings, "counts": dict(counts), "n_clean": len(rows),
            "split_source": src, "split_counts": split_counts, "unit_classes": unit_classes, "negative": neg}


# ------------------------------------------------------------------ registry
def registry() -> list[dict]:
    if os.path.exists(REGISTRY):
        return json.load(open(REGISTRY))
    shipped = []
    for name, ls, data in (("clm_regime_7way", "7way", "data"), ("clm_regime_5way", "5way", "data_5way")):
        p = os.path.join(CKPT_DIR, name + ".pt")
        if os.path.exists(p):
            classes = LABEL_SETS[ls]
            shipped.append({"name": name, "base_name": name, "version": 1, "parent": "CLM_v0.1-8B.pt (reference heads)",
                            "checkpoint": os.path.relpath(p, ROOT), "created": os.path.getmtime(p),
                            "labels": list(classes), "classes": classes, "source": "shipped (release)",
                            "train_data": os.path.join("regime_clf", data), "unit_classes": None, "metrics": {}})
    _save_registry(shipped)
    return shipped


def _save_registry(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    json.dump(entries, open(REGISTRY, "w"), indent=2)


def register(entry: dict) -> None:
    entries = [e for e in registry() if e["checkpoint"] != entry["checkpoint"]]
    entries.append(entry)
    _save_registry(entries)


def next_version(name: str) -> tuple[str, int]:
    """Versioned checkpoint name: the first model called X is X (v1), later ones X_v2, X_v3, ..."""
    versions = [e.get("version", 1) for e in registry() if e.get("base_name", e["name"]) == name]
    v = (max(versions) + 1) if versions else 1
    return (name if v == 1 else f"{name}_v{v}"), v


def registry_entry_for(checkpoint: str) -> dict | None:
    rel = os.path.relpath(checkpoint, ROOT) if os.path.isabs(checkpoint) else checkpoint
    return next((e for e in registry() if e["checkpoint"] == rel), None)


# ------------------------------------------------------------------ feedback as training data
def feedback_rows(labels: list[str], negative: str) -> dict:
    """Expert flags from the chat UI as (text, label, unit) rows for the given label space."""
    rows, skipped, by_turn = [], 0, {}
    if os.path.exists(FEEDBACK):
        for l in open(FEEDBACK):
            f = json.loads(l)
            lab = f.get("correct_label")
            if lab is None and f.get("kind") == "false_positive":
                lab = negative
            if lab in NEGATIVE_ALIASES and lab not in labels:
                lab = negative
            if lab not in labels:
                skipped += 1; continue
            by_turn[(f["turn_id"], f["target"])] = {"text": f["text"], "label": lab,
                                                    "unit": "client_message" if f["target"] == "user" else "assistant_response",
                                                    "note": f.get("note"), "ts": f.get("ts")}
    rows = list(by_turn.values())   # latest flag per turn wins
    return {"rows": rows, "n": len(rows), "skipped": skipped,
            "by_label": dict(collections.Counter(r["label"] for r in rows))}


# ------------------------------------------------------------------ jobs
def jobs() -> list[dict]:
    with _job_lock:
        return sorted(({k: v for k, v in j.items() if k != "_thread"} for j in _jobs.values()),
                      key=lambda j: j["created"], reverse=True)


def job(job_id: str) -> dict | None:
    with _job_lock:
        j = _jobs.get(job_id)
        return {k: v for k, v in j.items() if k != "_thread"} if j else None


def start_job(req: dict, embedder, active_ckpt: str | None) -> dict:
    """req: name, epochs, lr, init (reference|active|<checkpoint rel path>), and one of
       upload_id (train tab) or {post_train: true, base: active checkpoint, extra_upload_id?,
       use_feedback: bool, oversample: int}."""
    if _running.is_set():
        raise RuntimeError("a training job is already running")
    jid = uuid.uuid4().hex[:10]
    j = {"id": jid, "created": time.time(), "status": "queued", "stage": "queued", "progress": {}, "log": [],
         "request": req, "result": None, "error": None}
    with _job_lock:
        _jobs[jid] = j
    t = threading.Thread(target=_run, args=(j, embedder, active_ckpt), daemon=True)
    j["_thread"] = t
    _running.set()
    t.start()
    return job(jid)


def _log(j: dict, msg: str) -> None:
    j["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")


class _EncoderAdapter:
    """embed_cached wants ._encode(texts) -> [n, 4096]; wrap whichever embedder the server holds."""
    def __init__(self, embedder):
        self.e = embedder

    def _encode(self, texts):
        if hasattr(self.e, "_encode"):
            return self.e._encode(texts)
        return self.e.embed(texts)[0]


def _embed(j: dict, enc, texts: list[str], what: str, chunk: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(texts), chunk):
        j["progress"] = {"stage": "embedding", "what": what, "done": i, "total": len(texts)}
        out.append(embed_cached(enc, texts[i:i + chunk], what, chunk=chunk))
    j["progress"] = {"stage": "embedding", "what": what, "done": len(texts), "total": len(texts)}
    return np.concatenate(out) if out else np.zeros((0, 4096), np.float32)


def _run(j: dict, embedder, active_ckpt: str | None) -> None:
    try:
        j["status"] = "running"
        _run_inner(j, _EncoderAdapter(embedder), active_ckpt)
        j["status"] = "done"; j["stage"] = "done"
    except Exception as e:  # surface everything to the UI
        j["status"] = "failed"; j["error"] = f"{type(e).__name__}: {e}"
        _log(j, "failed: " + traceback.format_exc().splitlines()[-1])
    finally:
        _running.clear()


def _load_training_rows(req: dict, active_ckpt: str | None) -> tuple[list[dict], dict, dict, str, dict, list[str]]:
    """-> rows (with split), classes, names, negative, unit_classes, notes."""
    notes = []
    if req.get("post_train"):
        import torch
        base = req.get("base") or active_ckpt
        if not base:
            raise ValueError("post-training needs an active model to start from")
        ck = torch.load(os.path.join(ROOT, base) if not os.path.isabs(base) else base, map_location="cpu")
        classes = dict(ck["classes"]); labels = list(classes); neg = labels[0]
        entry = registry_entry_for(base) or {}
        names = entry.get("names") or {l: l for l in labels}
        unit_classes = entry.get("unit_classes")
        if unit_classes is None:   # shipped models: derive from inference.UNIT_CLASSES
            from inference import UNIT_CLASSES
            unit_classes = {u: [l for l in ls if l in classes] for u, ls in UNIT_CLASSES.items()}
        rows = []
        if req.get("include_base_data", True) and entry.get("train_data"):
            td = os.path.join(ROOT, entry["train_data"])
            if os.path.isdir(td):
                for s in ("train", "val", "test"):
                    for l in open(os.path.join(td, f"{s}.jsonl")):
                        r = json.loads(l)
                        rows.append({"text": r["text"], "label": r["label"] if r["label"] in classes else neg,
                                     "unit": r.get("unit"), "split": s, "source": "base"})
            elif os.path.isfile(td):
                for l in open(td):
                    r = json.loads(l); rows.append({**r, "source": "base"})
            notes.append(f"base training data: {len(rows)} rows from {entry['train_data']}")
        expert = []
        if req.get("use_feedback", True):
            fb = feedback_rows(labels, neg)
            expert += [{**r, "source": "feedback"} for r in fb["rows"]]
            notes.append(f"feedback rows: {fb['n']} used, {fb['skipped']} skipped (label outside this model)")
        if req.get("extra_upload_id"):
            for r in _rows(req["extra_upload_id"], "clean.jsonl"):
                if r["label"] not in classes:
                    raise ValueError(f"uploaded expert row has label '{r['label']}' which this model does not have")
                expert.append({**r, "source": "expert"})
            notes.append(f"expert upload rows: {len([r for r in expert if r['source'] == 'expert'])}")
        if not expert:
            raise ValueError("no expert rows: raise some flags in the chat or upload labelled rows")
        # expert rows: 80/20 train/val so the gate also sees them; oversample the train part
        rng = random.Random(7); rng.shuffle(expert)
        nv = max(1, len(expert) // 5) if len(expert) >= 5 else 0
        k = int(req.get("oversample", 5))
        for i, r in enumerate(expert):
            r["split"] = "val" if i < nv else "train"
        base_texts = {r["text"] for r in rows}
        rows = [r for r in rows if r["text"] not in {e["text"] for e in expert}]   # expert label wins
        rows += [r for r in expert if r["split"] == "val"]
        rows += [dict(r) for r in expert if r["split"] == "train" for _ in range(k)]
        notes.append(f"expert train rows oversampled x{k}; {len(expert) - nv} train / {nv} val; "
                     f"{len([e for e in expert if e['text'] in base_texts])} replaced base rows")
        return rows, classes, names, neg, unit_classes, notes
    spec = json.load(open(os.path.join(UPLOADS, req["upload_id"], "spec.json")))
    rows = [{**r, "source": "upload"} for r in _rows(req["upload_id"], "clean.jsonl")]
    return rows, spec["classes"], spec["names"], spec["negative"], spec["unit_classes"], notes


def _run_inner(j: dict, enc, active_ckpt: str | None) -> None:
    from train_api import REF_CKPT, fit_probe, mask_probs, metrics, predict_ckpt, train_heads
    req = j["request"]
    base_name = slug(req.get("name") or ("post_" if req.get("post_train") else "model_") + time.strftime("%m%d_%H%M"))
    name, version = next_version(base_name)
    j["stage"] = "loading data"
    rows, classes, names, neg, unit_classes, notes = _load_training_rows(req, active_ckpt)
    labels = list(classes)
    for n in notes:
        _log(j, n)
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if not splits["val"]:
        raise ValueError("no validation rows")
    if not splits["test"]:
        _log(j, "no held-out test rows in this job; reporting validation metrics as test")
        splits["test"] = splits["val"]
    _log(j, f"rows: train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}; classes: {labels}")

    j["stage"] = "embedding"
    uniq = list(dict.fromkeys(state_text(r["text"], INSTRUCTIONS) for s in splits.values() for r in s))
    _log(j, f"embedding {len(uniq)} distinct texts (cached ones are free)")
    E = _embed(j, enc, uniq, "ui_train")
    idx = {t: i for i, t in enumerate(uniq)}
    X = {s: E[[idx[state_text(r["text"], INSTRUCTIONS)] for r in rs]] for s, rs in splits.items()}
    y = {s: np.array([labels.index(r["label"]) for r in rs]) for s, rs in splits.items()}
    C = _embed(j, enc, list(classes.values()), "ui_candidates")

    j["stage"] = "training"
    init = req.get("init", "reference")
    init_ckpt = REF_CKPT if init == "reference" else os.path.join(ROOT, active_ckpt if init == "active" else init)
    epochs = int(req.get("epochs", 30))
    batch = int(min(256, max(16, len(X["train"]) // 10)))   # small datasets need more optimizer steps per epoch
    steps = epochs * ((len(X["train"]) + batch - 1) // batch)
    _log(j, f"training heads from {os.path.basename(init_ckpt)}: epochs {epochs}, lr {req.get('lr', 1e-3)}, "
            f"batch {batch} -> {steps} optimizer steps" + (" (fewer than 300: consider more epochs)" if steps < 300 else ""))
    out_ckpt = os.path.join(CKPT_DIR, name + ".pt")
    def prog(ep, eps, loss, f1):
        j["progress"] = {"stage": "training", "epoch": ep, "epochs": eps, "loss": round(loss, 4), "val_macro_f1": round(f1, 4)}
    hist = train_heads(X, y, C, classes, INSTRUCTIONS, out_ckpt, init_ckpt=init_ckpt,
                       epochs=epochs, lr=float(req.get("lr", 1e-3)), batch=batch, progress=prog)
    best = max(hist, key=lambda h: h["val_macro_f1"])
    _log(j, f"best validation macro-F1 {best['val_macro_f1']:.4f} at epoch {best['epoch']}")

    j["stage"] = "evaluating"
    units = [r.get("unit") for r in splits["test"]]
    p = predict_ckpt(out_ckpt, X["test"], C)
    pm = mask_probs(p, labels, units, unit_classes)
    res = {"labels": labels, "names": names, "label_set": name, "n": len(y["test"]),
           "clm_finetuned": metrics(y["test"], p.argmax(1), labels),
           "clm_finetuned_unit_masked": metrics(y["test"], pm.argmax(1), labels)}
    try:
        pz = predict_ckpt(REF_CKPT, X["test"], C)
        res["clm_zero_shot"] = metrics(y["test"], pz.argmax(1), labels, ci=False)
    except Exception as e:
        _log(j, f"zero-shot skipped: {e}")
    if len(X["train"]) >= 20:
        probe = fit_probe(X["train"], y["train"])
        res["linear_probe"] = metrics(y["test"], probe.predict(X["test"]), labels)
        res["linear_probe"]["C"] = 100.0
    _log(j, f"test macro-F1 {res['clm_finetuned_unit_masked']['macro_f1']:.4f}, accuracy {res['clm_finetuned_unit_masked']['accuracy']:.4f}")

    # gate against the active model, when it shares the label space
    gate = None
    if active_ckpt and os.path.exists(os.path.join(ROOT, active_ckpt)):
        import torch
        ack = torch.load(os.path.join(ROOT, active_ckpt), map_location="cpu")
        a_labels = list(ack["classes"])
        if set(a_labels) == set(labels) and a_labels[0] == labels[0]:
            j["stage"] = "gate"
            Ca = _embed(j, enc, list(ack["classes"].values()), "ui_candidates")
            a_unit = {u: [l for l in ls if l in a_labels] for u, ls in unit_classes.items()}
            remap = np.array([labels.index(l) for l in a_labels])   # active-order index -> candidate-order index
            def active_pred(Xs, us):
                pa = mask_probs(predict_ckpt(os.path.join(ROOT, active_ckpt), Xs, Ca), a_labels, us, a_unit)
                return remap[pa.argmax(1)]
            gate = {"sets": {}, "regressions": []}
            gate["sets"]["test"] = _compare(y["test"], pm.argmax(1), active_pred(X["test"], units), labels)
            for key, path in OOD_FILES.items():
                if os.path.exists(path) and all(l in LABEL_SETS["7way"] for l in labels[1:]):   # positives within the shipped taxonomy
                    rs = [json.loads(l) for l in open(path)]
                    yo = np.array([labels.index(r["label"] if r["label"] in classes else neg) for r in rs])
                    Xo = _embed(j, enc, [state_text(r["text"], INSTRUCTIONS) for r in rs], key)
                    uo = [r.get("unit") for r in rs]
                    pc = mask_probs(predict_ckpt(out_ckpt, Xo, C), labels, uo, unit_classes).argmax(1)
                    gate["sets"][key] = _compare(yo, pc, active_pred(Xo, uo), labels)
            for sname, cmpd in gate["sets"].items():
                if cmpd["candidate"]["macro_f1"] < cmpd["active"]["macro_f1"] - 0.01:
                    gate["regressions"].append(f"{sname}: macro-F1 {cmpd['active']['macro_f1']:.3f} -> {cmpd['candidate']['macro_f1']:.3f}")
                for l in labels[1:]:
                    ra, rc = cmpd["active"]["recall"][l], cmpd["candidate"]["recall"][l]
                    if rc < ra - 0.05:
                        gate["regressions"].append(f"{sname}: recall for {names.get(l, l)} {ra:.2f} -> {rc:.2f}")
            gate["passed"] = not gate["regressions"]
            _log(j, f"gate vs active model: {'passed' if gate['passed'] else str(len(gate['regressions'])) + ' regression(s)'}")
        else:
            _log(j, f"gate skipped: the active model has a different label space ({a_labels} vs {labels})")

    j["stage"] = "report"
    run_dir = os.path.join(TESTS, time.strftime("%Y-%m-%d") + "_ui_" + name)
    os.makedirs(os.path.join(run_dir, "eval_test"), exist_ok=True)
    json.dump(res, open(os.path.join(run_dir, "eval_test", "metrics.json"), "w"), indent=2)
    with open(os.path.join(run_dir, "eval_test", "predictions.jsonl"), "w") as f:
        for r, t, pr in zip(splits["test"], y["test"], pm):
            k = int(pr.argmax()); f.write(json.dumps({**r, "pred": labels[k], "p_pred": round(float(pr[k]), 4), "correct": bool(k == t)}) + "\n")
    json.dump({"labels": labels, "classes": classes, "clm_finetuned": {"history": hist, "macro_f1": res["clm_finetuned"]["macro_f1"],
               "hparams": {"epochs": epochs, "lr": float(req.get("lr", 1e-3)), "batch": batch, "init": os.path.basename(init_ckpt)}},
               "latency": {"clm_heads_ms_per_decision_cpu": float("nan")}, "gate": gate, "notes": notes},
              open(os.path.join(run_dir, "training_metrics.json"), "w"), indent=2)
    if gate:
        json.dump(gate, open(os.path.join(run_dir, "gate.json"), "w"), indent=2)
    try:
        sys.path.insert(0, TESTS)
        import make_reports
        make_reports.main()
        report = os.path.relpath(os.path.join(run_dir, "report.html"), ROOT)
    except Exception as e:
        _log(j, f"report generation failed: {e}"); report = None

    entry = {"name": name, "base_name": base_name, "version": version, "parent": os.path.relpath(init_ckpt, ROOT) if init_ckpt.startswith(ROOT) else os.path.basename(init_ckpt),
             "checkpoint": os.path.relpath(out_ckpt, ROOT), "created": time.time(), "labels": labels,
             "classes": classes, "names": names, "unit_classes": unit_classes,
             "source": "post-training" if req.get("post_train") else "trained from upload",
             "train_data": os.path.join("app", "data", "uploads", req["upload_id"], "clean.jsonl") if req.get("upload_id") else
                           (registry_entry_for(active_ckpt) or {}).get("train_data"),
             "job": j["id"], "report": report, "gate": gate and {"passed": gate["passed"], "regressions": gate["regressions"]},
             "metrics": {"test_macro_f1": res["clm_finetuned_unit_masked"]["macro_f1"],
                         "test_accuracy": res["clm_finetuned_unit_masked"]["accuracy"], "n_test": res["n"]}}
    register(entry)
    j["result"] = {"checkpoint": entry["checkpoint"], "name": name, "metrics": res, "gate": gate, "report": report,
                   "history": hist, "run_dir": os.path.relpath(run_dir, ROOT)}


def _compare(y: np.ndarray, pc: np.ndarray, pa: np.ndarray, labels: list[str]) -> dict:
    from sklearn.metrics import recall_score
    def one(pred):
        rec = recall_score(y, pred, average=None, labels=range(len(labels)), zero_division=0)
        return {"accuracy": float((y == pred).mean()),
                "macro_f1": float(f1_score_(y, pred, len(labels))),
                "recall": {l: float(r) for l, r in zip(labels, rec)}}
    return {"n": int(len(y)), "candidate": one(pc), "active": one(pa)}


def f1_score_(y, pred, k):
    from sklearn.metrics import f1_score
    return f1_score(y, pred, average="macro", labels=range(k), zero_division=0)
