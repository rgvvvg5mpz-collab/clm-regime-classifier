"""Data curation (tab 0): run Claude Code headlessly to research and build a labelled dataset.

Each job is a folder app/data/curation/<job_id>/ that Claude Code works in. It is told
(by an appended system prompt) to write:

  dataset.jsonl   one {"text", "label", "speaker"?, "source"?, "rationale"?} object per line
  classes.json    {"<label>": {"description": "...", "speaker": "user|assistant|either",
                   "negative": true|false}, ...}  (exactly one negative class)
  NOTES.md        what it researched, sources, judgement calls

The CLI runs with web search / fetch and file tools only (no shell), permission mode
acceptEdits (writes outside the job folder are refused), and a dollar budget cap. A
follow-up prompt resumes the same Claude Code session in the same folder, so the user can
iterate ("add 50 harder negatives", "rebalance the classes"). The result can be sent to the
Train tab (as an upload with the class table prefilled) or used as expert rows.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from name_check import scrub  # noqa: E402

JOBS_DIR = os.path.join(ROOT, "app", "data", "curation")
ALLOWED_TOOLS = ["WebSearch", "WebFetch", "Read", "Write", "Edit", "Glob", "Grep"]
DISALLOWED_TOOLS = ["Bash", "NotebookEdit", "Task", "Agent"]
_jobs: dict[str, dict] = {}
_lock = threading.Lock()

CONTRACT = """You are curating a labelled text dataset for a contrastive (System 1) text classifier,
working in the current directory only. Research the topic as needed (web search / fetch), then write:

1. dataset.jsonl: one JSON object per line, {"text": str, "label": str, "speaker": "user"|"assistant" (only
   if the data is conversational), "source": str (URL or "synthetic"), "rationale": str (one short sentence)}.
2. classes.json: {"<label>": {"description": "<one plain sentence defining the class; the model scores
   texts against it>", "speaker": "user"|"assistant"|"either", "negative": true|false}}. Exactly ONE class
   must be the negative / "none of the above" class, with its own examples (it is what lets the
   classifier abstain). Label keys are short snake_case.
3. NOTES.md: what you researched, the sources you used, how labels were decided, borderline cases,
   and known gaps.

Rules: every label in dataset.jsonl must be a key of classes.json; aim for balanced classes and at
least the requested rows per class; vary wording, length and style (avoid near-duplicates and
templates); include hard cases near class boundaries; use invented names for people, firms and
products and only obviously fake personal data (SSN 000-xx-xxxx style); do not copy copyrighted
text verbatim beyond short quotes; never write outside the current directory. When asked to refine,
edit the existing files in place. Finish with a two-line summary of what changed."""


# ------------------------------------------------------------------ CLI discovery
def find_cli(config: dict | None = None) -> str | None:
    """CLAUDE_CODE_BIN env -> config curation.claude_bin -> `claude` on PATH -> newest desktop-app bundle."""
    cand = os.environ.get("CLAUDE_CODE_BIN") or ((config or {}).get("curation") or {}).get("claude_bin")
    if cand:
        if os.path.exists(cand):
            return os.path.abspath(cand)   # the CLI runs inside the job folder, so relative paths would break
        return shutil.which(cand)
    if shutil.which("claude"):
        return shutil.which("claude")
    bundles = glob.glob(os.path.expanduser("~/Library/Application Support/Claude/claude-code/*/*/claude.app/Contents/MacOS/claude"))
    def ver(p):
        try:
            return tuple(int(x) for x in p.split("/claude-code/")[1].split("/")[0].split("."))
        except ValueError:
            return (0,)
    return max(bundles, key=ver) if bundles else None


def status(config: dict | None = None) -> dict:
    cli = find_cli(config)
    if not cli:
        return {"available": False, "logged_in": False, "cli": None,
                "help": "Claude Code CLI not found. Install it (npm i -g @anthropic-ai/claude-code) or set CLAUDE_CODE_BIN."}
    out = {"available": True, "cli": cli, "version": None, "logged_in": False, "auth": None}
    try:
        out["version"] = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
        a = subprocess.run([cli, "auth", "status"], capture_output=True, text=True, timeout=20).stdout
        auth = json.loads(a) if a.strip().startswith("{") else {}
        out["logged_in"] = bool(auth.get("loggedIn")) or bool(os.environ.get("ANTHROPIC_API_KEY"))
        out["auth"] = auth.get("authMethod") or ("api_key" if os.environ.get("ANTHROPIC_API_KEY") else "none")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        out["error"] = str(e)
    if not out["logged_in"]:
        out["help"] = (f'Claude Code is installed but not logged in for this server. In a terminal run:  "{cli}" /login  '
                       "(or start the server with ANTHROPIC_API_KEY set), then press Re-check.")
    return out


# ------------------------------------------------------------------ jobs
def _public(j: dict) -> dict:
    return {k: v for k, v in j.items() if not k.startswith("_")}


def jobs() -> list[dict]:
    with _lock:
        return sorted((_public(j) for j in _jobs.values()), key=lambda j: j["created"], reverse=True)


def job(job_id: str) -> dict | None:
    with _lock:
        j = _jobs.get(job_id)
        return _public(j) if j else None


def start(prompt: str, config: dict | None = None, model: str | None = None, max_turns: int = 40,
          budget_usd: float = 5.0, rows_per_class: int = 50, resume_of: str | None = None) -> dict:
    cli = find_cli(config)
    if not cli:
        raise RuntimeError("Claude Code CLI not found")
    if any(j["status"] == "running" for j in _jobs.values()):
        raise RuntimeError("a curation run is already in progress")
    parent = _jobs.get(resume_of) if resume_of else None
    if resume_of and not parent:
        raise RuntimeError(f"unknown curation job {resume_of}")
    jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5]
    workdir = parent["workdir"] if parent else os.path.join(JOBS_DIR, jid)
    os.makedirs(workdir, exist_ok=True)
    full_prompt = prompt.strip() if parent else f"{prompt.strip()}\n\nTarget: at least {rows_per_class} rows per class."
    cmd = [cli, "-p", full_prompt, "--output-format", "stream-json", "--verbose",
           "--max-turns", str(int(max_turns)), "--max-budget-usd", str(float(budget_usd)),
           "--permission-mode", "acceptEdits", "--strict-mcp-config",
           "--allowedTools", *ALLOWED_TOOLS, "--disallowedTools", *DISALLOWED_TOOLS,
           "--append-system-prompt", CONTRACT]
    if model:
        cmd += ["--model", model]
    if parent and parent.get("session_id"):
        cmd += ["--resume", parent["session_id"]]
    j = {"id": jid, "created": time.time(), "status": "running", "prompt": prompt, "model": model or "default",
         "workdir": os.path.relpath(workdir, ROOT), "parent": resume_of, "events": [], "session_id": None,
         "cost_usd": None, "turns": None, "error": None, "result_text": None, "summary": None,
         "_workdir": workdir}
    with _lock:
        _jobs[jid] = j
    with open(os.path.join(workdir, "prompts.log"), "a") as f:
        f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} job {jid}\n{full_prompt}\n")
    t = threading.Thread(target=_run, args=(j, cmd, workdir), daemon=True)
    j["_thread"] = t
    t.start()
    return _public(j)


def cancel(job_id: str) -> bool:
    j = _jobs.get(job_id)
    p = j and j.get("_proc")
    if p and p.poll() is None:
        p.terminate()
        j["status"] = "cancelled"
        return True
    return False


def _event(j: dict, kind: str, text: str) -> None:
    j["events"].append({"t": time.time(), "kind": kind, "text": text[:400]})
    if len(j["events"]) > 400:
        del j["events"][:100]


def _describe_tool(name: str, inp: dict) -> str:
    if name == "WebSearch":
        return f"searched: {inp.get('query', '')}"
    if name == "WebFetch":
        return f"fetched: {inp.get('url', '')}"
    if name in ("Write", "Edit"):
        return f"{'wrote' if name == 'Write' else 'edited'} {os.path.basename(inp.get('file_path', '') or '')}"
    if name == "Read":
        return f"read {os.path.basename(inp.get('file_path', '') or '')}"
    if name in ("Glob", "Grep"):
        return f"{name.lower()}: {inp.get('pattern', '')}"
    return name


def _run(j: dict, cmd: list[str], workdir: str) -> None:
    env = dict(os.environ)
    try:
        p = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, bufsize=1)
    except OSError as e:
        j["status"] = "failed"; j["error"] = str(e); return
    j["_proc"] = p
    _event(j, "start", "Claude Code started")
    for line in p.stdout:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = d.get("type")
        if typ == "system" and d.get("subtype") == "init":
            j["session_id"] = d.get("session_id") or j["session_id"]
            _event(j, "info", f"model {d.get('model')} · Claude Code {d.get('claude_code_version', '')}")
        elif typ == "assistant":
            j["session_id"] = d.get("session_id") or j["session_id"]
            if d.get("error"):
                _event(j, "error", str(d.get("error")))
            for b in (d.get("message") or {}).get("content") or []:
                if b.get("type") == "text" and b.get("text", "").strip():
                    _event(j, "say", b["text"].strip())
                elif b.get("type") == "tool_use":
                    _event(j, "tool", _describe_tool(b.get("name", ""), b.get("input") or {}))
        elif typ == "result":
            j["cost_usd"] = d.get("total_cost_usd"); j["turns"] = d.get("num_turns")
            j["result_text"] = d.get("result"); j["session_id"] = d.get("session_id") or j["session_id"]
            if d.get("is_error"):
                j["error"] = d.get("result") or "Claude Code reported an error"
            for den in d.get("permission_denials") or []:
                _event(j, "denied", f"denied: {den.get('tool_name')}")
    p.wait()
    err = (p.stderr.read() or "").strip() if p.stderr else ""
    if j["status"] == "cancelled":
        _event(j, "info", "cancelled")
    elif j.get("error") or p.returncode not in (0, None):
        j["status"] = "failed"
        j["error"] = j.get("error") or err[-500:] or f"exit code {p.returncode}"
        _event(j, "error", j["error"])
    else:
        j["status"] = "done"
        _event(j, "done", (j.get("result_text") or "finished").strip())
    j["summary"] = inspect(workdir)


# ------------------------------------------------------------------ results
def inspect(workdir: str) -> dict:
    """Validate what Claude Code wrote: dataset.jsonl rows, classes.json, NOTES.md."""
    out = {"files": sorted(os.listdir(workdir)) if os.path.isdir(workdir) else [], "errors": [], "warnings": [],
           "n_rows": 0, "by_label": {}, "classes": None, "notes": None, "preview": []}
    ds = os.path.join(workdir, "dataset.jsonl")
    if not os.path.exists(ds):
        out["errors"].append("dataset.jsonl was not written")
        return out
    rows, bad = [], 0
    for line in open(ds):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if not isinstance(r, dict) or not str(r.get("text", "")).strip() or not str(r.get("label", "")).strip():
                bad += 1; continue
            rows.append(r)
        except json.JSONDecodeError:
            bad += 1
    if bad:
        out["warnings"].append(f"{bad} malformed lines skipped")
    scrubbed = sum(1 for r in rows if scrub(str(r["text"])) != str(r["text"]))
    if scrubbed:
        out["warnings"].append(f"{scrubbed} rows contain a blocked real-firm name; they will be replaced on import")
    texts = collections.Counter(str(r["text"]).strip() for r in rows)
    dups = sum(c - 1 for c in texts.values() if c > 1)
    if dups:
        out["warnings"].append(f"{dups} duplicate texts")
    out["n_rows"] = len(rows)
    out["by_label"] = dict(collections.Counter(str(r["label"]) for r in rows).most_common())
    out["preview"] = rows[:12]
    cp = os.path.join(workdir, "classes.json")
    if os.path.exists(cp):
        try:
            classes = json.load(open(cp))
            out["classes"] = classes
            negs = [k for k, v in classes.items() if isinstance(v, dict) and v.get("negative")]
            if len(negs) != 1:
                out["errors"].append(f"classes.json must mark exactly one negative class (found {len(negs)})")
            missing = sorted(set(out["by_label"]) - set(classes))
            if missing:
                out["errors"].append(f"labels with no class definition: {missing}")
            empty = sorted(set(classes) - set(out["by_label"]))
            if empty:
                out["warnings"].append(f"classes with no rows: {empty}")
        except json.JSONDecodeError as e:
            out["errors"].append(f"classes.json is not valid JSON: {e.msg}")
    else:
        out["errors"].append("classes.json was not written")
    np_ = os.path.join(workdir, "NOTES.md")
    if os.path.exists(np_):
        out["notes"] = open(np_).read()[:20000]
    small = [l for l, n in out["by_label"].items() if n < 10]
    if small:
        out["warnings"].append(f"classes under 10 rows (the Train tab will refuse them): {small}")
    return out


def to_upload(job_id: str):
    """Stage the curated dataset as a Train-tab upload with the class table prefilled."""
    import training
    j = _jobs.get(job_id)
    if not j:
        raise FileNotFoundError(f"unknown curation job {job_id}")
    workdir = j["_workdir"]
    rows = [json.loads(l) for l in open(os.path.join(workdir, "dataset.jsonl")) if l.strip()]
    data = "\n".join(json.dumps({"text": r["text"], "label": r["label"], "speaker": r.get("speaker") or ""}) for r in rows)
    summary = training.parse_upload(f"curated_{job_id}.jsonl", data.encode())
    classes = json.load(open(os.path.join(workdir, "classes.json")))
    summary["spec"] = {k: {"name": k.replace("_", " "), "description": v.get("description", ""),
                           "speaker": {"user": "client", "assistant": "assistant"}.get(v.get("speaker"), "either"),
                           "negative": bool(v.get("negative"))} for k, v in classes.items()}
    summary["example"] = f"curated_{job_id}"
    return summary
