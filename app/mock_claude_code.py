#!/usr/bin/env python3
"""Offline stand-in for the Claude Code CLI, for developing and testing the Data curation tab.

    CLAUDE_CODE_BIN="$PWD/app/mock_claude_code.py" .venv/bin/python -m uvicorn app.server:app --port 8710

It accepts the flags the tab passes, emits the same stream-json event shapes (system init,
assistant text / tool_use, result), and writes a small sentiment dataset that follows the
curation contract (dataset.jsonl, classes.json, NOTES.md) in the working directory. A
follow-up (--resume) appends rows.
"""
import json
import os
import random
import sys
import time
import uuid

args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("0.0.0 (mock Claude Code)"); sys.exit(0)
if args[:2] == ["auth", "status"]:
    print(json.dumps({"loggedIn": True, "authMethod": "mock"})); sys.exit(0)
resume = args[args.index("--resume") + 1] if "--resume" in args else None
sid = resume or str(uuid.uuid4())


def emit(d):
    print(json.dumps(d), flush=True); time.sleep(0.15)


emit({"type": "system", "subtype": "init", "session_id": sid, "model": "mock-model", "claude_code_version": "0.0.0"})
emit({"type": "assistant", "session_id": sid, "message": {"content": [{"type": "text", "text": "I'll research customer-review sentiment and draft a dataset."}]}})
emit({"type": "assistant", "session_id": sid, "message": {"content": [{"type": "tool_use", "name": "WebSearch", "input": {"query": "product review sentiment labelling guidelines"}}]}})
emit({"type": "assistant", "session_id": sid, "message": {"content": [{"type": "tool_use", "name": "WebFetch", "input": {"url": "https://example.com/labelling-guide"}}]}})
random.seed(len(open("dataset.jsonl").readlines()) if os.path.exists("dataset.jsonl") else 0)
pos = ["Arrived early and works perfectly", "Best purchase I've made this year", "The fabric is soft and the fit is great",
       "Support fixed my issue in minutes", "Battery easily lasts two days", "Exactly as described, would buy again"]
neg = ["Broke after a week of light use", "The strap snapped on day one", "Customer service never replied",
       "Smells strongly of chemicals", "Charger stopped working within a month", "Much smaller than the photos suggest"]
neu = ["Delivered on Tuesday", "It comes in three colours", "The box contained the manual and a cable",
       "I ordered the medium size", "Assembly took about twenty minutes", "It replaced my old one"]
rows = []
for label, pool in (("positive", pos), ("negative", neg), ("neutral", neu)):
    for i in range(20):
        rows.append({"text": f"{random.choice(pool)}{random.choice(['.', '!', ' - honestly.', ', no complaints.' if label != 'negative' else ', not happy.'])} (#{random.randint(100, 999)})",
                     "label": label, "source": "synthetic", "rationale": f"clear {label} wording"})
with open("dataset.jsonl", "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
emit({"type": "assistant", "session_id": sid, "message": {"content": [{"type": "tool_use", "name": "Write", "input": {"file_path": os.path.abspath("dataset.jsonl")}}]}})
json.dump({"neutral": {"description": "None of the other classes applies: a factual statement with no opinion.", "speaker": "either", "negative": True},
           "positive": {"description": "The reviewer is satisfied or praises the product.", "speaker": "either", "negative": False},
           "negative": {"description": "The reviewer is dissatisfied or reports a problem.", "speaker": "either", "negative": False}},
          open("classes.json", "w"), indent=2)
emit({"type": "assistant", "session_id": sid, "message": {"content": [{"type": "tool_use", "name": "Write", "input": {"file_path": os.path.abspath("classes.json")}}]}})
with open("NOTES.md", "w") as f:
    f.write("# Notes\n\nMock run: synthetic product-review sentiment rows (positive / negative / neutral).\n")
emit({"type": "assistant", "session_id": sid, "message": {"content": [{"type": "tool_use", "name": "Write", "input": {"file_path": os.path.abspath("NOTES.md")}}]}})
n = len(open("dataset.jsonl").readlines())
emit({"type": "result", "subtype": "success", "is_error": False, "session_id": sid, "num_turns": 6, "total_cost_usd": 0.0,
      "result": f"Wrote {len(rows)} rows ({n} total) across 3 classes.\nNeutral is the negative class.", "permission_denials": []})
