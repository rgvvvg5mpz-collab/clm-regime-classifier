"""Inference: text -> class + probabilities with a trained CLASP checkpoint.

    from engine.classifier import Classifier
    clf = Classifier("models/clm_regime_5way.pt")                      # local encoder (CUDA / MPS / CPU)
    clf = Classifier(ckpt, emb_url="http://gpu:8090/v1/embeddings")     # or a vLLM pooling server
    clf.classify(["Can you text me on my cell instead?"], unit="client_message")

CLI (from the repo root):
    .venv/bin/python -m engine.classifier --checkpoint models/clm_regime_5way.pt "message text" --unit client_message
    .venv/bin/python -m engine.classifier --checkpoint <ckpt> --jsonl in.jsonl --out preds.jsonl

A checkpoint carries its own class descriptions and question. Optional per-model metadata
(display names, which classes each speaker can receive) is read from ``<checkpoint>.meta.json``
or passed in; without it every class applies to every speaker. ``unit`` ("client_message" =
a user turn, "assistant_response" = an assistant turn) restricts the softmax to that speaker's
classes; the negative class (always first) is allowed for both.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from clm.heads import HeadPair
from clm.schema import state_text

from .paths import GENERIC_INSTRUCTIONS, MODELS_DIR

UNITS = ("client_message", "assistant_response")


@dataclass
class Prediction:
    label: str
    rule: str            # display name of the label (kept as "rule" for API compatibility)
    flagged: bool        # label is not the negative class
    confidence: float
    probabilities: dict[str, float]


def read_meta(checkpoint: str) -> dict:
    p = os.path.splitext(checkpoint)[0] + ".meta.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def write_meta(checkpoint: str, names: dict | None, unit_classes: dict | None, **extra) -> None:
    meta = {"names": names or {}, "unit_classes": unit_classes, **extra}
    json.dump(meta, open(os.path.splitext(checkpoint)[0] + ".meta.json", "w"), indent=2)


class Classifier:
    def __init__(self, checkpoint: str, emb_url: str | None = None, emb_model: str = "qwen3-8b",
                 device: str = "cpu", embedder=None, unit_classes: dict | None = None, names: dict | None = None):
        """``embedder`` lets a caller reuse an already-loaded encoder (the UI swaps checkpoints
        without reloading Qwen3-8B). ``unit_classes`` / ``names`` override the .meta.json sidecar."""
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"{checkpoint} not found - run scripts/setup.sh to download the shipped models")
        ck = torch.load(checkpoint, map_location="cpu")
        if "classes" not in ck:
            raise ValueError(f"{checkpoint} has no class descriptions; it was not trained by CLASP")
        meta = read_meta(checkpoint)
        self.classes: dict[str, str] = dict(ck["classes"])
        self.instructions: str = ck.get("instructions", GENERIC_INSTRUCTIONS)
        self.labels = list(self.classes)
        self.negative = self.labels[0]
        self.names = {**{l: l for l in self.labels}, **(meta.get("names") or {}), **(names or {})}
        src = unit_classes if unit_classes is not None else meta.get("unit_classes")
        if not src:
            src = {u: self.labels[1:] for u in UNITS}
        self.unit_classes = {u: [self.negative] + [l for l in ls if l in self.classes and l != self.negative]
                             for u, ls in src.items()}
        self.heads = HeadPair("clasp", checkpoint, device).ensure()
        if embedder is not None:
            self.embedder = embedder
        elif emb_url:
            from clm.embedder import Embedder
            self.embedder = Embedder(url=emb_url, model=emb_model)
        else:
            from .encoder import MPSEmbedder
            self.embedder = MPSEmbedder()
        self.checkpoint = checkpoint
        cand, _ = self.embedder.embed(list(self.classes.values()))
        self._zc = self.heads.project_actions(cand)

    def logits(self, texts: list[str]) -> np.ndarray:
        states = [state_text(t, self.instructions) for t in texts]
        emb, _ = self.embedder.embed(states)
        with torch.no_grad():
            return (self.heads.scale * self.heads.project_states(emb) @ self._zc.T).cpu().numpy()

    def classify(self, texts: list[str], unit: str | None = None) -> list[Prediction]:
        lg = self.logits(texts)
        if unit:
            keep = self.unit_classes[unit]
            lg = np.where(np.isin(self.labels, keep)[None, :], lg, -np.inf)
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        out = []
        for row in p:
            j = int(row.argmax())
            label = self.labels[j]
            out.append(Prediction(label=label, rule=self.names.get(label, label), flagged=label != self.negative,
                                  confidence=float(row[j]),
                                  probabilities={l: round(float(v), 5) for l, v in zip(self.labels, row)}))
        return out


RegimeClassifier = Classifier   # backwards-compatible name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?")
    ap.add_argument("--unit", choices=UNITS)
    ap.add_argument("--jsonl", help="input JSONL with a 'text' field (and optional 'unit')")
    ap.add_argument("--out", help="output JSONL (default: stdout)")
    ap.add_argument("--checkpoint", default=os.path.join(MODELS_DIR, "clm_regime_5way.pt"))
    ap.add_argument("--emb-url", default=os.environ.get("CLM_EMB_URL"))
    a = ap.parse_args()
    clf = Classifier(a.checkpoint, emb_url=a.emb_url)
    rows = ([json.loads(l) for l in open(a.jsonl)] if a.jsonl else [{"text": a.text, "unit": a.unit}])
    t0 = time.perf_counter()
    preds = []
    for unit in {r.get("unit") or a.unit for r in rows}:
        idx = [i for i, r in enumerate(rows) if (r.get("unit") or a.unit) == unit]
        for i, p in zip(idx, clf.classify([rows[i]["text"] for i in idx], unit=unit)):
            preds.append((i, p))
    preds.sort()
    f = open(a.out, "w") if a.out else sys.stdout
    for i, p in preds:
        f.write(json.dumps({**rows[i], **asdict(p)}) + "\n")
    print(f"[classifier] {len(rows)} rows in {time.perf_counter() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
