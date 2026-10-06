"""Evaluate the regime classifiers on any labelled JSONL (in-distribution test or OOD).

    python evaluate.py --data ood/ood_hard_v1.jsonl --out ../Tests/2026-10-05_ood_fable

Rows need ``text`` and ``label``; an optional ``unit`` (client_message / assistant_response)
enables the unit-masked variant. Optional ``ood_axis`` / ``difficulty`` fields are used
for sliced metrics. Compares: fine-tuned CLM (with and without unit masking), zero-shot
CLM, and a linear probe refit on the cached train embeddings.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from classes import CLASSES, INSTRUCTIONS, LABELS
from clm.schema import state_text
from inference import UNIT_CLASSES
from mps_embedder import MPSEmbedder, embed_cached
from run_experiment import REF_CKPT, clm_predict, load_heads, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
FT_CKPT = os.path.join(HERE, "checkpoints", "clm_regime_7way.pt")
PROBE_C = 100.0


def embed(texts: list[str], name: str) -> np.ndarray:
    try:
        return embed_cached(None, texts, name)
    except AssertionError:
        return embed_cached(MPSEmbedder(), texts, name)


def mask_units(p: np.ndarray, units: list[str | None]) -> np.ndarray:
    p = p.copy()
    for i, u in enumerate(units):
        if u in UNIT_CLASSES:
            keep = np.isin(LABELS, UNIT_CLASSES[u])
            p[i, ~keep] = 0
            p[i] /= p[i].sum()
    return p


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """95% percentile bootstrap CI for accuracy and macro-F1 (resampling rows)."""
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        acc.append((y[i] == pred[i]).mean())
        f1.append(f1_score(y[i], pred[i], average="macro"))
    return {"accuracy_95ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
            "macro_f1_95ci": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]}


def sliced(rows: list[dict], y: np.ndarray, pred: np.ndarray, field: str) -> dict:
    out = {}
    for v in sorted({r.get(field) for r in rows if r.get(field)}):
        idx = np.array([r.get(field) == v for r in rows])
        out[v] = {"n": int(idx.sum()), "accuracy": float((y[idx] == pred[idx]).mean()),
                  "macro_f1": float(f1_score(y[idx], pred[idx], average="macro",
                                             labels=sorted(set(y[idx])), zero_division=0))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=FT_CKPT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    rows = [json.loads(l) for l in open(a.data)]
    y = np.array([LABELS.index(r["label"]) for r in rows])
    units = [r.get("unit") for r in rows]
    X = embed([state_text(r["text"], INSTRUCTIONS) for r in rows], os.path.basename(a.data))
    C = embed(list(CLASSES.values()), "candidates")

    probs = {}
    sh, ah, ls, _ = load_heads(a.checkpoint, dev)
    probs["clm_finetuned"] = clm_predict(sh, ah, ls, X, C, dev)
    probs["clm_finetuned_unit_masked"] = mask_units(probs["clm_finetuned"], units)
    sh, ah, ls, _ = load_heads(REF_CKPT, dev)
    probs["clm_zero_shot"] = clm_predict(sh, ah, ls, X, C, dev)

    train = [json.loads(l) for l in open(os.path.join(HERE, "data", "train.jsonl"))]
    Xtr = embed([state_text(r["text"], INSTRUCTIONS) for r in train], "state_train")
    ytr = np.array([LABELS.index(r["label"]) for r in train])
    probe = LogisticRegression(C=PROBE_C, max_iter=3000).fit(Xtr, ytr)
    probs["linear_probe"] = probe.predict_proba(X)

    out = {"data": os.path.relpath(a.data, HERE), "n": len(rows), "labels": LABELS,
           "label_counts": dict(collections.Counter(r["label"] for r in rows))}
    for name, p in probs.items():
        pred = p.argmax(1)
        m = metrics(y, pred)
        m.update(bootstrap_ci(y, pred))
        m["by_ood_axis"] = sliced(rows, y, pred, "ood_axis")
        m["by_difficulty"] = sliced(rows, y, pred, "difficulty")
        m["by_unit"] = sliced(rows, y, pred, "unit")
        out[name] = m
        print(f"{name:28s} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"95% CI [{m['macro_f1_95ci'][0]:.3f}, {m['macro_f1_95ci'][1]:.3f}]", flush=True)
    json.dump(out, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)

    best = probs["clm_finetuned_unit_masked"]
    with open(os.path.join(a.out, "predictions.jsonl"), "w") as f:
        for r, t, p in zip(rows, y, best):
            j = int(p.argmax())
            f.write(json.dumps({**r, "pred": LABELS[j], "p_pred": round(float(p[j]), 4),
                                "correct": bool(j == t)}) + "\n")


if __name__ == "__main__":
    main()
