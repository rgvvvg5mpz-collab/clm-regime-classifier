"""Inference pipeline: text -> regime label + probabilities with the fine-tuned CLM heads.

    from inference import RegimeClassifier
    clf = RegimeClassifier()                                   # local Qwen3-8B on MPS/CPU
    clf = RegimeClassifier(emb_url="http://gpu:8090/v1/embeddings")   # or a vLLM pooling server
    clf.classify(["Can you text me on my cell instead?"], unit="client_message")

CLI:
    python inference.py "message text" [--unit client_message|assistant_response]
    python inference.py --jsonl in.jsonl --out preds.jsonl     # rows need a "text" field

``unit`` restricts the candidate set to the regimes that apply to that speaker
(FINRA 2210 / Reg BI judge the assistant's words; 4530 / 17a-3/4 / S-P / S-ID judge
the client's). Leave it unset to score all seven classes.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from classes import CLASSES as DEFAULT_CLASSES, INSTRUCTIONS as DEFAULT_INSTRUCTIONS  # noqa: E402
from clm.heads import HeadPair  # noqa: E402
from clm.schema import state_text  # noqa: E402

DEFAULT_CKPT = os.environ.get("REGIME_CKPT", os.path.join(HERE, "checkpoints", "clm_regime_7way.pt"))
NEGATIVE = "compliant"
UNIT_CLASSES = {
    "client_message": ["compliant", "finra_4530", "sec_17a3_17a4", "reg_sp", "reg_sid"],
    "assistant_response": ["compliant", "finra_2210", "reg_bi"],
}
RULE_NAMES = {
    "compliant": "No issue",
    "finra_2210": "FINRA Rule 2210 (communications)",
    "reg_bi": "Regulation Best Interest",
    "finra_4530": "FINRA Rule 4530 (customer complaint)",
    "sec_17a3_17a4": "SEC 17a-3/17a-4 (books & records)",
    "reg_sp": "Regulation S-P (privacy)",
    "reg_sid": "Regulation S-ID (identity theft)",
}


@dataclass
class Prediction:
    label: str
    rule: str
    flagged: bool
    confidence: float
    probabilities: dict[str, float]


class RegimeClassifier:
    def __init__(self, checkpoint: str = DEFAULT_CKPT, emb_url: str | None = None,
                 emb_model: str = "qwen3-8b", device: str = "cpu"):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"{checkpoint} not found - download clm_regime_7way.pt from the GitHub release "
                "into regime_clf/checkpoints/ or set REGIME_CKPT")
        ck = torch.load(checkpoint, map_location="cpu")
        self.classes: dict[str, str] = ck.get("classes", DEFAULT_CLASSES)
        self.instructions: str = ck.get("instructions", DEFAULT_INSTRUCTIONS)
        self.labels = list(self.classes)
        self.heads = HeadPair("regime", checkpoint, device).ensure()
        if emb_url:
            from clm.embedder import Embedder
            self.embedder = Embedder(url=emb_url, model=emb_model)
        else:
            from mps_embedder import MPSEmbedder
            self.embedder = MPSEmbedder()
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
            keep = UNIT_CLASSES[unit]
            lg = np.where(np.isin(self.labels, keep)[None, :], lg, -np.inf)
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        out = []
        for row in p:
            j = int(row.argmax())
            label = self.labels[j]
            out.append(Prediction(label=label, rule=RULE_NAMES.get(label, label), flagged=label != NEGATIVE,
                                  confidence=float(row[j]),
                                  probabilities={l: round(float(v), 5) for l, v in zip(self.labels, row)}))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?")
    ap.add_argument("--unit", choices=list(UNIT_CLASSES))
    ap.add_argument("--jsonl", help="input JSONL with a 'text' field (and optional 'unit')")
    ap.add_argument("--out", help="output JSONL (default: stdout)")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--emb-url", default=os.environ.get("CLM_EMB_URL"))
    a = ap.parse_args()
    clf = RegimeClassifier(a.checkpoint, emb_url=a.emb_url)
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
    print(f"[inference] {len(rows)} rows in {time.perf_counter() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
