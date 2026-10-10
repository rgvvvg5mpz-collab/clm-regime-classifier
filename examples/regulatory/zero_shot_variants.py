"""Zero-shot ablation: does CLM-v0.1-8B separate the 7 regimes without training?

States are fixed (cached); only the candidate texts vary, so each variant costs 7 encodes.
``+centered`` subtracts each class's mean logit over the (unlabelled) test states — a
label-free prior correction for candidates that attract every state.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from classes import CLASSES, INSTRUCTIONS, LABEL_SET, LABELS

assert LABEL_SET == "7way", "the hand-written candidate variants below are for the 7-way label set"
from clm.schema import state_text
from engine.encoder import MPSEmbedder, embed_cached
from run_experiment import REF_CKPT, clm_logits, load_heads

VARIANTS = {
    "descriptions": list(CLASSES.values()),
    "short_names": ["No regulatory issue", "FINRA Rule 2210 communications with the public",
                    "Regulation Best Interest", "FINRA Rule 4530 customer complaint",
                    "SEC Rule 17a-4 books and records", "Regulation S-P customer privacy",
                    "Regulation S-ID identity theft red flags"],
    "plain_situations": ["This message is fine and raises no compliance issue.",
                         "The assistant exaggerated, promised returns, or left out risks.",
                         "The assistant told the customer what to buy, or which account to open.",
                         "The customer is complaining that something went wrong with their account.",
                         "The customer wants to talk by text or personal phone, or not be recorded.",
                         "The customer shared their social security number or asked about privacy.",
                         "Someone may be pretending to be the customer or has stolen their identity."],
}


def main(out_path: str):
    rows = [json.loads(l) for l in open("data/test.jsonl")]
    y = np.array([LABELS.index(r["label"]) for r in rows])
    X = embed_cached(None, [state_text(r["text"], INSTRUCTIONS) for r in rows], "state_test")
    enc = MPSEmbedder()
    sh, ah, ls, _ = load_heads(REF_CKPT, "cpu")
    res = {}
    with torch.no_grad():
        for name, cands in VARIANTS.items():
            C = torch.from_numpy(embed_cached(enc, cands, f"cand_{name}"))
            Xt = torch.from_numpy(X)
            for mode in ("heads", "raw"):
                lg = (clm_logits(sh, ah, ls, Xt, C) if mode == "heads" else 100.0 * Xt @ C.T).numpy()
                for centered in (False, True):
                    l2 = lg - lg.mean(0, keepdims=True) if centered else lg
                    p = l2.argmax(1)
                    key = f"{name}/{mode}{'+centered' if centered else ''}"
                    res[key] = {"accuracy": float(accuracy_score(y, p)),
                                "macro_f1": float(f1_score(y, p, average="macro"))}
                    print(f"{key:40s} acc={res[key]['accuracy']:.3f} macro_f1={res[key]['macro_f1']:.3f}", flush=True)
    json.dump(res, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/zero_shot_variants.json")
