"""7-way regime classification on the RegModels data with CLM-8B.

  1. clm_zero_shot  reference CLM-v0.1-8B heads, class descriptions as candidates
  2. clm_finetuned  heads initialised from the reference checkpoint, fine-tuned with
                    softmax-CE over the 7 candidates (the repo's ``choice --loss softce``)
  3. linear_probe   logistic regression on the frozen Qwen3-8B state embeddings (baseline)

Encoder embeddings are computed once (Qwen3-8B, engine/encoder.py) and cached in cache/embeddings/.
The fine-tuned head is saved in CLM checkpoint format, so Engine / clm-serve load it.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from classes import CKPT_NAME, CLASSES, DATA_DIR, INSTRUCTIONS, LABEL_SET, LABELS
from clm.heads import HeadPair
from clm.schema import state_text
from engine.paths import MODELS_DIR
from engine.trainer import REF_CKPT, clm_logits, clm_predict, load_heads  # noqa: F401  (re-exported for the other scripts)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CKPT = os.path.join(MODELS_DIR, CKPT_NAME)
RESULTS = os.path.join(HERE, "results")


def load_split(split: str) -> list[dict]:
    return [json.loads(l) for l in open(os.path.join(HERE, DATA_DIR, f"{split}.jsonl"))]


def embeddings(splits: dict[str, list[dict]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    from engine.encoder import MPSEmbedder, embed_cached
    texts = {s: [state_text(r["text"], INSTRUCTIONS) for r in rows] for s, rows in splits.items()}
    cand_texts = list(CLASSES.values())
    try:   # all cached -> skip loading the 16 GB encoder
        X = {s: embed_cached(None, t, f"state_{s}") for s, t in texts.items()}
        C = embed_cached(None, cand_texts, "candidates")
    except AssertionError:
        enc = MPSEmbedder()
        X = {s: embed_cached(enc, t, f"state_{s}") for s, t in texts.items()}
        C = embed_cached(enc, cand_texts, "candidates")
        t0 = time.perf_counter()
        enc._encode([state_text(r["text"], INSTRUCTIONS) for r in splits["test"][:64]])
        json.dump({"encoder_ms_per_message": 1000 * (time.perf_counter() - t0) / 64},
                  open(os.path.join(RESULTS, "encoder_latency.json"), "w"))
        del enc
    return X, C


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class": classification_report(y, pred, labels=range(len(LABELS)), target_names=LABELS,
                                           output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y, pred, labels=range(len(LABELS))).tolist(),
    }


# ----------------------------------------------------------------- CLM heads
def finetune(X: dict, y: dict, C: np.ndarray, device: str, epochs: int, lr: float, batch: int, seed: int):
    torch.manual_seed(seed)
    sh, ah, log_scale, cfg = load_heads(REF_CKPT, device)
    params = [*sh.parameters(), *ah.parameters(), log_scale]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * ((len(X["train"]) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Xtr, ytr = torch.from_numpy(X["train"]).to(device), torch.from_numpy(y["train"]).to(device)
    Ct = torch.from_numpy(C).to(device)
    best, best_f1, history = None, -1.0, []
    for ep in range(epochs):
        sh.train(); ah.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(clm_logits(sh, ah, log_scale, Xtr[idx], Ct), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        val_pred = clm_predict(sh, ah, log_scale, X["val"], C, device).argmax(1)
        vf1 = f1_score(y["val"], val_pred, average="macro")
        history.append({"epoch": ep + 1, "train_loss": tot / len(Xtr), "val_macro_f1": float(vf1)})
        print(f"[finetune] epoch {ep + 1}/{epochs} loss={tot / len(Xtr):.4f} val_macro_f1={vf1:.4f}", flush=True)
        if vf1 > best_f1:
            best_f1 = vf1
            best = (copy.deepcopy(sh.state_dict()), copy.deepcopy(ah.state_dict()), log_scale.detach().clone())
    sh.load_state_dict(best[0]); ah.load_state_dict(best[1]); log_scale.data = best[2]
    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    torch.save({"state_head": {k: v.cpu() for k, v in best[0].items()},
                "action_head": {k: v.cpu() for k, v in best[1].items()},
                "logit_scale": best[2].cpu(), "cfg": cfg,
                "classes": CLASSES, "instructions": INSTRUCTIONS}, OUT_CKPT)
    return sh, ah, log_scale, history


def head_latency_ms(C: np.ndarray, x: np.ndarray) -> float:
    """Per-decision cost of the CLM part (heads + 7-way softmax), candidates pre-projected."""
    hp = HeadPair("ft", OUT_CKPT, "cpu").ensure()
    zc = hp.project_actions(C)
    for _ in range(5):
        hp.project_states(x[:1]) @ zc.T
    t0 = time.perf_counter()
    for i in range(200):
        (hp.scale * hp.project_states(x[i % len(x):i % len(x) + 1]) @ zc.T).softmax(-1)
    return 1000 * (time.perf_counter() - t0) / 200


def main():
    global RESULTS, OUT_CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20933)
    ap.add_argument("--out", default=RESULTS, help="results directory (e.g. ../Tests/<date>_in_distribution)")
    a = ap.parse_args()
    RESULTS = a.out
    OUT_CKPT = os.path.join(RESULTS, CKPT_NAME)
    os.makedirs(RESULTS, exist_ok=True)
    from engine.paths import device as _device
    device = _device()

    splits = {s: load_split(s) for s in ("train", "val", "test")}
    y = {s: np.array([LABELS.index(r["label"]) for r in rows]) for s, rows in splits.items()}
    X, C = embeddings(splits)
    print({s: v.shape for s, v in X.items()}, C.shape, flush=True)

    out: dict = {"labels": LABELS, "label_set": LABEL_SET, "n": {s: len(v) for s, v in y.items()}}

    # 1. zero-shot
    sh, ah, ls, _ = load_heads(REF_CKPT, device)
    p0 = clm_predict(sh, ah, ls, X["test"], C, device)
    out["clm_zero_shot"] = metrics(y["test"], p0.argmax(1))
    print(f"[zero-shot] test acc={out['clm_zero_shot']['accuracy']:.4f} "
          f"macro_f1={out['clm_zero_shot']['macro_f1']:.4f}", flush=True)

    # 2. fine-tuned heads
    sh, ah, ls, hist = finetune(X, y, C, device, a.epochs, a.lr, a.batch, a.seed)
    p1 = clm_predict(sh, ah, ls, X["test"], C, device)
    out["clm_finetuned"] = metrics(y["test"], p1.argmax(1))
    out["clm_finetuned"]["history"] = hist
    out["clm_finetuned"]["hparams"] = {k: v for k, v in vars(a).items() if k != "out"}
    print(f"[finetuned] test acc={out['clm_finetuned']['accuracy']:.4f} "
          f"macro_f1={out['clm_finetuned']['macro_f1']:.4f}", flush=True)

    # 3. linear probe on raw encoder embeddings (C picked on val)
    best = None
    for c in (1.0, 10.0, 100.0):
        lr = LogisticRegression(C=c, max_iter=2000).fit(X["train"], y["train"])
        f = f1_score(y["val"], lr.predict(X["val"]), average="macro")
        if best is None or f > best[0]:
            best = (f, c, lr)
    out["linear_probe"] = metrics(y["test"], best[2].predict(X["test"]))
    out["linear_probe"]["C"] = best[1]
    print(f"[probe] C={best[1]} test acc={out['linear_probe']['accuracy']:.4f} "
          f"macro_f1={out['linear_probe']['macro_f1']:.4f}", flush=True)

    # misclassified test rows (fine-tuned), for review
    with open(os.path.join(RESULTS, "errors_finetuned.jsonl"), "w") as f:
        for r, t, p, pr in zip(splits["test"], y["test"], p1.argmax(1), p1):
            if t != p:
                f.write(json.dumps({"text": r["text"], "gold": LABELS[t], "pred": LABELS[p],
                                    "p_pred": float(pr[p]), "fine_labels": r["fine_labels"]}) + "\n")

    out["latency"] = {"clm_heads_ms_per_decision_cpu": head_latency_ms(C, X["test"])}
    lat = os.path.join(RESULTS, "encoder_latency.json")
    if os.path.exists(lat):
        out["latency"].update(json.load(open(lat)))
    json.dump(out, open(os.path.join(RESULTS, "metrics.json"), "w"), indent=2)
    print(json.dumps(out["latency"]), flush=True)


if __name__ == "__main__":
    main()
