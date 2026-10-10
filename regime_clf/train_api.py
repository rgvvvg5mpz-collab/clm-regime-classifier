"""Training and evaluation primitives that take the label space as data.

``run_experiment.py`` reads its classes from ``classes.py`` (selected by REGIME_LABEL_SET);
this module is the same recipe with the classes passed in, so the chat UI can train a
model on an uploaded dataset with user-defined classes. Checkpoints are the same format
(state_head / action_head / logit_scale / cfg + classes / instructions), so ``inference.py``
and ``clm.Engine`` load them unchanged.
"""
from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from run_experiment import REF_CKPT, clm_logits, clm_predict, load_heads  # noqa: F401  (re-exported)


def metrics(y: np.ndarray, pred: np.ndarray, labels: list[str], ci: bool = True) -> dict:
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(len(labels)), zero_division=0)),
        "per_class": classification_report(y, pred, labels=range(len(labels)), target_names=labels,
                                           output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y, pred, labels=range(len(labels))).tolist(),
    }
    if ci and len(y) >= 20:
        rng = np.random.default_rng(0)
        acc, f1 = [], []
        for _ in range(500):
            i = rng.integers(0, len(y), len(y))
            acc.append((y[i] == pred[i]).mean())
            f1.append(f1_score(y[i], pred[i], average="macro", labels=range(len(labels)), zero_division=0))
        out["accuracy_95ci"] = [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]
        out["macro_f1_95ci"] = [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]
    return out


def mask_probs(p: np.ndarray, labels: list[str], units: list[str | None], unit_classes: dict[str, list[str]]) -> np.ndarray:
    """Zero out classes the speaker cannot trigger (negative class always allowed) and renormalise."""
    p = p.copy()
    for i, u in enumerate(units):
        if u in unit_classes:
            keep = np.isin(labels, [labels[0], *unit_classes[u]])
            p[i, ~keep] = 0
            s = p[i].sum()
            p[i] = p[i] / s if s > 0 else p[i]
    return p


def train_heads(X: dict[str, np.ndarray], y: dict[str, np.ndarray], C: np.ndarray, classes: dict[str, str],
                instructions: str, out_path: str, init_ckpt: str = REF_CKPT, epochs: int = 30, lr: float = 1e-3,
                batch: int = 256, seed: int = 20933, device: str | None = None, progress=None) -> list[dict]:
    """Fine-tune state/action heads with softmax-CE over the class candidates.

    ``X``/``y`` need ``train`` and ``val``; the checkpoint kept is the best validation macro-F1
    epoch. ``progress(epoch, epochs, loss, val_f1)`` is called after every epoch.
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    sh, ah, log_scale, cfg = load_heads(init_ckpt, device)
    params = [*sh.parameters(), *ah.parameters(), log_scale]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = max(1, epochs * ((len(X["train"]) + batch - 1) // batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Xtr, ytr = torch.from_numpy(X["train"]).to(device), torch.from_numpy(y["train"]).to(device)
    Ct = torch.from_numpy(C).to(device)
    labels = list(classes)
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
        vf1 = float(f1_score(y["val"], val_pred, average="macro", labels=range(len(labels)), zero_division=0))
        history.append({"epoch": ep + 1, "train_loss": tot / len(Xtr), "val_macro_f1": vf1})
        if progress:
            progress(ep + 1, epochs, tot / len(Xtr), vf1)
        if vf1 > best_f1:
            best_f1 = vf1
            best = (copy.deepcopy(sh.state_dict()), copy.deepcopy(ah.state_dict()), log_scale.detach().clone())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_head": {k: v.cpu() for k, v in best[0].items()},
                "action_head": {k: v.cpu() for k, v in best[1].items()},
                "logit_scale": best[2].cpu(), "cfg": cfg,
                "classes": classes, "instructions": instructions,
                "history": history, "trained_at": time.time(), "init_ckpt": os.path.basename(init_ckpt)}, out_path)
    return history


def predict_ckpt(ckpt: str, X: np.ndarray, C: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    sh, ah, ls, _ = load_heads(ckpt, device)
    return clm_predict(sh, ah, ls, X, C, device)


def fit_probe(Xtr: np.ndarray, ytr: np.ndarray, C: float = 100.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
