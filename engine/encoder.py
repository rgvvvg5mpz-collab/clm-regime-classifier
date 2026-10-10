"""In-process Qwen3-8B encoder (CUDA / Apple Silicon / CPU): a drop-in for
clm.embedder.Embedder that needs no vLLM server, plus a content-addressed embedding store.

Matches the reference recipe: raw text, no special tokens, last-token pooling of
the final (post-norm) hidden state, L2-normalised.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from clm.embedder import l2

from .paths import CACHE_DIR, ENCODER as MODEL, device as default_device


class MPSEmbedder:
    def __init__(self, model: str = MODEL, max_tokens: int = 2048, batch: int = 16,
                 device: str | None = None, dtype=torch.bfloat16):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, dtype=dtype).to(self.device).eval()
        self.max_tokens, self.batch = max_tokens, batch
        self.cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> np.ndarray:
        ids = [self.tok(t, add_special_tokens=False)["input_ids"][: self.max_tokens] or [220] for t in texts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))   # length-bucketed batches
        out = np.zeros((len(ids), self.model.config.hidden_size), np.float32)
        pad = self.tok.pad_token_id or 0
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            L = max(len(ids[i]) for i in idx)
            inp = torch.full((len(idx), L), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                inp[r, : len(ids[i])] = torch.tensor(ids[i]); mask[r, : len(ids[i])] = 1
            h = self.model(input_ids=inp.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
            last = mask.sum(1) - 1
            out[idx] = h[torch.arange(len(idx)), last.to(self.device)].float().cpu().numpy()
        return l2(out)

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """Same contract as clm.embedder.Embedder.embed."""
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            for t, v in zip(todo, self._encode(todo)):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts]), 0

    def healthy(self) -> bool:
        return True


def _store_paths():
    return os.path.join(CACHE_DIR, "store.npy"), os.path.join(CACHE_DIR, "store_texts.json")


def _load_store() -> tuple[np.ndarray, dict[str, int]]:
    vp, tp = _store_paths()
    if not os.path.exists(vp):
        return np.zeros((0, 4096), np.float16), {}
    texts = json.load(open(tp))
    return np.load(vp), {t: i for i, t in enumerate(texts)}


def _save_store(vecs: np.ndarray, index: dict[str, int]) -> None:
    vp, tp = _store_paths()
    np.save(vp + ".tmp.npy", vecs)
    json.dump(sorted(index, key=index.get), open(tp + ".tmp", "w"))
    os.replace(vp + ".tmp.npy", vp); os.replace(tp + ".tmp", tp)


def embed_cached(embedder: "MPSEmbedder | None", texts: list[str], name: str = "", chunk: int = 512) -> np.ndarray:
    """Embed ``texts`` through a content-addressed float16 store in emb_cache/.

    Only texts never seen before are encoded (resumably, ``chunk`` at a time), so editing a
    few rows re-embeds just those rows. Raises AssertionError when texts are missing and no
    embedder is given.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    vecs, index = _load_store()
    todo = [t for t in dict.fromkeys(texts) if t not in index]
    if todo:
        assert embedder is not None, f"{len(todo)} texts not in the embedding store ({name}); pass an embedder"
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            vecs = np.concatenate([vecs, embedder._encode(part).astype(np.float16)])
            for t in part:
                index[t] = len(index)
            _save_store(vecs, index)
            print(f"[embed] {name}: {min(i + chunk, len(todo))}/{len(todo)} new", flush=True)
    return vecs[[index[t] for t in texts]].astype(np.float32)
