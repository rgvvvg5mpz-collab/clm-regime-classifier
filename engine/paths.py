"""Shared locations, defaults and device choice for the CLASP engine."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("CLASP_MODELS_DIR", os.path.join(ROOT, "models"))           # trained checkpoints (*.pt)
CACHE_DIR = os.environ.get("CLASP_CACHE_DIR", os.path.join(ROOT, "cache", "embeddings"))  # text-keyed encoder store
REF_CKPT = os.environ.get("CLM_CKPT", os.path.expanduser("~/.cache/clm/CLM_v0.1-8B.pt"))  # published CLM heads (clm-download)
ENCODER = os.environ.get("CLASP_ENCODER", "Qwen/Qwen3-8B")

# The question appended to every input before encoding (CLM's state = context + question).
# Checkpoints store the one they were trained with; this is only the default for new models.
GENERIC_INSTRUCTIONS = "Which of the defined classes best describes this text, if any?"


def device() -> str:
    """CUDA, then Apple Silicon (MPS), then CPU. Override with CLASP_DEVICE."""
    d = os.environ.get("CLASP_DEVICE")
    if d:
        return d
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
