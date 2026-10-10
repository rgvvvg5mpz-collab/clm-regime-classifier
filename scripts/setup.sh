#!/usr/bin/env bash
# One-time setup for a fresh clone of CLASP. Safe to re-run: every step is skipped when done.
#
#   scripts/setup.sh                 # everything, including the 16 GB encoder download
#   scripts/setup.sh --skip-encoder  # let the encoder download on first use instead
#   scripts/setup.sh --with-video    # also download the walkthrough video (~53 MB) into docs/demo/
#
# Needs: Python 3.10+, git, curl, ~20 GB free disk. Runs best on Apple Silicon with 24 GB+
# unified memory, or a CUDA GPU with 24 GB+.
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_ENCODER=0
WITH_VIDEO=0
for a in "$@"; do
  case "$a" in
    --skip-encoder) SKIP_ENCODER=1 ;;
    --with-video) WITH_VIDEO=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.14 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "Python 3.10+ not found (set PYTHON=/path/to/python3)" >&2; exit 1; }

step "Python environment (.venv, using $("$PY" --version))"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "ok"

step "CLM library (third_party/CLM)"
[ -d third_party/CLM/.git ] || git clone -q https://github.com/Contrastive-LM/CLM.git third_party/CLM
.venv/bin/pip install -q --no-deps -e third_party/CLM   # --no-deps: skip vLLM (CUDA-only); the in-process encoder replaces it
echo "ok"

step "Reference CLM heads (~75 MB, ~/.cache/clm/)"
.venv/bin/clm-download >/dev/null && echo "ok"

step "Shipped example models (models/, ~75 MB each)"
REPO_URL="$(git remote get-url origin 2>/dev/null | sed -E 's#^git@github.com:#https://github.com/#; s#\.git$##' || true)"
case "$REPO_URL" in https://github.com/*) ;; *) REPO_URL="https://github.com/rgvvvg5mpz-collab/clasp" ;; esac
mkdir -p models
fetch() {   # fetch <release tag> <asset>, then verify the published checksum
  local tag="$1" name="$2" f="models/$2"
  if [ ! -s "$f" ]; then
    curl -fL --progress-bar -o "$f.part" "$REPO_URL/releases/download/$tag/$name" && mv "$f.part" "$f"
  fi
  local want; want="$(curl -fsL "$REPO_URL/releases/download/$tag/$name.sha256" 2>/dev/null | cut -d' ' -f1 || true)"
  if [ -n "$want" ] && [ "$want" != "$(sha256 "$f")" ]; then echo "checksum mismatch for $f; delete it and re-run" >&2; exit 1; fi
  echo "$f ok"
}
fetch v1.1 clm_regime_5way.pt
fetch v1.0 clm_regime_7way.pt

if [ "$WITH_VIDEO" = 1 ]; then
  step "Walkthrough video (docs/demo/clasp_demo.mp4, ~53 MB)"
  f=docs/demo/clasp_demo.mp4
  if [ ! -s "$f" ]; then curl -fL --progress-bar -o "$f.part" "$REPO_URL/releases/download/demo-v1/clasp_demo.mp4" && mv "$f.part" "$f"; fi
  want="$(curl -fsL "$REPO_URL/releases/download/demo-v1/clasp_demo.mp4.sha256" 2>/dev/null | cut -d' ' -f1 || true)"
  if [ -n "$want" ] && [ "$want" != "$(sha256 "$f")" ]; then echo "checksum mismatch for $f (a newer local build?); delete it to re-download" >&2; else echo "$f ok"; fi
fi

step "Encoder Qwen/Qwen3-8B (~16 GB, Hugging Face cache)"
if [ "$SKIP_ENCODER" = 1 ]; then
  echo "skipped: it downloads the first time the UI loads a model"
else
  .venv/bin/hf download Qwen/Qwen3-8B --include "*.json" "*.safetensors" "*.txt" >/dev/null && echo "ok"
fi

step "Hardware check"
.venv/bin/python - <<'EOF'
import os, torch
mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30 if hasattr(os, "sysconf") else 0
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {dev}, memory: {mem:.0f} GB")
if dev == "cpu":
    print("warning: no GPU found; the 8B encoder will run on CPU and be slow")
elif dev == "mps" and mem < 20:
    print("warning: under 20 GB of unified memory; the 16 GB encoder may swap")
EOF

step "Done. Start the UI with:  scripts/run_ui.sh   (then open http://localhost:8710)"
