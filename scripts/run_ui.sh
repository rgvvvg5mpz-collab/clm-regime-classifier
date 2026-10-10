#!/usr/bin/env bash
# Start the CLASP workbench UI (runs scripts/setup.sh first if the clone is not set up yet).
#
#   scripts/run_ui.sh                   # http://localhost:8710
#   scripts/run_ui.sh --port 8800
#   scripts/run_ui.sh --mock-curation   # Data curation tab uses an offline stand-in for Claude Code
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8710}"
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --mock-curation) export CLAUDE_CODE_BIN="$PWD/app/mock_claude_code.py"; shift ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import clm, fastapi" 2>/dev/null || [ ! -s models/clm_regime_5way.pt ]; then
  echo "First run: setting up the clone (scripts/setup.sh)..."
  scripts/setup.sh --skip-encoder
fi

if command -v lsof >/dev/null && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Is CLASP already running? Open http://localhost:$PORT, or pass --port." >&2
  exit 1
fi

echo "CLASP workbench: http://localhost:$PORT"
echo "The encoder (Qwen3-8B) loads in the background: ~20 s, or longer the first time while it downloads (~16 GB)."
exec .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT"
