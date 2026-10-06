"""Blocked real-firm names: the scrub list for the dataset build and a repo-wide check.

Names are stored ROT13-encoded so the repository text never contains them.

    .venv/bin/python tools/name_check.py          # scan tracked + untracked (non-ignored) files; exit 1 on a hit
"""
from __future__ import annotations

import codecs
import re
import subprocess
import sys
from pathlib import Path

# (encoded blocked name, fictional replacement)
_BLOCKED = [("inathneq", "Harborline")]
SCRUB = [(re.compile(codecs.decode(enc, "rot13"), re.I), repl) for enc, repl in _BLOCKED]
ROOT = Path(__file__).resolve().parent.parent


def scrub(text: str) -> str:
    for pat, repl in SCRUB:
        text = pat.sub(repl, text)
    return text


def main() -> int:
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    hits = 0
    for rel in filter(None, files):
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size > 50_000_000:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for pat, _ in SCRUB:
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                print(f"{rel}:{line}: blocked name")
                hits += 1
    print(f"name check: {hits} hit(s) in {len([f for f in files if f])} files")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
