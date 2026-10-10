"""Write docs/demo/README.md from docs/demo/build_info.json (chapters, duration, voice). Run after make_demo.py."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
info = json.load(open(os.path.join(ROOT, "docs", "demo", "build_info.json")))
mp4 = os.path.join(ROOT, info["video"])
m, s = divmod(info["duration_s"], 60)
chapters = "\n".join(f"| {int(t // 60)}:{int(t % 60):02d} | {title} |" for t, title in info["chapters"])
open(os.path.join(ROOT, "docs", "demo", "README.md"), "w").write(f"""# CLASP walkthrough video

**[Download `clasp_demo.mp4`](https://github.com/rgvvvg5mpz-collab/clm-regime-classifier/releases/download/demo-v1/clasp_demo.mp4)**
([release `demo-v1`](https://github.com/rgvvvg5mpz-collab/clm-regime-classifier/releases/tag/demo-v1)): {int(m)} min {int(s)} s, 1080p H.264,
narrated by macOS {info["voice"]}, with an English subtitle track (also here as [`clasp_demo.srt`](clasp_demo.srt)).
{os.path.getsize(mp4) / 1e6:.0f} MB, so it is a release asset rather than a file in git; `scripts/setup.sh --with-video` saves it here.

The first third explains what CLASP is and how it works (slides); the rest is a recorded walkthrough of the
live workbench, tab by tab, using the regulatory example.

| Time | Chapter |
|---|---|
{chapters}

**Slides:** [`slides.html`](slides.html) is the deck used in the video, and works on its own as a presentation
(open in a browser; arrow keys or click to advance).

## Regenerate it

The video is built from the running app, so it stays in step with the UI:

```bash
.venv/bin/pip install -r scripts/demo/requirements-demo.txt   # playwright, imageio-ffmpeg, pillow (once)
scripts/run_ui.sh --mock-curation                               # in another terminal; wait for "Model ready"
.venv/bin/python scripts/demo/make_demo.py                      # ~6 min: capture, voice, render
.venv/bin/python scripts/demo/write_readme.py                   # refresh this file's chapter list
cd docs/demo && shasum -a 256 clasp_demo.mp4 > clasp_demo.mp4.sha256 \
  && gh release upload demo-v1 clasp_demo.mp4 clasp_demo.mp4.sha256 clasp_demo.srt --clobber   # publish the new build
```

`docs/demo/*.mp4` is gitignored; the build writes the video locally and the release holds the published copy.

- Narration and the screen for each line live in `SCENES` in `scripts/demo/make_demo.py`; pronunciation
  fixes (e.g. "Qwen" → "Kwen") are in `SPEECH`. Captions keep the written form.
- `scripts/demo/render.py` does the camera moves, spotlights, crossfades and chapter cards.
- Options: `--voice "Zoe (Premium)"`, `--rate 170`, `--height 2160` (4K), `--skip-curation`, `--reuse-shots`
  (re-voice / re-render without re-capturing), `--keep` (keep the models and flags the recording creates;
  by default everything it changes in `app/data/`, `models/` and `Tests/` is restored).
- Needs macOS (for `say` and the premium voices: System Settings → Accessibility → Spoken Content → System voice
  → Manage voices) and Google Chrome.
""")
print("wrote docs/demo/README.md")
