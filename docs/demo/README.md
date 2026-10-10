# CLASP walkthrough video

**[Download `clasp_demo.mp4`](https://github.com/rgvvvg5mpz-collab/clm-regime-classifier/releases/download/demo-v1/clasp_demo.mp4)**
([release `demo-v1`](https://github.com/rgvvvg5mpz-collab/clm-regime-classifier/releases/tag/demo-v1)): 9 min 27 s, 1080p H.264,
narrated by macOS Ava (Premium), with an English subtitle track (also here as [`clasp_demo.srt`](clasp_demo.srt)).
53 MB, so it is a release asset rather than a file in git; `scripts/setup.sh --with-video` saves it here.

The first third explains what CLASP is and how it works (slides); the rest is a recorded walkthrough of the
live workbench, tab by tab, using the regulatory example.

| Time | Chapter |
|---|---|
| 0:00 | What CLASP is |
| 0:21 | System-1 classifiers |
| 0:53 | How a text is scored |
| 1:29 | The contrastive loss |
| 2:07 | The four-tab loop |
| 2:41 | The worked example |
| 3:22 | Getting started |
| 3:52 | A tour of the workbench |
| 4:19 | 0 · Data curation |
| 5:23 | 1 · Train |
| 6:40 | 2 · Chat & flag |
| 7:39 | 3 · Post-train |
| 8:47 | The classes |
| 9:05 | Where to go next |

**Slides:** [`slides.html`](slides.html) is the deck used in the video, and works on its own as a presentation
(open in a browser; arrow keys or click to advance).

## Regenerate it

The video is built from the running app, so it stays in step with the UI:

```bash
.venv/bin/pip install -r scripts/demo/requirements-demo.txt   # playwright, imageio-ffmpeg, pillow (once)
scripts/run_ui.sh --mock-curation                               # in another terminal; wait for "Model ready"
.venv/bin/python scripts/demo/make_demo.py                      # ~6 min: capture, voice, render
.venv/bin/python scripts/demo/write_readme.py                   # refresh this file's chapter list
cd docs/demo && shasum -a 256 clasp_demo.mp4 > clasp_demo.mp4.sha256   && gh release upload demo-v1 clasp_demo.mp4 clasp_demo.mp4.sha256 clasp_demo.srt --clobber   # publish the new build
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
