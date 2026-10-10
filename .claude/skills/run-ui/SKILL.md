---
name: run-ui
description: Start the CLASP workbench web UI for this repo (setup on first run, then the server on http://localhost:8710). Use when the user asks to run, start, launch, open or demo the UI / app / workbench.
---

# Run the CLASP workbench UI

1. **Set up (fresh clone only).** If `.venv/bin/python` or `models/clm_regime_5way.pt` is missing, run
   `scripts/setup.sh --skip-encoder` with the shell tool first (a few minutes: `.venv`, requirements, CLM into
   `third_party/`, the reference heads, and the two shipped models from the GitHub releases). Tell the user it is happening.

2. **Start it.**
   - In the Claude desktop app: `preview_start` with the launch configuration **`clasp-ui`** (`.claude/launch.json`;
     it runs `.venv/bin/python -m uvicorn app.server:app` directly, because the app's preview runner may not be
     allowed to execute shell scripts in the user's folders). Use **`clasp-ui-mock-curation`** if the user wants the
     Data curation tab to work without a logged-in Claude Code.
   - Anywhere else: run `scripts/run_ui.sh` in the background (it does step 1 itself, then serves; it never exits).

3. **Wait for the model.** The server loads the Qwen3-8B encoder in the background. Poll
   `curl -s localhost:8710/api/health` until `"state": "ready"` (about 20 s; the very first time it also
   downloads ~16 GB, which can take a long while, so report progress instead of waiting silently).
   - `missing_model` → run `scripts/setup.sh`.
   - `error` → read the server log; the usual cause is memory (one Qwen3-8B per 24 GB machine: stop any other
     CLASP server or evaluation first).

4. **Hand over.** Give the user http://localhost:8710 and one line on where to start: the welcome card's
   **Play the demo**, or the **Train** tab's walkthrough (the regulatory example, about 15 minutes).

Do not ask for API keys. The demo (scripted transcripts), training and post-training need none. Live LLM chat
needs `ANTHROPIC_API_KEY` or a key typed in the sidebar; the Data curation tab needs Claude Code logged in
(`claude /login`) on this machine. Mention these only if the user wants those features.
