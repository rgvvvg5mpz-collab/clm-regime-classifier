# CLAUDE.md

Orientation for Claude Code sessions in this repo. Human docs are the HTML files in `docs/` and `app/DEV_README.html`.

## If you are asked to run the UI

Use the project skill **`run-ui`** (`.claude/skills/run-ui/SKILL.md`). Short version:
- Fresh clone (no `.venv` or no `models/clm_regime_5way.pt`): run `scripts/setup.sh --skip-encoder` with the shell tool first (`.venv`, requirements, CLM into `third_party/`, reference heads, the two shipped models from GitHub releases v1.0/v1.1). The Qwen3-8B encoder (~16 GB) downloads on first model load.
- Desktop app: `preview_start` with launch config **`clasp-ui`** (runs uvicorn via `.venv/bin/python` directly; the preview runner may not be allowed to exec shell scripts in user folders). Elsewhere: `scripts/run_ui.sh` in the background (does the setup itself).
- Poll `curl -s localhost:8710/api/health` until `"state": "ready"`, then give the user http://localhost:8710.
- No API key is needed for the demo, training or post-training. Don't ask for keys unless the user wants live LLM chat (`ANTHROPIC_API_KEY`) or the Data curation tab (Claude Code logged in; or `clasp-ui-mock-curation` offline).

## What this is

**CLASP** (Contrastive Labelling And System-1 Prediction): a workbench for text classifiers built on CLM-8B. Two trained MLP heads (4096→1536→1536→512, 18.9 M params) sit over a **frozen Qwen3-8B** encoder (last-token pooled, L2-normalised). Score = `exp(logit_scale)·cos(state_head(enc(text + question)), action_head(enc(class description)))`, then a softmax over the classes the speaker can receive. Training is softmax cross-entropy over the class descriptions (a contrastive loss: a row's own class is the positive, the other classes are negatives). The UI tabs: 0 Data curation (Claude Code), 1 Train, 2 Chat & flag, 3 Post-train, ? The classes.

The **regulatory regime screener** (`examples/regulatory/`) is the shipped worked example, not the product. UI text, tooltips and instructions must stay generic (user turn / assistant turn, classes, negative class). Regulatory wording belongs only in the example walkthrough, the example's files and the docs about the example.

## Layout

- `engine/`: the generic core, importable as `engine.*` from the repo root.
  - `paths.py`: models dir, embedding cache, reference heads, `GENERIC_INSTRUCTIONS`, `device()` (CUDA → MPS → CPU).
  - `encoder.py`: in-process Qwen3-8B (`MPSEmbedder`, a drop-in for `clm.embedder.Embedder`). `embed_cached()` is a text-keyed fp16 store in `cache/embeddings/`.
  - `trainer.py`: `load_heads`, `clm_logits`, `clm_predict`, `train_heads` (classes passed as data), `metrics`, `mask_probs`, `fit_probe`.
  - `classifier.py`: `Classifier(checkpoint, …).classify(texts, unit=)` + CLI (`python -m engine.classifier`). Class descriptions and the question come from the checkpoint. Display names and speaker masks come from `<ckpt>.meta.json`, or args, or default to all classes for every speaker.
- `app/`: the workbench.
  - `server.py`: routes; loads the model in the background at startup; `/api/health` gives the load state; hot-swaps checkpoints with the encoder kept.
  - `training.py`: uploads, validation, training jobs, the promotion gate, the registry (`app/data/models.json`, auto-migrates old paths), and the Project data folders (`PROJECT_DATA`).
  - `curation.py`: runs the Claude Code CLI headlessly for tab 0. Tools are limited, `acceptEdits`, budget cap, `--resume` for follow-ups.
  - `llm.py`: Anthropic SDK + OpenAI-compatible adapters.
  - `static/`: `app.js` (chat + shared), `train.js`, `curate.js`, `styles.css`.
  - `mock_llm.py`, `mock_claude_code.py`: offline stand-ins.
  - `config.yaml`: presets; `classifier.checkpoint` is the default model (`models/clm_regime_5way.pt`).
- `examples/regulatory/`: the example.
  - `classes.py`: label sets `7way` / `5way` via `REGIME_LABEL_SET`, `INSTRUCTIONS`, `UNIT_CLASSES`. Importing it puts the repo root on `sys.path`.
  - `build_dataset.py`: from the external RegModels repo.
  - Research scripts: `run_experiment.py`, `evaluate.py`, `sweep.sh`, `zero_shot_variants.py`, `sanity_check.py`. Run them from that folder with `../../.venv/bin/python`.
  - `data/`, `data_5way/`: the splits.
  - `ood/`: `SPEC.md`, the Fable-written raw parts, `build_ood.py`, `ood_*_v1.jsonl`.
  - `samples/`: the Train-tab examples, with `.spec.json` files that carry classes and the example's question.
  - `transcripts/demo_conversations.json`: the chat demo.
- `models/`: `*.pt` (gitignored; the shipped ones are release assets) + `*.meta.json` (tracked).
- `data/`: the user's files (`*.private.*` gitignored).
- `scripts/`: `setup.sh`, `run_ui.sh`.
- `Tests/<YYYY-MM-DD>_<name>/`: one folder per run (`report.html`, optional hand-written `notes.html`). `Tests/make_reports.py` regenerates all reports, `index.html` and `METRICS.md`; quote numbers from `METRICS.md`.
- `third_party/CLM` (gitignored): github.com/Contrastive-LM/CLM, installed `--no-deps -e`.

## Current results (the regulatory example)

- **5-way** (no_flag + FINRA 2210, Reg BI, FINRA 4530, SEC 17a-3/4; S-P/S-ID rows are out-of-class `no_flag`): test **0.983**, OOD hard **0.800**, OOD low/medium **0.830** macro-F1. The 7-way model collapsed to these five classes scores 0.985 / 0.822 / 0.839, so the gain is from the label change, not the retraining.
- **7-way**: test **0.977**, OOD hard **0.732**, OOD low/medium **0.748**.
- Main weaknesses:
  - Reg BI → FINRA 2210 (Reg BI recall 0.53 on hard OOD).
  - S-ID → S-P (7-way).
  - Explicit (low-difficulty) messages are no easier than hard ones: the drop comes from distribution shift.
- Recipe: lr 1e-3, 30 epochs, batch 256 (smaller for small data), AdamW wd 0.01, one-cycle, heads initialised from `CLM_v0.1-8B.pt`, best-validation epoch kept.

## Commands

```bash
scripts/run_ui.sh [--port N] [--mock-curation]
.venv/bin/python -m engine.classifier --checkpoint models/clm_regime_5way.pt "text" --unit client_message
.venv/bin/python Tests/make_reports.py
.venv/bin/python tools/name_check.py                 # must report 0 hits before any push
cd examples/regulatory && ../../.venv/bin/python evaluate.py --data ood/ood_hard_v1.jsonl --out ../../Tests/<date>_ood
```

## Rules and gotchas

- Never write real financial-firm names into repo text. Use invented names and run `tools/name_check.py`; it stores the blocked names ROT13-encoded.
- One Qwen3-8B per 24 GB machine. Stop the UI server before running evaluations from the shell.
- Don't use raw encoder embeddings for similarity or dedup: cos ≈ 0.997 between unrelated states. Use TF-IDF or the head space.
- Each model has its own question (stored in the checkpoint). Changing a question means re-encoding, which the cache handles, but compare models on their own questions (the gate does).
- New test runs go in a new dated `Tests/` folder; don't overwrite old runs.
- Ship new release models as GitHub release assets with a `.sha256`, and add a tracked `models/<name>.meta.json`.
- Compare label sets like for like: collapse predictions before crediting a retrain.
- For Claude calls use the `anthropic` SDK (`app/llm.py`); the default model is `claude-opus-5-5`.

## Next work

See `docs/developer_guide.html` §6. Top items:
- SME adjudication of the S-ID/S-P boundary
- Recall-first thresholds and calibration
- pytest + CI
- More varied training data
- Wiring `feedback.jsonl` into a scheduled, gated retrain
