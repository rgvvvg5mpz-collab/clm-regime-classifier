# CLAUDE.md

Orientation for Claude Code sessions in this repo. Human docs are the HTML files in `docs/`.

## What this is
A 7-way **regime classifier** for broker-dealer GenAI chat. It labels a message as one of `compliant`, `finra_2210`, `reg_bi`, `finra_4530`, `sec_17a3_17a4`, `reg_sp` or `reg_sid`. The model is **CLM-8B**: two trained MLP projection heads (4096→1536→1536→512, 18.9 M params) over a **frozen Qwen3-8B** encoder (last-token pooled, L2-normalised). Score = `exp(logit_scale)·cos(state_head(msg+question), action_head(class description))`, then softmax over the 7 classes.

## Current results (2026-10-05)
- In-dist test (RegModels v2 test, n=2,501): fine-tuned CLM **0.977** macro-F1 [0.970, 0.983]; linear probe 0.955; zero-shot CLM 0.064.
- OOD hard (Fable 5.1, n=315): fine-tuned CLM **0.732** [0.680, 0.778]; probe 0.747 (CIs overlap). Main errors: S-ID→S-P, Reg BI→2210, violations predicted compliant (22/270).
- Selected config: lr 1e-3, 30 epochs, batch 256, AdamW wd 0.01, one-cycle, softmax-CE over candidates, heads initialised from `CLM_v0.1-8B.pt`, best-val epoch kept.

## Layout
- `regime_clf/classes.py`: label space, candidate texts, state question. **Editing this changes model inputs: retrain.**
- `regime_clf/build_dataset.py`: RegModels `slm-supervisory-suite/data/*_v2.jsonl` → `data/{train,val,test}.jsonl` (keeps source splits; adds `unit`; scrubs blocked names via `tools/name_check.py`).
- `regime_clf/mps_embedder.py`: in-process Qwen3-8B (MPS/CPU) drop-in for `clm.embedder.Embedder`; `embed_cached()` is a text-keyed fp16 store in `regime_clf/emb_cache/` (gitignored, ~200 MB).
- `regime_clf/run_experiment.py`: zero-shot + fine-tune + probe; `--out` dir gets `metrics.json` and the checkpoint. `sweep.sh`: LR sweep.
- `regime_clf/evaluate.py`: any labelled JSONL → metrics with bootstrap CIs and slices (`ood_axis`, `difficulty`, `unit`).
- `regime_clf/inference.py`: production API (`RegimeClassifier.classify(texts, unit=)`) + CLI. `UNIT_CLASSES` masks classes by speaker.
- `regime_clf/ood/`: `SPEC.md` (generation spec), `raw_*.jsonl` (3 Fable agents × 105), `build_ood.py` → `ood_hard_v1.jsonl`.
- `app/`: FastAPI chat (`server.py`), providers (`llm.py`: Anthropic SDK + OpenAI-compatible), `config.yaml`, `static/`, `mock_llm.py`. Writes `app/data/turns.jsonl` and `app/data/feedback.jsonl` (FP/FN flags).
- `Tests/<YYYY-MM-DD>_<name>/`: one folder per run, each with `report.html`; `Tests/make_reports.py` regenerates all reports + `Tests/index.html`.
- `CLM/` (gitignored): clone of github.com/Contrastive-LM/CLM, installed `--no-deps -e`.

## Commands
```bash
cd regime_clf
../.venv/bin/python build_dataset.py
../.venv/bin/python run_experiment.py --epochs 30 --lr 1e-3 --out ../Tests/<date>_x/run
../.venv/bin/python evaluate.py --data ood/ood_hard_v1.jsonl --out ../Tests/<date>_ood
../.venv/bin/python inference.py "text" --unit client_message
cd .. && .venv/bin/python Tests/make_reports.py
.venv/bin/python -m uvicorn app.server:app --port 8710      # + app.mock_llm:app --port 8799 for offline
.venv/bin/python tools/name_check.py                         # must report 0 hits before any push
```

## Rules and gotchas
- Never write real financial-firm names into repo text. Use invented names and run `tools/name_check.py` (it stores blocked names ROT13-encoded).
- One Qwen3-8B per 24 GB machine: stop the UI server before running evaluations.
- Don't use raw encoder embeddings for similarity or dedup (cos ≈ 0.997 between unrelated states); use TF-IDF or the head space.
- Checkpoints embed their own `classes`/`instructions`, and `inference.py` prefers them over `classes.py`.
- New test runs go in a new dated `Tests/` folder; don't overwrite old runs.
- Weights (`*.pt`) are gitignored; ship `regime_clf/checkpoints/clm_regime_7way.pt` as a GitHub release asset.
- For Claude calls use the `anthropic` SDK (see `app/llm.py`); the default model is `claude-opus-5-5`.

## Next work
See `docs/developer_guide.html` §6. The top items: SME adjudication of the S-ID/S-P boundary, recall-first thresholds, calibration, pytest + CI, harder training data, and consuming `feedback.jsonl` for gated retraining.
