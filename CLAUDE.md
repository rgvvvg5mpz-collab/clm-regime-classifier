# CLAUDE.md

Orientation for Claude Code sessions in this repo. Human docs are the HTML files in `docs/`.

## What this is
A **regime classifier** for broker-dealer GenAI chat with two label sets (`regime_clf/classes.py`, env `REGIME_LABEL_SET`, default `7way`):
- `7way`: `compliant`, `finra_2210`, `reg_bi`, `finra_4530`, `sec_17a3_17a4`, `reg_sp`, `reg_sid` → `data/`, `clm_regime_7way.pt` (release v1.0)
- `5way`: `no_flag`, `finra_2210`, `reg_bi`, `finra_4530`, `sec_17a3_17a4`; all Reg S-P / S-ID rows are out-of-class samples labelled `no_flag` → `data_5way/`, `clm_regime_5way.pt` (release v1.1, chat UI default)
The negative class is always first; `INSTRUCTIONS` is shared so cached state embeddings are reused across sets. The model is **CLM-8B**: two trained MLP projection heads (4096→1536→1536→512, 18.9 M params) over a **frozen Qwen3-8B** encoder (last-token pooled, L2-normalised). Score = `exp(logit_scale)·cos(state_head(msg+question), action_head(class description))`, then softmax over the 7 classes.

## Current results
5-way (2026-10-06): test **0.983** [0.977, 0.989]; OOD hard **0.800** [0.748, 0.846]; OOD low/medium **0.830** [0.770, 0.882]; probe 0.960 / 0.833 / 0.817. The 7-way model collapsed to 5 classes scores 0.985 / 0.822 / 0.839, so the OOD gain is from dropping S-P/S-ID, not retraining. Out-of-class S-P/S-ID rows left unflagged 87% (hard) / 80% (low-med); false flags are identity-theft reports → 4530/17a-3/4. Reg BI → 2210 unchanged (16/45, 9/30).

7-way (2026-10-05):
- In-dist test (RegModels v2 test, n=2,501): fine-tuned CLM **0.977** macro-F1 [0.970, 0.983]; linear probe 0.955; zero-shot CLM 0.064.
- OOD hard/very-hard (Fable 5.1, n=315): fine-tuned CLM **0.732** [0.680, 0.778]; probe 0.747 (CIs overlap). Main errors: S-ID→S-P, Reg BI→2210, violations predicted compliant (22/270).
- OOD low/medium (Fable 5.1, n=210): fine-tuned CLM **0.748** [0.688, 0.801]; probe 0.720. Low rows 73.3% < medium 80.0%: difficulty does not drive the drop, distribution shift does. Same two confusions (S-ID F1 0.38; blunt Reg BI → 2210 at p≈0.99).
- Selected config: lr 1e-3, 30 epochs, batch 256, AdamW wd 0.01, one-cycle, softmax-CE over candidates, heads initialised from `CLM_v0.1-8B.pt`, best-val epoch kept.

## Layout
- `regime_clf/classes.py`: label space, candidate texts, state question. **Editing this changes model inputs: retrain.**
- `regime_clf/build_dataset.py`: RegModels `slm-supervisory-suite/data/*_v2.jsonl` → `data/` or `data_5way/` `{train,val,test}.jsonl` (keeps source splits; adds `unit`; dropped tracks → negative; scrubs blocked names via `tools/name_check.py`).
- `regime_clf/mps_embedder.py`: in-process Qwen3-8B (MPS/CPU) drop-in for `clm.embedder.Embedder`; `embed_cached()` is a text-keyed fp16 store in `regime_clf/emb_cache/` (gitignored, ~200 MB).
- `regime_clf/run_experiment.py`: zero-shot + fine-tune + probe; `--out` dir gets `metrics.json` and the checkpoint. `sweep.sh`: LR sweep.
- `regime_clf/evaluate.py`: any labelled JSONL → metrics with bootstrap CIs and slices (`ood_axis`, `difficulty`, `unit`, `source_label`). Labels outside the active set map to the negative class (`classes.to_label_set`), so the 7-way OOD files evaluate the 5-way model directly.
- `regime_clf/inference.py`: production API (`RegimeClassifier.classify(texts, unit=)`) + CLI. Negative class, label list and speaker masks come from the checkpoint (`UNIT_CLASSES` lists positives only; the negative is added at load).
- `regime_clf/ood/`: `SPEC.md` (generation spec incl. difficulty definitions), `raw_*.jsonl` (5 Fable agents × 105), `build_ood.py [--parts … --out …]` → `ood_hard_v1.jsonl` (default) / `ood_low_medium_v1.jsonl`.
- `app/`: FastAPI chat (`server.py`), providers (`llm.py`: Anthropic SDK + OpenAI-compatible), `config.yaml` (`classifier.checkpoint` picks the model; default 5-way), `static/`, `mock_llm.py`. `/api/config` returns the checkpoint's `labels` and `unit_classes`; the front end adapts. Writes `app/data/turns.jsonl` and `app/data/feedback.jsonl` (FP/FN flags).
- `Tests/<YYYY-MM-DD>_<name>/`: one folder per run, each with `report.html` and an optional hand-written `notes.html` (interpretation); `Tests/make_reports.py` regenerates all reports + `Tests/index.html`.
- `CLM/` (gitignored): clone of github.com/Contrastive-LM/CLM, installed `--no-deps -e`.

## Commands
```bash
cd regime_clf                                                 # prefix with REGIME_LABEL_SET=5way for the 5-way set
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
- Weights (`*.pt`) are gitignored; ship `regime_clf/checkpoints/clm_regime_*.pt` as GitHub release assets (v1.0 = 7-way, v1.1 = 5-way).
- Compare label sets like for like: collapse the 7-way predictions to 5 classes before crediting a retrain.
- For Claude calls use the `anthropic` SDK (see `app/llm.py`); the default model is `claude-opus-5-5`.

## Next work
See `docs/developer_guide.html` §6. The top items: SME adjudication of the S-ID/S-P boundary, recall-first thresholds, calibration, pytest + CI, harder training data, and consuming `feedback.jsonl` for gated retraining.
