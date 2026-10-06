# CLM Regime Classifier

A classifier for a broker-dealer's GenAI chat channel. Given a message, it names the securities-regulation regime the message implicates, or flags nothing. Two label sets: **5-way** (`no_flag`, FINRA 2210, Reg BI, FINRA 4530, SEC 17a-3/4; Reg S-P and Reg S-ID content is out of class and left unflagged) and **7-way** (`compliant` plus all six regimes). It's built on [CLM-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) (Stanford/NVIDIA contrastive language model), with fine-tuned projection heads on a frozen Qwen3-8B encoder, and trained on the [RegModels](https://github.com/rgvvvg5mpz-collab/RegModels) datasets.

| Macro-F1 | 5-way (v1.1) | 7-way (v1.0) | 7-way collapsed to 5 classes | n |
|---|---|---|---|---|
| Held-out test (in-distribution) | **0.983** [0.977–0.989] | 0.977 [0.970–0.983] | 0.985 | 2,501 |
| OOD hard/very-hard (Claude Fable 5.1) | **0.800** [0.748–0.846] | 0.732 [0.680–0.778] | 0.822 | 315 |
| OOD low/medium (Claude Fable 5.1) | **0.830** [0.770–0.882] | 0.748 [0.688–0.801] | 0.839 | 210 |

Plan around the OOD numbers. The 5-way model's higher OOD scores come from no longer scoring the two hardest regimes, not from retraining (last column). Explicit, easy-to-read messages are no easier for either model than hard ones: the drop comes from distribution shift, not difficulty. See [`Tests/`](Tests/index.html) for every run.

**Documentation (HTML).** Open the files locally, or view them via GitHub Pages / htmlpreview:
- [Overview](docs/README.html) · [Methodology](docs/methodology.html) · [Architecture](docs/architecture.html)
- [Developer handoff guide and improvements](docs/developer_guide.html) · [Chat UI dev README](app/DEV_README.html)
- [Test runs](Tests/index.html)

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone https://github.com/Contrastive-LM/CLM.git && .venv/bin/pip install --no-deps -e CLM
.venv/bin/clm-download                                                      # reference CLM heads
gh release download v1.1 -p clm_regime_5way.pt -D regime_clf/checkpoints   # 5-way heads (UI default)
gh release download v1.0 -p clm_regime_7way.pt -D regime_clf/checkpoints   # 7-way heads

cd regime_clf && ../.venv/bin/python inference.py "Can you text me on my personal cell?" --unit client_message
cd .. && .venv/bin/python -m uvicorn app.server:app --port 8710            # chat UI: http://localhost:8710
```

Qwen3-8B (~16 GB) downloads on first use. It runs in-process on Apple Silicon/CPU, or you can point `--emb-url` at a vLLM pooling server.

## Layout

```
regime_clf/   dataset build, Qwen3-8B embedder, training, evaluation, inference pipeline
  data/       7-way train/val/test built from RegModels; data_5way/ the 5-way variant
  ood/        Fable OOD sets (spec, raw parts, merged ood_hard_v1.jsonl and ood_low_medium_v1.jsonl)
app/          chat UI: FastAPI server, Anthropic + OpenAI-compatible adapters, static front end, mock LLM
Tests/        dated test runs, each with report.html; index.html summarises them
docs/         HTML documentation
tools/        name_check.py (blocked real-firm names)
CLAUDE.md     orientation for Claude Code
```

## Model weights

`clm_regime_5way.pt` ([v1.1 release](../../releases/tag/v1.1)) and `clm_regime_7way.pt` ([v1.0 release](../../releases/tag/v1.0)), 75 MB each, are the fine-tuned heads. Both need `Qwen/Qwen3-8B` and are Apache 2.0, like CLM and Qwen3. Select the label set for the training/evaluation scripts with `REGIME_LABEL_SET=5way|7way`; `inference.py` and the chat UI read the class list from whichever checkpoint they load.

*Research code. It is not a compliance system of record and its output is not legal advice.*
