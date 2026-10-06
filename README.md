# CLM Regime Classifier

A 7-way classifier for a broker-dealer's GenAI chat channel. Given a message, it names the securities-regulation regime the message implicates: **FINRA 2210, Reg BI, FINRA 4530, SEC 17a-3/4, Reg S-P, Reg S-ID**, or **compliant**. It's built on [CLM-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) (Stanford/NVIDIA contrastive language model), with fine-tuned projection heads on a frozen Qwen3-8B encoder, and trained on the [RegModels](https://github.com/rgvvvg5mpz-collab/RegModels) datasets.

| | Macro-F1 | 95% CI | n |
|---|---|---|---|
| Held-out test (in-distribution) | **0.977** | 0.970–0.983 | 2,501 |
| OOD hard set, written by Claude Fable 5.1 | **0.732** | 0.680–0.778 | 315 |

Plan around the OOD number. See [`Tests/`](Tests/index.html) for every run.

**Documentation (HTML).** Open the files locally, or view them via GitHub Pages / htmlpreview:
- [Overview](docs/README.html) · [Methodology](docs/methodology.html) · [Architecture](docs/architecture.html)
- [Developer handoff guide and improvements](docs/developer_guide.html) · [Chat UI dev README](app/DEV_README.html)
- [Test runs](Tests/index.html)

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone https://github.com/Contrastive-LM/CLM.git && .venv/bin/pip install --no-deps -e CLM
.venv/bin/clm-download                                                      # reference CLM heads
gh release download v1.0 -p clm_regime_7way.pt -D regime_clf/checkpoints   # fine-tuned heads

cd regime_clf && ../.venv/bin/python inference.py "Can you text me on my personal cell?" --unit client_message
cd .. && .venv/bin/python -m uvicorn app.server:app --port 8710            # chat UI: http://localhost:8710
```

Qwen3-8B (~16 GB) downloads on first use. It runs in-process on Apple Silicon/CPU, or you can point `--emb-url` at a vLLM pooling server.

## Layout

```
regime_clf/   dataset build, Qwen3-8B embedder, training, evaluation, inference pipeline
  data/       7-class train/val/test built from RegModels
  ood/        Fable OOD hard set (spec, raw parts, merged ood_hard_v1.jsonl)
app/          chat UI: FastAPI server, Anthropic + OpenAI-compatible adapters, static front end, mock LLM
Tests/        dated test runs, each with report.html; index.html summarises them
docs/         HTML documentation
tools/        name_check.py (blocked real-firm names)
CLAUDE.md     orientation for Claude Code
```

## Model weights

`clm_regime_7way.pt` (75 MB, the fine-tuned heads) is attached to the [v1.0 release](../../releases/tag/v1.0). It needs `Qwen/Qwen3-8B` and is Apache 2.0, like CLM and Qwen3.

*Research code. It is not a compliance system of record and its output is not legal advice.*
