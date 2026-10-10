# CLASP: Contrastive Labelling And System-1 Prediction

A workbench for building **System-1 text classifiers** trained with a contrastive loss. Two small heads ([CLM-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), Stanford/NVIDIA) sit on a frozen Qwen3-8B encoder and score each text against a one-sentence description of every class. A few hundred labelled rows per class train a model in about a minute.

The browser UI covers the whole loop:

| Tab | What you do |
|---|---|
| **0 · Data curation** | Prompt Claude Code to research a topic and write a labelled dataset, class definitions and notes |
| **1 · Train** | Pick a dataset (yours, curated, or an example), define the classes, validate, train, read the metrics |
| **2 · Chat & flag** | Run conversations (scripted transcripts or a live LLM) through the selected model; flag its mistakes |
| **3 · Post-train** | Fold the flags and any SME file back in, gate the candidate against the current model, promote it |
| **? · The classes** | See what the selected model can assign, per speaker |

Every model is versioned in a registry, and any version can be selected from the sidebar.

## Run it

```bash
git clone https://github.com/rgvvvg5mpz-collab/clm-regime-classifier.git && cd clm-regime-classifier
scripts/run_ui.sh
```

Then open <http://localhost:8710>. The first run sets everything up: `.venv`, requirements, CLM into `third_party/`, the reference heads, and the two shipped models from the GitHub releases. The Qwen3-8B encoder (~16 GB) downloads the first time the UI loads a model.

**With Claude Code:** open the clone and ask it to *"run the UI"*. The repo ships a `run-ui` skill and launch configurations for exactly that.

**Requirements:** Python 3.10+, git, curl, ~20 GB of disk, and either Apple Silicon with 24 GB+ of unified memory or a CUDA GPU with 24 GB+.

**What needs keys:**
- Nothing, for the demo (scripted transcripts), training and post-training.
- Live LLM chat needs `ANTHROPIC_API_KEY`, or a key typed in the sidebar. Any OpenAI-compatible endpoint also works.
- The Data curation tab needs Claude Code logged in on the machine (`claude /login`), or run with `scripts/run_ui.sh --mock-curation` to try it offline.

## Repository layout

```
app/                  the workbench: FastAPI server (server.py), training jobs + registry (training.py),
                      Claude Code runner (curation.py), LLM adapters (llm.py), static UI, offline mocks
engine/               generic core: encoder + embedding store, contrastive head trainer, classifier (API + CLI)
examples/regulatory/  the worked example: label sets, dataset build, research scripts, data splits, OOD sets,
                      samples (Train-tab examples) and the demo transcript
models/               checkpoints (*.pt, downloaded or trained, gitignored) + .meta.json sidecars
data/                 your own datasets and transcripts (listed first in the UI's pickers)
scripts/              setup.sh (one-time, idempotent), run_ui.sh
Tests/                dated evaluation runs with report.html; index.html and METRICS.md summarise them
docs/                 HTML documentation        tools/  name_check.py        CLAUDE.md  orientation for Claude Code
```

## Documentation

These are HTML files: open them locally, or from the running UI at `/docs/…`.

- [Overview](docs/README.html) · [Architecture](docs/architecture.html) · [Developer guide](docs/developer_guide.html) · [UI dev README](app/DEV_README.html)
- The worked example: [methodology](docs/methodology.html) · [all test runs](Tests/index.html) · [metrics summary](Tests/METRICS.md)

## The worked example: a regulatory regime screener

The repo ships one complete example: a classifier for a broker-dealer's GenAI chat. It names the regime a message implicates (FINRA 2210, Reg BI, FINRA 4530, SEC 17a-3/4, Reg S-P, Reg S-ID) or flags nothing. It's trained on the [RegModels](https://github.com/rgvvvg5mpz-collab/RegModels) datasets.

| Macro-F1 | 5-way model (v1.1, UI default) | 7-way model (v1.0) | n |
|---|---|---|---|
| Held-out test | **0.983** [0.977–0.989] | 0.977 [0.970–0.983] | 2,501 |
| Out-of-distribution, hard (written by Claude Fable 5.1) | **0.800** [0.748–0.846] | 0.732 [0.680–0.778] | 315 |
| Out-of-distribution, low/medium (Claude Fable 5.1) | **0.830** [0.770–0.882] | 0.748 [0.688–0.801] | 210 |

Plan around the out-of-distribution numbers. The main weaknesses:
- Reg BI recommendations get read as FINRA 2210 (Reg BI recall is 0.53 on the hard set).
- The 7-way model confuses Reg S-ID with Reg S-P.

The 5-way model's higher scores come from no longer scoring those two regimes, not from the retraining. See [`Tests/METRICS.md`](Tests/METRICS.md) for per-class precision and recall.

The Train tab's walkthrough runs this example end to end in about 15 minutes.

*Research code. It is not a compliance system of record, and its output is not legal advice.*
