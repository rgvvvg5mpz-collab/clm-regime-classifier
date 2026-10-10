# Models

Trained CLASP checkpoints (`*.pt`, gitignored) and their metadata sidecars (`*.meta.json`, tracked).

- `clm_regime_5way.pt` and `clm_regime_7way.pt` are the regulatory example's shipped models. They are
  GitHub release assets (v1.1 and v1.0), downloaded by `scripts/setup.sh`.
- Models you train in the workbench are saved here too, with a `.meta.json` holding display names and
  which classes each speaker can receive. The registry of all versions is `app/data/models.json`.

A checkpoint is the two contrastive heads plus the temperature (~75 MB) and carries its own class
descriptions and question; it needs the frozen `Qwen/Qwen3-8B` encoder.
