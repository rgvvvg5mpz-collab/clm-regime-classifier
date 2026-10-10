# Your data

Drop labelled files here and they appear at the top of the **Project data** pickers in the
workbench (Train, Post-train and the Chat transcript picker), no upload needed.

- Training / expert rows: `.csv` or `.jsonl` with `text` and `label` columns; optional `speaker`
  (user / assistant) and `split` (train / val / test). A sibling `<name>.spec.json` (same format as
  `examples/regulatory/samples/*.spec.json`) prefills the class table.
- Chat transcripts: `.json` in the format of `app/transcripts/FORMAT.md`.

Files here are committed with the repo like anything else; keep private data out of git (for
example name it `*.private.csv`, which is gitignored).
