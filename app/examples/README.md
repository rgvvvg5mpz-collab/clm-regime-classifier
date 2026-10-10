# Example datasets for the Train / Post-train tabs

Pick these from the "Use an example dataset" dropdown in the chat UI's **Train** tab (or
**Post-train** tab for the corrections file). All rows come from the RegModels-derived data in
`regime_clf/data*/`; the samples are small enough to train in about a minute on a laptop.

| File | Rows | Use |
|---|---:|---|
| `regime_5way_sample.csv` | 2,100 | Train tab: 5 classes (`no_flag` + FINRA 2210, Reg BI, FINRA 4530, SEC 17a-3/4). Prefilled class spec in `regime_5way_sample.spec.json`. |
| `regime_7way_sample.csv` | 2,300 | Train tab: 7 classes (`compliant` + six regimes incl. Reg S-P / S-ID). |
| `regime_5way_expert_corrections.csv` | 80 | Post-train tab: "SME-corrected" rows for a 5-way model, drawn from the held-out test split (disjoint from the training samples). |

Columns: `text`, `label`, `speaker` (client / assistant). Your own files need the same two
required columns; `speaker` and `split` (train / val / test) are optional.
