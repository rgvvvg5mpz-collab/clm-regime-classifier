# Transcript file format (chat UI replay mode)

A transcript file is JSON. The UI replays each conversation turn by turn through the
screening pipeline without calling any LLM: the user turn is screened, then the scripted
assistant turn is screened, exactly as a live turn would be.

```json
{
  "title": "Demo conversations",
  "description": "one line shown in the UI",
  "conversations": [
    {
      "id": "c01",
      "title": "Short label shown in the dropdown",
      "scenario": "one sentence: what this conversation exercises",
      "turns": [
        {"role": "user",      "content": "...", "expected": "finra_4530", "note": "optional: why"},
        {"role": "assistant", "content": "...", "expected": "none"}
      ]
    }
  ]
}
```

Rules

- `turns` strictly alternate `user`, `assistant`, `user`, ... and start with `user`. A conversation
  may end on a user turn.
- `expected` is what a compliance officer would flag for that turn, as a label key from the full
  7-way set (`finra_2210`, `reg_bi`, `finra_4530`, `sec_17a3_17a4`, `reg_sp`, `reg_sid`) or `"none"`.
  Speaker matters: `finra_2210` / `reg_bi` only on assistant turns; the other four only on user turns.
  The 5-way model treats `reg_sp` / `reg_sid` as no flag; the UI shows expected vs predicted so the
  demo can discuss both.
- `note` is optional free text (shown in the turn's details).
- Plain prose only; invented firm, fund and product names; fake PII only (SSN 000-xx-xxxx style).
- Keep each turn under ~120 words.
