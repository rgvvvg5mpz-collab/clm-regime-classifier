# OOD hard validation set — generation spec

Goal: a small, **out-of-distribution**, **hard** test set for a 7-way classifier that assigns a
message from a broker-dealer's GenAI chat channel to the securities-regulation regime it
implicates. Training data is synthetic and templated (see `../data/train.jsonl`; skim ~30 lines so
you know what *not* to imitate). Your examples must differ from it in style, scenario, or
difficulty while keeping an unambiguous gold label.

## Classes (gold label must be exactly one of these keys)

The authoritative descriptions are in `../classes.py` (`CLASSES`). Summary:

| key | unit | fires when |
|---|---|---|
| `compliant` | either | no regulatory issue: ordinary question, education, properly balanced/disclosed answer, benign account chat |
| `finra_2210` | assistant response | misleading/promissory/exaggerated, unbalanced risk, performance projection, tax mischaracterization, missing comparison/testimonial/recommendation disclosure |
| `reg_bi` | assistant response | personalized recommendation of a security, strategy, account type or rollover; steering to proprietary products |
| `finra_4530` | client message | written customer grievance: sales practice, unauthorized trading, operational errors, theft/misappropriation/forgery, harm from the AI assistant itself |
| `sec_17a3_17a4` | client message | references/invites off-channel communication (text, personal phone/email, WhatsApp), asks that something not be recorded, or reports an account update that must be captured in records |
| `reg_sp` | client message | shares sensitive personal info (SSN, DOB, account numbers), privacy/opt-out request, reports a data incident, asks for another person's info |
| `reg_sid` | client message | identity-theft red flags: reported stolen identity, inconsistent/suspicious ID info, account-takeover signals, bypassing or delegating verification |

Note the unit: `finra_2210`/`reg_bi` are the **assistant's** words; the other four regimes are the
**client's** words. `compliant` can be either.

## Hard rules

- One JSON object per line, written to the output file you are given:
  `{"text": str, "label": <key>, "unit": "client_message"|"assistant_response", "ood_axis": <your axis>, "difficulty": "hard"|"very_hard", "rationale": "<one sentence: why this label and not the nearest confuser>"}`
- The gold label must be defensible to a compliance officer. If a message genuinely implicates two
  regimes equally, do not include it. A dominant regime with incidental mention of another is fine
  and is exactly what makes it hard; say so in the rationale.
- No real financial-firm names or fund names. Invent neutral names.
- No real people. Fake PII only (e.g. SSN 000-xx-xxxx style or obviously fake digits).
- No near-copies of training templates. Vary length (one line up to ~200 words).
- Exactly the per-class counts you are asked for.
- After writing, validate the file: every line parses as JSON, labels are valid keys, counts match.
