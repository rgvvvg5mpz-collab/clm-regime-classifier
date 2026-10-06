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

## Difficulty levels

`difficulty` is one of `low`, `medium`, `hard`, `very_hard`. Every set still has to be OOD (new
style/scenario versus the training templates); difficulty is about how much inference the label
needs, not about how similar the text is to training data.

| level | what it means |
|---|---|
| `low` | The regime is stated almost outright: explicit vocabulary ("I want to file a complaint", "here is my SSN", "text me on my cell", "I guarantee 10%"), one topic per message, no competing regime, plain register, short to medium length. A careful first-year compliance analyst would label it in seconds. |
| `medium` | The signal is clear but not spelled out: a grievance without the word "complaint", PII given in passing, an off-channel request phrased as convenience, a recommendation framed as "many clients like you…". At most one incidental mention of another regime. Register and scenario may be unusual. |
| `hard` | Requires inference and an explicit judgment about the nearest competing class; register or scenario shift is substantial. (See the three hard axes above.) |
| `very_hard` | As hard, plus negation, hypotheticals, sarcasm, quoted speech, or a buried signal; a reasonable reviewer might pause. |

`ood_axis` is still required on every row (`register_shift`, `adversarial_confuser`, `novel_scenario`);
for low/medium sets pick whichever axis best describes why the row is OOD and spread rows across the
three axes roughly evenly.

## Hard rules

- One JSON object per line, written to the output file you are given:
  `{"text": str, "label": <key>, "unit": "client_message"|"assistant_response", "ood_axis": <your axis>, "difficulty": "low"|"medium"|"hard"|"very_hard", "rationale": "<one sentence: why this label and not the nearest confuser>"}`
- The gold label must be defensible to a compliance officer. If a message genuinely implicates two
  regimes equally, do not include it. A dominant regime with incidental mention of another is fine
  and is exactly what makes it hard; say so in the rationale.
- No real financial-firm names or fund names. Invent neutral names.
- No real people. Fake PII only (e.g. SSN 000-xx-xxxx style or obviously fake digits).
- No near-copies of training templates. Vary length (one line up to ~200 words).
- Exactly the per-class counts you are asked for.
- After writing, validate the file: every line parses as JSON, labels are valid keys, counts match.
