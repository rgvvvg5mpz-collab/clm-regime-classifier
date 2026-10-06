"""Merge the Fable-generated OOD parts into one validated set.

Checks: schema, label/unit consistency, no blocked real firm names (tools/name_check.py), exact duplicates (within the set and
against train/val/test), and near-duplicates against train by character 3-5-gram TF-IDF cosine on the raw text (rows at
>= NEAR_DUP_COS to any training text are dropped, since they are not OOD). Encoder
embeddings are not used for this: every state ends with the same instruction suffix, so
last-token embeddings of different messages sit at cosine ~0.997 and cannot separate them.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from classes import LABELS  # noqa: E402
from inference import UNIT_CLASSES  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "tools"))
from name_check import scrub  # noqa: E402

OUT = os.path.join(HERE, "ood_hard_v1.jsonl")
NEAR_DUP_COS = 0.80
KEYS = {"text", "label", "unit", "ood_axis", "difficulty", "rationale"}


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "raw_*.jsonl"))):
        for i, line in enumerate(open(f)):
            r = json.loads(line)
            assert set(r) == KEYS, f"{f}:{i} keys {set(r)}"
            assert r["label"] in LABELS, f"{f}:{i} label {r['label']}"
            assert r["label"] in UNIT_CLASSES[r["unit"]], f"{f}:{i} label/unit mismatch"
            assert scrub(r["text"]) == r["text"], f"{f}:{i} contains a blocked real firm name"
            rows.append(r)
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"]); uniq.append(r)
    print(f"parts: {len(rows)} rows, {len(rows) - len(uniq)} in-set duplicates")

    data = os.path.join(os.path.dirname(HERE), "data")
    known = {json.loads(l)["text"] for s in ("train", "val", "test") for l in open(os.path.join(data, f"{s}.jsonl"))}
    exact = [r for r in uniq if r["text"] in known]
    uniq = [r for r in uniq if r["text"] not in known]
    print(f"exact matches with train/val/test: {len(exact)}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    train = [json.loads(l)["text"] for l in open(os.path.join(data, "train.jsonl"))]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(train)
    max_cos = (vec.transform([r["text"] for r in uniq]) @ vec.transform(train).T).max(1).toarray().ravel()
    keep = [r | {"max_cos_to_train": round(float(c), 4)} for r, c in zip(uniq, max_cos) if c < NEAR_DUP_COS]
    print(f"near-duplicates of train (cos >= {NEAR_DUP_COS}): {len(uniq) - len(keep)}; "
          f"max-cos median {np.median(max_cos):.3f}, p95 {np.percentile(max_cos, 95):.3f}")
    with open(OUT, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(keep)} rows -> {OUT}")
    print(dict(Counter(r["label"] for r in keep)))
    print(dict(Counter(r["ood_axis"] for r in keep)), dict(Counter(r["difficulty"] for r in keep)))


if __name__ == "__main__":
    main()
