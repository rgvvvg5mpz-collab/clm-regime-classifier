"""Merge the six RegModels tracks into one regime dataset for the active label set.

class = track name when the row has any violation/complaint label, else the negative class.
Tracks that the label set does not cover (classes.DROPPED) map entirely to the negative
class: their messages are the out-of-class samples. Select the set with REGIME_LABEL_SET
(7way -> data/, 5way -> data_5way/).
Source splits (train/val/test v2) are kept; exact-duplicate texts are resolved so a
text appears once, in one split, with one class.
"""
import collections
import json
import os
import sys

SUITE = os.path.expanduser("~/Desktop/ClaudeCode_GIT/RegModels/slm-supervisory-suite/data")
TRACKS = {  # class name -> (data dir, file prefix)
    "finra_2210": ("finra_2210", "finra_2210"),
    "reg_bi": ("regbi", "regbi"),
    "finra_4530": ("finra_4530", "finra_4530"),
    "sec_17a3_17a4": ("records_17a4", "rec_17a4"),
    "reg_sp": ("regsp", "regsp"),
    "reg_sid": ("regsid", "regsid"),
}
ASSISTANT_TRACKS = {"finra_2210", "reg_bi"}   # these tracks score the assistant's words
SPLITS = ["train", "val", "test"]
SPLIT_RANK = {"test": 0, "val": 1, "train": 2}  # a text seen in several splits stays in the held-out one
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from name_check import scrub  # noqa: E402
from classes import DATA_DIR, DROPPED, LABEL_SET, NEGATIVE  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR)


def main():
    rows = []
    for cls, (d, prefix) in TRACKS.items():
        for split in SPLITS:
            for line in open(os.path.join(SUITE, d, f"{prefix}_{split}_v2.jsonl")):
                r = json.loads(line)
                text = r["text"].strip()
                text = scrub(text)   # real firm names -> fictional (tools/name_check.py)
                rows.append({"text": text,
                             "label": cls if r["labels"] and cls not in DROPPED else NEGATIVE,
                             "track": cls,
                             "unit": "assistant_response" if cls in ASSISTANT_TRACKS else "client_message",
                             "fine_labels": r["labels"], "split": split})

    by_text = collections.defaultdict(list)
    for r in rows:
        by_text[r["text"]].append(r)
    kept, conflicts, dups = [], 0, 0
    for text, group in by_text.items():
        labels = {r["label"] for r in group}
        if len(labels) > 1:   # same text, different classes: ambiguous, drop
            conflicts += 1
            continue
        dups += len(group) - 1
        kept.append(min(group, key=lambda r: SPLIT_RANK[r["split"]]))

    os.makedirs(OUT, exist_ok=True)
    for split in SPLITS:
        part = [r for r in kept if r["split"] == split]
        with open(os.path.join(OUT, f"{split}.jsonl"), "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        c = collections.Counter(r["label"] for r in part)
        print(f"{split:5s} {len(part):6d}  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"label set={LABEL_SET} -> {OUT}")
    print(f"raw rows={len(rows)}  duplicates removed={dups}  conflicting texts dropped={conflicts}")


if __name__ == "__main__":
    main()
