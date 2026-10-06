"""Label spaces and the text CLM sees for each class.

A message (client message or assistant response from the firm's GenAI channel) is
assigned to the rule regime it implicates, or to the negative class when it implicates
none of the covered regimes.

Two label sets are defined; pick one with ``REGIME_LABEL_SET`` (default ``7way``):

  7way  compliant + FINRA 2210, Reg BI, FINRA 4530, SEC 17a-3/4, Reg S-P, Reg S-ID
  5way  no_flag  + FINRA 2210, Reg BI, FINRA 4530, SEC 17a-3/4
        Reg S-P and Reg S-ID are not covered; their messages are out-of-class samples
        and belong to ``no_flag``.

The negative class is always the first entry. ``INSTRUCTIONS`` is shared by both sets
on purpose: the encoder embedding of a state depends on it, so keeping it fixed lets
the cached embeddings be reused across label sets.
"""
import os

INSTRUCTIONS = ("Which securities regulation does this message from a broker-dealer's "
                "AI chat channel potentially violate or trigger, if any?")

_POSITIVE = {
    "finra_2210": "FINRA Rule 2210 communications with the public: the assistant's response is "
                  "misleading, promissory, exaggerated, unbalanced on risks and benefits, projects "
                  "performance, mischaracterizes taxes, or omits required disclosures.",
    "reg_bi": "Regulation Best Interest: the assistant makes a personalized recommendation of a "
              "security, investment strategy, account type or rollover, or steers the customer "
              "toward proprietary products.",
    "finra_4530": "FINRA Rule 4530 customer complaint: the customer is making a written grievance "
                  "about sales practices, unauthorized trading, account errors, theft, "
                  "misappropriation or forgery that must be reported.",
    "sec_17a3_17a4": "SEC Rules 17a-3 and 17a-4 books and records: the customer references or "
                     "invites off-channel communication (texts, personal phone, personal email), asks "
                     "that something not be recorded or written down, or reports an account update "
                     "that must be captured in the firm's records.",
    "reg_sp": "Regulation S-P privacy: the customer shares sensitive personal information such as an "
              "SSN, date of birth or account numbers, makes a privacy or opt-out request, reports a "
              "data breach or incident, or asks for another person's information.",
    "reg_sid": "Regulation S-ID identity theft red flags: the message shows signs of identity theft "
               "or account takeover, such as a reported stolen identity, inconsistent or suspicious "
               "identifying information, unauthorized contact-detail changes, or attempts to bypass "
               "or delegate identity verification.",
}

LABEL_SETS: dict[str, dict[str, str]] = {
    "7way": {
        "compliant": "None. The message is compliant: an ordinary question, general education, "
                     "or a properly balanced and disclosed answer that raises no regulatory issue.",
        **{k: _POSITIVE[k] for k in ("finra_2210", "reg_bi", "finra_4530", "sec_17a3_17a4", "reg_sp", "reg_sid")},
    },
    "5way": {
        "no_flag": "No flag. The message raises none of the covered issues: it is an ordinary "
                   "question, general education, a properly balanced and disclosed answer, or a "
                   "matter outside the covered rules such as personal data, privacy, identity "
                   "verification or account security.",
        **{k: _POSITIVE[k] for k in ("finra_2210", "reg_bi", "finra_4530", "sec_17a3_17a4")},
    },
}

# Tracks whose rows are out-of-class for a label set: all their rows map to the negative class.
DROPPED_TRACKS: dict[str, set[str]] = {"7way": set(), "5way": {"reg_sp", "reg_sid"}}

LABEL_SET = os.environ.get("REGIME_LABEL_SET", "7way")
if LABEL_SET not in LABEL_SETS:
    raise ValueError(f"REGIME_LABEL_SET={LABEL_SET!r}; expected one of {list(LABEL_SETS)}")
CLASSES = LABEL_SETS[LABEL_SET]
LABELS = list(CLASSES)
NEGATIVE = LABELS[0]
DROPPED = DROPPED_TRACKS[LABEL_SET]
DATA_DIR = "data" if LABEL_SET == "7way" else f"data_{LABEL_SET}"
CKPT_NAME = "clm_regime_7way.pt" if LABEL_SET == "7way" else f"clm_regime_{LABEL_SET}.pt"


def to_label_set(label: str) -> str:
    """Map a label from any set into the active one (out-of-class labels -> negative)."""
    return label if label in CLASSES else NEGATIVE
