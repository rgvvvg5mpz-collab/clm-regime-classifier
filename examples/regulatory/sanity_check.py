"""Reproduce the model-card examples to confirm the local encoder matches the reference.

Expected (from huggingface.co/Contrastive-LM/CLM-v0.1-8B):
  department -> billing ~0.94
  tides      -> "The Moon's gravitational pull." ~0.99
"""
import classes  # noqa: F401  (puts the repo root on sys.path)
from clm import Choice, Engine, Noul, Score
from engine.encoder import MPSEmbedder

engine = Engine(embedder=MPSEmbedder(), device="cpu", action_cache=0)

r = engine.answer(
    "Customer: my invoice was charged twice and nobody answers the phone!",
    {
        "urgency": Noul(instructions="Is this urgent?"),
        "department": Choice(instructions="Which team should handle this?",
                             criteria={"billing": "Charges, invoices, refunds",
                                       "technical": "Bugs and outages"}),
        "frustration": Score(instructions="How frustrated is the customer?",
                             criteria=["Calm", "Frustrated", "Very angry"]),
    },
)
for k, a in r["answers"].items():
    print(k, a)
print(engine.rank("What causes tides on Earth?",
                  ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."]))
