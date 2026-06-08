"""
The task contract. The docstring is deliberately BARE — it states the question but gives NO
criteria for what makes a clause unfair. That's the point: the rubric is what GEPA must
discover and write in. A generic prompt like this sits near chance on the balanced set.
"""
from typing import Literal

import dspy

Verdict = Literal["fair", "unfair"]


class ClauseFairness(dspy.Signature):
    """Decide whether this Terms-of-Service clause is unfair to the consumer."""

    text: str = dspy.InputField(desc="A single clause from an online Terms-of-Service contract.")
    label: Verdict = dspy.OutputField(desc="'unfair' if the clause is unfair to the consumer, else 'fair'.")
