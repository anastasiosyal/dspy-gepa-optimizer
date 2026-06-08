"""The program: a single ChainOfThought predictor over the ClauseFairness signature."""
import dspy

from signatures import ClauseFairness


class FairnessClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ClauseFairness)

    def forward(self, text: str) -> dspy.Prediction:
        return self.classify(text=text)
