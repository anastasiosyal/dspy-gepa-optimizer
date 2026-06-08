"""
Dataset loader: unfair_tos (LexGLUE) — "is this Terms-of-Service clause UNFAIR to the consumer?"

Source: `coastalcph/lex_glue`, config `unfair_tos` — clean, script-free. Each clause has a
multi-label `labels` list of unfairness categories; empty list = fair. We binarise to
fair / unfair.

Why this task: the decision criteria — what makes a clause unfair under consumer-protection
principles (unilateral termination, limitation of liability, mandatory arbitration, ...) — are
NON-OBVIOUS and must be encoded in the prompt. A generic prompt scores near chance; GEPA's job is
to discover and write that rubric. It's the same shape as any regulatory-compliance task whose
rule is a rubric the model doesn't already know.

The raw data is ~89% fair / 11% unfair. We BALANCE it 50/50 so accuracy is honest (no
majority-class gaming) and a no-rubric baseline sits near chance — leaving clear headroom.
"""
from collections import Counter, defaultdict
import random

import dspy
from datasets import load_dataset

CLASSES = ("fair", "unfair")


def _rows(split: str) -> list[dict]:
    ds = load_dataset("coastalcph/lex_glue", "unfair_tos", split=split)
    out = []
    for r in ds:
        labels = r["labels"]
        out.append({"text": r["text"], "label": "unfair" if labels else "fair"})
    return out


def _balanced_take(rows, per_class, seed, exclude=None):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        if exclude and r["text"] in exclude:
            continue
        by[r["label"]].append(r)
    out = []
    for c in CLASSES:
        g = by[c]
        rng.shuffle(g)
        out.extend(g[:per_class])
    rng.shuffle(out)
    return out


def _to_examples(rows):
    return [dspy.Example(text=r["text"], label=r["label"]).with_inputs("text") for r in rows]


def load_splits(train_per_class=100, val_per_class=60, test_per_class=150, seed=13):
    train_pool = _rows("train")
    test_pool = _rows("test")
    train = _balanced_take(train_pool, train_per_class, seed)
    tr = {r["text"] for r in train}
    val = _balanced_take(train_pool, val_per_class, seed + 1, exclude=tr)
    test = _balanced_take(test_pool, test_per_class, seed + 2)
    return {"train": _to_examples(train), "val": _to_examples(val), "test": _to_examples(test)}


def describe(splits):
    for name, ex in splits.items():
        c = Counter(e.label for e in ex)
        print(f"{name:6s} n={len(ex):4d}  fair={c['fair']}  unfair={c['unfair']}")


if __name__ == "__main__":
    describe(load_splits())
