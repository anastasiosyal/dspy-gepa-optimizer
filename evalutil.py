"""Scoring helpers: accuracy, macro-F1, and unfair-class recall/precision."""
from collections import Counter

from data import CLASSES


def score(examples, preds) -> dict:
    conf = Counter()
    correct = 0
    records = []
    for ex, p in zip(examples, preds):
        got = getattr(p, "label", None) if p is not None else None
        ok = got == ex.label
        correct += ok
        conf[(ex.label, got)] += 1
        records.append({
            "text": ex.text, "gold": ex.label, "pred": got, "correct": ok,
            "reasoning": getattr(p, "reasoning", None) if p is not None else None,
        })
    n = len(examples)
    return {"acc": correct / n if n else 0.0, "n": n, "correct": correct,
            "conf": conf, "records": records}


def _prf(conf, positive="unfair"):
    tp = conf.get((positive, positive), 0)
    fn = sum(c for (g, p), c in conf.items() if g == positive and p != positive)
    fp = sum(c for (g, p), c in conf.items() if g != positive and p == positive)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def macro_f1(conf) -> float:
    return sum(_prf(conf, c)[2] for c in CLASSES) / len(CLASSES)


def unfair_recall(conf) -> float:
    return _prf(conf, "unfair")[1]


def unfair_precision(conf) -> float:
    return _prf(conf, "unfair")[0]
