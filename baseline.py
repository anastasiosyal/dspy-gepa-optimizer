"""Run the baseline FairnessClassifier; report accuracy, macro-F1, unfair-recall, confusion."""
import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

import config
from data import load_splits
from evalutil import macro_f1, score, unfair_recall
from modules import FairnessClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    config.configure_lm()
    ex = load_splits()[args.split]
    if args.limit:
        ex = ex[: args.limit]
    print(f"Running baseline on {len(ex)} '{args.split}' examples...")

    clf = FairnessClassifier()
    preds = clf.batch(ex, num_threads=args.threads, max_errors=len(ex))
    r = score(ex, preds)
    print(f"\nAccuracy {r['correct']}/{r['n']} = {r['acc']:.1%}   "
          f"macro-F1 {macro_f1(r['conf']):.3f}   unfair-recall {unfair_recall(r['conf']):.0%}")
    print("confusion (gold->pred):", dict(r["conf"]))

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/predictions_{args.split}.json", "w") as f:
        json.dump(r["records"], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
