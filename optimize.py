"""
baseline -> GEPA (discover the unfairness rubric) -> optimized, on the held-out test set,
and print the instruction GEPA evolved.

    python optimize.py
"""
import os

from dotenv import load_dotenv

load_dotenv()

import dspy

import config
from data import load_splits
from evalutil import macro_f1, score, unfair_recall
from metric import fairness_metric
from modules import FairnessClassifier


def build_reflection_lm() -> dspy.LM:
    host = os.getenv("LLM_PROXY_HOST", "").rstrip("/")
    # NB: claude-opus-4-8 now rejects an explicit `temperature` ("deprecated for this model"),
    # so we omit it — dspy then sends no temperature and the model uses its own default.
    return dspy.LM(os.getenv("LLM_PROXY_REFLECTION_MODEL"), api_key=os.getenv("LLM_PROXY_API_KEY"),
                   api_base=host, max_tokens=8000)


def report(program, test, name: str, threads: int = 12):
    preds = program.batch(test, num_threads=threads, max_errors=len(test))
    r = score(test, preds)
    f1 = macro_f1(r["conf"])
    print(f"[{name}] acc {r['acc']:.1%}  macro-F1 {f1:.3f}  unfair-recall {unfair_recall(r['conf']):.0%}  ({r['correct']}/{r['n']})")
    return r["acc"], f1


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--budget", type=int, default=6000, help="GEPA max_metric_calls")
    args = ap.parse_args()
    out = args.out or f"optimized_utos_seed{args.seed}.json"
    logdir = args.logdir or f"gepa_logs_utos_seed{args.seed}"

    config.configure_lm()
    s = load_splits()
    test = s["test"]
    print(f"seed={args.seed}  train={len(s['train'])}  val={len(s['val'])}  test={len(test)}")

    program = FairnessClassifier()
    print("=" * 64, "\nBASELINE")
    b_acc, b_f1 = report(program, test, "baseline", args.threads)

    print("=" * 64, f"\nGEPA optimizing (discovering the unfairness rubric)... budget={args.budget}")
    optimized = dspy.GEPA(
        metric=fairness_metric,
        max_metric_calls=args.budget,
        reflection_lm=build_reflection_lm(),
        reflection_minibatch_size=5,
        add_format_failure_as_feedback=True,
        num_threads=args.threads,
        track_stats=True,
        seed=args.seed,
        log_dir=logdir,
    ).compile(program, trainset=s["train"], valset=s["val"])

    print("=" * 64, "\nOPTIMIZED")
    o_acc, o_f1 = report(optimized, test, "optimized", args.threads)
    optimized.save(out, save_program=False)

    print("=" * 64)
    print(f"RESULT  seed={args.seed}  accuracy {b_acc:.1%} -> {o_acc:.1%}   macro-F1 {b_f1:.3f} -> {o_f1:.3f}")


if __name__ == "__main__":
    main()
