# dspy-gepa-optimizer

When does automatic **prompt optimization** (DSPy + GEPA) actually help — and when is it a waste?
This repo answers it honestly on real data, and lands two findings most "GEPA boosted my prompt"
posts skip.

**The task:** classify a Terms-of-Service clause as **unfair to the consumer** or **fair**
([LexGLUE `unfair_tos`](https://huggingface.co/datasets/coastalcph/lex_glue), balanced 50/50).
Every concept is defined from scratch in [`CONCEPTS.md`](CONCEPTS.md).

## Result — GEPA on the held-out test (300 balanced clauses)

Four runs, identical config, each a 6,000-call GEPA budget; the prompt-writer is Opus, the
classifier is cheap Haiku throughout.

| | accuracy | macro-F1 | unfair-recall |
|---|:--:|:--:|:--:|
| baseline (bare prompt, no GEPA) | 77.7% | 0.773 | 65% |
| **GEPA · Opus reflection (mean of 4 runs)** | **87.5%** | **0.876** | **86.5%** |
| GEPA · Opus reflection (best of 4 runs) | 90.3% | 0.903 | 91% |

From a one-line prompt with no criteria, GEPA **discovered and wrote the unfairness rubric**
(unilateral termination/modification, price-change-at-will, forced arbitration, choice of law…)
plus a *"don't over-flag routine clauses"* guardrail — lifting violation-catch (unfair-recall)
from **65% → 86.5% on average, 91% on the best of four runs**.

## Two findings

1. **GEPA wins when the task has a hidden rubric.** The same setup gave **+1pp on sentiment** and
   **regressed on 77-way intent** — tasks where the model already knows the criteria, so there's
   nothing to discover. It won here because the legal criteria are non-obvious and must live in the
   prompt. *Reach for a prompt optimizer when the decision rule is a rubric, not when it's obvious.*
2. **Spend on the reflection model, not the task model.** The prompt-writer is a strong model
   (Opus 4.8); the classifier stays cheap Haiku the whole time. Pay once for a smart prompt-writer,
   keep serving cheap.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # point at any OpenAI-compatible endpoint (see .env.example)
python baseline.py --split test     # baseline accuracy / unfair-recall / confusion
python optimize.py                  # baseline -> GEPA -> optimized, + the evolved rubric
```

Set `LLM_PROXY_MODEL` to a cheap task model and `LLM_PROXY_REFLECTION_MODEL` to a strong one.

## Layout

| file | role |
|------|------|
| `data.py` | `unfair_tos` loader; binarise fair/unfair; balanced stratified splits |
| `signatures.py` / `modules.py` | the DSPy task contract (deliberately bare prompt) + classifier |
| `metric.py` | code-based correctness + rich feedback that names the unfairness mechanism |
| `evalutil.py` | accuracy, macro-F1, unfair-class recall/precision |
| `baseline.py` | run the un-optimized classifier; per-class + confusion |
| `optimize.py` | baseline → GEPA → optimized; prints the evolved rubric |
| `CONCEPTS.md` | every concept from zero |
| `docs/error-analysis.md` | the baseline's misses, by legal category |

## License

MIT — see [LICENSE](LICENSE).
