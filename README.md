# Loop engineering with DSPy

There's always been a person inside the prompt-tuning loop: write a prompt, run it over a batch
of examples, squint at the failures, adjust a sentence, run it again. It's unglamorous, but it's
genuinely how prompts get good: try, inspect, rewrite.

Nothing about that loop actually needs the person inside it.

This repo is a reproducible experiment in **automatic prompt optimisation** with DSPy + GEPA. You
hand over the same things the human was working from anyway — a starting prompt, labelled
examples, a metric, and feedback on what went wrong — and the system runs the iteration itself:
try a prompt variant, score it, read the misses, rewrite the instruction, repeat.

The human's place is outside the loop, orchestrating: you define the goal, the metric, and the
feedback. The loop finds the wording.

Full write-up: [Loop engineering with DSPy: let GEPA evolve the prompt](https://medium.com/empirical-engineer/gepa-wrote-its-own-legal-rubric-and-caught-33-more-unfair-contract-clauses-913a2d7d8ad5)

**The task:** classify a Terms-of-Service clause as **unfair to the consumer** or **fair**
([LexGLUE `unfair_tos`](https://huggingface.co/datasets/coastalcph/lex_glue), balanced 50/50).
Every concept is defined from scratch in [`CONCEPTS.md`](CONCEPTS.md).

## What this shows

Instead of hand-tuning a prompt, you define the loop:

1. a DSPy task with typed inputs and outputs;
2. labelled examples from public data;
3. a metric that scores correctness;
4. written feedback explaining why misses are wrong;
5. a GEPA budget for evolving better instructions.

On this task, the loop turns a one-line prompt with no criteria into an explicit unfairness rubric.
The useful lesson is not "legal AI". It is that prompts can be treated as software artefacts:
generated, evaluated, versioned, and improved against a metric.

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
from **65% → 86.5% on average, 91% on the best of four runs**. Nobody hand-tuned a word.

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
uv venv
uv pip install -r requirements.txt
cp .env.example .env                # point at any OpenAI-compatible endpoint
.venv/bin/python baseline.py --split test
.venv/bin/python optimize.py
```

Set `LLM_PROXY_MODEL` to a cheap task model and `LLM_PROXY_REFLECTION_MODEL` to a strong one.
`optimize.py` defaults to a 6,000-call GEPA budget, so run `baseline.py` first as the quick smoke
test.

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
