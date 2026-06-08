# Concepts, from zero

Every idea this project uses, explained assuming no prior knowledge. Order ≈ the order we apply
them. This doubles as source material for the blog post, which is written the same way.

---

## 1. The task and the dataset

We build something that reads **one clause from a Terms-of-Service contract** and decides whether
it is **unfair to the consumer** or **fair**.

- *"we may suspend or terminate your account at any time"* → unfair (unilateral provider power)
- *"you may cancel your subscription at any time without penalty"* → fair

The dataset is **`unfair_tos`** (from LexGLUE): real ToS clauses, each labelled by legal experts.
The labels follow a **rubric** — specific legal criteria for what makes a clause unfair
(unilateral termination/modification, limitation of liability, mandatory arbitration, choice of
forum/law, broad content licences, contract-by-using). That rubric is the whole story (see §4).

## 2. Classes and class balance

The two answers — fair / unfair — are the **classes**; sorting inputs into them is
**classification**. **Class balance** is how many examples fall in each. The raw data is heavily
skewed: **~89% fair, ~11% unfair**. Skew matters: a lazy model that always says "fair" scores 89%
and is useless. So we **balance** the data 50/50 (equal fair and unfair). Now accuracy is honest
(a coin-flip scores 50%), and there's clear room to improve.

## 3. Train / validation / test split

You can't grade a student on the questions they studied. So we cut the data into non-overlapping
parts: **train** (what the optimizer learns from), **validation** (used *during* optimization to
pick the best candidate prompt), and **test** (locked away, looked at once, for the honest final
number). We sample each split **balanced** (equal fair/unfair) so every number is meaningful.

## 4. The hidden rubric — the core idea

This is *why this task is the right one*. A clause is unfair only under specific legal criteria
that a general model **doesn't reliably know**. With a bare prompt ("is this clause unfair?") the
model has rough intuition but misses whole categories — choice-of-law, unilateral price changes,
etc. (see `docs/error-analysis.md`). The decision criteria are **non-obvious and must live in the
prompt.**

That is exactly when prompt optimization pays off. (Contrast: "is this review positive or
negative?" has *no* hidden rubric — the model already knows — so optimizing the prompt does almost
nothing. We confirmed this; it's the negative result in the post.)

## 5. Baseline

The **baseline** is the simplest thing that runs end-to-end, measured before any optimization:
the bare prompt. It's the number every later change must beat. Here it's **77.7% accuracy / 65%
unfair-recall**.

## 6. Error analysis

Before optimizing, **read the mistakes.** Pull the wrong predictions, note *why* each failed,
cluster them. Here the misses cluster into legal categories the model didn't know were unfair —
which tells us precisely what the prompt is missing. The unglamorous step that everything else
builds on.

## 7. Precision, recall, and the *objective*

Measure a classifier one class at a time. For the "unfair" class:
- **Recall** = of the truly-unfair clauses, what fraction did we catch? `caught / all-unfair`.
- **Precision** = of the ones we *called* unfair, how many really were?

In compliance the costly mistake is a **false negative** (a missed violation), so the **objective**
is to maximise **unfair-recall** without over-flagging. Naming the objective is what makes the
optimization meaningful — you optimize *toward* something, not just "accuracy."

## 8. Rich-feedback metric

A **metric** scores one prediction. A thin metric returns 1/0 (right/wrong). A **rich-feedback**
metric also returns *text* explaining the miss — here, naming the kind of mechanism in play
("you missed a unilateral-termination clause…"). The optimizer reads that text to improve the
prompt. The feedback encodes the **objective** (catch violations) and the **rubric hints**; the
optimizer turns them into a usable instruction. Correctness itself is checked in **code** (the
label is one of two words) — no second LLM needed to grade it.

## 9. The reflection model (the prompt-writer)

The optimizer uses two models: a cheap **task model** (Haiku) that does the classification, and a
strong **reflection model** that *reads the feedback and rewrites the prompt*. They're separate —
so you can keep inference cheap while paying for a smart prompt-writer once. Here, Opus 4.8 as the
prompt-writer wrote a sharp, explicit rubric with the task model left as cheap Haiku. **Spend on
the reflection model, not the task model.**

## 10. GEPA (the optimizer)

**GEPA** is a reflective optimizer: it proposes prompt variants, scores them with your metric,
keeps a frontier of the best, and uses the reflection model to read the feedback and propose
smarter variants. On this task it took the bare prompt and **wrote out the unfairness rubric** —
an explicit list of disadvantaging mechanisms plus a "don't over-flag routine clauses" guardrail —
lifting unfair-recall from 65% to 86.5% on average (91% on the best of four 6,000-call runs). GEPA
is only as good as the metric (objective) you give it,
and it only has something to discover when the task has a hidden rubric.

## 11. The DSPy program (what GEPA actually rewrites)

The classifier is a `dspy.Module` wrapping a single `dspy.Signature` (`ClauseFairness`). A Signature
declares the task's typed inputs and outputs, and its **docstring is the instruction** the model
runs on. The "bare prompt" was that one-line docstring.

That's what lets GEPA optimise at all: the prompt is a *declared field*, not a string hard-coded
into a model call, so the optimiser has something well-defined to rewrite. GEPA rewrites the
instruction string and leaves the field names, the types and your code untouched. (It can also
evolve few-shot demos; in our runs it added none, so the entire gain is one rewritten instruction.)
