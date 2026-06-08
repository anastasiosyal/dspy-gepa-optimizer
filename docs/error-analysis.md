# Error analysis — baseline FairnessClassifier (claude-haiku)

Run: `baseline.py --split test`, 300 balanced clauses (150 fair / 150 unfair). **Accuracy 77.7%.**
But for a compliance task the headline number isn't the point — *which* errors, and at what cost.

## The errors split asymmetrically

| | count | cost |
|---|------:|------|
| **False negatives** (unfair → "fair") | **52 / 150** | **high** — a missed violation slips through |
| False positives (fair → "unfair") | 15 / 150 | low — a fair clause gets a needless review |

Unfair-recall is only **65%** — the model misses a third of the violations. In compliance the
false negative is the expensive error, so recall on the unfair class is the metric that matters,
not raw accuracy.

## What it misses (read the false negatives)

The 52 missed clauses cluster into recognisable legal categories the model *doesn't know are
unfair* — because nothing in the prompt tells it the rubric:

- **Choice-of-law / jurisdiction** — *"these terms shall be governed by … the laws of the
  Netherlands"*, *"if you reside outside the US, Canada and Brazil, these terms…"*. The model
  reads these as neutral boilerplate; under consumer-protection criteria they disadvantage the
  consumer.
- **Unilateral termination / modification** — *"we may cancel … inactive accounts or modify or
  discontinue our services"*. Reads as routine; it's unilateral provider power.
- **Unilateral price change** — *"may update the pricing of virtual items at any time in its sole
  discretion"*.
- **Contract-by-use** — *"by using the services, you agree to the terms"*.

And the 15 false positives are the mirror: it **over-flags** generic disclaimers and plain
prohibitions (*"academia.edu makes no warranty…"*, *"create a false identity"*) as unfair
because, lacking the rubric, it pattern-matches "legalistic = unfair."

## The one insight

The model has rough intuition but **no rubric**: it doesn't know the specific contractual
mechanisms (unilateral power, jurisdiction, liability limits, forced arbitration) that make a
clause unfair, nor that "legalistic ≠ unfair." That rubric is *non-obvious* and has to live in
the prompt — which is exactly the gap a reflective optimiser can close. It's the same reason any
rubric-based compliance prompt scores near chance until the domain's criteria are written in.

That's the setup for the optimizer: feed the model's misses back as feedback that names the
*kind* of mechanism in play, weight the costly error (missed violations), and let GEPA assemble
the rubric. See the result in `README.md`.
