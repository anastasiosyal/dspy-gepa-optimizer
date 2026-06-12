"""
Metric — code-based correctness (binary exact match) + RICH feedback that points at the
*kind* of contractual feature in play, so GEPA's reflection can assemble the unfairness rubric.
We hint at the dimensions; GEPA has to generalise them into a usable instruction.
"""
import dspy


def fairness_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    want = gold.label
    got = getattr(pred, "label", None)
    if got == want:
        return dspy.Prediction(score=1.0, feedback=f"Correct: '{want}'.")
    if want == "unfair":
        fb = (
            "Wrong: you said 'fair' but this clause is UNFAIR to the consumer. Pinpoint the "
            "specific mechanism that disadvantages the consumer — e.g. the provider can change "
            "terms or terminate unilaterally, liability is limited/excluded, disputes are forced "
            "to arbitration or a specific court, the consumer grants a broad content/data licence, "
            "or contract terms bind the consumer to things outside the document — and name the "
            f'principle it offends. Clause: "{gold.text[:300]}"'
        )
    else:
        fb = (
            "Wrong: you said 'unfair' but this clause is FAIR. A clause being legalistic, dry, or "
            "merely one-sided in topic is NOT itself unfair — it is unfair only if it imposes an "
            "unreasonable disadvantage on the consumer. Do not over-flag routine or informational "
            f'terms. Clause: "{gold.text[:300]}"'
        )
    return dspy.Prediction(score=0.0, feedback=fb)


def fairness_metric_recall(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Recall-leaning variant. A missed violation (false negative) scores 0.0; over-flagging a
    fair clause (false positive) is the cheaper error and scores 0.5. That asymmetry lowers the
    effective decision threshold to ~P(unfair) > 1/3, trading precision for higher unfair-recall
    — the right bias when a missed violation is the expensive mistake. Feedback is aligned: the
    miss is framed as the costly error, the over-flag as the minor one."""
    want = gold.label
    got = getattr(pred, "label", None)
    if got == want:
        return dspy.Prediction(score=1.0, feedback=f"Correct: '{want}'.")
    if want == "unfair":  # false negative — the expensive error
        fb = (
            "Wrong, and this is the costly miss: you said 'fair' but this clause is UNFAIR to the "
            "consumer. Pinpoint the specific mechanism that disadvantages the consumer — e.g. the "
            "provider can change terms or terminate unilaterally, liability is limited/excluded, "
            "disputes are forced to arbitration or a specific court, the consumer grants a broad "
            "content/data licence, or contract terms bind the consumer to things outside the "
            f'document — and name the principle it offends. Clause: "{gold.text[:300]}"'
        )
        return dspy.Prediction(score=0.0, feedback=fb)
    # false positive — over-flagging a fair clause is the minor error here
    fb = (
        "You said 'unfair' but this clause is FAIR. Over-flagging is the lesser error in compliance, "
        "but this clause carries no concrete disadvantaging mechanism — reserve 'unfair' for clauses "
        f'with a real mechanism, not merely legalistic or one-sided wording. Clause: "{gold.text[:300]}"'
    )
    return dspy.Prediction(score=0.5, feedback=fb)
