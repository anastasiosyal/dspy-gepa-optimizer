"""
Headline bar chart: accuracy + unfair-recall for baseline vs GEPA (mean of 4 runs)
vs GEPA's best run. Task model is cheap Haiku throughout; the lift is the prompt.

    python make_headline_chart.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

GREY = "#9aa7b6"
BLUE = "#2f6fb0"
DARK = "#334155"
OUT = "charts/headline_chart.png"   # local + gitignored; copy into a post if you want it there

groups = ["Baseline\n(no GEPA)", "GEPA · Opus\n(mean of 4 runs)", "GEPA · Opus\n(best of 4 runs)"]
acc = [77.7, 87.5, 90.3]
rec = [65.0, 86.5, 91.0]

x = np.arange(len(groups))
w = 0.38

fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
b1 = ax.bar(x - w / 2, acc, w, color=GREY, label="Accuracy")
b2 = ax.bar(x + w / 2, rec, w, color=BLUE, label="Unfair-recall (violations caught)")

for xi, v in zip(x - w / 2, acc):
    ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", color=DARK, fontsize=12)
for xi, v in zip(x + w / 2, rec):
    ax.text(xi, v + 1.2, f"{v:g}", ha="center", va="bottom", color=BLUE,
            fontsize=12, fontweight="bold")

# callout on the best-run group
ax.annotate("+33% recall on average\n+40% on the best run", xy=(2 + w / 2, 91),
            xytext=(2 + w / 2, 99.5), ha="center", color=BLUE, fontsize=12,
            fontweight="bold")

ax.set_ylim(0, 104)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11.5, color=DARK)
ax.set_ylabel("%", fontsize=12, color=DARK)
ax.set_title("Detecting unfair Terms-of-Service clauses   ·   task model stays cheap Haiku throughout",
             fontsize=13, color=DARK, pad=12)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=DARK)
ax.legend(frameon=False, fontsize=11.5, loc="upper left")

fig.tight_layout()
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
