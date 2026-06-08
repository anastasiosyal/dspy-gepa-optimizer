"""
Reconstruct GEPA's optimisation trajectory from run logs. For each run, take the
metric of the prompt GEPA would *return* at each budget (its accuracy-best-so-far
candidate); plot a shaded band between the best and worst run, with the average
across runs highlighted.

    --metric accuracy      → validation accuracy
    --metric unfair_recall → unfair-recall (violations caught) on the validation set

    python make_trajectory_chart.py --metric unfair_recall \
        --dirs gepa_logs_utos_6k_seed10,...,gepa_logs_utos_6k_seed13 \
        --out charts/trajectory_recall_chart.png
"""
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BLUE = "#2f6fb0"      # average
BAND = "#8aa6c2"      # band edges / fill
DARK = "#334155"
GREEN = "#2f8f5b"     # best-endpoint label
ORANGE = "#c2410c"    # worst-endpoint label

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", default="gepa_logs_utos_6k_seed10,gepa_logs_utos_6k_seed11,"
                                   "gepa_logs_utos_6k_seed12,gepa_logs_utos_6k_seed13")
ap.add_argument("--out", default="charts/trajectory_chart.png")   # local + gitignored
ap.add_argument("--metric", choices=["accuracy", "unfair_recall"], default="accuracy")
ap.add_argument("--title", default=None)
args = ap.parse_args()
DIRS = [d for d in args.dirs.split(",") if d]

unfair_idx = None
if args.metric == "unfair_recall":
    from data import load_splits
    val = load_splits()["val"]
    unfair_idx = [i for i, e in enumerate(val) if e.label == "unfair"]


def load(d):
    s = pickle.load(open(f"{d}/gepa_state.bin", "rb"))
    calls = np.array(s["num_metric_calls_by_discovery"], dtype=float)
    sub = s["prog_candidate_val_subscores"]
    accs = np.array([float(np.mean(list(x.values()))) for x in sub])
    if unfair_idx is not None:
        vals = np.array([float(np.mean([x[i] for i in unfair_idx])) for x in sub])
    else:
        vals = accs
    return calls, accs, vals


def returned_curve(calls, accs, vals, grid):
    """Metric of the candidate GEPA would return at each budget = its accuracy-best-so-far."""
    out = np.empty_like(grid, dtype=float)
    for k, b in enumerate(grid):
        elig = np.where(calls <= b)[0]
        if len(elig) == 0:
            out[k] = np.nan
            continue
        sel = elig[int(np.argmax(accs[elig]))]
        out[k] = vals[sel]
    return out


runs = [load(d) for d in DIRS]
start = runs[0][2][0]
gmax = max(c.max() for c, _, _ in runs)
grid = np.arange(0, gmax + 1, 10)

curves = np.array([returned_curve(c, a, v, grid) for c, a, v in runs])
top = np.nanmax(curves, axis=0)      # best of the 4 runs at each budget
bot = np.nanmin(curves, axis=0)      # worst of the 4 runs at each budget
mean_curve = np.nanmean(curves, axis=0)
best_f, worst_f, avg_f = top[-1], bot[-1], mean_curve[-1]

is_rec = args.metric == "unfair_recall"
ylo, yhi = (45, 100) if is_rec else (55, 95)
ylab = "Unfair-recall — violations caught (validation)" if is_rec else "Validation accuracy"
title = args.title or (
    "GEPA over a 6,000-call budget   ·   unfair-recall (best–worst band + average)" if is_rec
    else "GEPA over a 6,000-call budget   ·   validation accuracy (best–worst band + average)")

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=130)

ax.fill_between(grid, bot * 100, top * 100, color=BAND, alpha=0.20, zorder=1,
                label="Best–worst range (4 runs)")
ax.plot(grid, top * 100, color=BAND, lw=1.2, zorder=2)
ax.plot(grid, bot * 100, color=BAND, lw=1.2, zorder=2)
ax.plot(grid, mean_curve * 100, color=BLUE, lw=2.8, zorder=4,
        label=f"Average of 4 runs  →{avg_f*100:.1f}%")
ax.axhline(start * 100, ls="--", lw=1.3, color=DARK, alpha=0.7, zorder=3,
           label=f"Starting prompt, no GEPA ({start*100:.1f}%)")

ax.text(gmax, best_f * 100 + 0.5, f"best run  {best_f*100:.1f}%",
        ha="right", va="bottom", color=GREEN, fontsize=11, fontweight="bold")
ax.text(gmax, worst_f * 100 - 1.8, f"worst run  {worst_f*100:.1f}%",
        ha="right", va="top", color=ORANGE, fontsize=11, fontweight="bold")

ax.set_xlim(-gmax * 0.015, gmax * 1.02)
ax.set_ylim(ylo, yhi)
ax.set_xlabel("Optimisation budget  (metric calls)", fontsize=12, color=DARK)
ax.set_ylabel(ylab, fontsize=12, color=DARK)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title(title, fontsize=13.5, color=DARK, pad=14)

for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=DARK)
ax.grid(axis="y", color="#e2e8f0", lw=0.8, zorder=0)
ax.legend(frameon=False, fontsize=10.5, loc="lower right")

fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, bbox_inches="tight", facecolor="white")
print(f"wrote {args.out}  [{args.metric}]")
print(f"runs={len(runs)}  start={start:.4f}  best={best_f:.4f}  worst={worst_f:.4f}  avg={avg_f:.4f}")
