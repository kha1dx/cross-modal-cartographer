"""Regenerate main_results.png for Chapter 5 from the Notion-recorded numbers.

Run from V0/ with the project venv activated:
    source .venv/bin/activate
    python generate_main_results.py

Writes ../SS26_MyThesisTemplate_Ashry_NeSy4/img/main_results.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "SS26_MyThesisTemplate_Ashry_NeSy4" / "img" / "main_results.png"

CONDITIONS = ["E1\nText\n(precise)", "E2\nText\n(vague)", "E3\nSketch + vague\n($\\alpha=0.3$)", "E4\nSketch + vague\n($\\alpha=0.7$)"]
P_AT_1 = [0.833, 0.667, 0.833, 0.667]
P_AT_5 = [0.933, 0.533, 0.667, 0.533]
MEAN_SIM = [0.340, 0.310, 0.487, 0.683]
TAU = 0.60

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.15, 1.0]})

# Left panel: Precision@k grouped bars
x = np.arange(len(CONDITIONS))
width = 0.36
bars1 = ax_left.bar(x - width / 2, P_AT_1, width, label=r"$P@1$", color="#1f77b4", edgecolor="black", linewidth=0.6)
bars5 = ax_left.bar(x + width / 2, P_AT_5, width, label=r"$P@5$", color="#aec7e8", edgecolor="black", linewidth=0.6)
ax_left.set_ylim(0, 1.05)
ax_left.set_ylabel("Precision")
ax_left.set_xticks(x)
ax_left.set_xticklabels(CONDITIONS, fontsize=9)
ax_left.set_title("Class-level retrieval accuracy", fontsize=11)
ax_left.grid(axis="y", linestyle=":", alpha=0.5)
ax_left.legend(loc="upper right", frameon=False)
for bar in list(bars1) + list(bars5):
    h = bar.get_height()
    ax_left.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

# Right panel: mean top-5 cosine similarity with tau line
colors = ["#d62728", "#d62728", "#d62728", "#2ca02c"]  # E1-E3 below tau (red), E4 above tau (green)
bars_sim = ax_right.bar(x, MEAN_SIM, width=0.55, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
ax_right.axhline(TAU, color="black", linestyle="--", linewidth=1.2)
ax_right.text(len(CONDITIONS) - 0.5, TAU + 0.02, r"$\tau = 0.60$", ha="right", va="bottom", fontsize=10)
ax_right.set_ylim(0, 0.85)
ax_right.set_ylabel("Mean top-5 cosine similarity")
ax_right.set_xticks(x)
ax_right.set_xticklabels(CONDITIONS, fontsize=9)
ax_right.set_title("Verification regime (modality gap)", fontsize=11)
ax_right.grid(axis="y", linestyle=":", alpha=0.5)
for bar, val in zip(bars_sim, MEAN_SIM):
    ax_right.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

# Annotate which conditions clear tau
ax_right.text(1.0, 0.05, r"below $\tau$: 0/90 accepted, all $\tau$-rejected", ha="center", va="bottom",
              fontsize=9, color="#a32424")
ax_right.annotate("above $\\tau$: 25/30 accepted", xy=(3, 0.683), xytext=(3, 0.79), ha="center", fontsize=9,
                  color="#1f6f1f", arrowprops=dict(arrowstyle="-", color="#1f6f1f", lw=0.8))

fig.suptitle("Modality ablation: accuracy vs verification confidence", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Wrote {OUT}")
