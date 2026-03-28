"""
gen_completion_fallacy_illustration.py

Redesigned fig_completion_fallacy: clean 2-panel layout.
Left panel:  "What completion sees" — Model A and B both score 0.90
Right panel: "What WMF-AM reveals" — Model A: 0.983, Model B: 0.300

Size: 7 × 3.6 inches (fits in a paper column or half-width).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# ── Colorblind-safe palette (Wong 2011) ───────────────────────────────────────
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
VERMIL = "#D55E00"
GRAY   = "#777777"
LGRAY  = "#DDDDDD"

OUT_DIR = Path(__file__).parent

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.2, 3.4),
                                          gridspec_kw={"wspace": 0.42})

MODELS   = ["deepseek-r1:14b", "mixtral:8x7b"]
ABBREVS  = ["deepseek-r1\n:14b", "mixtral\n:8x7b"]
X        = np.array([0, 1])
BAR_W    = 0.38

# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — Completion (both 0.90)
# ─────────────────────────────────────────────────────────────────────────────
comp_scores = [0.90, 0.90]
bars_l = ax_left.bar(X, comp_scores, width=BAR_W,
                     color=[GREEN, GREEN], edgecolor="white",
                     linewidth=0.8, zorder=3)

ax_left.set_ylim(0, 1.18)
ax_left.set_xlim(-0.55, 1.55)
ax_left.set_xticks(X)
ax_left.set_xticklabels(ABBREVS, fontsize=8.5)
ax_left.set_ylabel(r"Task Completion Score", fontsize=9)
ax_left.set_title("What Completion Sees", fontsize=10, fontweight="bold", pad=6)
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)
ax_left.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
ax_left.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

# Value labels above bars
for bar, val in zip(bars_l, comp_scores):
    ax_left.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.025, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color=GREEN)

# "Indistinguishable" annotation
ax_left.annotate("", xy=(X[1] + BAR_W/2, 1.09),
                 xytext=(X[0] - BAR_W/2, 1.09),
                 arrowprops=dict(arrowstyle="<->", color=GREEN,
                                 lw=1.5, mutation_scale=10))
ax_left.text(0.5, 1.10, "Indistinguishable",
             ha="center", va="bottom", fontsize=8,
             color=GREEN, fontweight="bold",
             transform=ax_left.get_xaxis_transform())

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — WMF-AM (0.983 vs 0.300)
# ─────────────────────────────────────────────────────────────────────────────
wmf_scores = [0.983, 0.300]
bar_colors = [BLUE, VERMIL]
bars_r = ax_right.bar(X, wmf_scores, width=BAR_W,
                      color=bar_colors, edgecolor="white",
                      linewidth=0.8, zorder=3)

ax_right.set_ylim(0, 1.18)
ax_right.set_xlim(-0.55, 1.55)
ax_right.set_xticks(X)
ax_right.set_xticklabels(ABBREVS, fontsize=8.5)
ax_right.set_ylabel("WMF-AM Score", fontsize=9)
ax_right.set_title("What WMF-AM Reveals", fontsize=10, fontweight="bold", pad=6)
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)
ax_right.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

# Value labels
label_colors = [BLUE, VERMIL]
for bar, val, col in zip(bars_r, wmf_scores, label_colors):
    ax_right.text(bar.get_x() + bar.get_width() / 2,
                  val + 0.025, f"{val:.3f}",
                  ha="center", va="bottom", fontsize=9, fontweight="bold",
                  color=col)

# ΔWMF-AM annotation
ax_right.annotate("", xy=(X[0] + BAR_W/2, wmf_scores[0]),
                  xytext=(X[1] + BAR_W/2, wmf_scores[1]),
                  arrowprops=dict(arrowstyle="<->", color=GRAY,
                                  lw=1.4, mutation_scale=10))
mid_x = (X[0] + BAR_W/2 + X[1] + BAR_W/2) / 2 + 0.18
mid_y = (wmf_scores[0] + wmf_scores[1]) / 2
ax_right.text(mid_x + 0.02, mid_y,
              r"$\Delta = 0.683$",
              ha="left", va="center", fontsize=8.5, color=GRAY,
              style="italic")

# Sub-labels: blue inside bar (tall), orange below bar (too short)
ax_right.text(X[0], wmf_scores[0] / 2,
              "Genuine\nstate\ntracking",
              ha="center", va="center", fontsize=7.5, color="white",
              fontweight="bold", style="italic")
ax_right.text(X[1] + BAR_W/2 + 0.08, 0.55,
              "Per-step\nshortcuts",
              ha="left", va="center", fontsize=7.5, color=VERMIL,
              style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# Shared caption below figure
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.5, -0.04,
         r"Both models complete the same task at 0.90 accuracy, yet WMF-AM reveals"
         "\n"
         r"a 0.683-point gap in cumulative arithmetic state-tracking competence.",
         ha="center", va="top", fontsize=8, color=GRAY, style="italic")

fig.tight_layout(rect=[0, 0.0, 1, 1])

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
for fmt in ("pdf", "png", "svg"):
    out = OUT_DIR / f"fig_completion_fallacy.{fmt}"
    kw = {"dpi": 300} if fmt == "png" else {}
    fig.savefig(out, format=fmt, bbox_inches="tight", facecolor="white", **kw)
    print(f"Saved: {out}")

plt.close()
