#!/usr/bin/env python3
"""
gen_figures_v8.py — Master figure generator for CEF paper.

Generates all 4 figures with PDF + PNG + SVG output:
  fig1_dissociation      — N=20 scatter (Completion vs WMF-AM, WMF-AM vs Agent)
  fig2_depth_profile     — N=15 depth K profile
  fig3_framework         — CEF taxonomy diagram (adds SVG)
  fig4_validity_heatmap  — Kendall tau validity matrix

Usage:
    python gen_figures_v8.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Shared academic style
# ─────────────────────────────────────────────────────────────────────────────
SERIF_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

SANS_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def save_fig(fig, stem, dpi_png=300):
    """Save figure as PDF, PNG, and SVG."""
    for ext in ("pdf", "png", "svg"):
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = dpi_png
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, **kw)
        print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Dissociation scatter (N=20)
# ─────────────────────────────────────────────────────────────────────────────
def gen_fig1():
    plt.rcParams.update(SERIF_STYLE)

    # Data: (completion, wmf_am, agent_overall)
    orig = {
        "qwen2.5:7b":      (0.870, 0.350, 0.90),
        "qwen2.5:14b":     (0.920, 0.467, 0.90),
        "qwen2.5:32b":     (0.910, 0.650, 0.90),
        "llama3.1:8b":     (0.780, 0.183, 0.60),
        "gemma2:27b":      (0.830, 0.450, 0.80),
        "deepseek-r1:14b": (0.840, 0.983, 0.70),
        "mistral:7b":      (0.860, 0.350, 0.30),
    }
    expan = {
        "phi3:14b":      (0.790, 0.267, 0.20),
        "gemma2:9b":     (0.750, 0.400, 0.90),
        "qwen2.5:3b":    (0.820, 0.200, 0.40),
        "llama3.2:3b":   (0.820, 0.133, 0.30),
        "deepseek-r1:7b":(0.760, 0.150, 0.40),
        "mixtral:8x7b":  (0.880, 0.300, 0.40),
        "command-r:35b": (0.810, 0.350, 0.70),
        "yi:34b":        (0.880, 0.250, 0.30),
    }
    small = {
        "qwen2.5:0.5b":  (0.580, 0.050, 0.00),
        "qwen2.5:1.5b":  (0.800, 0.117, 0.30),
        "llama3.2:1b":   (0.720, 0.067, 0.20),
        "gemma2:2b":     (0.720, 0.217, 0.40),
        "tinyllama:1.1b":(0.440, 0.117, 0.00),
    }

    abbrev = {
        "qwen2.5:7b": "qw7b", "qwen2.5:14b": "qw14b", "qwen2.5:32b": "qw32b",
        "llama3.1:8b": "ll8b", "gemma2:27b": "gm27b",
        "deepseek-r1:14b": "ds-r1:14b", "mistral:7b": "mis7b",
        "phi3:14b": "phi14b", "gemma2:9b": "gm9b", "qwen2.5:3b": "qw3b",
        "llama3.2:3b": "ll3b", "deepseek-r1:7b": "ds-r1:7b",
        "mixtral:8x7b": "mix8x7b", "command-r:35b": "cmd35b", "yi:34b": "yi34b",
        "qwen2.5:0.5b": "qw0.5b", "qwen2.5:1.5b": "qw1.5b",
        "llama3.2:1b": "ll1b", "gemma2:2b": "gm2b", "tinyllama:1.1b": "tl1.1b",
    }

    # Colors
    C_ORIG  = "#1f77b4"
    C_EXPAN = "#e67e22"
    C_SMALL = "#2ca02c"
    C_ORIG_TXT  = "#1f77b4"
    C_EXPAN_TXT = "#c0650a"
    C_SMALL_TXT = "#1a7a1a"

    # Per-model label offsets for Panel A (completion vs wmf_am).
    # Format: (ox, oy, ha)  — ha='right' means text right-edge lands at cx+ox,
    # so the label appears to the LEFT of that anchor point (avoids right-axis overflow).
    off_A = {
        # orig group
        "qwen2.5:7b":      (-0.004,  0.025, "right"),  # right-align; stays left of right edge
        "qwen2.5:14b":     (-0.004, -0.050, "right"),  # right-align below; cx=0.920 near edge
        "qwen2.5:32b":     (-0.004,  0.025, "right"),  # right-align above; cx=0.910
        "llama3.1:8b":     (-0.004,  0.025, "right"),  # right-align; cx=0.780
        "gemma2:27b":      ( 0.008,  0.025, "left"),
        "deepseek-r1:14b": (-0.030,  0.040, "left"),   # above top point, shifted left
        "mistral:7b":      ( 0.008, -0.055, "left"),   # below; same y as qw7b so go down
        # expan group
        "phi3:14b":        ( 0.008,  0.025, "left"),
        "gemma2:9b":       (-0.004,  0.025, "right"),
        "qwen2.5:3b":      (-0.004, -0.050, "right"),  # right-align below; cx=0.820
        "llama3.2:3b":     ( 0.008, -0.055, "left"),   # below; cx=0.820, low wmf
        "deepseek-r1:7b":  ( 0.008, -0.055, "left"),   # below; cx=0.760
        "mixtral:8x7b":    (-0.004,  0.025, "right"),  # right-align; cx=0.880
        "command-r:35b":   (-0.004,  0.025, "right"),  # right-align; avoids clash w/ qw7b
        "yi:34b":          ( 0.008, -0.055, "left"),   # below; cx=0.880
        # small group
        "qwen2.5:0.5b":    ( 0.008,  0.025, "left"),   # isolated, no issue
        "qwen2.5:1.5b":    ( 0.008,  0.025, "left"),
        "llama3.2:1b":     (-0.004,  0.025, "right"),  # right-align; cx=0.720
        "gemma2:2b":       ( 0.008,  0.025, "left"),
        "tinyllama:1.1b":  ( 0.008,  0.025, "left"),   # isolated lower-left
    }

    # Per-model label offsets for Panel B (wmf_am vs agent).
    # Many models share the same discrete agent score; alternate above/below to avoid stacking.
    off_B = {
        # orig group
        "qwen2.5:7b":      ( 0.015,  0.030, "left"),   # agent=0.90, above
        "qwen2.5:14b":     ( 0.015,  0.030, "left"),   # agent=0.90, above (x=0.467 clear)
        "qwen2.5:32b":     ( 0.015,  0.030, "left"),   # agent=0.90, above (x=0.650 rightmost)
        "llama3.1:8b":     ( 0.015,  0.030, "left"),   # agent=0.60, isolated
        "gemma2:27b":      ( 0.015,  0.030, "left"),   # agent=0.80, isolated
        "deepseek-r1:14b": (-0.015,  0.030, "right"),  # agent=0.70, wmf=0.983; right-align
        "mistral:7b":      ( 0.015, -0.058, "left"),   # agent=0.30, below
        # expan group
        "phi3:14b":        ( 0.015,  0.030, "left"),   # agent=0.20
        "gemma2:9b":       ( 0.015, -0.058, "left"),   # agent=0.90, below; stagger from qw7b/qw14b
        "qwen2.5:3b":      ( 0.015,  0.030, "left"),   # agent=0.40, above
        "llama3.2:3b":     ( 0.015, -0.058, "left"),   # agent=0.30, below
        "deepseek-r1:7b":  (-0.015, -0.058, "right"),  # agent=0.40, right-align below; crowded x
        "mixtral:8x7b":    ( 0.015,  0.030, "left"),   # agent=0.40, above (x=0.300 clear)
        "command-r:35b":   ( 0.015,  0.030, "left"),   # agent=0.70
        "yi:34b":          ( 0.015,  0.030, "left"),   # agent=0.30, above; x=0.250
        # small group
        "qwen2.5:0.5b":    ( 0.015,  0.030, "left"),   # agent=0.00, above; x=0.050
        "qwen2.5:1.5b":    ( 0.015,  0.030, "left"),   # agent=0.30, above; x=0.117
        "llama3.2:1b":     ( 0.015,  0.030, "left"),   # agent=0.20; x=0.067
        "gemma2:2b":       ( 0.015, -0.058, "left"),   # agent=0.40, below; x=0.217
        "tinyllama:1.1b":  ( 0.015, -0.042, "left"),   # agent=0.00, below (just above y-floor)
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Completion vs WMF-AM
    for m, (cx, cy, _) in orig.items():
        ax1.plot(cx, cy, 'o', color=C_ORIG, markersize=5, zorder=3)
        ox, oy, ha = off_A[m]
        ax1.text(cx + ox, cy + oy, abbrev[m], fontsize=6.5, color=C_ORIG_TXT,
                 ha=ha, zorder=4)

    for m, (cx, cy, _) in expan.items():
        ax1.plot(cx, cy, 's', color=C_EXPAN, markersize=4.5, zorder=3)
        ox, oy, ha = off_A[m]
        ax1.text(cx + ox, cy + oy, abbrev[m], fontsize=6.5, color=C_EXPAN_TXT,
                 ha=ha, zorder=4)

    for m, (cx, cy, _) in small.items():
        ax1.plot(cx, cy, '^', color=C_SMALL, markersize=4.5, zorder=3)
        ox, oy, ha = off_A[m]
        ax1.text(cx + ox, cy + oy, abbrev[m], fontsize=6.5, color=C_SMALL_TXT,
                 ha=ha, zorder=4)

    ax1.text(0.97, 0.03,
             r"$\tau = 0.279,\ p = 0.150$" + "\n" + r"($N{=}15$, $\geq$3B models)",
             transform=ax1.transAxes, fontsize=7.5, ha="right", va="bottom",
             bbox=dict(facecolor="white", edgecolor="#999999", linewidth=0.5, pad=3, alpha=0.85))

    h1 = ax1.plot([], [], 'o', color=C_ORIG,  markersize=5,   label="Original 7")[0]
    h2 = ax1.plot([], [], 's', color=C_EXPAN, markersize=4.5, label="Expansion 8")[0]
    h3 = ax1.plot([], [], '^', color=C_SMALL, markersize=4.5, label="Small (0.5B\u20132B)")[0]
    ax1.legend(handles=[h1, h2, h3], loc="upper left", fontsize=7.5,
               framealpha=0.9, edgecolor="#cccccc")

    ax1.set_xlabel("Completion Score (100-item battery)")
    ax1.set_ylabel("WMF-AM Score (multi-seed mean)")
    ax1.set_xlim(0.40, 0.96)
    ax1.set_ylim(-0.02, 1.08)
    ax1.set_title("(a) Completion vs WMF-AM", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: WMF-AM vs Agent Performance
    for m, (cx, cy, ag) in orig.items():
        ax2.plot(cy, ag, 'o', color=C_ORIG, markersize=5, zorder=3)
        ox, oy, ha = off_B[m]
        ax2.text(cy + ox, ag + oy, abbrev[m], fontsize=6.5, color=C_ORIG_TXT,
                 ha=ha, zorder=4)

    for m, (cx, cy, ag) in expan.items():
        ax2.plot(cy, ag, 's', color=C_EXPAN, markersize=4.5, zorder=3)
        ox, oy, ha = off_B[m]
        ax2.text(cy + ox, ag + oy, abbrev[m], fontsize=6.5, color=C_EXPAN_TXT,
                 ha=ha, zorder=4)

    for m, (cx, cy, ag) in small.items():
        ax2.plot(cy, ag, '^', color=C_SMALL, markersize=4.5, zorder=3)
        ox, oy, ha = off_B[m]
        ax2.text(cy + ox, ag + oy, abbrev[m], fontsize=6.5, color=C_SMALL_TXT,
                 ha=ha, zorder=4)

    ax2.text(0.03, 0.97,
             r"$\tau = 0.612,\ p = 0.0003$" + "\n" +
             r"partial $\tau|$compl $= 0.411,\ p = 0.011$",
             transform=ax2.transAxes, fontsize=7.5, ha="left", va="top",
             bbox=dict(facecolor="white", edgecolor="#999999", linewidth=0.5, pad=3, alpha=0.85))

    ax2.set_xlabel("WMF-AM Score (multi-seed mean)")
    ax2.set_ylabel("Agent Task Performance")
    ax2.set_xlim(-0.02, 1.08)
    ax2.set_ylim(-0.05, 1.00)
    ax2.set_title("(b) WMF-AM predicts agent performance", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    print("Fig 1:")
    save_fig(fig, "fig1_dissociation")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Depth profile (N=15)
# ─────────────────────────────────────────────────────────────────────────────
def gen_fig2():
    plt.rcParams.update(SERIF_STYLE)

    data = {
        "qwen2.5:7b":      (0.733, 0.400, 0.133),
        "qwen2.5:14b":     (0.933, 0.400, 0.333),
        "qwen2.5:32b":     (1.000, 0.600, 0.400),
        "llama3.1:8b":     (0.800, 0.400, 0.200),
        "gemma2:27b":      (0.800, 0.400, 0.200),
        "deepseek-r1:14b": (1.000, 1.000, 0.933),
        "mistral:7b":      (0.933, 0.467, 0.200),
        "phi3:14b":        (0.800, 0.400, 0.133),
        "gemma2:9b":       (0.800, 0.467, 0.333),
        "qwen2.5:3b":      (0.400, 0.200, 0.000),
        "llama3.2:3b":     (0.800, 0.200, 0.067),
        "deepseek-r1:7b":  (0.844, 0.200, 0.000),
        "mixtral:8x7b":    (0.800, 0.600, 0.267),
        "command-r:35b":   (0.800, 0.600, 0.267),
        "yi:34b":          (0.600, 0.500, 0.400),
    }

    means = {m: np.mean(v) for m, v in data.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    top5 = ranked[:5]

    top5_style = {
        "deepseek-r1:14b": ("#d62728", "o", "-",   2.5, 7),
        "qwen2.5:32b":     ("#1f77b4", "s", "-",   1.8, 5.5),
        "mixtral:8x7b":    ("#2ca02c", "^", "--",  1.8, 5.5),
        "command-r:35b":   ("#9467bd", "D", "-.",  1.8, 5),
        "qwen2.5:14b":     ("#ff7f0e", "v", "-",   1.8, 5.5),
    }
    palette_fb = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    ci = 0
    for m in top5:
        if m not in top5_style:
            top5_style[m] = (palette_fb[ci % len(palette_fb)], "o", "-", 1.8, 5)
            ci += 1

    depths = [3, 5, 7]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    # Background models
    for m in ranked:
        if m in top5:
            continue
        vals = data[m]
        ax.plot(depths, vals, color="#cccccc", linewidth=1.0, alpha=0.7,
                marker='o', markersize=3, zorder=1)

    h_other = ax.plot([], [], color="#cccccc", linewidth=1.0, marker='o',
                      markersize=3, label=f"Other models (N={15 - len(top5)})")[0]

    # Top 5
    handles_top = []
    for m in top5:
        vals = data[m]
        col, mkr, ls, lw, ms = top5_style[m]
        h, = ax.plot(depths, vals, color=col, linewidth=lw, linestyle=ls,
                     marker=mkr, markersize=ms, zorder=3, label=m)
        handles_top.append(h)

    ax.legend(handles=handles_top + [h_other], fontsize=7.5, loc="upper right",
              framealpha=0.9, edgecolor="#cccccc")

    ax.set_xlabel("Depth K (number of state-update operations)")
    ax.set_ylabel("WMF-AM Accuracy")
    ax.set_xticks(depths)
    ax.set_xticklabels(["K = 3", "K = 5", "K = 7"])
    ax.set_xlim(2.5, 7.5)
    ax.set_ylim(-0.03, 1.06)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: deepseek-r1 robustness
    ax.annotate("Reasoning arch.\nrobust at K=7",
                xy=(7, data["deepseek-r1:14b"][2]),
                xytext=(4.8, 0.75),
                arrowprops=dict(arrowstyle="-|>", color="#d62728",
                                lw=1.0, mutation_scale=9,
                                connectionstyle="arc3,rad=-0.2"),
                fontsize=7, color="#d62728", ha="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#d62728", linewidth=0.7, alpha=0.9))

    fig.tight_layout()
    print("Fig 2:")
    save_fig(fig, "fig2_depth_profile")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — CEF Framework diagram (now also saves SVG)
# ─────────────────────────────────────────────────────────────────────────────
def gen_fig3():
    plt.rcParams.update(SANS_STYLE)

    FIG_W, FIG_H = 10.0, 5.0
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ROOT_FILL  = "#2C3E50"
    DIM_COLORS = ["#2980B9", "#16A085", "#8E44AD", "#D35400"]
    DIM_LIGHT  = ["#D6EAF8", "#D1F2EB", "#E8DAEF", "#FAE5D3"]
    CONN_COLOR = "#95A5A6"

    ROOT_X, ROOT_Y, ROOT_W, ROOT_H = 2.5, 4.35, 5.0, 0.48
    COL_XS  = [0.60, 2.90, 5.65, 8.40]
    DIM_W   = 1.80
    DIM_H   = 0.78
    DIM_Y   = 3.32
    SUB_W   = 1.62
    SUB_H   = 0.36
    SUB_GAP = 0.12
    SUB_TOP_Y = 2.55

    DIMS = [
        {"abbr": "WMF", "name": "Working Memory\nFidelity",
         "anchor": "Miller 1956 · Baddeley 1974",
         "subs": [("AM", True), ("IM", False), ("IR", False)]},
        {"abbr": "MCC", "name": "Metacognitive\nCalibration",
         "anchor": "Nelson & Narens 1990",
         "subs": [("MA", True), ("CE", False)]},
        {"abbr": "EMC", "name": "Episodic Memory\nCoherence",
         "anchor": "Tulving 1972 · Johnson 1993",
         "subs": [("TO", True), ("EI", True), ("SA", False)]},
        {"abbr": "CLA", "name": "Cognitive Load\nAdaptation",
         "anchor": "Sweller 1988",
         "subs": [("DC", True), ("RA", True), ("CR", True)]},
    ]

    def rounded_box(cx, cy, w, h, fc, ec, lw=1.2, radius=0.06, alpha=1.0, zorder=3):
        patch = FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle=f"round,pad={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            zorder=zorder, alpha=alpha)
        ax.add_patch(patch)

    def conn(x1, y1, x2, y2, lw=1.0):
        ax.plot([x1, x2], [y1, y2], color=CONN_COLOR, lw=lw, zorder=1,
                solid_capstyle="round")

    root_cx = ROOT_X + ROOT_W / 2
    root_cy = ROOT_Y + ROOT_H / 2

    rounded_box(root_cx, root_cy, ROOT_W, ROOT_H,
                fc=ROOT_FILL, ec=ROOT_FILL, lw=0, radius=0.06)
    ax.text(root_cx, root_cy, "Cognitive Evaluation Framework (CEF)",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color="white", zorder=4)

    spine_y = 3.18
    conn(root_cx, ROOT_Y, root_cx, spine_y + 0.04, lw=1.2)
    conn(COL_XS[0], spine_y, COL_XS[-1], spine_y, lw=1.2)
    for cx in COL_XS:
        conn(cx, spine_y, cx, DIM_Y + DIM_H / 2, lw=1.2)

    for i, (dim, cx, dc, dl) in enumerate(zip(DIMS, COL_XS, DIM_COLORS, DIM_LIGHT)):
        cy = DIM_Y
        rounded_box(cx + 0.04, cy - 0.04, DIM_W, DIM_H,
                    fc="#D0D3D4", ec="none", lw=0, radius=0.07, zorder=2)
        rounded_box(cx, cy, DIM_W, DIM_H,
                    fc=dc, ec=dc, lw=0, radius=0.07, zorder=3)

        ax.text(cx, cy + 0.17, dim["abbr"],
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white", zorder=4)
        ax.text(cx, cy - 0.04, dim["name"],
                ha="center", va="center", fontsize=7.5, color="white",
                linespacing=1.3, zorder=4)
        ax.text(cx, cy - 0.31, dim["anchor"],
                ha="center", va="center", fontsize=6.2,
                color="white", alpha=0.80, style="italic", zorder=4)

        sub_spine_y = DIM_Y - DIM_H / 2 - 0.04
        conn(cx, DIM_Y - DIM_H / 2, cx, sub_spine_y, lw=1.0)

        for j, (sub_abbr, is_t1) in enumerate(dim["subs"]):
            sub_cy = SUB_TOP_Y - j * (SUB_H + SUB_GAP)
            if j == 0:
                conn(cx, sub_spine_y, cx, sub_cy + SUB_H / 2, lw=1.0)
            else:
                prev_cy = SUB_TOP_Y - (j-1) * (SUB_H + SUB_GAP)
                conn(cx, prev_cy - SUB_H / 2, cx, sub_cy + SUB_H / 2, lw=1.0)

            rounded_box(cx, sub_cy, SUB_W, SUB_H,
                        fc="white", ec=dc, lw=1.5, radius=0.04, zorder=3)

            badge_x  = cx + SUB_W / 2 - 0.26
            badge_y  = sub_cy
            badge_fc = dc if is_t1 else "#BDC3C7"
            rounded_box(badge_x, badge_y, 0.36, 0.22,
                        fc=badge_fc, ec="none", lw=0, radius=0.03, zorder=4)
            ax.text(badge_x, badge_y, "T1" if is_t1 else "T2",
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white", zorder=5)

            ax.text(cx - 0.12, sub_cy, f"{dim['abbr']}-{sub_abbr}",
                    ha="center", va="center", fontsize=9, fontweight="semibold",
                    color="#2C3E50", zorder=4)

    # Legend
    legend_y = 0.26
    lx = 2.80
    for fc, label in [(DIM_COLORS[0], "T1  Tested in pilot"),
                      ("#BDC3C7",     "T2  Protocol specified, full study pending")]:
        rounded_box(lx, legend_y, 0.32, 0.22,
                    fc=fc, ec="none", lw=0, radius=0.03, zorder=3)
        ax.text(lx + 0.24, legend_y, label,
                ha="left", va="center", fontsize=8.0, color="#555")
        lx += 2.95

    fig.tight_layout(pad=0.3)
    print("Fig 3:")
    save_fig(fig, "fig3_framework", dpi_png=200)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Validity heatmap
# ─────────────────────────────────────────────────────────────────────────────
def gen_fig4():
    plt.rcParams.update(SERIF_STYLE)

    MEASURES = ["WMF-AM", "WMF-Yoked", "MCC-MA", "Completion", "AGENT-PQ"]
    LABELS   = ["WMF-AM", "WMF-Yoked", "MCC-MA", "Compl.", "Agent-PQ"]

    tau_data = {
        ("WMF-AM",    "WMF-AM"):    1.0,
        ("WMF-Yoked", "WMF-Yoked"): 1.0,
        ("MCC-MA",    "MCC-MA"):    1.0,
        ("Completion","Completion"): 1.0,
        ("AGENT-PQ",  "AGENT-PQ"):  1.0,
        # N=20 (final)
        ("WMF-AM",    "AGENT-PQ"):  0.612,   # p<0.001; main result
        ("Completion","AGENT-PQ"):  0.428,   # p=0.012; N=20
        # N=15 (≥3B models)
        ("WMF-AM",    "Completion"): 0.279,  # p=0.150 (n.s.)
        ("WMF-Yoked", "Completion"): 0.408,  # p=0.047
        # N=7 (pilot; pending N=15 full update)
        ("WMF-AM",    "MCC-MA"):    0.169,
        ("WMF-AM",    "WMF-Yoked"): 0.600,
        ("WMF-Yoked", "MCC-MA"):    0.100,
        ("WMF-Yoked", "AGENT-PQ"):  0.200,
        ("MCC-MA",    "Completion"):-0.183,
        ("MCC-MA",    "AGENT-PQ"):  -0.282,
    }
    sig_data = {
        ("WMF-AM",    "AGENT-PQ"):  True,   # p<0.001, N=20
        ("WMF-Yoked", "Completion"): True,  # p=0.047, N=15
        ("Completion","AGENT-PQ"):  True,   # p=0.012, N=20
    }

    n = len(MEASURES)
    mat = np.full((n, n), np.nan)
    for i, m1 in enumerate(MEASURES):
        for j, m2 in enumerate(MEASURES):
            key = (m1, m2) if (m1, m2) in tau_data else (m2, m1)
            if key in tau_data:
                mat[i, j] = tau_data[key]

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    fig.patch.set_facecolor("white")

    im = ax.imshow(mat, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if np.isnan(val) or i == j:
                continue
            key     = (MEASURES[i], MEASURES[j])
            key_rev = (MEASURES[j], MEASURES[i])
            is_sig  = sig_data.get(key, False) or sig_data.get(key_rev, False)
            txt    = f"{val:.2f}" + ("*" if is_sig else "")
            color  = "white" if abs(val) > 0.65 else "black"
            weight = "bold" if is_sig else "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(LABELS, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(r"Kendall's $\tau$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", linewidth=1.5)
        ax.axvline(k - 0.5, color="white", linewidth=1.5)

    ax.spines[:].set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    rect = Rectangle((-0.5, -0.5), 3, 3, linewidth=2, edgecolor="#333333",
                     facecolor="none", linestyle="--", zorder=5)
    ax.add_patch(rect)
    ax.text(1.0, -0.75, "CEF dimensions", ha="center", va="top",
            fontsize=7.5, color="#333333", style="italic")

    ax.text(0.0, 1.02, r"$\dagger$ $N{=}20$; others $N{=}7$--$15$; * $p < 0.05$",
            transform=ax.transAxes, fontsize=7.5, color="#666666",
            ha="left", va="bottom")

    fig.tight_layout(pad=0.5)
    print("Fig 4:")
    save_fig(fig, "fig4_validity_heatmap")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== gen_figures_v8.py ===")
    gen_fig1()
    gen_fig2()
    gen_fig3()
    gen_fig4()
    print(f"\nAll figures saved to: {OUT}")
    print("Format: PDF + PNG + SVG for each figure")
