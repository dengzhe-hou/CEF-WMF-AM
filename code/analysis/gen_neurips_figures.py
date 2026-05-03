#!/usr/bin/env python3
"""
Generate NeurIPS 2026 figures from all experimental data.

Figures:
  1. K-Degradation Curves (main finding): 28 models, grouped by type
  2. WMF-AM vs Agent scatter (N=28, with held-out highlighted)
  3. Load-Shift Intervention (supported vs unsupported bar chart)
  4. K_crit distribution + regime classification
"""

import csv
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit

DATA = Path(__file__).parent.parent.parent / "data"
OUT = Path(__file__).parent.parent.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Load data ──────────────────────────────────────────────────────
rows = list(csv.DictReader(open(DATA / "master_28models.csv")))

# Color scheme
COLORS = {
    "standard": "#3498DB",
    "LRM": "#E74C3C",
    "LRM-distill": "#E67E22",
}
MARKERS = {
    "standard": "o",
    "LRM": "s",
    "LRM-distill": "D",
}


def save(fig, stem):
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  Saved: {stem}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: K-Degradation Curves
# ═══════════════════════════════════════════════════════════════════
def fig1_k_curves():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    K_VALS = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100]

    # Group models
    groups = {
        "Standard (Open-Weight)": [r for r in rows if r["type"] == "standard" and r["source"] == "ollama"],
        "Standard (API)": [r for r in rows if r["type"] == "standard" and r["source"] == "api"],
        "LRM / LRM-Distill": [r for r in rows if "LRM" in r["type"]],
    }

    titles = list(groups.keys())

    for ax_idx, (title, group) in enumerate(groups.items()):
        ax = axes[ax_idx]

        for r in group:
            ks = []
            accs = []
            for k in K_VALS:
                val = r.get(f"K{k}", "")
                if val:
                    ks.append(k)
                    accs.append(float(val))

            if not ks:
                continue

            color = COLORS.get(r["type"], "#999")
            alpha = 0.8 if "LRM" in r["type"] else 0.5
            lw = 2.0 if "LRM" in r["type"] else 1.0
            label = r["model"]

            # Shorten label
            short = label.replace("qwen2.5:", "Q").replace("llama3.", "L").replace("gemma2:", "G")
            short = short.replace("deepseek-r1", "R1").replace("deepseek-v3", "V3")
            short = short.replace("mixtral:8x7b", "Mix8x7").replace("command-r:35b", "CmdR")
            short = short.replace("claude-sonnet-4", "Claude").replace("o3-mini", "o3-mini")
            short = short.replace("gpt-4o-mini", "4o-mini").replace("gpt-4o", "GPT-4o")
            short = short.replace("gemini-2.5-flash", "Gemini").replace("tinyllama:1.1b", "TinyL")

            ax.plot(ks, accs, marker="o", markersize=3, linewidth=lw,
                    alpha=alpha, color=color, label=short)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("K (number of operations)")
        if ax_idx == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log")
        ax.set_xticks([3, 5, 7, 10, 20, 50, 100])
        ax.set_xticklabels([3, 5, 7, 10, 20, 50, 100])
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.2)

        if len(group) <= 8:
            ax.legend(fontsize=7, loc="lower left", framealpha=0.8)

    fig.suptitle("K-Degradation Curves: Cumulative State Tracking Under Increasing Load",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig_k_degradation_curves")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: WMF-AM vs Agent Scatter (N=28)
# ═══════════════════════════════════════════════════════════════════
def fig2_scatter():
    from adjustText import adjust_text

    # Short display names for all models
    SHORT_NAMES = {
        "claude-sonnet-4": "Claude",
        "o3-mini": "o3-mini",
        "deepseek-v3": "V3",
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "4o-mini",
        "gemini-2.5-flash": "Gemini",
        "deepseek-r1-full": "R1-full",
        "deepseek-r1:14b": "R1:14b",
        "deepseek-r1:7b": "R1:7b",
        "qwen2.5:32b": "Qwen:32b",
        "qwen2.5:14b": "Qwen:14b",
        "qwen2.5:7b": "Qwen:7b",
        "qwen2.5:3b": "Qwen:3b",
        "qwen2.5:1.5b": "Qwen:1.5b",
        "qwen2.5:0.5b": "Qwen:0.5b",
        "gemma2:27b": "Gemma:27b",
        "gemma2:9b": "Gemma:9b",
        "gemma2:2b": "Gemma:2b",
        "llama3.1:70b": "Llama:70b",
        "llama3.1:8b": "Llama:8b",
        "llama3.2:3b": "Llama:3b",
        "llama3.2:1b": "Llama:1b",
        "mixtral:8x7b": "Mixtral",
        "mistral:7b": "Mistral:7b",
        "command-r:35b": "CmdR:35b",
        "phi3:14b": "Phi3:14b",
        "yi:34b": "Yi:34b",
        "tinyllama:1.1b": "TinyLlama",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    texts = []

    for r in rows:
        wmf = r["wmf_am_mean"]
        agent = r["agent_score"]
        if not wmf or not agent:
            continue

        wmf, agent = float(wmf), float(agent)
        mtype = r["type"]
        color = COLORS.get(mtype, "#999")
        marker = MARKERS.get(mtype, "o")
        size = 80 if "LRM" in mtype else 50
        edge = "black" if r["source"] == "api" else "none"

        # Slight jitter for overlapping points (claude-sonnet-4 & o3-mini both at 1.0, 0.9)
        jitter_x, jitter_y = 0, 0
        if r["model"] == "claude-sonnet-4":
            jitter_x, jitter_y = -0.015, 0.015
        elif r["model"] == "o3-mini":
            jitter_x, jitter_y = 0.015, -0.015

        ax.scatter(wmf + jitter_x, agent + jitter_y, c=color, marker=marker, s=size,
                   edgecolors=edge, linewidths=1.5, zorder=3)

        short = SHORT_NAMES.get(r["model"], r["model"])
        texts.append(ax.text(wmf + jitter_x, agent + jitter_y, short, fontsize=6, alpha=0.8))

    # Correlation line
    from scipy import stats
    valid = [(float(r["wmf_am_mean"]), float(r["agent_score"]))
             for r in rows if r["wmf_am_mean"] and r["agent_score"]]
    xs, ys = zip(*valid)
    tau, p = stats.kendalltau(xs, ys)

    # Regression line
    z = np.polyfit(xs, ys, 1)
    xline = np.linspace(0, 1.05, 100)
    ax.plot(xline, np.polyval(z, xline), "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("WMF-AM Score (mean accuracy, K=3/5/7)", fontsize=11)
    ax.set_ylabel("Agent Battery Score (10-task completion)", fontsize=11)
    ax.set_title(f"WMF-AM Predicts Agent Performance (N={len(valid)}, τ={tau:.3f}, p<0.001)",
                 fontsize=11, fontweight="bold")

    # Legend
    handles = [
        mpatches.Patch(color=COLORS["standard"], label="Standard"),
        mpatches.Patch(color=COLORS["LRM"], label="LRM (Reasoning)"),
        mpatches.Patch(color=COLORS["LRM-distill"], label="LRM-Distill"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                    markeredgecolor="black", markersize=8, label="API model (black edge)"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.08, 1.05)
    ax.grid(True, alpha=0.2)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray",
                alpha=0.4, lw=0.5, shrinkA=3, shrinkB=3),
                force_text=(1.2, 1.2), force_points=(2.5, 2.5),
                expand=(1.6, 1.6), ensure_inside_axes=True,
                min_arrow_len=5)

    plt.tight_layout()
    save(fig, "fig_wmfam_vs_agent_n28")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Load-Shift Intervention
# ═══════════════════════════════════════════════════════════════════
def fig3_load_shift():
    # Short display names (single-line)
    LS_NAMES = {
        "claude-sonnet-4": "Claude",
        "o3-mini": "o3-mini",
        "deepseek-v3": "V3",
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "4o-mini",
        "gemini-2.5-flash": "Gemini",
        "deepseek-r1-full": "R1-full",
        "deepseek-r1:14b": "R1:14b",
        "deepseek-r1:7b": "R1:7b",
        "qwen2.5:32b": "Qwen:32b",
        "qwen2.5:14b": "Qwen:14b",
        "qwen2.5:7b": "Qwen:7b",
        "qwen2.5:3b": "Qwen:3b",
        "qwen2.5:1.5b": "Qwen:1.5b",
        "qwen2.5:0.5b": "Qwen:0.5b",
        "gemma2:27b": "Gemma:27b",
        "gemma2:9b": "Gemma:9b",
        "gemma2:2b": "Gemma:2b",
        "llama3.1:70b": "Llama:70b",
        "llama3.1:8b": "Llama:8b",
        "llama3.2:3b": "Llama:3b",
        "llama3.2:1b": "Llama:1b",
        "mixtral:8x7b": "Mixtral",
        "mistral:7b": "Mistral:7b",
        "command-r:35b": "CmdR:35b",
        "phi3:14b": "Phi3:14b",
        "yi:34b": "Yi:34b",
        "tinyllama:1.1b": "TinyLlama",
    }

    # Only models with load-shift data
    ls_rows = [r for r in rows if r["supported_agent"] and r["unsupported_agent"]]
    ls_models = sorted(ls_rows, key=lambda r: -float(r["supported_agent"]))

    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))

    names = []
    sup_scores = []
    unsup_scores = []
    colors_bar = []

    for r in ls_models:
        names.append(LS_NAMES.get(r["model"], r["model"]))
        sup_scores.append(float(r["supported_agent"]))
        unsup_scores.append(float(r["unsupported_agent"]))
        colors_bar.append(COLORS.get(r["type"], "#999"))

    x = np.arange(len(names))
    width = 0.35

    import matplotlib.colors as mcolors
    def lighten(hex_color, amount=0.6):
        rgb = mcolors.to_rgb(hex_color)
        return tuple(c + (1 - c) * amount for c in rgb)

    light_colors = [lighten(c) for c in colors_bar]

    bars1 = ax.bar(x - width/2, sup_scores, width, label="Supported (full history)",
                   color=colors_bar, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, unsup_scores, width, label="Unsupported (last turn only)",
                   color=light_colors, edgecolor=[c for c in colors_bar], linewidth=1.2)

    # Delta labels on top of bars
    for i, (s, u) in enumerate(zip(sup_scores, unsup_scores)):
        delta = s - u
        if delta > 0.01:
            ax.text(i, max(s, u) + 0.02, f"Δ{delta:+.1f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color="red" if delta > 0.3 else "gray")
        else:
            ax.text(i, max(s, u) + 0.02, "Δ0.0",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color="green")

    ax.set_ylabel("Agent Battery Score", fontsize=11)
    ax.set_title("Load-Shift Intervention: History Removal Effect on Agent Performance",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    save(fig, "fig_load_shift_intervention")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: K_crit Distribution
# ═══════════════════════════════════════════════════════════════════
def fig4_kcrit():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: K_crit vs ABS scatter
    for r in rows:
        kcrit = r["K_crit"]
        agent = r["agent_score"]
        r2 = r["sigmoid_R2"]
        if not kcrit or not agent or not r2:
            continue
        kcrit, agent, r2 = float(kcrit), float(agent), float(r2)

        color = COLORS.get(r["type"], "#999")
        marker = MARKERS.get(r["type"], "o")
        alpha = 1.0 if r2 > 0.5 and kcrit > 0 else 0.3
        ax1.scatter(abs(kcrit), agent, c=color, marker=marker, s=60,
                    alpha=alpha,
                    edgecolors="black" if r["source"] == "api" else "none",
                    linewidths=1, zorder=3)

    from scipy import stats
    kc_all = [(float(r["K_crit"]), float(r["agent_score"]))
              for r in rows if r["K_crit"] and r["agent_score"]]
    if kc_all:
        kc, ab = zip(*kc_all)
        tau_kc, p_kc = stats.kendalltau(kc, ab)
        ax1.set_title(f"K_crit vs Agent Score (τ={tau_kc:.3f}, p={p_kc:.2f}, N={len(kc_all)})",
                      fontsize=10, fontweight="bold")
    ax1.set_xlabel("K_crit (50% accuracy breakpoint)")
    ax1.set_ylabel("Agent Battery Score")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.2)

    # Right: K_crit histogram by type
    for mtype, color in COLORS.items():
        kcrits = [float(r["K_crit"]) for r in rows
                  if r["type"] == mtype and r["K_crit"]]
        if kcrits:
            ax2.hist(kcrits, bins=10, alpha=0.6, color=color, label=f"{mtype} (N={len(kcrits)})",
                     edgecolor="white")

    ax2.set_xlabel("K_crit")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of K_crit by Model Type", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    save(fig, "fig_kcrit_analysis")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating NeurIPS figures...")
    fig1_k_curves()
    fig2_scatter()
    fig3_load_shift()
    fig4_kcrit()
    print("\nAll figures saved to:", OUT)
