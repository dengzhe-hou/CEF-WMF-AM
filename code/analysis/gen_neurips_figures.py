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
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

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

        ax.scatter(wmf, agent, c=color, marker=marker, s=size,
                   edgecolors=edge, linewidths=1.5, zorder=3)

        # Label notable models
        label = r["model"]
        if any(x in label for x in ["o3-mini", "deepseek-r1-full", "claude", "gpt-4o",
                                      "deepseek-v3", "gemini", "tinyllama", "qwen2.5:0.5b"]):
            short = label.replace("deepseek-r1-full", "R1-full").replace("claude-sonnet-4", "Claude")
            short = short.replace("gpt-4o-mini", "4o-mini").replace("gpt-4o", "GPT-4o")
            short = short.replace("gemini-2.5-flash", "Gemini").replace("deepseek-v3", "V3")
            short = short.replace("tinyllama:1.1b", "TinyLlama").replace("qwen2.5:0.5b", "Q0.5B")
            ax.annotate(short, (wmf, agent), fontsize=7, alpha=0.8,
                        xytext=(5, 5), textcoords="offset points")

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
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save(fig, "fig_wmfam_vs_agent_n28")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Load-Shift Intervention
# ═══════════════════════════════════════════════════════════════════
def fig3_load_shift():
    # Only models with load-shift data
    ls_rows = [r for r in rows if r["supported_agent"] and r["unsupported_agent"]]
    # Add GPT-4o-mini from sanity test
    ls_models = sorted(ls_rows, key=lambda r: -float(r["supported_agent"]))

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    names = []
    sup_scores = []
    unsup_scores = []
    colors_bar = []

    for r in ls_models:
        short = r["model"].replace("gpt-4o-mini", "GPT-4o-mini").replace("gpt-4o", "GPT-4o")
        short = short.replace("claude-sonnet-4", "Claude\nSonnet 4")
        short = short.replace("deepseek-v3", "DeepSeek\nV3")
        short = short.replace("gemini-2.5-flash", "Gemini\n2.5 Flash")
        short = short.replace("o3-mini", "o3-mini\n(LRM)")
        short = short.replace("deepseek-r1", "DeepSeek\nR1 (LRM)")
        names.append(short)
        sup_scores.append(float(r["supported_agent"]))
        unsup_scores.append(float(r["unsupported_agent"]))
        colors_bar.append(COLORS.get(r["type"], "#999"))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, sup_scores, width, label="Supported (full history)",
                   color=[c for c in colors_bar], alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width/2, unsup_scores, width, label="Unsupported (last turn only)",
                   color=[c for c in colors_bar], alpha=0.35, edgecolor="white",
                   hatch="//")

    # Delta labels
    for i, (s, u) in enumerate(zip(sup_scores, unsup_scores)):
        delta = s - u
        if delta > 0.01:
            ax.annotate(f"Δ={delta:+.1f}", (i, max(s, u) + 0.03),
                        ha="center", fontsize=8, fontweight="bold",
                        color="red" if delta > 0.3 else "gray")
        else:
            ax.annotate("Δ=0.0", (i, max(s, u) + 0.03),
                        ha="center", fontsize=8, fontweight="bold", color="green")

    ax.set_ylabel("Agent Battery Score", fontsize=11)
    ax.set_title("Load-Shift Intervention: History Removal Effect on Agent Performance",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
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
        if r2 < 0.5 or kcrit < 0:
            continue

        color = COLORS.get(r["type"], "#999")
        marker = MARKERS.get(r["type"], "o")
        ax1.scatter(kcrit, agent, c=color, marker=marker, s=60,
                    edgecolors="black" if r["source"] == "api" else "none",
                    linewidths=1, zorder=3)

    from scipy import stats
    kc_valid = [(float(r["K_crit"]), float(r["agent_score"]))
                for r in rows if r["K_crit"] and r["agent_score"] and r["sigmoid_R2"]
                and float(r["sigmoid_R2"]) > 0.5 and float(r["K_crit"]) > 0]
    if kc_valid:
        kc, ab = zip(*kc_valid)
        tau_kc, p_kc = stats.kendalltau(kc, ab)
        ax1.set_title(f"K_crit vs Agent Score (τ={tau_kc:.3f}, p={p_kc:.3f}, n.s.)",
                      fontsize=10, fontweight="bold")
    ax1.set_xlabel("K_crit (50% accuracy breakpoint)")
    ax1.set_ylabel("Agent Battery Score")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.2)

    # Right: K_crit histogram by type
    for mtype, color in COLORS.items():
        kcrits = [float(r["K_crit"]) for r in rows
                  if r["type"] == mtype and r["K_crit"] and r["sigmoid_R2"]
                  and float(r["sigmoid_R2"]) > 0.5 and float(r["K_crit"]) > 0]
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
