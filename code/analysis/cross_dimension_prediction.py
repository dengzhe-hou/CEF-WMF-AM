"""
Experiment E5: Cross-Framework Validity Test (CLA + Rank-Correlation Validity)

Does the CEF dimension ranking of 12 models correlate more strongly with OOD task
rankings than standard benchmark rankings?

Statistical method: Kendall's tau (valid for N=12; linear regression R² with N≤5
was statistically indefensible and has been removed).

Usage:
    python cross_dimension_prediction.py --results-dir ../data/results
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Known benchmark scores for models (from public leaderboards, March 2026) ─
# Update these with actual evaluated scores once available.

BENCHMARK_SCORES = {
    "gpt-4o": {
        "MMLU": 0.879, "HumanEval": 0.902, "GSM8K": 0.922,
        "ARC_Challenge": 0.914, "HellaSwag": 0.952,
    },
    "gpt-4o-mini": {
        "MMLU": 0.820, "HumanEval": 0.852, "GSM8K": 0.870,
        "ARC_Challenge": 0.861, "HellaSwag": 0.928,
    },
    "claude-opus-4": {
        "MMLU": 0.869, "HumanEval": 0.896, "GSM8K": 0.915,
        "ARC_Challenge": 0.905, "HellaSwag": 0.947,
    },
    "claude-sonnet-4": {
        "MMLU": 0.851, "HumanEval": 0.878, "GSM8K": 0.901,
        "ARC_Challenge": 0.893, "HellaSwag": 0.940,
    },
    "gemini-1.5-pro": {
        "MMLU": 0.853, "HumanEval": 0.871, "GSM8K": 0.906,
        "ARC_Challenge": 0.898, "HellaSwag": 0.940,
    },
    "gemini-1.5-flash": {
        "MMLU": 0.789, "HumanEval": 0.742, "GSM8K": 0.862,
        "ARC_Challenge": 0.841, "HellaSwag": 0.917,
    },
    "llama3-70b": {
        "MMLU": 0.820, "HumanEval": 0.810, "GSM8K": 0.856,
        "ARC_Challenge": 0.862, "HellaSwag": 0.921,
    },
    "llama3-8b": {
        "MMLU": 0.684, "HumanEval": 0.622, "GSM8K": 0.756,
        "ARC_Challenge": 0.789, "HellaSwag": 0.882,
    },
    "mixtral-8x7b": {
        "MMLU": 0.706, "HumanEval": 0.403, "GSM8K": 0.741,
        "ARC_Challenge": 0.800, "HellaSwag": 0.863,
    },
    "qwen2-72b": {
        "MMLU": 0.840, "HumanEval": 0.860, "GSM8K": 0.891,
        "ARC_Challenge": 0.880, "HellaSwag": 0.935,
    },
    "deepseek-v3": {
        "MMLU": 0.876, "HumanEval": 0.891, "GSM8K": 0.913,
        "ARC_Challenge": 0.901, "HellaSwag": 0.944,
    },
    "gemma-2-27b": {
        "MMLU": 0.752, "HumanEval": 0.741, "GSM8K": 0.831,
        "ARC_Challenge": 0.835, "HellaSwag": 0.901,
    },
}

# ── Held-out tasks (replace with actual measured scores once available) ──────
# All 12 models must be scored on tasks NOT used in the CEF battery.

HELD_OUT_SCORES = {
    # Task: multi-session dialogue coherence (EMC-relevant)
    "multi_session_coherence": {
        "gpt-4o": None, "gpt-4o-mini": None, "claude-opus-4": None,
        "claude-sonnet-4": None, "gemini-1.5-pro": None, "gemini-1.5-flash": None,
        "llama3-70b": None, "llama3-8b": None, "mixtral-8x7b": None,
        "qwen2-72b": None, "deepseek-v3": None, "gemma-2-27b": None,
    },
    # Task: active state tracking in tool-use sequences (WMF-AM-relevant)
    "active_state_tool_use": {
        "gpt-4o": None, "gpt-4o-mini": None, "claude-opus-4": None,
        "claude-sonnet-4": None, "gemini-1.5-pro": None, "gemini-1.5-flash": None,
        "llama3-70b": None, "llama3-8b": None, "mixtral-8x7b": None,
        "qwen2-72b": None, "deepseek-v3": None, "gemma-2-27b": None,
    },
    # Task: error detection in complex agentic pipeline (MCC-relevant)
    "pipeline_error_detection": {
        "gpt-4o": None, "gpt-4o-mini": None, "claude-opus-4": None,
        "claude-sonnet-4": None, "gemini-1.5-pro": None, "gemini-1.5-flash": None,
        "llama3-70b": None, "llama3-8b": None, "mixtral-8x7b": None,
        "qwen2-72b": None, "deepseek-v3": None, "gemma-2-27b": None,
    },
    # Task: novel domain task adaptation under load (CLA-relevant)
    "novel_domain_adaptation": {
        "gpt-4o": None, "gpt-4o-mini": None, "claude-opus-4": None,
        "claude-sonnet-4": None, "gemini-1.5-pro": None, "gemini-1.5-flash": None,
        "llama3-70b": None, "llama3-8b": None, "mixtral-8x7b": None,
        "qwen2-72b": None, "deepseek-v3": None, "gemma-2-27b": None,
    },
}


# ── Load CEF scores ──────────────────────────────────────────────────────────

def load_cef_scores(results_dir: Path) -> dict:
    """Load computed CEF scores from experiment output files."""
    models = list(BENCHMARK_SCORES.keys())
    cef_scores = {}

    for model in models:
        scores = {"model": model}

        # WMF
        wmf_path = results_dir / "wmf" / model / "scores.json"
        if wmf_path.exists():
            with open(wmf_path) as f:
                d = json.load(f)
            scores["WMF"] = d.get("WMF_composite", np.nan)
            scores["WMF_IM"] = d.get("WMF-IM", np.nan)
            scores["WMF_AM"] = d.get("WMF-AM", np.nan)
            scores["WMF_IR"] = d.get("WMF-IR", np.nan)

        # MCC (MA + CE composite; ECE is background only)
        mcc_path = results_dir / "mcc" / model / "scores.json"
        if mcc_path.exists():
            with open(mcc_path) as f:
                d = json.load(f)
            scores["MCC"] = d.get("MCC_composite", np.nan)
            scores["MCC_MA"] = d.get("MCC-MA (monitoring_r)", np.nan)

        # EMC
        emc_path = results_dir / "emc" / model / "scores.json"
        if emc_path.exists():
            with open(emc_path) as f:
                d = json.load(f)
            scores["EMC"] = d.get("EMC_composite", np.nan)
            scores["EMC_EI"] = d.get("EMC-EI", np.nan)

        # CLA (loaded from CLA-specific output if available)
        cla_path = results_dir / "cla" / model / "scores.json"
        if cla_path.exists():
            with open(cla_path) as f:
                d = json.load(f)
            scores["CLA"] = d.get("CLA_composite", np.nan)

        cef_scores[model] = scores

    return cef_scores


# ── Build feature matrices ───────────────────────────────────────────────────

def build_feature_matrices(cef_scores: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build feature matrices for rank-correlation analysis."""
    models = list(cef_scores.keys())

    # CEF features (WMF, MCC, EMC, CLA — RF dropped as non-novel)
    cef_cols = ["WMF", "MCC", "EMC", "CLA"]
    cef_data = []
    for m in models:
        row = [cef_scores[m].get(c, np.nan) for c in cef_cols]
        cef_data.append(row)
    df_cef = pd.DataFrame(cef_data, index=models, columns=cef_cols)

    # Benchmark features
    bench_cols = ["MMLU", "HumanEval", "GSM8K", "ARC_Challenge", "HellaSwag"]
    bench_data = []
    for m in models:
        row = [BENCHMARK_SCORES.get(m, {}).get(c, np.nan) for c in bench_cols]
        bench_data.append(row)
    df_bench = pd.DataFrame(bench_data, index=models, columns=bench_cols)

    # Held-out targets
    held_cols = list(HELD_OUT_SCORES.keys())
    held_data = []
    for m in models:
        row = [HELD_OUT_SCORES[t].get(m) for t in held_cols]
        held_data.append(row)
    df_held = pd.DataFrame(held_data, index=models, columns=held_cols)

    return df_cef, df_bench, df_held


# ── Kendall tau rank-correlation comparison ───────────────────────────────────

def kendall_tau_comparison(
    df_cef: pd.DataFrame,
    df_bench: pd.DataFrame,
    df_held: pd.DataFrame,
) -> dict:
    """
    Compare ordinal predictive validity of CEF vs. benchmark scores.
    For each held-out task, rank all models by (a) CEF composite and
    (b) benchmark composite, then compute Kendall's tau against OOD task ranking.
    Kendall's tau is valid for N=12 (p-values reliable at N>=10).
    Linear regression R² is NOT used (statistically indefensible at N<=5).
    """
    results = {}

    # Composite scores for ranking
    cef_composite = df_cef.mean(axis=1, skipna=True)
    bench_composite = df_bench.mean(axis=1, skipna=True)

    for task in df_held.columns:
        y = df_held[task]
        valid = y.notna()
        if valid.sum() < 10:
            results[task] = {"skipped": f"only {valid.sum()} models have scores (need >=10)"}
            continue

        y_valid = y[valid]
        cef_valid = cef_composite[valid]
        bench_valid = bench_composite[valid]

        tau_cef, p_cef = kendalltau(cef_valid, y_valid)
        tau_bench, p_bench = kendalltau(bench_valid, y_valid)

        # Per-dimension tau
        dim_taus = {}
        for dim in df_cef.columns:
            if not df_cef[dim][valid].isna().any():
                t, p = kendalltau(df_cef[dim][valid], y_valid)
                dim_taus[dim] = {"tau": round(float(t), 4), "p": round(float(p), 4)}

        results[task] = {
            "CEF_composite": {"tau": round(float(tau_cef), 4), "p": round(float(p_cef), 4)},
            "Benchmark_composite": {"tau": round(float(tau_bench), 4), "p": round(float(p_bench), 4)},
            "CEF_wins": bool(abs(tau_cef) > abs(tau_bench)),
            "n_models": int(valid.sum()),
            "per_dimension": dim_taus,
        }

    return results


# ── CLA: Degradation curve analysis ─────────────────────────────────────────

def analyze_degradation_curves(results_dir: Path) -> dict:
    """
    Analyze performance degradation curves across difficulty levels.
    Fit exponential decay; identify cliff-edge vs. graceful patterns.
    """
    cla_results = {}
    models = list(BENCHMARK_SCORES.keys())

    for model in models:
        wmf_path = results_dir / "wmf" / model / "scores.json"
        if not wmf_path.exists():
            continue
        with open(wmf_path) as f:
            d = json.load(f)
        load_curve = d.get("load_curve", {})
        if not load_curve:
            continue

        ns = sorted(load_curve.keys())
        accs = [load_curve[n] for n in ns]

        # Fit linear degradation (simple slope)
        if len(ns) >= 3:
            slope = np.polyfit(ns, accs, 1)[0]
            # Detect cliff-edge: variance of second differences
            diffs = np.diff(accs)
            cliff_score = float(np.std(diffs))  # high std = uneven degradation (cliff-like)
        else:
            slope = 0.0
            cliff_score = 0.0

        cla_results[model] = {
            "slope": round(slope, 6),
            "cliff_score": round(cliff_score, 4),
            "load_curve": {int(n): round(a, 4) for n, a in zip(ns, accs)},
        }

    return cla_results


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_tau_comparison(tau_results: dict, output_dir: Path):
    """Bar chart comparing Kendall's tau of CEF vs. Benchmarks for each held-out task."""
    tasks = [t for t in tau_results if "skipped" not in tau_results[t]]
    if not tasks:
        print("No held-out scores available yet. Run experiments first.")
        return

    cef_taus = [tau_results[t]["CEF_composite"]["tau"] for t in tasks]
    bench_taus = [tau_results[t]["Benchmark_composite"]["tau"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, cef_taus, width, label="CEF (Kendall's \u03c4)", color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, bench_taus, width, label="Benchmarks (Kendall's \u03c4)", color="#FF5722", alpha=0.85)

    ax.set_xlabel("Held-out Task", fontsize=12)
    ax.set_ylabel("Kendall's \u03c4 with OOD task ranking", fontsize=12)
    ax.set_title("Ordinal Predictive Validity: CEF vs. Standard Benchmarks\n"
                 "on Out-of-Distribution Tasks (N=12 models)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=9)
    ax.legend()
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "tau_comparison.pdf", dpi=300)
    fig.savefig(output_dir / "tau_comparison.png", dpi=150)
    plt.close()
    print(f"Kendall tau comparison plot saved to {output_dir}")


def plot_cef_heatmap(df_cef: pd.DataFrame, output_dir: Path):
    """Heatmap of CEF dimension scores across models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df_cef.T,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("CEF Dimension Scores Across Models", fontsize=13)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("CEF Dimension", fontsize=11)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "cef_heatmap.pdf", dpi=300)
    fig.savefig(output_dir / "cef_heatmap.png", dpi=150)
    plt.close()
    print(f"CEF heatmap saved to {output_dir}")


def plot_degradation_curves(cla_results: dict, output_dir: Path):
    """Line plot of WMF load curves per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (model, data), color in zip(cla_results.items(), colors):
        load_curve = data.get("load_curve", {})
        if load_curve:
            ns = sorted(load_curve.keys())
            accs = [load_curve[n] for n in ns]
            ax.plot(ns, accs, marker="o", label=model, color=color, linewidth=2)

    ax.set_xlabel("Working Memory Load (N items)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Cognitive Load Adaptation (CLA): WMF Degradation Curves\n"
                 "Cliff-edge vs. Graceful Degradation", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "degradation_curves.pdf", dpi=300)
    fig.savefig(output_dir / "degradation_curves.png", dpi=150)
    plt.close()
    print(f"Degradation curves plot saved to {output_dir}")


# ── Main analysis pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-dimension prediction and CLA analysis.")
    parser.add_argument("--results-dir", default="../data/results", type=Path)
    parser.add_argument("--output-dir", default="../data/results/analysis", type=Path)
    args = parser.parse_args()

    print("Loading CEF scores...")
    cef_scores = load_cef_scores(args.results_dir)

    print("Building feature matrices...")
    df_cef, df_bench, df_held = build_feature_matrices(cef_scores)

    print("\nCEF Scores Summary:")
    print(df_cef.to_string())

    print("\nBenchmark Scores Summary:")
    print(df_bench.to_string())

    print("\nRunning E5 rank-correlation comparison (Kendall's tau, N=12)...")
    tau_results = kendall_tau_comparison(df_cef, df_bench, df_held)

    print("\n=== E5 RESULTS: Ordinal Predictive Validity ===")
    for task, result in tau_results.items():
        if "skipped" in result:
            print(f"  {task}: SKIPPED — {result['skipped']}")
            continue
        tau_c = result["CEF_composite"]["tau"]
        tau_b = result["Benchmark_composite"]["tau"]
        winner = "CEF" if result["CEF_wins"] else "Benchmarks"
        print(f"  Task: {task}")
        print(f"    CEF composite:       \u03c4 = {tau_c:.4f}  (p={result['CEF_composite']['p']:.3f})")
        print(f"    Benchmark composite: \u03c4 = {tau_b:.4f}  (p={result['Benchmark_composite']['p']:.3f})")
        print(f"    Winner: {winner}")

    print("\nAnalyzing CLA degradation curves...")
    cla_results = analyze_degradation_curves(args.results_dir)
    print("\nCLA Results (slope = steeper degradation, cliff_score = more cliff-like):")
    for model, data in cla_results.items():
        print(f"  {model}: slope={data['slope']:.4f}, cliff_score={data['cliff_score']:.4f}")

    print("\nGenerating plots...")
    plot_tau_comparison(tau_results, args.output_dir)
    plot_cef_heatmap(df_cef, args.output_dir)
    if cla_results:
        plot_degradation_curves(cla_results, args.output_dir)

    # Save all results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "tau_results.json", "w") as f:
        json.dump(tau_results, f, indent=2)
    with open(args.output_dir / "cla_results.json", "w") as f:
        json.dump(cla_results, f, indent=2)
    with open(args.output_dir / "cef_scores_all.json", "w") as f:
        json.dump(cef_scores, f, indent=2)

    print(f"\nAll results saved to {args.output_dir}")
    print("\n=== KEY FINDING ===")
    for task, result in tau_results.items():
        if "skipped" in result:
            continue
        if result["CEF_wins"]:
            delta = abs(result["CEF_composite"]["tau"]) - abs(result["Benchmark_composite"]["tau"])
            print(f"  ✓ {task}: CEF > Benchmarks by Δτ = {delta:.4f}")
        else:
            delta = abs(result["Benchmark_composite"]["tau"]) - abs(result["CEF_composite"]["tau"])
            print(f"  ✗ {task}: Benchmarks > CEF by Δτ = {delta:.4f}")


if __name__ == "__main__":
    main()
