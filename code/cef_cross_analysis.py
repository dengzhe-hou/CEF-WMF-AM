"""
CEF Cross-Experiment Analysis

Merges data from all completed experiments and computes:
1. Comprehensive per-model CEF profile (all dimensions)
2. Kendall's τ matrix across ALL dimensions (Phase 1 + convergent + MCC-CE v2 + agent)
3. Completion-CEF dissociation with full data
4. Agent validation correlation: do CEF scores predict agent process quality?
5. Summary statistics for paper Table 1
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

RESULTS = Path("/home/hou/Research/Cognitive-LLM-Agent-Position/project/data/results")


def load_phase1():
    """Load Phase 1 main benchmark."""
    p = RESULTS / "cef_phase1_full.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    results = d.get("results", [])
    scores = {}
    for r in results:
        m = r["model"]
        sd = r["sub_dim"]
        if m not in scores:
            scores[m] = {}
        if sd not in scores[m]:
            scores[m][sd] = []
        if "accurate" in r:
            scores[m][sd].append(r["accurate"])
        elif "ma_jaccard" in r:
            scores[m][sd].append(r["ma_jaccard"])
        elif "tau" in r:
            scores[m][sd].append(r["tau"])
        elif "recovery" in r:
            scores[m][sd].append(r["recovery"])
    return {m: {k: float(np.mean(v)) for k, v in s.items()} for m, s in scores.items()}


def load_convergent():
    """Load convergent validity results."""
    p = RESULTS / "cef_convergent_validity_ollama.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    results = d.get("results", [])
    scores = {}
    for r in results:
        m = r["model"]
        if m not in scores:
            scores[m] = {}
        probe = r.get("probe", "?")
        if probe == "RF-POC":
            scores[m].setdefault("CONV-RFPOC", []).append(r.get("process_score", 0))
        elif probe == "Self-Knowledge":
            scores[m].setdefault("CONV-SELFKNOW", []).append(r.get("accurate", 0))
        elif probe == "Factual-Retrieval":
            scores[m].setdefault("CONV-FACTUAL", []).append(r.get("accurate", 0))
    return {m: {k: float(np.mean(v)) for k, v in s.items()} for m, s in scores.items()}


def load_mcc_ce_v2():
    """Load MCC-CE v2 results."""
    p = RESULTS / "cef_mcc_ce_v2_ollama.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    results = d.get("results", [])
    scores = {}
    for r in results:
        m = r["model"]
        if m not in scores:
            scores[m] = {}
        if "mcc_ce_v2" in r and r["mcc_ce_v2"] is not None:
            scores[m].setdefault("MCC-CE-v2", []).append(r["mcc_ce_v2"])
        if "discrimination" in r and r["discrimination"] is not None:
            scores[m].setdefault("MCC-CE-disc", []).append(r["discrimination"])
    return {m: {k: float(np.mean(v)) for k, v in s.items()} for m, s in scores.items()}


def load_agent():
    """Load agent validation results."""
    p = RESULTS / "cef_agent_validation_ollama.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    results = d.get("results", [])
    scores = {}
    for r in results:
        m = r["model"]
        if m not in scores:
            scores[m] = {}
        if "task_completion" in r:
            scores[m].setdefault("AGENT-TC", []).append(r["task_completion"])
        if "process_quality" in r:
            scores[m].setdefault("AGENT-PQ", []).append(r["process_quality"])
    return {m: {k: float(np.mean(v)) for k, v in s.items()} for m, s in scores.items()}


def merge_scores(*dicts):
    """Merge multiple score dictionaries."""
    merged = {}
    for d in dicts:
        for m, s in d.items():
            if m not in merged:
                merged[m] = {}
            merged[m].update(s)
    return merged


def print_profile_table(scores):
    """Print comprehensive model profiles."""
    models = sorted(scores.keys())
    all_dims = sorted(set(k for s in scores.values() for k in s.keys()))

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL PROFILES")
    print("=" * 80)

    # Print header
    hdr = "{:<25}".format("Model")
    for d in all_dims:
        hdr += " {:>8}".format(d[:8])
    print(hdr)
    print("-" * len(hdr))

    for m in models:
        row = "{:<25}".format(m.replace("ollama:", ""))
        for d in all_dims:
            v = scores[m].get(d)
            if v is not None:
                row += " {:>8.3f}".format(v)
            else:
                row += " {:>8}".format("-")
        print(row)
    return all_dims


def compute_correlations(scores, dims):
    """Compute pairwise Kendall's τ for all dimensions."""
    models = sorted(scores.keys())
    n = len(models)

    print("\n" + "=" * 80)
    print("KENDALL'S τ MATRIX (N={} models)".format(n))
    print("=" * 80)

    # Only use dims with variance
    valid_dims = []
    for d in dims:
        vals = [scores[m].get(d, None) for m in models]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 3 and len(set(vals)) > 1:
            valid_dims.append(d)

    results = {}
    significant_pairs = []

    for i, d1 in enumerate(valid_dims):
        for j, d2 in enumerate(valid_dims):
            if j <= i:
                continue
            v1 = [scores[m].get(d1, None) for m in models]
            v2 = [scores[m].get(d2, None) for m in models]
            # Only use models with both values
            pairs = [(a, b) for a, b in zip(v1, v2) if a is not None and b is not None]
            if len(pairs) < 3:
                continue
            a, b = zip(*pairs)
            if len(set(a)) <= 1 or len(set(b)) <= 1:
                continue
            tau, p = kendalltau(a, b)
            results["{} vs {}".format(d1, d2)] = {"tau": tau, "p": p, "n": len(pairs)}
            if p < 0.1:
                significant_pairs.append((d1, d2, tau, p))

    # Print significant correlations
    print("\nSignificant or notable correlations (p < 0.1):")
    for d1, d2, tau, p in sorted(significant_pairs, key=lambda x: x[3]):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."
        print("  {} vs {}: τ={:.3f} (p={:.3f}) {}".format(d1, d2, tau, p, sig))

    # Key correlations for paper
    print("\n--- KEY CORRELATIONS FOR PAPER ---")
    key_pairs = [
        ("WMF-AM", "CONV-RFPOC", "convergent: WMF-AM should correlate with reasoning process"),
        ("WMF-AM", "CONV-FACTUAL", "divergent: WMF-AM should NOT correlate with factual retrieval"),
        ("WMF-AM", "AGENT-PQ", "predictive: WMF-AM should predict agent process quality"),
        ("WMF-AM", "VALIDITY-GSM8K", "dissociation: WMF-AM vs completion"),
        ("MCC-CE-v2", "AGENT-PQ", "predictive: metacognition should predict agent quality"),
        ("AGENT-TC", "AGENT-PQ", "agent: completion vs process quality gap"),
        ("EMC-TO", "AGENT-PQ", "predictive: episodic memory vs agent quality"),
    ]
    for d1, d2, desc in key_pairs:
        v1 = [scores[m].get(d1, None) for m in models]
        v2 = [scores[m].get(d2, None) for m in models]
        pairs = [(a, b) for a, b in zip(v1, v2) if a is not None and b is not None]
        if len(pairs) < 3:
            print("  {}: insufficient data".format(desc))
            continue
        a, b = zip(*pairs)
        if len(set(a)) <= 1 or len(set(b)) <= 1:
            print("  {}: no variance".format(desc))
            continue
        tau, p = kendalltau(a, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print("  {}: τ={:.3f} (p={:.3f}) {} [N={}]".format(desc, tau, p, sig, len(pairs)))

    return results


def agent_prediction_analysis(scores):
    """Key analysis: do CEF scores predict agent behavior beyond completion?"""
    models = sorted(scores.keys())

    print("\n" + "=" * 80)
    print("AGENT PREDICTION ANALYSIS")
    print("=" * 80)

    # Get models with both CEF and agent data
    cef_dims = ["WMF-AM", "EMC-TO", "CLA-DC", "MCC-CE-v2", "MCC-MA"]
    agent_dims = ["AGENT-TC", "AGENT-PQ"]

    for target in agent_dims:
        print("\n--- Predicting {} ---".format(target))
        target_vals = [scores[m].get(target) for m in models]
        valid_models = [m for m, v in zip(models, target_vals) if v is not None]
        target_clean = [v for v in target_vals if v is not None]

        if len(valid_models) < 3:
            print("  Insufficient data")
            continue

        for pred in cef_dims + ["VALIDITY-GSM8K", "VALIDITY-MMLU", "CONV-RFPOC"]:
            pred_vals = [scores[m].get(pred) for m in valid_models]
            pairs = [(a, b) for a, b in zip(pred_vals, target_clean) if a is not None]
            if len(pairs) < 3:
                continue
            a, b = zip(*pairs)
            if len(set(a)) <= 1 or len(set(b)) <= 1:
                continue
            tau, p = kendalltau(a, b)
            rho, p_rho = spearmanr(a, b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print("  {} → {}: τ={:.3f} (p={:.3f}) {} | ρ={:.3f}".format(
                pred, target, tau, p, sig, rho))

    # Dissociation: agent completion vs agent process quality
    tc = [scores[m].get("AGENT-TC") for m in models]
    pq = [scores[m].get("AGENT-PQ") for m in models]
    pairs = [(a, b) for a, b in zip(tc, pq) if a is not None and b is not None]
    if len(pairs) >= 3:
        a, b = zip(*pairs)
        tau, p = kendalltau(a, b)
        print("\n  AGENT completion vs process quality: τ={:.3f} (p={:.3f})".format(tau, p))
        gap = [b_i - a_i for a_i, b_i in zip(a, b)]
        print("  Mean gap (PQ - TC): {:.3f}".format(np.mean(gap)))


def main():
    print("Loading all experiment data...")

    p1 = load_phase1()
    conv = load_convergent()
    mcc2 = load_mcc_ce_v2()
    agent = load_agent()

    print("  Phase 1: {} models".format(len(p1)))
    print("  Convergent: {} models".format(len(conv)))
    print("  MCC-CE v2: {} models".format(len(mcc2)))
    print("  Agent: {} models".format(len(agent)))

    scores = merge_scores(p1, conv, mcc2, agent)
    print("  Merged: {} models".format(len(scores)))

    # 1. Profile table
    all_dims = print_profile_table(scores)

    # 2. Correlation matrix
    corr = compute_correlations(scores, all_dims)

    # 3. Agent prediction
    agent_prediction_analysis(scores)

    # 4. Summary stats for paper
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS FOR PAPER")
    print("=" * 80)

    models = sorted(scores.keys())
    key_dims = ["WMF-AM", "EMC-TO", "CLA-DC", "MCC-CE-v2", "MCC-MA",
                "VALIDITY-MMLU", "VALIDITY-GSM8K", "AGENT-TC", "AGENT-PQ"]
    for d in key_dims:
        vals = [scores[m].get(d) for m in models if scores[m].get(d) is not None]
        if vals:
            print("  {}: mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}, range={:.3f}, N={}".format(
                d, np.mean(vals), np.std(vals), min(vals), max(vals),
                max(vals) - min(vals), len(vals)))

    # Save
    output = {
        "n_models": len(scores),
        "model_profiles": {m: {k: round(v, 4) for k, v in s.items()}
                           for m, s in scores.items()},
        "correlations": {k: {"tau": round(v["tau"], 4), "p": round(v["p"], 4), "n": v["n"]}
                         for k, v in corr.items()},
    }
    out_path = RESULTS / "cef_cross_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to {}".format(out_path))


if __name__ == "__main__":
    main()
