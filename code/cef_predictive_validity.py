"""
CEF Predictive Validity Analysis

Tests: Does WMF-AM predict downstream agent task performance?

Combines:
  1. WMF-AM probe scores (from multi-seed data)
  2. Agent validation task scores (from cef_agent_validation)
  3. Completion battery scores (100-item)

Computes:
  - τ(WMF-AM, agent_WMF-AM_tasks)  — WMF-AM predicts WMF-relevant agent tasks
  - τ(WMF-AM, agent_overall)        — WMF-AM predicts general agent competence
  - τ(Completion, agent_overall)     — Completion predicts agent competence
  - τ(WMF-AM, agent_overall | Completion) — WMF-AM adds value beyond completion
  - Bootstrap CIs for all correlations

Usage:
    python cef_predictive_validity.py
    python cef_predictive_validity.py --agent-file cef_agent_validation_all_20260318.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

# ── Data loading ────────────────────────────────────────────────────────────

def load_wmf_am_scores() -> dict[str, float]:
    """Load WMF-AM 4-seed mean scores for all available models."""
    scores = {}

    # Original 7: from cef_expanded.json + cef_wmf_multiseed.json
    expanded_path = RESULTS_DIR / "cef_expanded.json"
    if expanded_path.exists():
        with open(expanded_path) as f:
            data = json.load(f)
        for model_data in data.get("models", data) if isinstance(data, dict) else data:
            if isinstance(model_data, dict):
                model = model_data.get("model", "")
                wmf = model_data.get("wmf_am", model_data.get("WMF-AM"))
                if model and wmf is not None:
                    scores[model] = float(wmf)

    # Multi-seed original 7
    multiseed_path = RESULTS_DIR / "cef_wmf_multiseed.json"
    if multiseed_path.exists():
        with open(multiseed_path) as f:
            data = json.load(f)
        for m in data.get("per_model", []):
            model = m["model"]
            scores[model] = m["mean_accuracy"]

    # Multi-seed expansion 8
    for p in RESULTS_DIR.glob("cef_wmf_multiseed_expansion8_*.json"):
        with open(p) as f:
            data = json.load(f)
        for m in data.get("per_model", []):
            model = m["model"]
            scores[model] = m["mean_accuracy"]

    return scores


def load_completion_scores() -> dict[str, float]:
    """Load 100-item completion battery scores."""
    scores = {}
    for p in sorted(RESULTS_DIR.glob("cef_completion_battery_v2_*.json")):
        if ".details." in str(p):
            continue
        with open(p) as f:
            data = json.load(f)
        for r in data.get("results", []):
            scores[r["model"]] = r["score"]
    return scores


def load_agent_scores(agent_file: str = None) -> dict[str, dict]:
    """Load agent validation results.

    Returns dict: model -> {overall, wmf_am_tasks, mcc_tasks, emc_tasks, general_tasks}
    """
    # Find the most recent agent validation file
    if agent_file:
        path = RESULTS_DIR / agent_file
    else:
        candidates = sorted(RESULTS_DIR.glob("cef_agent_validation_*.json"))
        # Prefer summary files
        summaries = [c for c in candidates if "summary" in str(c)]
        if summaries:
            path = summaries[-1]
        elif candidates:
            path = candidates[-1]
        else:
            return {}

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    model_scores = {}

    for r in results:
        model = r["model"]
        if model not in model_scores:
            model_scores[model] = {
                "all_tasks": [],
                "wmf_am_tasks": [],
                "mcc_tasks": [],
                "emc_tasks": [],
                "general_tasks": [],
            }

        tc = r["task_completion"]
        pq = r.get("process_quality", tc)
        model_scores[model]["all_tasks"].append(tc)

        dim = r.get("cef_dim", "")
        if dim == "WMF-AM":
            model_scores[model]["wmf_am_tasks"].append(tc)
        elif dim == "MCC-MA":
            model_scores[model]["mcc_tasks"].append(tc)
        elif dim == "EMC":
            model_scores[model]["emc_tasks"].append(tc)
        elif dim == "GENERAL":
            model_scores[model]["general_tasks"].append(tc)

    # Compute means
    out = {}
    for model, data in model_scores.items():
        out[model] = {}
        for key, vals in data.items():
            out[model][key] = float(np.mean(vals)) if vals else None
    return out


# ── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_kendall_ci(x, y, n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap CI for Kendall's tau."""
    rng = np.random.RandomState(seed)
    n = len(x)
    taus = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        tau_b, _ = kendalltau(x[idx], y[idx])
        if not np.isnan(tau_b):
            taus.append(tau_b)
    taus = np.array(taus)
    lo = np.percentile(taus, 100 * alpha / 2)
    hi = np.percentile(taus, 100 * (1 - alpha / 2))
    return lo, hi


# ── Partial rank correlation ────────────────────────────────────────────────

def partial_kendall(x, y, z):
    """Partial Kendall's tau: τ(x, y | z) via residualization.

    Approximation: rank-residualize x and y on z, then compute τ on residuals.
    """
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    # Linear residuals of ranks
    def residualize(r, rz):
        # Simple OLS residual
        z_mean = np.mean(rz)
        r_mean = np.mean(r)
        beta = np.sum((rz - z_mean) * (r - r_mean)) / (np.sum((rz - z_mean)**2) + 1e-12)
        return r - beta * rz

    rx_res = residualize(rx, rz)
    ry_res = residualize(ry, rz)

    tau, p = kendalltau(rx_res, ry_res)
    return tau, p


# ── Main analysis ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CEF Predictive Validity Analysis")
    parser.add_argument("--agent-file", type=str, default=None)
    args = parser.parse_args()

    wmf_scores = load_wmf_am_scores()
    completion_scores = load_completion_scores()
    agent_scores = load_agent_scores(args.agent_file)

    print("=" * 70)
    print("CEF PREDICTIVE VALIDITY ANALYSIS")
    print("=" * 70)

    print(f"\nData loaded:")
    print(f"  WMF-AM scores: {len(wmf_scores)} models")
    print(f"  Completion scores: {len(completion_scores)} models")
    print(f"  Agent validation: {len(agent_scores)} models")

    # Find common models across all three sources
    common = set(wmf_scores) & set(completion_scores) & set(agent_scores)
    # Also check models with agent + wmf only (no completion needed for primary test)
    common_wmf_agent = set(wmf_scores) & set(agent_scores)

    print(f"  Common (WMF + Completion + Agent): {len(common)}")
    print(f"  Common (WMF + Agent): {len(common_wmf_agent)}")

    if len(common_wmf_agent) < 5:
        print("\nWARNING: Need ≥5 models with both WMF-AM and agent scores.")
        print("Available WMF-AM models:", sorted(wmf_scores.keys()))
        print("Available agent models:", sorted(agent_scores.keys()))
        print("\nRun cef_agent_validation.py on more models first.")
        sys.exit(0)

    # Build arrays for common_wmf_agent models
    models = sorted(common_wmf_agent)
    wmf = np.array([wmf_scores[m] for m in models])
    agent_overall = np.array([agent_scores[m]["all_tasks"] for m in models])
    agent_wmf = np.array([agent_scores[m]["wmf_am_tasks"] for m in models])

    print(f"\n{'='*70}")
    print("MODEL SCORES")
    print(f"{'='*70}")
    print(f"{'Model':<28} {'WMF-AM':>8} {'Agent':>8} {'AgWMF':>8} {'Compl':>8}")
    print("-" * 70)
    for m in models:
        comp = completion_scores.get(m, float('nan'))
        print(f"{m:<28} {wmf_scores[m]:>8.3f} {agent_scores[m]['all_tasks']:>8.3f} "
              f"{agent_scores[m]['wmf_am_tasks']:>8.3f} {comp:>8.3f}")

    # ── Primary test: τ(WMF-AM, agent WMF-AM tasks) ──
    print(f"\n{'='*70}")
    print("KENDALL'S τ CORRELATIONS")
    print(f"{'='*70}")

    results = {}

    # 1. WMF-AM → agent WMF-AM tasks
    tau1, p1 = kendalltau(wmf, agent_wmf)
    ci1 = bootstrap_kendall_ci(wmf, agent_wmf)
    results["wmf_am_vs_agent_wmf"] = {
        "tau": round(tau1, 4), "p": round(p1, 4),
        "ci_95": [round(ci1[0], 4), round(ci1[1], 4)],
        "n": len(models),
    }
    print(f"  τ(WMF-AM, Agent WMF tasks) = {tau1:.4f} (p={p1:.4f}), 95% CI [{ci1[0]:.3f}, {ci1[1]:.3f}]")

    # 2. WMF-AM → agent overall
    tau2, p2 = kendalltau(wmf, agent_overall)
    ci2 = bootstrap_kendall_ci(wmf, agent_overall)
    results["wmf_am_vs_agent_overall"] = {
        "tau": round(tau2, 4), "p": round(p2, 4),
        "ci_95": [round(ci2[0], 4), round(ci2[1], 4)],
        "n": len(models),
    }
    print(f"  τ(WMF-AM, Agent overall)   = {tau2:.4f} (p={p2:.4f}), 95% CI [{ci2[0]:.3f}, {ci2[1]:.3f}]")

    # 3. Completion → agent overall (if available)
    common_all = sorted(set(models) & set(completion_scores))
    if len(common_all) >= 5:
        comp = np.array([completion_scores[m] for m in common_all])
        agent_o = np.array([agent_scores[m]["all_tasks"] for m in common_all])
        wmf_c = np.array([wmf_scores[m] for m in common_all])

        tau3, p3 = kendalltau(comp, agent_o)
        ci3 = bootstrap_kendall_ci(comp, agent_o)
        results["completion_vs_agent_overall"] = {
            "tau": round(tau3, 4), "p": round(p3, 4),
            "ci_95": [round(ci3[0], 4), round(ci3[1], 4)],
            "n": len(common_all),
        }
        print(f"  τ(Completion, Agent overall)= {tau3:.4f} (p={p3:.4f}), 95% CI [{ci3[0]:.3f}, {ci3[1]:.3f}]")

        # 4. Partial: τ(WMF-AM, Agent | Completion)
        tau4, p4 = partial_kendall(wmf_c, agent_o, comp)
        results["wmf_am_vs_agent_partial_completion"] = {
            "tau": round(tau4, 4), "p": round(p4, 4),
            "n": len(common_all),
            "note": "partial τ, controlling for completion",
        }
        print(f"  τ(WMF-AM, Agent | Compl)   = {tau4:.4f} (p={p4:.4f})  [partial, controlling completion]")
    else:
        print(f"  (Need ≥5 models with completion scores for comparison; have {len(common_all)})")

    # ── Interpretation ──
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    if tau1 > 0 and p1 < 0.05:
        print("  ✓ WMF-AM PREDICTS WMF-relevant agent task performance (significant)")
    elif tau1 > 0:
        print(f"  ~ WMF-AM shows positive trend with WMF agent tasks (τ={tau1:.3f}) but p={p1:.3f}")
    else:
        print(f"  ✗ WMF-AM does NOT predict WMF agent tasks (τ={tau1:.3f}, p={p1:.3f})")

    if "wmf_am_vs_agent_partial_completion" in results:
        tau4 = results["wmf_am_vs_agent_partial_completion"]["tau"]
        p4 = results["wmf_am_vs_agent_partial_completion"]["p"]
        if tau4 > 0:
            print(f"  {'✓' if p4 < 0.05 else '~'} WMF-AM adds predictive value BEYOND completion (partial τ={tau4:.3f})")
        else:
            print(f"  ✗ WMF-AM does not add value beyond completion (partial τ={tau4:.3f})")

    # ── Save results ──
    output = {
        "analysis": "cef_predictive_validity",
        "n_models": len(models),
        "models": models,
        "correlations": results,
        "per_model": {
            m: {
                "wmf_am": wmf_scores[m],
                "completion": completion_scores.get(m),
                "agent_overall": agent_scores[m]["all_tasks"],
                "agent_wmf_tasks": agent_scores[m]["wmf_am_tasks"],
                "agent_mcc_tasks": agent_scores[m].get("mcc_tasks"),
                "agent_emc_tasks": agent_scores[m].get("emc_tasks"),
                "agent_general_tasks": agent_scores[m].get("general_tasks"),
            }
            for m in models
        },
    }

    out_path = RESULTS_DIR / "cef_predictive_validity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
