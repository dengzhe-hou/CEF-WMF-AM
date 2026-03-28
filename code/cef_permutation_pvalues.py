"""
Permutation p-values for CEF primary results.

Computes exact or Monte Carlo permutation p-values for Kendall's tau
correlations, addressing the N=7 small-sample concern.

Primary hypotheses:
  H1: WMF-AM ≠ completion (dissociation) — τ ≈ 0.07
  H2: CONV-RFPOC ~ AGENT-PQ (convergent) — τ = 0.905
  H3: WMF-AM ≠ MCC-MA (divergent) — τ = 0.169

Also computes:
  - Bootstrap 95% CIs for each τ
  - Leave-one-family-out robustness (Qwen family = 3 models)

Usage:
    python cef_permutation_pvalues.py
"""

import json
import itertools
import numpy as np
from scipy import stats
from pathlib import Path

N_PERM = 100000
N_BOOT = 10000
np.random.seed(42)

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

def load_cross_analysis():
    with open(RESULTS_DIR / "cef_cross_analysis.json") as f:
        d = json.load(f)
    return d["model_profiles"]

def get_vectors(profiles, key1, key2):
    """Extract paired vectors for two measures, skipping NaN."""
    models = sorted(profiles.keys())
    x, y, names = [], [], []
    for m in models:
        v1 = profiles[m].get(key1)
        v2 = profiles[m].get(key2)
        if v1 is not None and v2 is not None:
            # Check for NaN
            try:
                if np.isnan(v1) or np.isnan(v2):
                    continue
            except (TypeError, ValueError):
                continue
            x.append(v1)
            y.append(v2)
            names.append(m.replace("ollama:", ""))
    return np.array(x), np.array(y), names

def permutation_tau(x, y, n_perm=N_PERM, alternative="two-sided"):
    """Permutation p-value for Kendall's tau."""
    n = len(x)
    observed_tau, _ = stats.kendalltau(x, y)
    
    count = 0
    for _ in range(n_perm):
        perm_y = np.random.permutation(y)
        perm_tau, _ = stats.kendalltau(x, perm_y)
        if alternative == "two-sided":
            if abs(perm_tau) >= abs(observed_tau):
                count += 1
        elif alternative == "greater":
            if perm_tau >= observed_tau:
                count += 1
        elif alternative == "less":
            if perm_tau <= observed_tau:
                count += 1
    
    p_perm = (count + 1) / (n_perm + 1)  # +1 for observed
    return observed_tau, p_perm

def bootstrap_ci(x, y, n_boot=N_BOOT, alpha=0.05):
    """Bootstrap 95% CI for Kendall's tau."""
    n = len(x)
    taus = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        tau, _ = stats.kendalltau(x[idx], y[idx])
        taus.append(tau)
    taus = np.array(taus)
    lo = np.percentile(taus, 100 * alpha / 2)
    hi = np.percentile(taus, 100 * (1 - alpha / 2))
    return lo, hi

def leave_one_family_out(profiles, key1, key2, family_prefix="qwen"):
    """Leave-one-family-out robustness check."""
    x_full, y_full, names_full = get_vectors(profiles, key1, key2)
    tau_full, _ = stats.kendalltau(x_full, y_full)
    
    # Remove family
    mask = [not n.startswith(family_prefix) for n in names_full]
    x_sub = x_full[mask]
    y_sub = y_full[mask]
    names_sub = [n for n, m in zip(names_full, mask) if m]
    
    if len(x_sub) < 3:
        return tau_full, None, names_sub
    
    tau_sub, p_sub = stats.kendalltau(x_sub, y_sub)
    return tau_full, tau_sub, names_sub

def main():
    profiles = load_cross_analysis()
    
    # Define primary hypotheses
    hypotheses = [
        {
            "name": "H1: WMF-AM vs Completion (dissociation)",
            "key1": "WMF-AM",
            "key2": "AGENT-TC",  # task completion
            "alternative": "two-sided",
            "expected": "near zero (dissociation)",
        },
        {
            "name": "H2: CONV-RFPOC vs AGENT-PQ (convergent validity)",
            "key1": "CONV-RFPOC",
            "key2": "AGENT-PQ",
            "alternative": "greater",
            "expected": "large positive (convergent)",
        },
        {
            "name": "H3: WMF-AM vs MCC-MA (divergent validity)",
            "key1": "WMF-AM",
            "key2": "MCC-MA",
            "alternative": "two-sided",
            "expected": "near zero (divergent)",
        },
        {
            "name": "H4: WMF-AM vs CLA-DC (construct separation check)",
            "key1": "WMF-AM",
            "key2": "CLA-DC",
            "alternative": "two-sided",
            "expected": "high positive (may collapse)",
        },
        {
            "name": "H5: WMF-AM vs AGENT-PQ (process quality prediction)",
            "key1": "WMF-AM",
            "key2": "AGENT-PQ",
            "alternative": "greater",
            "expected": "positive (predictive)",
        },
    ]
    
    results = []
    
    print("=" * 70)
    print("CEF Permutation P-Values and Bootstrap CIs")
    print(f"N_perm={N_PERM:,}, N_boot={N_BOOT:,}")
    print("=" * 70)
    
    for h in hypotheses:
        x, y, names = get_vectors(profiles, h["key1"], h["key2"])
        n = len(x)
        
        if n < 3:
            print(f"\n{h['name']}: SKIPPED (N={n} < 3)")
            continue
        
        # Parametric
        tau_param, p_param = stats.kendalltau(x, y)
        
        # Permutation
        tau_perm, p_perm = permutation_tau(x, y, alternative=h["alternative"])
        
        # Bootstrap CI
        ci_lo, ci_hi = bootstrap_ci(x, y)
        
        # Leave-one-family-out
        _, tau_lofo, lofo_names = leave_one_family_out(profiles, h["key1"], h["key2"])
        
        print(f"\n{h['name']}")
        print(f"  N = {n}, models: {names}")
        print(f"  Expected: {h['expected']}")
        print(f"  τ = {tau_param:.3f}")
        print(f"  Parametric p = {p_param:.4f}")
        print(f"  Permutation p ({h['alternative']}) = {p_perm:.4f}")
        print(f"  Bootstrap 95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]")
        if tau_lofo is not None:
            print(f"  Leave-Qwen-out τ = {tau_lofo:.3f} (N={len(lofo_names)})")
        else:
            print(f"  Leave-Qwen-out: insufficient N")
        
        results.append({
            "hypothesis": h["name"],
            "key1": h["key1"],
            "key2": h["key2"],
            "n": n,
            "models": names,
            "tau": round(tau_param, 4),
            "p_parametric": round(p_param, 4),
            "p_permutation": round(p_perm, 4),
            "alternative": h["alternative"],
            "ci_95_lo": round(ci_lo, 3),
            "ci_95_hi": round(ci_hi, 3),
            "tau_leave_qwen_out": round(tau_lofo, 3) if tau_lofo is not None else None,
        })
    
    # Save
    out = RESULTS_DIR / "cef_permutation_pvalues.json"
    with open(out, "w") as f:
        json.dump({"n_perm": N_PERM, "n_boot": N_BOOT, "seed": 42, "results": results}, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
