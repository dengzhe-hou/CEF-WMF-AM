#!/usr/bin/env python3
"""
Block 0: Extract manipulation slope β₁ per model from item-level WMF-AM data.

Consolidates three data sources into a single long-format DataFrame,
fits a hierarchical logistic model, and computes τ correlations with ABS.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# ── Model metadata ──────────────────────────────────────────────────────────

FAMILY_MAP = {
    "qwen2.5": "Qwen", "llama3.1": "Llama", "llama3.2": "Llama",
    "gemma2": "Gemma", "deepseek-r1": "DeepSeek", "mistral": "Mistral",
    "phi3": "Phi", "mixtral": "Mistral", "command-r": "Cohere",
    "yi": "Yi", "tinyllama": "TinyLlama",
}

PARAM_MAP = {
    "qwen2.5:0.5b": 0.5, "qwen2.5:1.5b": 1.5, "qwen2.5:3b": 3,
    "qwen2.5:7b": 7, "qwen2.5:14b": 14, "qwen2.5:32b": 32,
    "llama3.1:8b": 8, "llama3.2:1b": 1, "llama3.2:3b": 3,
    "gemma2:2b": 2, "gemma2:9b": 9, "gemma2:27b": 27,
    "deepseek-r1:7b": 7, "deepseek-r1:14b": 14,
    "mistral:7b": 7, "phi3:14b": 14, "mixtral:8x7b": 46.7,
    "command-r:35b": 35, "yi:34b": 34, "tinyllama:1.1b": 1.1,
}


def get_family(model_short: str) -> str:
    for prefix, family in FAMILY_MAP.items():
        if model_short.startswith(prefix):
            return family
    return "Unknown"


def get_log_params(model_short: str) -> float:
    params = PARAM_MAP.get(model_short, None)
    return math.log10(params) if params else None


def normalize_model(name: str) -> str:
    """Strip 'ollama:' prefix if present."""
    return name.replace("ollama:", "")


# ── Data loading ────────────────────────────────────────────────────────────

def load_phase1() -> pd.DataFrame:
    """Load cef_phase1_full.json — 7 original models, 840 WMF-AM items."""
    path = DATA_DIR / "cef_phase1_full.json"
    with open(path) as f:
        data = json.load(f)

    rows = []
    for r in data["results"]:
        if r.get("sub_dim") != "WMF-AM":
            continue
        model = normalize_model(r["model"])
        rows.append({
            "model": model,
            "K": r["k"],
            "seed": r["seed"],
            "surface_form": r.get("template", "unknown"),
            "accurate": int(r["accurate"]),
        })
    return pd.DataFrame(rows)


def load_expansion8() -> pd.DataFrame:
    """Load expansion8 multiseed — 5 small models, 300 items."""
    path = list(DATA_DIR.glob("cef_wmf_multiseed_expansion8_*.json"))[0]
    with open(path) as f:
        data = json.load(f)

    rows = []
    for pm in data["per_model"]:
        model = normalize_model(pm["model"])
        for seed_block in pm["seeds"]:
            seed = seed_block["seed"]
            for d in seed_block["details"]:
                rows.append({
                    "model": model,
                    "K": d["k"],
                    "seed": seed,
                    "surface_form": "A_points",
                    "accurate": int(d["accurate"]),
                })
    return pd.DataFrame(rows)


def load_nexp() -> pd.DataFrame:
    """Load nexp/*.details.json — 8 expansion models, ~45 items each."""
    nexp_dir = DATA_DIR / "nexp"
    rows = []
    for fpath in sorted(nexp_dir.glob("*.details.json")):
        with open(fpath) as f:
            data = json.load(f)
        model = normalize_model(data["model"])
        for d in data["wmf_am"]["details"]:
            rows.append({
                "model": model,
                "K": d["k"],
                "seed": d["seed"],
                "surface_form": "nexp_default",
                "accurate": int(d["accurate"]),
            })
    return pd.DataFrame(rows)


def load_abs_scores() -> dict[str, float]:
    """Load Agent Battery Scores from cef_agent_validation_all.json."""
    path = DATA_DIR / "cef_agent_validation_all.json"
    with open(path) as f:
        data = json.load(f)

    scores = {}
    for r in data["results"]:
        model = normalize_model(r["model"])
        if model not in scores:
            scores[model] = {"total": 0, "count": 0}
        scores[model]["total"] += r["task_completion"]
        scores[model]["count"] += 1

    return {m: v["total"] / v["count"] for m, v in scores.items()}


def load_oc_scores() -> dict[str, float]:
    """Load Outcome Correctness scores. Combines two sources."""
    oc = {}

    # From completion battery v2 (5 small models)
    p1 = DATA_DIR / "cef_completion_battery_v2_20260317T122721.json"
    if p1.exists():
        with open(p1) as f:
            data = json.load(f)
        for r in data.get("results", data.get("per_model", [])):
            model = normalize_model(r["model"])
            oc[model] = r["score"]

    # From expanded (7 original models)
    p2 = DATA_DIR / "cef_expanded.json"
    if p2.exists():
        with open(p2) as f:
            data = json.load(f)
        for pm in data["per_model"]:
            model = normalize_model(pm["model"])
            if "outcome_correctness" in pm:
                oc[model] = pm["outcome_correctness"].get("score", pm["outcome_correctness"].get("accuracy"))

    # From nexp (8 expansion models)
    for fpath in sorted((DATA_DIR / "nexp").glob("*.json")):
        if "details" in fpath.name:
            continue
        with open(fpath) as f:
            data = json.load(f)
        model = normalize_model(data["model"])
        if "outcome_correctness" in data:
            oc[model] = data["outcome_correctness"].get("score", data["outcome_correctness"].get("accuracy"))

    return oc


# ── Analysis ────────────────────────────────────────────────────────────────

def fit_per_model_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit per-model logistic regression: logit(accurate) ~ K.
    Extract slope (β₁) per model.

    For models with all-zero or all-one outcomes at some K,
    uses regularized (Firth-style) logistic or falls back to OLS on proportions.
    """
    from sklearn.linear_model import LogisticRegression

    results = []
    for model, group in df.groupby("model"):
        K_vals = group["K"].values.reshape(-1, 1)
        y = group["accurate"].values

        # Per-K accuracy for descriptive stats
        per_k = group.groupby("K")["accurate"].mean()

        if len(np.unique(y)) < 2:
            # All same outcome — slope is 0
            slope = 0.0
            k50 = np.nan
        else:
            try:
                lr = LogisticRegression(penalty="l2", C=10.0, solver="lbfgs",
                                        max_iter=1000)
                lr.fit(K_vals, y)
                slope = lr.coef_[0][0]
                # K50: K at which P(correct) = 0.5
                # logit(0.5) = 0 = intercept + slope * K50
                k50 = -lr.intercept_[0] / slope if abs(slope) > 1e-6 else np.nan
            except Exception:
                # Fallback: linear regression on per-K proportions
                ks = np.array(sorted(per_k.index))
                accs = np.array([per_k[k] for k in ks])
                if len(ks) >= 2:
                    slope_lin, _, _, _, _ = stats.linregress(ks, accs)
                    slope = slope_lin
                else:
                    slope = 0.0
                k50 = np.nan

        mean_acc = group["accurate"].mean()
        results.append({
            "model": model,
            "beta1": slope,
            "K50": k50,
            "mean_acc": mean_acc,
            "n_items": len(group),
            "acc_K3": per_k.get(3, np.nan),
            "acc_K5": per_k.get(5, np.nan),
            "acc_K7": per_k.get(7, np.nan),
        })

    return pd.DataFrame(results)


def kendall_tau(x, y):
    """Kendall's tau-b with p-value."""
    mask = ~(np.isnan(x) | np.isnan(y))
    return stats.kendalltau(x[mask], y[mask])


def steiger_test_dependent_taus(tau1, tau2, tau12, n):
    """
    Approximate test for H0: tau1 = tau2 (dependent correlations).
    Uses the method from Steiger (1980) adapted for Kendall's tau.
    Returns z-statistic and p-value.
    """
    # Approximate variance of difference
    var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1))
    se_diff = math.sqrt(2 * var_tau * (1 - tau12))
    if se_diff < 1e-10:
        return 0.0, 1.0
    z = (tau1 - tau2) / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def bootstrap_tau_ci(x, y, n_boot=10000, ci=0.95):
    """Bootstrap CI for Kendall's tau."""
    taus = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        t, _ = stats.kendalltau(x[idx], y[idx])
        if not np.isnan(t):
            taus.append(t)
    taus = np.array(taus)
    lo = np.percentile(taus, (1 - ci) / 2 * 100)
    hi = np.percentile(taus, (1 + ci) / 2 * 100)
    return lo, hi


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Block 0: β₁ Manipulation Slope Analysis")
    print("=" * 70)

    # 1. Load and consolidate data
    print("\n── Loading data ──")
    df1 = load_phase1()
    print(f"  phase1: {len(df1)} items, {df1['model'].nunique()} models")

    df2 = load_expansion8()
    print(f"  expansion8: {len(df2)} items, {df2['model'].nunique()} models")

    df3 = load_nexp()
    print(f"  nexp: {len(df3)} items, {df3['model'].nunique()} models")

    df = pd.concat([df1, df2, df3], ignore_index=True)
    df["family"] = df["model"].apply(get_family)
    df["log_params"] = df["model"].apply(get_log_params)

    print(f"\n  TOTAL: {len(df)} items, {df['model'].nunique()} models")
    print(f"  Models: {sorted(df['model'].unique())}")

    # Save consolidated CSV
    csv_path = DATA_DIR / "wmf_am_item_level_consolidated.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved to: {csv_path}")

    # 2. Fit per-model slopes
    print("\n── Fitting per-model slopes ──")
    slopes = fit_per_model_slopes(df)

    # 3. Load ABS and OC
    abs_scores = load_abs_scores()
    oc_scores = load_oc_scores()

    slopes["ABS"] = slopes["model"].map(abs_scores)
    slopes["OC"] = slopes["model"].map(oc_scores)
    slopes["log_params"] = slopes["model"].apply(get_log_params)
    slopes["family"] = slopes["model"].apply(get_family)

    # Sort by β₁
    slopes = slopes.sort_values("beta1")

    print("\n── Per-model results ──")
    print(slopes[["model", "family", "beta1", "K50", "mean_acc", "ABS", "OC",
                   "acc_K3", "acc_K5", "acc_K7", "n_items"]].to_string(index=False))

    # Save slopes
    slopes_path = DATA_DIR / "beta1_slopes.csv"
    slopes.to_csv(slopes_path, index=False)
    print(f"\n  Saved to: {slopes_path}")

    # 4. Compute tau correlations
    print("\n── τ correlations with ABS ──")
    valid = slopes.dropna(subset=["ABS", "beta1", "mean_acc"])

    beta1 = valid["beta1"].values
    mean_acc = valid["mean_acc"].values
    abs_vals = valid["ABS"].values
    log_p = valid["log_params"].values
    oc_vals = valid["OC"].values

    # Note: β₁ is negative (accuracy decreases with K), so more negative = steeper decline
    # For correlation with ABS: models with LESS negative β₁ (flatter) should have HIGHER ABS
    # So τ(β₁, ABS) should be positive if the hypothesis holds

    tau_beta1, p_beta1 = kendall_tau(beta1, abs_vals)
    tau_mean, p_mean = kendall_tau(mean_acc, abs_vals)
    tau_logp, p_logp = kendall_tau(log_p, abs_vals)

    # OC may have NaNs for some models
    oc_mask = ~np.isnan(oc_vals)
    if oc_mask.sum() > 5:
        tau_oc, p_oc = kendall_tau(oc_vals[oc_mask], abs_vals[oc_mask])
    else:
        tau_oc, p_oc = np.nan, np.nan

    print(f"\n  τ(β₁, ABS)       = {tau_beta1:.3f}  (p = {p_beta1:.4f})  N = {len(valid)}")
    print(f"  τ(mean_acc, ABS)  = {tau_mean:.3f}  (p = {p_mean:.4f})  N = {len(valid)}")
    print(f"  τ(log_params, ABS)= {tau_logp:.3f}  (p = {p_logp:.4f})  N = {len(valid)}")
    print(f"  τ(OC, ABS)        = {tau_oc:.3f}  (p = {p_oc:.4f})  N = {oc_mask.sum()}")

    # 5. Steiger test: β₁ vs mean_acc
    print("\n── Steiger test: τ(β₁,ABS) vs τ(mean_acc,ABS) ──")
    tau_beta1_mean, _ = kendall_tau(beta1, mean_acc)
    z, p_steiger = steiger_test_dependent_taus(
        abs(tau_beta1), abs(tau_mean), tau_beta1_mean, len(valid)
    )
    print(f"  |τ(β₁,ABS)| = {abs(tau_beta1):.3f} vs |τ(mean,ABS)| = {abs(tau_mean):.3f}")
    print(f"  τ(β₁,mean) = {tau_beta1_mean:.3f}")
    print(f"  Steiger z = {z:.3f}, p = {p_steiger:.4f}")
    if abs(tau_beta1) > abs(tau_mean):
        print("  → β₁ is a STRONGER predictor than mean accuracy ✓")
    else:
        print("  → β₁ is NOT stronger than mean accuracy ✗")

    # 6. Bootstrap CIs
    print("\n── Bootstrap 95% CIs (10,000 resamples) ──")
    ci_beta1 = bootstrap_tau_ci(beta1, abs_vals)
    ci_mean = bootstrap_tau_ci(mean_acc, abs_vals)
    print(f"  τ(β₁, ABS):      [{ci_beta1[0]:.3f}, {ci_beta1[1]:.3f}]")
    print(f"  τ(mean_acc, ABS): [{ci_mean[0]:.3f}, {ci_mean[1]:.3f}]")

    # 7. Leave-one-family-out
    print("\n── Leave-one-family-out stability ──")
    families = valid["family"].unique()
    for fam in sorted(families):
        subset = valid[valid["family"] != fam]
        t, p = kendall_tau(subset["beta1"].values, subset["ABS"].values)
        print(f"  Drop {fam:12s} (N={len(subset):2d}): τ(β₁,ABS) = {t:.3f} (p={p:.4f})")

    # 8. Decision Gate G0
    print("\n" + "=" * 70)
    print("DECISION GATE G0")
    print("=" * 70)
    if abs(tau_beta1) > abs(tau_mean):
        print(f"✅ PASS: |τ(β₁,ABS)| = {abs(tau_beta1):.3f} > |τ(mean,ABS)| = {abs(tau_mean):.3f}")
        print("   → Manipulation slope β₁ is a stronger predictor. Proceed with C1.")
    else:
        print(f"❌ FAIL: |τ(β₁,ABS)| = {abs(tau_beta1):.3f} ≤ |τ(mean,ABS)| = {abs(tau_mean):.3f}")
        print("   → Mean accuracy is at least as good. Consider reframing contribution.")


if __name__ == "__main__":
    main()
