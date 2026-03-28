"""
CEF Incremental Validity Analysis

Tests: Do CEF scores predict outcomes BEYOND what model size/family/standard benchmarks predict?

Method:
  1. Load Phase 1 results (cef_phase1_full.json or cef_benchmark_ollama_phase1.json)
  2. Extract per-model scores: WMF-AM, EMC-lite, MCC-MA, MMLU, GSM8K
  3. Build model metadata (family, parameter count)
  4. Run hierarchical regression:
     - Base model: MMLU + GSM8K + param_count → target (e.g., CLA-DC degradation slope)
     - Full model: Base + WMF-AM + MCC-MA + EMC → target
     - Report ΔR², F-test for CEF block

Usage:
    python cef_incremental_validity.py --input cef_phase1_full.json
    python cef_incremental_validity.py --input cef_benchmark_ollama_phase1.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

# ── Model metadata ──────────────────────────────────────────────────────────

MODEL_META = {
    # Ollama models
    "ollama:qwen2.5:7b":      {"family": "qwen",     "params_b": 7,   "arch": "dense"},
    "ollama:qwen2.5:14b":     {"family": "qwen",     "params_b": 14,  "arch": "dense"},
    "ollama:qwen2.5:32b":     {"family": "qwen",     "params_b": 32,  "arch": "dense"},
    "ollama:llama3.1:8b":     {"family": "llama",    "params_b": 8,   "arch": "dense"},
    "ollama:llama3.1:70b":    {"family": "llama",    "params_b": 70,  "arch": "dense"},
    "ollama:gemma2:27b":      {"family": "gemma",    "params_b": 27,  "arch": "dense"},
    "ollama:deepseek-r1:14b": {"family": "deepseek", "params_b": 14,  "arch": "reasoning"},
    "ollama:mistral:7b":      {"family": "mistral",  "params_b": 7,   "arch": "dense"},
    # API models
    "gpt-4o":                 {"family": "openai",   "params_b": 200, "arch": "dense"},
    "gpt-4o-mini":            {"family": "openai",   "params_b": 8,   "arch": "dense"},
    "claude-opus-4":          {"family": "anthropic", "params_b": 200, "arch": "dense"},
    "claude-sonnet-4":        {"family": "anthropic", "params_b": 70,  "arch": "dense"},
    "gemini-1.5-pro":         {"family": "google",   "params_b": 200, "arch": "moe"},
    "gemini-1.5-flash":       {"family": "google",   "params_b": 30,  "arch": "dense"},
    "gemma-2-27b":            {"family": "gemma",    "params_b": 27,  "arch": "dense"},
    "llama3-70b":             {"family": "llama",    "params_b": 70,  "arch": "dense"},
    "llama3-8b":              {"family": "llama",    "params_b": 8,   "arch": "dense"},
    "mixtral-8x7b":           {"family": "mistral",  "params_b": 47,  "arch": "moe"},
    "qwen2-72b":              {"family": "qwen",     "params_b": 72,  "arch": "dense"},
    "deepseek-v3":            {"family": "deepseek", "params_b": 671, "arch": "moe"},
}


def extract_model_scores(results: list[dict]) -> dict:
    """Extract per-model, per-dimension mean scores from raw results."""
    scores = {}
    for r in results:
        model = r["model"]
        sd = r["sub_dim"]
        if model not in scores:
            scores[model] = {}
        if sd not in scores[model]:
            scores[model][sd] = []

        if "accurate" in r:
            scores[model][sd].append(r["accurate"])
        elif "ma_jaccard" in r:
            scores[model][sd].append(r["ma_jaccard"])
        elif "tau" in r:
            scores[model][sd].append(r["tau"])
        elif "recovery" in r:
            scores[model][sd].append(r["recovery"])

    # Compute means
    means = {}
    for model in scores:
        means[model] = {}
        for sd in scores[model]:
            vals = scores[model][sd]
            means[model][sd] = float(np.mean(vals)) if vals else 0.0

    return means


def compute_kendall_tau_matrix(model_scores: dict, dimensions: list[str]) -> dict:
    """Compute pairwise Kendall's τ between all dimensions."""
    from scipy.stats import kendalltau

    models = sorted(model_scores.keys())
    matrix = {}

    for d1 in dimensions:
        for d2 in dimensions:
            vals1 = [model_scores[m].get(d1, 0.0) for m in models]
            vals2 = [model_scores[m].get(d2, 0.0) for m in models]

            # Skip if all identical
            if len(set(vals1)) <= 1 or len(set(vals2)) <= 1:
                matrix[f"{d1}_vs_{d2}"] = {"tau": None, "p": None, "note": "no variance"}
                continue

            tau, p = kendalltau(vals1, vals2)
            matrix[f"{d1}_vs_{d2}"] = {"tau": round(tau, 4), "p": round(p, 4)}

    return matrix


def hierarchical_regression(model_scores: dict) -> dict:
    """
    Hierarchical regression: do CEF scores add predictive power beyond baselines?

    Base: log(params) + MMLU + GSM8K → target
    Full: Base + WMF-AM + MCC-MA → target

    Targets: CLA-DC accuracy, WMF-IR accuracy (if available)
    """
    from sklearn.linear_model import LinearRegression
    from scipy.stats import f as f_dist

    models = sorted(model_scores.keys())
    if len(models) < 5:
        return {"error": f"Need ≥5 models for regression, have {len(models)}"}

    # Build feature matrices
    log_params = []
    mmlu = []
    gsm8k = []
    wmf_am = []
    mcc_ma = []
    cla_dc = []

    for m in models:
        s = model_scores[m]
        meta = MODEL_META.get(m, {"params_b": 10})
        log_params.append(np.log(meta["params_b"]))
        mmlu.append(s.get("VALIDITY-MMLU", 0.0))
        gsm8k.append(s.get("VALIDITY-GSM8K", 0.0))
        wmf_am.append(s.get("WMF-AM", 0.0))
        mcc_ma.append(s.get("MCC-MA", 0.0))
        cla_dc.append(s.get("CLA-DC", 0.0))

    n = len(models)
    X_base = np.column_stack([log_params, mmlu, gsm8k])
    X_full = np.column_stack([log_params, mmlu, gsm8k, wmf_am, mcc_ma])
    y = np.array(cla_dc)

    # Check variance
    if np.std(y) < 1e-6:
        return {"error": "No variance in target variable"}

    # Fit models
    reg_base = LinearRegression().fit(X_base, y)
    reg_full = LinearRegression().fit(X_full, y)

    r2_base = reg_base.score(X_base, y)
    r2_full = reg_full.score(X_full, y)
    delta_r2 = r2_full - r2_base

    # F-test for the CEF block
    p_base = X_base.shape[1]
    p_full = X_full.shape[1]
    df_num = p_full - p_base  # number of added predictors
    df_den = n - p_full - 1

    if df_den > 0 and (1 - r2_full) > 0:
        f_stat = (delta_r2 / df_num) / ((1 - r2_full) / df_den)
        p_value = 1 - f_dist.cdf(f_stat, df_num, df_den)
    else:
        f_stat = None
        p_value = None

    return {
        "n_models": n,
        "r2_base": round(r2_base, 4),
        "r2_full": round(r2_full, 4),
        "delta_r2": round(delta_r2, 4),
        "f_statistic": round(f_stat, 4) if f_stat else None,
        "p_value": round(p_value, 4) if p_value else None,
        "base_predictors": ["log_params", "MMLU", "GSM8K"],
        "added_predictors": ["WMF-AM", "MCC-MA"],
        "target": "CLA-DC",
        "interpretation": (
            f"CEF scores add ΔR²={delta_r2:.3f} beyond baselines"
            + (f", F({df_num},{df_den})={f_stat:.2f}, p={p_value:.4f}"
               if f_stat else "")
        ),
    }


def compute_completion_cef_dissociation(model_scores: dict) -> dict:
    """
    Key analysis: do models with similar completion (MMLU/GSM8K) differ on CEF?
    This is the core "completion fallacy" evidence.
    """
    from scipy.stats import kendalltau

    models = sorted(model_scores.keys())

    # Composite completion = mean(MMLU, GSM8K)
    completion = []
    wmf_am = []
    for m in models:
        s = model_scores[m]
        comp = np.mean([s.get("VALIDITY-MMLU", 0.0), s.get("VALIDITY-GSM8K", 0.0)])
        completion.append(comp)
        wmf_am.append(s.get("WMF-AM", 0.0))

    if len(set(completion)) <= 1 or len(set(wmf_am)) <= 1:
        return {"tau": None, "note": "insufficient variance"}

    tau, p = kendalltau(completion, wmf_am)

    # Find ceiling models (completion > 0.85) with divergent WMF-AM
    ceiling_models = [(m, completion[i], wmf_am[i])
                      for i, m in enumerate(models) if completion[i] > 0.85]

    if ceiling_models:
        wmf_at_ceiling = [w for _, _, w in ceiling_models]
        wmf_spread = max(wmf_at_ceiling) - min(wmf_at_ceiling) if len(wmf_at_ceiling) > 1 else 0.0
    else:
        wmf_spread = 0.0

    return {
        "tau_completion_vs_wmf": round(tau, 4),
        "p_value": round(p, 4),
        "n_ceiling_models": len(ceiling_models),
        "wmf_spread_at_ceiling": round(wmf_spread, 4),
        "ceiling_models": [
            {"model": m, "completion": round(c, 3), "wmf_am": round(w, 3)}
            for m, c, w in ceiling_models
        ],
        "interpretation": (
            f"τ(completion, WMF-AM) = {tau:.3f} (p={p:.3f}). "
            f"Among {len(ceiling_models)} ceiling models (completion>0.85), "
            f"WMF-AM spread = {wmf_spread:.3f}"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="CEF Incremental Validity Analysis")
    parser.add_argument("--input", type=str, default="cef_phase1_full.json",
                        help="Input results file from cef_benchmark.py")
    args = parser.parse_args()

    input_path = RESULTS_DIR / args.input
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run cef_benchmark.py first.")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    results = data.get("results", data) if isinstance(data, dict) else data
    print(f"Loaded {len(results)} trials from {args.input}")
    print(f"Models: {data.get('models_completed', 'unknown')}")

    # Extract scores
    model_scores = extract_model_scores(results)
    print(f"\nModels with scores: {len(model_scores)}")
    for m in sorted(model_scores):
        dims = ", ".join(f"{k}={v:.3f}" for k, v in sorted(model_scores[m].items()))
        print(f"  {m}: {dims}")

    # 1. Kendall's τ matrix
    dims = ["WMF-AM", "WMF-IM", "WMF-IR", "MCC-MA", "CLA-DC",
            "VALIDITY-MMLU", "VALIDITY-GSM8K"]
    available_dims = [d for d in dims if any(d in model_scores[m] for m in model_scores)]

    print(f"\n{'='*60}")
    print("KENDALL'S τ CORRELATION MATRIX")
    print(f"{'='*60}")
    tau_matrix = compute_kendall_tau_matrix(model_scores, available_dims)
    for pair, vals in sorted(tau_matrix.items()):
        if vals["tau"] is not None:
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            print(f"  {pair}: τ={vals['tau']:.3f} (p={vals['p']:.3f}) {sig}")

    # 2. Completion-CEF dissociation
    print(f"\n{'='*60}")
    print("COMPLETION-CEF DISSOCIATION (Core Thesis Test)")
    print(f"{'='*60}")
    dissoc = compute_completion_cef_dissociation(model_scores)
    print(f"  {dissoc['interpretation']}")
    if dissoc.get("ceiling_models"):
        for cm in dissoc["ceiling_models"]:
            print(f"    {cm['model']}: completion={cm['completion']}, WMF-AM={cm['wmf_am']}")

    # 3. Hierarchical regression
    print(f"\n{'='*60}")
    print("HIERARCHICAL REGRESSION (Incremental Validity)")
    print(f"{'='*60}")
    reg = hierarchical_regression(model_scores)
    if "error" in reg:
        print(f"  {reg['error']}")
    else:
        print(f"  {reg['interpretation']}")
        print(f"  Base R² (params+MMLU+GSM8K): {reg['r2_base']:.4f}")
        print(f"  Full R² (+WMF-AM+MCC-MA):    {reg['r2_full']:.4f}")
        print(f"  ΔR²:                         {reg['delta_r2']:.4f}")

    # Save results
    output = {
        "input_file": args.input,
        "n_models": len(model_scores),
        "model_scores": {m: {k: round(v, 4) for k, v in s.items()}
                         for m, s in model_scores.items()},
        "tau_matrix": tau_matrix,
        "dissociation": dissoc,
        "incremental_regression": reg,
    }

    out_path = RESULTS_DIR / "cef_incremental_validity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
