#!/usr/bin/env python3
"""
Master consolidation & analysis script for WMF-AM paper (N=28).
Reads all raw data files, computes every statistic claimed in main.tex,
and saves to a single consolidated JSON for reproducibility auditing.

Usage:
    python consolidate_n28.py
"""

import json
import glob
import re
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau, rankdata
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "consolidated_n28.json"

# ── Canonical 28 models ─────────────────────────────────────────────────────
# Maps display_name -> list of possible raw-data keys
CANONICAL_MODELS = {
    # API / LRM (8)
    "claude-sonnet-4":   ["openrouter:claude-sonnet-4"],
    "o3-mini":           ["openrouter:o3-mini"],
    "deepseek-r1-full":  ["openrouter:deepseek-r1"],
    "deepseek-v3":       ["openrouter:deepseek-v3"],
    "gpt-4o":            ["openrouter:gpt-4o"],
    "gemini-2.5-flash":  ["openrouter:gemini-2.5-flash", "google:gemini-2.5-flash"],
    "gpt-4o-mini":       ["openrouter:gpt-4o-mini"],
    "llama3.1:70b":      ["ollama:llama3.1:70b"],
    # Open-weight original 7
    "deepseek-r1:14b":   ["ollama:deepseek-r1:14b"],
    "qwen2.5:32b":       ["ollama:qwen2.5:32b"],
    "qwen2.5:14b":       ["ollama:qwen2.5:14b"],
    "gemma2:27b":        ["ollama:gemma2:27b"],
    "qwen2.5:7b":        ["ollama:qwen2.5:7b"],
    "mistral:7b":        ["ollama:mistral:7b"],
    "llama3.1:8b":       ["ollama:llama3.1:8b"],
    # Open-weight expansion 8
    "gemma2:9b":         ["ollama:gemma2:9b"],
    "command-r:35b":     ["ollama:command-r:35b"],
    "mixtral:8x7b":      ["ollama:mixtral:8x7b"],
    "phi3:14b":          ["ollama:phi3:14b"],
    "yi:34b":            ["ollama:yi:34b"],
    "qwen2.5:3b":        ["ollama:qwen2.5:3b"],
    "deepseek-r1:7b":    ["ollama:deepseek-r1:7b"],
    "llama3.2:3b":       ["ollama:llama3.2:3b"],
    # Small baselines (5)
    "gemma2:2b":         ["ollama:gemma2:2b"],
    "qwen2.5:1.5b":      ["ollama:qwen2.5:1.5b"],
    "tinyllama:1.1b":    ["ollama:tinyllama:1.1b"],
    "llama3.2:1b":       ["ollama:llama3.2:1b"],
    "qwen2.5:0.5b":      ["ollama:qwen2.5:0.5b"],
}

MODEL_TYPE = {
    "claude-sonnet-4": "API", "o3-mini": "LRM", "deepseek-r1-full": "LRM",
    "deepseek-v3": "API", "gpt-4o": "API", "gemini-2.5-flash": "API",
    "gpt-4o-mini": "API", "llama3.1:70b": "Ollama",
    "deepseek-r1:14b": "Ollama", "qwen2.5:32b": "Ollama",
    "qwen2.5:14b": "Ollama", "gemma2:27b": "Ollama",
    "qwen2.5:7b": "Ollama", "mistral:7b": "Ollama",
    "llama3.1:8b": "Ollama", "gemma2:9b": "Ollama",
    "command-r:35b": "Ollama", "mixtral:8x7b": "Ollama",
    "phi3:14b": "Ollama", "yi:34b": "Ollama",
    "qwen2.5:3b": "Ollama", "deepseek-r1:7b": "Ollama",
    "llama3.2:3b": "Ollama", "gemma2:2b": "Ollama",
    "qwen2.5:1.5b": "Ollama", "tinyllama:1.1b": "Ollama",
    "llama3.2:1b": "Ollama", "qwen2.5:0.5b": "Ollama",
}

MODEL_FAMILY = {
    "claude-sonnet-4": "Anthropic", "o3-mini": "OpenAI", "deepseek-r1-full": "DeepSeek",
    "deepseek-v3": "DeepSeek", "gpt-4o": "OpenAI", "gemini-2.5-flash": "Google",
    "gpt-4o-mini": "OpenAI", "llama3.1:70b": "Llama",
    "deepseek-r1:14b": "DeepSeek", "qwen2.5:32b": "Qwen",
    "qwen2.5:14b": "Qwen", "gemma2:27b": "Gemma",
    "qwen2.5:7b": "Qwen", "mistral:7b": "Mistral",
    "llama3.1:8b": "Llama", "gemma2:9b": "Gemma",
    "command-r:35b": "Cohere", "mixtral:8x7b": "Mistral",
    "phi3:14b": "Phi", "yi:34b": "Yi",
    "qwen2.5:3b": "Qwen", "deepseek-r1:7b": "DeepSeek",
    "llama3.2:3b": "Llama", "gemma2:2b": "Gemma",
    "qwen2.5:1.5b": "Qwen", "tinyllama:1.1b": "TinyLlama",
    "llama3.2:1b": "Llama", "qwen2.5:0.5b": "Qwen",
}


def resolve(raw_key):
    """Map a raw data key to canonical display name."""
    for display, aliases in CANONICAL_MODELS.items():
        if raw_key in aliases:
            return display
    return None


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── Statistics helpers ──────────────────────────────────────────────────────

def bootstrap_kendall_ci(x, y, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(x)
    taus = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        t, _ = kendalltau(x[idx], y[idx])
        if not np.isnan(t):
            taus.append(t)
    return np.percentile(taus, 100 * alpha / 2), np.percentile(taus, 100 * (1 - alpha / 2))


def partial_kendall(x, y, z):
    """Partial Kendall tau: tau(x, y | z) via rank-residualization."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(r, c):
        cm = c - np.mean(c)
        beta = np.dot(cm, r - np.mean(r)) / (np.dot(cm, cm) + 1e-12)
        return r - beta * c
    return kendalltau(resid(rx, rz), resid(ry, rz))


def sigmoid_fit(k_vals, acc_vals):
    """Fit 4-param sigmoid: a / (1 + exp(alpha*(K - K_crit))) + offset."""
    from scipy.optimize import curve_fit
    def sigmoid(K, a, alpha, K_crit, offset):
        return a / (1.0 + np.exp(alpha * (K - K_crit))) + offset
    k = np.array(k_vals, dtype=float)
    a = np.array(acc_vals, dtype=float)
    try:
        popt, _ = curve_fit(sigmoid, k, a,
                            p0=[1.0, 0.5, 10.0, 0.0],
                            bounds=([0, 0.01, 0.1, -0.5], [2, 5, 200, 0.5]),
                            maxfev=10000)
        pred = sigmoid(k, *popt)
        ss_res = np.sum((a - pred) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        return {"a": popt[0], "alpha": popt[1], "K_crit": popt[2],
                "offset": popt[3], "R2": r2}
    except Exception as e:
        return {"error": str(e), "K_crit": None, "R2": None}


# ── Data loading ────────────────────────────────────────────────────────────

def load_wmf_am_scores():
    """Load WMF-AM mean accuracy for all 28 models."""
    scores = {}

    # 1. Ollama multiseed original (7 models)
    d = load_json(DATA_DIR / "cef_wmf_multiseed.json")
    for m in d["per_model"]:
        name = resolve(m["model"])
        if name:
            all_acc = [s["mean_accuracy"] for s in m["seeds"]]
            scores[name] = round(np.mean(all_acc), 4)

    # 2. Ollama multiseed expansion (5 small models)
    d = load_json(DATA_DIR / "cef_wmf_multiseed_expansion8_20260317T122849.json")
    for m in d["per_model"]:
        name = resolve(m["model"])
        if name and name not in scores:
            all_acc = [s["mean_accuracy"] for s in m["seeds"]]
            scores[name] = round(np.mean(all_acc), 4)

    # 3. Ollama N-expansion (8 models) — nexp format: {"wmf_am": {"mean": ...}}
    for f in sorted(glob.glob(str(DATA_DIR / "nexp" / "cef_nexp_ollama_*.json"))):
        if ".details." in f:
            continue
        d = load_json(f)
        raw = d.get("model", "")
        name = resolve(raw)
        if name and name not in scores:
            if isinstance(d.get("wmf_am"), dict) and "mean" in d["wmf_am"]:
                scores[name] = round(d["wmf_am"]["mean"], 4)
            elif "wmf_am_score" in d:
                scores[name] = round(d["wmf_am_score"], 4)

    # 4. API models (from api_held_out_final, latest wins)
    for f in sorted(glob.glob(str(RESULTS_DIR / "api_held_out_final_*.json"))):
        d = load_json(f)
        for r in d["results"]:
            name = resolve(r["model"])
            if name:
                scores[name] = round(r["wmf_am_score"], 4)

    # 5. Fallback: template harmonization "bare" results for remaining models
    for f in sorted(glob.glob(str(DATA_DIR / "wmf_am_template_harmonization_*.json"))) + \
             sorted(glob.glob(str(RESULTS_DIR / "wmf_am_template_harmonization_*.json"))):
        d = load_json(f)
        for r in d.get("results", []):
            name = resolve(r["model"])
            if name and name not in scores and r.get("template") == "bare":
                scores[name] = round(r["mean"], 4)

    return scores


def load_agent_scores():
    """Load Agent Battery Score for all 28 models."""
    scores = {}

    # 1. Ollama (20 models from cef_agent_validation_all)
    d = load_json(DATA_DIR / "cef_agent_validation_all.json")
    per_model = {}
    for r in d["results"]:
        m = r["model"]
        if m not in per_model:
            per_model[m] = {"correct": 0, "total": 0}
        per_model[m]["total"] += 1
        per_model[m]["correct"] += r["task_completion"]
    for raw, v in per_model.items():
        name = resolve(raw)
        if name:
            scores[name] = round(v["correct"] / v["total"], 2)

    # 2. API models (from api_held_out files with agent_score, latest wins)
    for f in sorted(glob.glob(str(RESULTS_DIR / "api_held_out_2026041*.json"))):
        d = load_json(f)
        for r in d["results"]:
            if "agent_score" in r:
                name = resolve(r["model"])
                if name:
                    scores[name] = round(r["agent_score"], 2)

    # 3. Fallback: load-shift supported score as agent proxy
    for fname in ["load_shift_ollama_all.json", "load_shift_ollama_full.json",
                   "load_shift_api_full.json"]:
        p = RESULTS_DIR / fname
        if p.exists():
            d = load_json(p)
            for r in d["results"]:
                name = resolve(r["model"])
                if name and name not in scores:
                    scores[name] = round(r["supported_score"], 2)

    return scores


def load_baseline_scores():
    """Load MMLU and GSM8K scores."""
    scores = {}
    for f in ["baseline_api.json", "baseline_ollama.json"]:
        p = RESULTS_DIR / f
        if p.exists():
            d = load_json(p)
            for r in d["results"]:
                name = resolve(r["model"])
                if name:
                    scores[name] = {
                        "mmlu": round(r["mmlu_score"], 4),
                        "gsm8k": round(r["gsm8k_score"], 4),
                    }
    return scores


def load_ksweep_data():
    """Load K-sweep per_k_accuracy for all 28 models."""
    data = {}
    # Load Ollama (21 models)
    p = RESULTS_DIR / "extended_k_20260413T180649.json"
    if p.exists():
        d = load_json(p)
        for r in d["results"]:
            name = resolve(r["model"])
            if name:
                data[name] = {int(k): v for k, v in r["per_k_accuracy"].items()}
    # Load API (7 models)
    p = RESULTS_DIR / "extended_k_20260413T030646.json"
    if p.exists():
        d = load_json(p)
        for r in d["results"]:
            name = resolve(r["model"])
            if name and name not in data:
                data[name] = {int(k): v for k, v in r["per_k_accuracy"].items()}

    # Merge WMF-AM K=3,5,7 data for models that only have high-K sweep
    return data


def load_loadshift_data():
    """Load supported/unsupported agent scores for all 28 models."""
    data = {}
    # Ollama (from load_shift_ollama_all, 17 models)
    for fname in ["load_shift_ollama_all.json", "load_shift_ollama_full.json"]:
        p = RESULTS_DIR / fname
        if p.exists():
            d = load_json(p)
            for r in d["results"]:
                name = resolve(r["model"])
                if name and name not in data:
                    data[name] = {
                        "supported": r["supported_score"],
                        "unsupported": r["unsupported_score"],
                        "delta": round(r["supported_score"] - r["unsupported_score"], 2),
                    }
    # API (6 models from full file + individual runs)
    for fname in ["load_shift_api_full.json"] + \
                 sorted([f.name for f in RESULTS_DIR.glob("load_shift_2026041*.json")]):
        p = RESULTS_DIR / fname
        if p.exists():
            d = load_json(p)
            for r in d["results"]:
                name = resolve(r["model"])
                if name and name not in data:
                    data[name] = {
                        "supported": r["supported_score"],
                        "unsupported": r["unsupported_score"],
                        "delta": round(r["supported_score"] - r["unsupported_score"], 2),
                    }
    return data


def load_nonarith_data():
    """Load non-arithmetic ablation accuracy per model."""
    data = {}
    # Load ALL nonarith files, latest per model wins
    for f in sorted(glob.glob(str(RESULTS_DIR / "wmf_am_nonarith_*.json"))) + \
             sorted(glob.glob(str(DATA_DIR / "wmf_am_nonarith_*.json"))):
        d = load_json(f)
        per_model = {}
        for r in d["results"]:
            m = r["model"]
            if m not in per_model:
                per_model[m] = {"correct": 0, "total": 0}
            per_model[m]["total"] += 1
            per_model[m]["correct"] += int(r["accurate"])
        for raw, v in per_model.items():
            name = resolve(raw)
            if name:
                data[name] = round(v["correct"] / v["total"], 4)
    return data


def load_yoked_data():
    """Load yoked cancellation accuracy per model."""
    data = {}
    # Per-model files
    for f in sorted(glob.glob(str(RESULTS_DIR / "wmf_am_yoked_control_*.json"))) + \
             sorted(glob.glob(str(DATA_DIR / "wmf_am_yoked_control_*.json"))):
        d = load_json(f)
        if isinstance(d, dict) and "summary" in d:
            name = resolve(d["metadata"]["model"])
            if name:
                data[name] = round(d["summary"]["overall_accuracy"], 4)
        elif isinstance(d, list):
            if d:
                raw = d[0].get("model", "")
                name = resolve(raw)
                correct = sum(1 for r in d if r.get("accurate", False))
                if name:
                    data[name] = round(correct / len(d), 4)
    return data


def load_template_data():
    """Load template harmonization data (bare/chat/cot accuracy per model)."""
    data = {}
    # Original 15 models
    p = DATA_DIR / "wmf_am_template_harmonization_20260317T073226.json"
    if p.exists():
        d = load_json(p)
        for r in d["results"]:
            name = resolve(r["model"])
            if name:
                if name not in data:
                    data[name] = {}
                data[name][r["template"]] = round(r["mean"], 4)
    # Additional models from results dir
    for f in sorted(glob.glob(str(RESULTS_DIR / "wmf_am_template_harmonization_*.json"))):
        d = load_json(f)
        for r in d.get("results", []):
            name = resolve(r["model"])
            if name:
                if name not in data:
                    data[name] = {}
                data[name][r["template"]] = round(r["mean"], 4)
    return data


def load_k1_control():
    """Load K=1 (single-step) control accuracy per model."""
    data = {}
    for f in sorted(glob.glob(str(RESULTS_DIR / "wmf_am_control_*.json"))) + \
             sorted(glob.glob(str(DATA_DIR / "wmf_am_control_*.json"))):
        d = load_json(f)
        if isinstance(d, list):
            if not d:
                continue
            raw = d[0].get("model", "")
            correct = sum(1 for r in d if r.get("accurate", False))
            name = resolve(raw)
            if name:
                data[name] = round(correct / len(d), 4)
        elif isinstance(d, dict) and "summary" in d:
            name = resolve(d.get("metadata", {}).get("model", ""))
            if name:
                data[name] = round(d["summary"]["overall_accuracy"], 4)
    return data


# ── Main analysis ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WMF-AM Consolidation Script (N=28)")
    print("=" * 60)

    # Load all data
    wmf = load_wmf_am_scores()
    agent = load_agent_scores()
    baseline = load_baseline_scores()
    ksweep = load_ksweep_data()
    loadshift = load_loadshift_data()
    nonarith = load_nonarith_data()
    yoked = load_yoked_data()
    template = load_template_data()
    k1_control = load_k1_control()

    # ── Coverage report ─────────────────────────────────────────────────
    print("\n── Coverage Report ──")
    all_models = sorted(CANONICAL_MODELS.keys())
    for src_name, src_data in [("WMF-AM", wmf), ("Agent", agent),
                                ("MMLU/GSM8K", baseline), ("K-sweep", ksweep),
                                ("Load-shift", loadshift), ("Non-arith", nonarith),
                                ("Yoked", yoked), ("Template", template),
                                ("K=1 control", k1_control)]:
        missing = [m for m in all_models if m not in src_data]
        print(f"  {src_name:15s}: {len(src_data):2d}/28  missing: {missing if missing else '(none)'}")

    # ── Build per-model table ───────────────────────────────────────────
    per_model = {}
    for m in all_models:
        per_model[m] = {
            "type": MODEL_TYPE[m],
            "family": MODEL_FAMILY[m],
            "wmf_am": wmf.get(m),
            "agent": agent.get(m),
            "mmlu": baseline.get(m, {}).get("mmlu"),
            "gsm8k": baseline.get(m, {}).get("gsm8k"),
            "nonarith": nonarith.get(m),
            "yoked": yoked.get(m),
            "k1_control": k1_control.get(m),
            "loadshift_supported": loadshift.get(m, {}).get("supported") if m in loadshift else None,
            "loadshift_unsupported": loadshift.get(m, {}).get("unsupported") if m in loadshift else None,
            "loadshift_delta": loadshift.get(m, {}).get("delta") if m in loadshift else None,
        }

    # ── Compute primary statistics ──────────────────────────────────────
    print("\n── Primary Statistics ──")
    stats = {}

    # Get aligned arrays for models with both WMF-AM and Agent
    both = [m for m in all_models if wmf.get(m) is not None and agent.get(m) is not None]
    w = np.array([wmf[m] for m in both])
    a = np.array([agent[m] for m in both])
    n = len(both)

    tau, p = kendalltau(w, a)
    ci_lo, ci_hi = bootstrap_kendall_ci(w, a)
    stats["wmf_am_vs_agent"] = {
        "tau": round(tau, 3), "p": round(p, 6), "n": n,
        "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
        "models": both,
    }
    print(f"  τ(WMF-AM, Agent) = {tau:.3f}, p = {p:.4f}, N = {n}, CI = [{ci_lo:.3f}, {ci_hi:.3f}]")

    # MMLU vs Agent
    both_mmlu = [m for m in all_models if baseline.get(m, {}).get("mmlu") is not None and agent.get(m) is not None]
    if both_mmlu:
        mm = np.array([baseline[m]["mmlu"] for m in both_mmlu])
        aa = np.array([agent[m] for m in both_mmlu])
        tau_mmlu, p_mmlu = kendalltau(mm, aa)
        stats["mmlu_vs_agent"] = {"tau": round(tau_mmlu, 3), "p": round(p_mmlu, 6), "n": len(both_mmlu)}
        print(f"  τ(MMLU, Agent) = {tau_mmlu:.3f}, p = {p_mmlu:.4f}, N = {len(both_mmlu)}")

    # GSM8K vs Agent
    both_gsm = [m for m in all_models if baseline.get(m, {}).get("gsm8k") is not None and agent.get(m) is not None]
    if both_gsm:
        gg = np.array([baseline[m]["gsm8k"] for m in both_gsm])
        aa = np.array([agent[m] for m in both_gsm])
        tau_gsm, p_gsm = kendalltau(gg, aa)
        stats["gsm8k_vs_agent"] = {"tau": round(tau_gsm, 3), "p": round(p_gsm, 6), "n": len(both_gsm)}
        print(f"  τ(GSM8K, Agent) = {tau_gsm:.3f}, p = {p_gsm:.4f}, N = {len(both_gsm)}")

    # Partial τ(WMF-AM, Agent | OC) — on open-weight subset
    oc_models = [m for m in all_models if wmf.get(m) is not None and agent.get(m) is not None
                 and baseline.get(m, {}).get("mmlu") is not None and MODEL_TYPE[m] == "Ollama"]
    if len(oc_models) >= 5:
        w_oc = np.array([wmf[m] for m in oc_models])
        a_oc = np.array([agent[m] for m in oc_models])
        # Use MMLU as OC proxy (or completion score if available)
        mm_oc = np.array([baseline[m]["mmlu"] for m in oc_models])
        pt, pp = partial_kendall(w_oc, a_oc, mm_oc)
        stats["partial_tau_wmf_agent_given_oc"] = {
            "tau": round(pt, 3), "p": round(pp, 4), "n": len(oc_models),
            "note": "partial tau(WMF-AM, Agent | MMLU) on Ollama subset",
        }
        print(f"  partial τ(WMF-AM, Agent | MMLU) = {pt:.3f}, p = {pp:.4f}, N = {len(oc_models)}")

    # Partial τ(WMF-AM, Agent | MMLU) — full N=28
    both_full = [m for m in all_models if wmf.get(m) is not None and agent.get(m) is not None
                 and baseline.get(m, {}).get("mmlu") is not None]
    if len(both_full) >= 5:
        w_f = np.array([wmf[m] for m in both_full])
        a_f = np.array([agent[m] for m in both_full])
        mm_f = np.array([baseline[m]["mmlu"] for m in both_full])
        pt2, pp2 = partial_kendall(w_f, a_f, mm_f)
        stats["partial_tau_wmf_agent_given_mmlu_full"] = {
            "tau": round(pt2, 3), "p": round(pp2, 4), "n": len(both_full),
        }
        print(f"  partial τ(WMF-AM, Agent | MMLU, full) = {pt2:.3f}, p = {pp2:.4f}, N = {len(both_full)}")

    # Yoked vs Agent
    both_yoked = [m for m in all_models if yoked.get(m) is not None and agent.get(m) is not None]
    if both_yoked:
        yw = np.array([yoked[m] for m in both_yoked])
        ya = np.array([agent[m] for m in both_yoked])
        tau_y, p_y = kendalltau(yw, ya)
        stats["yoked_vs_agent"] = {"tau": round(tau_y, 3), "p": round(p_y, 4), "n": len(both_yoked)}
        print(f"  τ(Yoked, Agent) = {tau_y:.3f}, p = {p_y:.4f}, N = {len(both_yoked)}")

    # Template stability: bare vs chat
    bare_models = [m for m in all_models if template.get(m, {}).get("bare") is not None
                   and template.get(m, {}).get("chat") is not None]
    if bare_models:
        tb = np.array([template[m]["bare"] for m in bare_models])
        tc = np.array([template[m]["chat"] for m in bare_models])
        tau_t, p_t = kendalltau(tb, tc)
        stats["template_bare_vs_chat"] = {"tau": round(tau_t, 3), "p": round(p_t, 4), "n": len(bare_models)}
        print(f"  τ(bare, chat) = {tau_t:.3f}, p = {p_t:.4f}, N = {len(bare_models)}")

    # K=1 ceiling
    k1_models = [m for m in all_models if k1_control.get(m) is not None]
    if k1_models:
        k1_vals = [k1_control[m] for m in k1_models]
        n_ceiling = sum(1 for v in k1_vals if v >= 0.95)
        stats["k1_control"] = {
            "n_models": len(k1_models),
            "n_ceiling_095": n_ceiling,
            "mean": round(np.mean(k1_vals), 3),
        }
        print(f"  K=1 control: {n_ceiling}/{len(k1_models)} at ceiling (≥0.95), mean = {np.mean(k1_vals):.3f}")

    # Non-arith ceiling
    na_models = [m for m in all_models if nonarith.get(m) is not None]
    if na_models:
        na_vals = [nonarith[m] for m in na_models]
        stats["nonarith_ceiling"] = {
            "n_models": len(na_models),
            "mean": round(np.mean(na_vals), 3),
            "range": [round(min(na_vals), 3), round(max(na_vals), 3)],
        }
        print(f"  Non-arith ceiling: mean = {np.mean(na_vals):.3f}, N = {len(na_models)}, range = [{min(na_vals):.3f}, {max(na_vals):.3f}]")

    # MMLU ceiling %
    mmlu_models = [m for m in all_models if baseline.get(m, {}).get("mmlu") is not None]
    if mmlu_models:
        mmlu_vals = [baseline[m]["mmlu"] for m in mmlu_models]
        n_ceiling_mmlu = sum(1 for v in mmlu_vals if v >= 0.95)
        pct = round(100 * n_ceiling_mmlu / len(mmlu_vals))
        stats["mmlu_ceiling"] = {"pct_at_095": pct, "n": len(mmlu_models)}
        print(f"  MMLU ceiling: {pct}% at ≥0.95 ({n_ceiling_mmlu}/{len(mmlu_models)})")

    # ── K-sweep analysis ────────────────────────────────────────────────
    print("\n── K-sweep Analysis ──")
    kcrit_data = {}
    for m in all_models:
        if m not in ksweep:
            continue
        kd = ksweep[m]
        k_vals = sorted(kd.keys())
        acc_vals = [kd[k] for k in k_vals]
        fit = sigmoid_fit(k_vals, acc_vals)
        kcrit_data[m] = fit
        if fit.get("K_crit") is not None:
            print(f"  {m:25s}: K_crit = {fit['K_crit']:6.1f}, R² = {fit['R2']:.3f}")

    # K_crit vs Agent
    kcrit_models = [m for m in all_models if m in kcrit_data and
                    kcrit_data[m].get("K_crit") is not None and agent.get(m) is not None]
    if kcrit_models:
        kc = np.array([kcrit_data[m]["K_crit"] for m in kcrit_models])
        ka = np.array([agent[m] for m in kcrit_models])
        tau_kc, p_kc = kendalltau(kc, ka)
        stats["kcrit_vs_agent"] = {"tau": round(tau_kc, 3), "p": round(p_kc, 4), "n": len(kcrit_models)}
        print(f"\n  τ(K_crit, Agent) = {tau_kc:.3f}, p = {p_kc:.4f}, N = {len(kcrit_models)}")

        # Excluding R1 (poor fit)
        kcrit_no_r1 = [m for m in kcrit_models if m != "deepseek-r1-full"]
        if kcrit_no_r1:
            kc2 = np.array([kcrit_data[m]["K_crit"] for m in kcrit_no_r1])
            ka2 = np.array([agent[m] for m in kcrit_no_r1])
            tau_kc2, p_kc2 = kendalltau(kc2, ka2)
            stats["kcrit_vs_agent_no_r1"] = {"tau": round(tau_kc2, 3), "p": round(p_kc2, 4), "n": len(kcrit_no_r1)}
            print(f"  τ(K_crit, Agent) excl R1 = {tau_kc2:.3f}, p = {p_kc2:.4f}, N = {len(kcrit_no_r1)}")

    kcrit_vals = [kcrit_data[m]["K_crit"] for m in kcrit_models if kcrit_data[m].get("K_crit")]
    if kcrit_vals:
        stats["kcrit_range"] = {"min": round(min(kcrit_vals), 1), "max": round(max(kcrit_vals), 1)}
        print(f"  K_crit range: {min(kcrit_vals):.1f} – {max(kcrit_vals):.1f}")

    # ── Load-shift analysis ─────────────────────────────────────────────
    print("\n── Load-Shift Analysis ──")
    ls_models = [m for m in all_models if m in loadshift]
    if ls_models:
        deltas = [loadshift[m]["delta"] for m in ls_models]
        mean_delta = np.mean(deltas)
        sd_delta = np.std(deltas, ddof=1)
        stats["loadshift"] = {
            "n_models": len(ls_models),
            "mean_delta": round(mean_delta, 2),
            "sd_delta": round(sd_delta, 2),
        }
        print(f"  Mean Δ = {mean_delta:.2f}, SD = {sd_delta:.2f}, N = {len(ls_models)}")

        # Load-shift delta vs baseline agent
        ls_with_agent = [m for m in ls_models if agent.get(m) is not None]
        if ls_with_agent:
            ld = np.array([loadshift[m]["delta"] for m in ls_with_agent])
            la = np.array([agent[m] for m in ls_with_agent])
            # Note: paper uses supported score as baseline
            ls_sup = np.array([loadshift[m]["supported"] for m in ls_with_agent])
            tau_ls, p_ls = kendalltau(ld, ls_sup)
            stats["loadshift_delta_vs_supported"] = {
                "tau": round(tau_ls, 3), "p": round(p_ls, 4), "n": len(ls_with_agent),
            }
            print(f"  τ(Δ, supported) = {tau_ls:.3f}, p = {p_ls:.4f}, N = {len(ls_with_agent)}")

    # ── Leave-one-task-out ──────────────────────────────────────────────
    print("\n── Leave-One-Task-Out (WMF-AM → Agent) ──")
    # Recompute agent scores excluding each task
    agent_task_data = {}
    d = load_json(DATA_DIR / "cef_agent_validation_all.json")
    for r in d["results"]:
        name = resolve(r["model"])
        if name:
            if name not in agent_task_data:
                agent_task_data[name] = {}
            agent_task_data[name][r["task_id"]] = r["task_completion"]

    # Also load API agent task-level data
    for f in sorted(glob.glob(str(RESULTS_DIR / "api_held_out_2026041*.json"))):
        dd = load_json(f)
        for r in dd["results"]:
            if "task_results" in r:
                name = resolve(r["model"])
                if name and name not in agent_task_data:
                    agent_task_data[name] = {}
                    for t in r["task_results"]:
                        agent_task_data[name][t["task_id"]] = int(t["correct"])

    if agent_task_data:
        all_tasks = set()
        for m in agent_task_data:
            all_tasks.update(agent_task_data[m].keys())
        loto_taus = {}
        loto_models = [m for m in all_models if m in agent_task_data and wmf.get(m) is not None]
        for task in sorted(all_tasks):
            # Recompute agent score without this task
            agent_no_task = {}
            for m in loto_models:
                tasks = agent_task_data[m]
                remaining = {t: v for t, v in tasks.items() if t != task}
                if remaining:
                    agent_no_task[m] = sum(remaining.values()) / len(remaining)
            ms = [m for m in loto_models if m in agent_no_task]
            if len(ms) >= 5:
                ww = np.array([wmf[m] for m in ms])
                aa = np.array([agent_no_task[m] for m in ms])
                t, p = kendalltau(ww, aa)
                loto_taus[task] = round(t, 3)
        if loto_taus:
            stats["leave_one_task_out"] = loto_taus
            print(f"  τ range: {min(loto_taus.values()):.3f} – {max(loto_taus.values()):.3f}")
            for t, v in sorted(loto_taus.items(), key=lambda x: x[1]):
                print(f"    drop {t:25s}: τ = {v:.3f}")

    # ── WMF-AM subgroup: ST vs NST ──────────────────────────────────────
    print("\n── Subgroup Analysis (ST vs NST) ──")
    ST_TASKS = {"multi_step_calc", "entity_tracking", "sequential_search"}
    NST_TASKS = {"uncertain_lookup", "multi_source_conflict", "conversation_recall",
                 "source_attribution", "shopping_assistant", "schedule_coordination",
                 "data_pipeline"}
    sub_models = [m for m in all_models if m in agent_task_data and wmf.get(m) is not None]
    if sub_models:
        for label, task_set in [("ST (3 tasks)", ST_TASKS), ("NST (7 tasks)", NST_TASKS)]:
            agent_sub = {}
            for m in sub_models:
                tasks = agent_task_data[m]
                sub = {t: v for t, v in tasks.items() if t in task_set}
                if sub:
                    agent_sub[m] = sum(sub.values()) / len(sub)
            ms = [m for m in sub_models if m in agent_sub]
            ww = np.array([wmf[m] for m in ms])
            aa = np.array([agent_sub[m] for m in ms])
            t, p = kendalltau(ww, aa)
            key = f"wmf_vs_agent_{label[:3].strip().lower()}"
            stats[key] = {"tau": round(t, 3), "p": round(p, 4), "n": len(ms)}
            print(f"  τ(WMF-AM, {label}) = {t:.3f}, p = {p:.4f}, N = {len(ms)}")

    # ── Save ────────────────────────────────────────────────────────────
    output = {
        "generated": datetime.now().isoformat(),
        "n_models": len(all_models),
        "models": all_models,
        "per_model": per_model,
        "statistics": stats,
        "ksweep_fits": {m: kcrit_data[m] for m in sorted(kcrit_data.keys())},
        "data_sources": {
            "wmf_am": "cef_wmf_multiseed.json + expansion + nexp + api_held_out_final",
            "agent": "cef_agent_validation_all.json + api_held_out",
            "baseline": "baseline_api.json + baseline_ollama.json",
            "ksweep": "extended_k_20260413T180649.json + extended_k_20260413T030646.json",
            "loadshift": "load_shift_ollama_all.json + load_shift_ollama_full.json + load_shift_api_full.json",
            "nonarith": "wmf_am_nonarith_*.json (all files)",
            "yoked": "wmf_am_yoked_control_*.json (all files)",
            "template": "wmf_am_template_harmonization_*.json (all files)",
            "k1_control": "wmf_am_control_*.json (all files)",
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Saved to {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    # ── Paper claim verification ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PAPER CLAIM VERIFICATION")
    print("=" * 60)
    claims = [
        ("τ=0.595, N=28", stats.get("wmf_am_vs_agent", {})),
        ("τ(MMLU)=0.688", stats.get("mmlu_vs_agent", {})),
        ("τ(GSM8K)=0.603", stats.get("gsm8k_vs_agent", {})),
        ("partial τ|MMLU=0.284", stats.get("partial_tau_wmf_agent_given_mmlu_full", {})),
        ("partial τ|OC=0.411", stats.get("partial_tau_wmf_agent_given_oc", {})),
        ("yoked τ=-0.031", stats.get("yoked_vs_agent", {})),
        ("template τ(bare,chat)=0.524", stats.get("template_bare_vs_chat", {})),
        ("K_crit vs Agent τ=0.216", stats.get("kcrit_vs_agent", {})),
        ("K_crit vs Agent excl R1 τ=0.195", stats.get("kcrit_vs_agent_no_r1", {})),
        ("load-shift Δ=-0.30", stats.get("loadshift", {})),
    ]
    for label, s in claims:
        if s:
            actual = s.get("tau", s.get("mean_delta", "?"))
            n = s.get("n", s.get("n_models", "?"))
            print(f"  {label:40s} → actual: {actual}, N={n}")
        else:
            print(f"  {label:40s} → NOT COMPUTED (missing data)")


if __name__ == "__main__":
    import os
    os.chdir(str(DATA_DIR))
    main()
