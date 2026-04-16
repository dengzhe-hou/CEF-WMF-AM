#!/usr/bin/env python3
"""
Block 1: Held-Out API Model Validation.

Runs the full WMF-AM + Agent Battery protocol on 7 API models
(5 new families + 2 LRM) for out-of-sample validation.

Reuses run_wmf_am() and run_agent_battery() from oos_validation.py.

Usage:
    # Run all 7 API models:
    python api_held_out_validation.py

    # Run specific models:
    python api_held_out_validation.py --models openrouter:gpt-4o-mini openrouter:deepseek-v3

    # WMF-AM only (skip agent battery):
    python api_held_out_validation.py --wmf-only

    # Agent battery only (if WMF-AM already done):
    python api_held_out_validation.py --agent-only
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from oos_validation import (
    run_wmf_am, run_agent_battery, compute_kendall_tau, STUDY_RESULTS
)
from config import RESULTS_DIR, PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"

# ── Held-out API models ──────────────────────────────────────────────────────

API_MODELS = [
    # Standard models
    "openrouter:gpt-4o",
    "openrouter:gpt-4o-mini",
    "openrouter:claude-sonnet-4",
    "openrouter:deepseek-v3",
    "google:gemini-2.5-flash",
    # Long Reasoning Models (LRM)
    "openrouter:o3-mini",
    "openrouter:deepseek-r1",
]

# llama3.1:70b already has OOS results — include in analysis
EXISTING_OOS = {
    "ollama:llama3.1:70b": {"wmf_am": 0.500, "agent": 0.70},  # from oos_validation results
}


def load_existing_results(results_dir: Path) -> dict:
    """Load any previously completed API results to allow resuming."""
    existing = {}
    for f in sorted(results_dir.glob("api_held_out_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for r in data.get("results", []):
            model = r["model"]
            if "wmf_am_score" in r and "agent_score" in r:
                existing[model] = r
    return existing


def run_single_model(model_name: str, wmf_only: bool = False,
                     agent_only: bool = False) -> dict:
    """Run full protocol on one model."""
    result = {"model": model_name, "timestamp": datetime.now().isoformat()}

    if not agent_only:
        print(f"\n{'='*60}")
        print(f"WMF-AM: {model_name}")
        print(f"{'='*60}")
        t0 = time.time()
        wmf_result = run_wmf_am(model_name)
        wmf_elapsed = time.time() - t0
        result.update(wmf_result)
        result["wmf_elapsed_s"] = round(wmf_elapsed, 1)
        print(f"  WMF-AM score: {wmf_result['wmf_am_score']:.4f} ({wmf_elapsed:.0f}s)")

    if not wmf_only:
        print(f"\n{'='*60}")
        print(f"Agent Battery: {model_name}")
        print(f"{'='*60}")
        t0 = time.time()
        agent_result = run_agent_battery(model_name)
        agent_elapsed = time.time() - t0
        result.update(agent_result)
        result["agent_elapsed_s"] = round(agent_elapsed, 1)
        print(f"  Agent score: {agent_result['agent_score']:.3f} ({agent_elapsed:.0f}s)")

    return result


def compute_combined_tau(api_results: list[dict]) -> dict:
    """Compute τ at N=20+API and for held-out subset only."""
    import numpy as np

    # N=20 study set
    study_wmf = [STUDY_RESULTS[m]["wmf_am"] for m in STUDY_RESULTS]
    study_agent = [STUDY_RESULTS[m]["agent"] for m in STUDY_RESULTS]
    tau_20, p_20 = compute_kendall_tau(study_wmf, study_agent)

    # Add existing OOS
    all_wmf = study_wmf.copy()
    all_agent = study_agent.copy()
    held_out_wmf = []
    held_out_agent = []

    for m, scores in EXISTING_OOS.items():
        all_wmf.append(scores["wmf_am"])
        all_agent.append(scores["agent"])
        held_out_wmf.append(scores["wmf_am"])
        held_out_agent.append(scores["agent"])

    # Add new API results
    for r in api_results:
        if "wmf_am_score" in r and "agent_score" in r:
            all_wmf.append(r["wmf_am_score"])
            all_agent.append(r["agent_score"])
            held_out_wmf.append(r["wmf_am_score"])
            held_out_agent.append(r["agent_score"])

    n_total = len(all_wmf)
    n_held_out = len(held_out_wmf)

    tau_all, p_all = compute_kendall_tau(all_wmf, all_agent)

    # Held-out subset τ (if enough data)
    if n_held_out >= 5:
        tau_held, p_held = compute_kendall_tau(held_out_wmf, held_out_agent)
    else:
        tau_held, p_held = float("nan"), float("nan")

    print(f"\n{'='*60}")
    print(f"COMBINED ANALYSIS")
    print(f"{'='*60}")
    print(f"  τ(WMF-AM, ABS) N=20 study:       {tau_20:.3f} (p={p_20:.4f})")
    print(f"  τ(WMF-AM, ABS) N={n_total} combined:  {tau_all:.3f} (p={p_all:.4f})")
    if n_held_out >= 5:
        print(f"  τ(WMF-AM, ABS) N={n_held_out} held-out: {tau_held:.3f} (p={p_held:.4f})")
    print()

    # Decision Gate G1
    print(f"DECISION GATE G1:")
    if tau_all >= 0.3 and p_all < 0.05:
        print(f"  ✅ PASS: τ={tau_all:.3f} ≥ 0.3 and p={p_all:.4f} < 0.05")
    else:
        print(f"  ❌ FAIL: τ={tau_all:.3f}, p={p_all:.4f}")

    return {
        "tau_n20": round(tau_20, 4), "p_n20": round(p_20, 4),
        "tau_combined": round(tau_all, 4), "p_combined": round(p_all, 4),
        "n_combined": n_total,
        "tau_held_out": round(tau_held, 4) if not np.isnan(tau_held) else None,
        "p_held_out": round(p_held, 4) if not np.isnan(p_held) else None,
        "n_held_out": n_held_out,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="API Held-Out Validation")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to run (default: all API_MODELS)")
    parser.add_argument("--wmf-only", action="store_true",
                        help="Run WMF-AM only, skip agent battery")
    parser.add_argument("--agent-only", action="store_true",
                        help="Run agent battery only, skip WMF-AM")
    parser.add_argument("--resume", action="store_true",
                        help="Skip models that already have results")
    args = parser.parse_args()

    models = args.models or API_MODELS

    print("=" * 60)
    print("Block 1: API Held-Out Validation")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Mode: {'WMF only' if args.wmf_only else 'Agent only' if args.agent_only else 'Full'}")

    # Check for existing results
    existing = load_existing_results(RESULTS_DIR) if args.resume else {}
    if existing:
        print(f"Found existing results for: {list(existing.keys())}")

    results = []
    for model in models:
        if args.resume and model in existing:
            print(f"\n  Skipping {model} (already done)")
            results.append(existing[model])
            continue

        try:
            result = run_single_model(model, args.wmf_only, args.agent_only)
            results.append(result)

            # Save incrementally after each model
            out_path = RESULTS_DIR / f"api_held_out_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
            with open(out_path, "w") as f:
                json.dump({"results": results}, f, indent=2)
            print(f"  Saved to {out_path}")

        except Exception as e:
            print(f"\n  ERROR on {model}: {e}")
            results.append({"model": model, "error": str(e)})

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'WMF-AM':>8} {'Agent':>8} {'Type':>10}")
    print("-" * 65)

    for r in results:
        wmf = r.get("wmf_am_score", "N/A")
        agent = r.get("agent_score", "N/A")
        model_type = "LRM" if any(x in r["model"] for x in ["o3", "deepseek-r1"]) else "Standard"
        wmf_str = f"{wmf:.4f}" if isinstance(wmf, float) else wmf
        agent_str = f"{agent:.3f}" if isinstance(agent, float) else agent
        print(f"  {r['model']:<33} {wmf_str:>8} {agent_str:>8} {model_type:>10}")

    # Combined τ analysis
    valid_results = [r for r in results
                     if "wmf_am_score" in r and "agent_score" in r]
    if valid_results:
        tau_results = compute_combined_tau(valid_results)

        # Save final results
        final_path = RESULTS_DIR / f"api_held_out_final_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        with open(final_path, "w") as f:
            json.dump({
                "results": results,
                "tau_analysis": tau_results,
                "study_results": {k: v for k, v in STUDY_RESULTS.items()},
                "existing_oos": EXISTING_OOS,
            }, f, indent=2)
        print(f"\nFinal results saved to {final_path}")


if __name__ == "__main__":
    main()
