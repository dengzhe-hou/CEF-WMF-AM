"""
WMF-AM Multi-Seed Reliability Check (CEF — NeurIPS 2026 supplement).

Goal: Establish per-model mean accuracy, standard deviation, and cross-seed
rank stability across 4 independent random seeds (addressing reviewer concern
about llama3.1:8b's seed sensitivity revealed in the template robustness check).

Design:
  - Template A only (original points format — matches main pilot surface form)
  - 4 seeds: 2026 (already in robustness check), 100, 200, 300
  - 7 models × 4 seeds × 15 probes (K=3/5/7, 5 each) = 420 calls
  - Reports per-model mean ± SD, cross-seed Kendall τ, per-model 95% CI

Output: JSON + console table with reliability statistics.
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, sem

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, call_model
from wm_fidelity import build_wmf_am_problem

SEEDS = [2026, 100, 200, 300]
DEPTHS = [3, 5, 7]
TRIALS_PER_DEPTH = 5


def format_template_A(initial_state: dict, operations: list, query_entity: str) -> str:
    state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())
    ops_str = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(operations))
    return (
        f"You will track a sequence of point updates.\n\n"
        f"Initial state:\n{state_str}\n\n"
        f"Operations (apply in order):\n{ops_str}\n\n"
        f"After all operations, how many points does {query_entity} have?\n\n"
        f"Respond with ONLY the final number."
    )


def run_one_seed(model_name: str, seed: int) -> dict:
    """Run Template A with one random seed. Returns mean accuracy and per-depth breakdown."""
    per_depth = {}
    all_correct = 0
    total = 0
    for depth in DEPTHS:
        correct_count = 0
        for trial in range(TRIALS_PER_DEPTH):
            random.seed(seed + depth * 100 + trial)
            initial_state, operations, correct_answer, query_entity = build_wmf_am_problem(depth)
            prompt = format_template_A(initial_state, operations, query_entity)
            try:
                response = call_model(model_name, prompt)
                nums = re.findall(r"\d+", response)
                predicted = int(nums[0]) if nums else -1
                is_correct = (predicted == correct_answer)
            except Exception as e:
                predicted = -1
                is_correct = False
                print(f"    ERR [seed={seed} K={depth} t={trial}]: {e}", flush=True)
            if is_correct:
                correct_count += 1
            total += 1
        per_depth[str(depth)] = correct_count / TRIALS_PER_DEPTH
        all_correct += correct_count
    return {
        "seed": seed,
        "mean_accuracy": all_correct / total,
        "per_depth": per_depth,
    }


def run_model(model_name: str) -> dict:
    seed_results = []
    for seed in SEEDS:
        r = run_one_seed(model_name, seed)
        seed_results.append(r)
        accs = r["per_depth"]
        print(f"    seed={seed}: mean={r['mean_accuracy']:.3f}  " +
              "  ".join(f"K{k}={accs[str(k)]:.2f}" for k in DEPTHS), flush=True)
    accs_per_seed = [r["mean_accuracy"] for r in seed_results]
    mean_acc = float(np.mean(accs_per_seed))
    std_acc = float(np.std(accs_per_seed, ddof=1)) if len(accs_per_seed) > 1 else 0.0
    se = float(sem(accs_per_seed)) if len(accs_per_seed) > 1 else 0.0
    ci95_half = 1.96 * se
    print(f"    → mean={mean_acc:.3f} ± {std_acc:.3f} (95% CI ±{ci95_half:.3f})", flush=True)
    return {
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "seeds": seed_results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "se_accuracy": se,
        "ci95_half": ci95_half,
        "accs_per_seed": accs_per_seed,
    }


DEFAULT_MODELS = [
    "ollama:qwen2.5:7b",
    "ollama:qwen2.5:14b",
    "ollama:qwen2.5:32b",
    "ollama:llama3.1:8b",
    "ollama:gemma2:27b",
    "ollama:deepseek-r1:14b",
    "ollama:mistral:7b",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", default=str(RESULTS_DIR / "cef_wmf_multiseed.json"))
    args = parser.parse_args()

    print(f"WMF-AM Multi-Seed Reliability — {len(args.models)} models × {len(SEEDS)} seeds", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Started: {datetime.utcnow().isoformat()}", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    for model in args.models:
        print(f"\nModel: {model}", flush=True)
        try:
            r = run_model(model)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    # ── Cross-seed rank stability ─────────────────────────────────────────────
    print("\n\n" + "=" * 70, flush=True)
    print("RELIABILITY SUMMARY", flush=True)
    print("=" * 70, flush=True)

    names = [r["model"].replace("ollama:", "") for r in all_results]
    means = [r["mean_accuracy"] for r in all_results]
    stds = [r["std_accuracy"] for r in all_results]
    ci95s = [r["ci95_half"] for r in all_results]

    print(f"\n{'Model':<22} {'Mean':>6} {'±SD':>6} {'95%CI':>8}  {'Tier'}")
    print("-" * 55)
    for name, mu, sd, ci in zip(names, means, stds, ci95s):
        tier = "HIGH" if mu >= 0.7 else ("MID" if mu >= 0.4 else "LOW")
        print(f"{name:<22} {mu:>6.3f} {sd:>6.3f} {ci:>8.3f}  {tier}")

    print(f"\nOverall spread: {max(means):.3f} - {min(means):.3f} = Δ{max(means)-min(means):.3f}")

    # Cross-seed tau: for each pair of seeds, compute rank correlation
    n_seeds = len(SEEDS)
    print(f"\nCross-seed Kendall τ (Template A, ranking stability):")
    all_tau = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            seed_i_accs = [r["seeds"][i]["mean_accuracy"] for r in all_results]
            seed_j_accs = [r["seeds"][j]["mean_accuracy"] for r in all_results]
            tau, p = kendalltau(seed_i_accs, seed_j_accs)
            all_tau.append(tau)
            print(f"  τ(seed{SEEDS[i]}, seed{SEEDS[j]}) = {tau:.3f}  p={p:.3f}")
    print(f"  Mean τ across all seed pairs: {np.mean(all_tau):.3f}")

    # Consistent high performers: always HIGH across all seeds
    print(f"\nConsistency across seeds:")
    for r in all_results:
        name = r["model"].replace("ollama:", "")
        seed_accs = r["accs_per_seed"]
        min_a, max_a = min(seed_accs), max(seed_accs)
        stable = "STABLE" if (max_a - min_a) <= 0.2 else "VARIABLE"
        print(f"  {name:<22}: {[f'{a:.2f}' for a in seed_accs]}  range={max_a-min_a:.3f}  [{stable}]")

    # Save
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_models": len(all_results),
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "template": "A_points",
        "per_model": all_results,
        "summary": {
            "means": dict(zip(names, means)),
            "stds": dict(zip(names, stds)),
            "spread_delta": max(means) - min(means),
        },
        "cross_seed_tau": {
            f"seed{SEEDS[i]}_vs_seed{SEEDS[j]}": float(
                kendalltau(
                    [r["seeds"][i]["mean_accuracy"] for r in all_results],
                    [r["seeds"][j]["mean_accuracy"] for r in all_results],
                )[0]
            )
            for i in range(n_seeds)
            for j in range(i + 1, n_seeds)
        },
        "mean_cross_seed_tau": float(np.mean(all_tau)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")
    print(f"Finished: {datetime.utcnow().isoformat()}")
