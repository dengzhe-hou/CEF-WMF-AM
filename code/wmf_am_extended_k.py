#!/usr/bin/env python3
"""
Block 3: Extended K-Sweep for WMF-AM Phase Transition Analysis.

Runs WMF-AM at K={3,5,7,10,15,20,30,50} (+ K=75,100 for LRM models)
to map the full degradation curve per model.

Purpose: identify distinct failure regimes (sigmoid cliff vs gradual decay)
between standard models and LRMs.
"""

import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import call_model, RESULTS_DIR
from oos_validation import build_wmf_am_problem

# ── Config ────────────────────────────────────────────────────────────────────

K_VALUES_STANDARD = [3, 5, 7, 10, 15, 20, 30, 50]
K_VALUES_LRM = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100]

SEEDS = [2026, 100, 200, 300]
PROBES_PER_K_PER_SEED = 5

LRM_MODELS = {"openrouter:o3-mini", "openrouter:deepseek-r1"}


def run_extended_k(model_name: str, k_values: list[int] = None) -> dict:
    """Run WMF-AM at multiple K values for one model."""
    if k_values is None:
        k_values = K_VALUES_LRM if model_name in LRM_MODELS else K_VALUES_STANDARD

    print(f"\n[Extended K] {model_name} — K={k_values}")
    by_k = {k: [] for k in k_values}
    all_trials = []

    for k in k_values:
        for seed in SEEDS:
            for i in range(PROBES_PER_K_PER_SEED):
                initial_state, operations, correct, query_entity = \
                    build_wmf_am_problem(k, seed, i)

                state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())
                ops_str = "\n".join(f"  {j+1}. {op}" for j, op in enumerate(operations))
                prompt = (
                    "You will track a sequence of point updates. "
                    "You cannot refer back to the initial state after reading it once.\n\n"
                    f"Initial state:\n{state_str}\n\n"
                    f"Operations (apply in order):\n{ops_str}\n\n"
                    f"After all operations, how many points does {query_entity} have?\n\n"
                    "Respond with ONLY the final number."
                )

                try:
                    response = call_model(model_name, prompt)
                    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
                    nums = re.findall(r"-?\d+", clean)
                    predicted = int(nums[-1]) if nums else -1
                    accurate = int(predicted == correct)
                except Exception as e:
                    print(f"  ERROR K={k} seed={seed} i={i}: {e}")
                    predicted = -1
                    accurate = 0

                by_k[k].append(accurate)
                all_trials.append({
                    "k": k, "seed": seed, "probe_idx": i,
                    "correct": correct, "predicted": predicted, "accurate": accurate,
                })
                print("." if accurate else "x", end="", flush=True)
        # Print per-K summary
        k_acc = sum(by_k[k]) / len(by_k[k]) if by_k[k] else 0
        print(f"  K={k}: {k_acc:.3f} ({sum(by_k[k])}/{len(by_k[k])})")

    # Compute per-K accuracy
    per_k = {}
    for k in k_values:
        trials = by_k[k]
        per_k[k] = round(sum(trials) / len(trials), 4) if trials else 0.0

    overall = sum(t["accurate"] for t in all_trials) / len(all_trials) if all_trials else 0
    print(f"  Overall: {overall:.4f}")

    return {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "k_values": k_values,
        "per_k_accuracy": per_k,
        "overall_accuracy": round(overall, 4),
        "n_trials": len(all_trials),
        "n_seeds": len(SEEDS),
        "probes_per_k_per_seed": PROBES_PER_K_PER_SEED,
        "trials": all_trials,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WMF-AM Extended K-Sweep")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=None,
                        help="Override K values (default: auto based on model type)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("WMF-AM Extended K-Sweep (Phase Transition Analysis)")
    print("=" * 60)

    results = []
    for model in args.models:
        t0 = time.time()
        result = run_extended_k(model, args.k_values)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        results.append(result)

        # Print curve
        print(f"\n  Degradation curve for {model}:")
        for k, acc in sorted(result["per_k_accuracy"].items(), key=lambda x: int(x[0])):
            bar = "█" * int(acc * 30)
            print(f"    K={int(k):>3}: {acc:.3f} {bar}")

    # Save
    out_path = args.output or str(
        RESULTS_DIR / f"extended_k_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
