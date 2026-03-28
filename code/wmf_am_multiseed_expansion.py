"""
CEF Multi-Seed Replication for Expansion 8 Models

Original 7 models have 4-seed WMF-AM data. This script runs 3 additional seeds
(100, 200, 300) for the expansion 8 models so all 15 have 4-seed estimates.

Usage:
    python wmf_am_multiseed_expansion.py --models all-8
    python wmf_am_multiseed_expansion.py --models ollama:phi3:14b ollama:gemma2:9b
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS, RESULTS_DIR, call_model
from wm_fidelity import build_wmf_am_problem

EXPANSION_8 = [
    "ollama:phi3:14b", "ollama:gemma2:9b", "ollama:qwen2.5:3b",
    "ollama:llama3.2:3b", "ollama:deepseek-r1:7b", "ollama:mixtral:8x7b",
    "ollama:command-r:35b", "ollama:yi:34b",
]

# The N-expansion already used seed cycling (42, 137, 256, 999) within its 15 probes per depth.
# For multi-seed replication, we use the same 4 seeds as the original 7:
SEEDS = [2026, 100, 200, 300]
DEPTHS = [3, 5, 7]
PROBES_PER_DEPTH = 5  # per seed; 4 seeds × 5 probes = 20 per depth


def run_seed(model_name: str, seed: int) -> dict:
    """Run WMF-AM for one model at one seed."""
    details = []
    by_depth = {k: [] for k in DEPTHS}

    for k in DEPTHS:
        for i in range(PROBES_PER_DEPTH):
            random.seed(seed + k * 1000 + i)
            initial_state, ops, correct, query_entity = build_wmf_am_problem(k)

            state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())
            ops_str = "\n".join(f"  {j+1}. {op}" for j, op in enumerate(ops))

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
            except Exception as exc:
                print(f"    ERROR: {exc}")
                response = ""

            nums = re.findall(r"\d+", response)
            predicted = int(nums[0]) if nums else -1
            accurate = int(predicted == correct)

            by_depth[k].append(accurate)
            details.append({
                "k": k, "seed": seed, "probe_idx": i,
                "correct": correct, "predicted": predicted, "accurate": accurate,
            })

    all_acc = [d["accurate"] for d in details]
    mean_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
    depth_means = {str(k): round(sum(v)/len(v), 4) if v else 0.0 for k, v in by_depth.items()}

    return {
        "seed": seed,
        "mean_accuracy": round(mean_acc, 4),
        "per_depth": depth_means,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-seed WMF-AM for expansion 8")
    parser.add_argument("--models", nargs="+", default=["all-8"])
    args = parser.parse_args()

    if "all-8" in args.models:
        models = EXPANSION_8
    else:
        models = args.models

    for m in models:
        if m not in MODELS:
            print(f"ERROR: {m} not in MODELS registry")
            sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"cef_wmf_multiseed_expansion8_{ts}.json"

    print(f"WMF-AM Multi-Seed Replication (Expansion 8)")
    print(f"  Models: {len(models)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Depths: {DEPTHS}")
    print(f"  Probes per seed per depth: {PROBES_PER_DEPTH}")
    print(f"  Total calls per model: {len(SEEDS) * len(DEPTHS) * PROBES_PER_DEPTH}")
    print()

    all_results = []
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        model_seeds = []
        accs = []
        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            result = run_seed(model, seed)
            model_seeds.append(result)
            accs.append(result["mean_accuracy"])
            print(f"    Mean: {result['mean_accuracy']:.4f}")
            print(f"    By depth: {result['per_depth']}")

        mean_across = sum(accs) / len(accs)
        std_across = (sum((a - mean_across)**2 for a in accs) / len(accs)) ** 0.5

        model_result = {
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds": model_seeds,
            "mean_accuracy": round(mean_across, 4),
            "std_accuracy": round(std_across, 4),
            "accs_per_seed": accs,
        }
        all_results.append(model_result)
        print(f"\n  Overall: mean={mean_across:.4f} ± {std_across:.4f}")

    output = {
        "timestamp": ts,
        "n_models": len(models),
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "template": "A_points",
        "per_model": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<28} {'Mean':>8} {'SD':>8} {'Seeds':>20}")
    print("-" * 60)
    for r in all_results:
        seeds_str = ", ".join(f"{a:.2f}" for a in r["accs_per_seed"])
        print(f"{r['model']:<28} {r['mean_accuracy']:>8.4f} {r['std_accuracy']:>8.4f} {seeds_str:>20}")


if __name__ == "__main__":
    main()
