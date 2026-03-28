"""
CEF Template Harmonization Sensitivity Check

Tests whether WMF-AM results are sensitive to prompt template format.
Reruns WMF-AM with a standardized instruction wrapper across all 15 models.

Three template conditions:
  1. "bare"    — current format (no system prompt, direct instruction)
  2. "chat"    — wrapped in explicit system + user role markers
  3. "cot"     — adds "Think step by step" before "Respond with ONLY the final number"

If WMF-AM rankings are stable across templates (Kendall's tau ≥ 0.8),
the reviewer concern about template confounding is addressed.

Usage:
    python wmf_am_template_harmonization.py --models all-15 --seeds 42 137
    python wmf_am_template_harmonization.py --models ollama:qwen2.5:7b --templates bare chat cot
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

# All 15 models
ORIGINAL_7 = [
    "ollama:qwen2.5:7b", "ollama:qwen2.5:14b", "ollama:qwen2.5:32b",
    "ollama:llama3.1:8b", "ollama:gemma2:27b", "ollama:deepseek-r1:14b",
    "ollama:mistral:7b",
]
EXPANSION_8 = [
    "ollama:phi3:14b", "ollama:gemma2:9b", "ollama:qwen2.5:3b",
    "ollama:llama3.2:3b", "ollama:deepseek-r1:7b", "ollama:mixtral:8x7b",
    "ollama:command-r:35b", "ollama:yi:34b",
]
ALL_15 = ORIGINAL_7 + EXPANSION_8

DEPTHS = [3, 5, 7]
PROBES_PER_DEPTH = 15
TEMPLATES = ["bare", "chat", "cot"]


def build_prompt(template: str, initial_state: dict, ops: list[str], query_entity: str) -> tuple[str, str]:
    """Build WMF-AM prompt in the specified template format.
    Returns (system_prompt, user_prompt)."""
    state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())
    ops_str = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(ops))

    if template == "bare":
        system = ""
        user = (
            "You will track a sequence of point updates. "
            "You cannot refer back to the initial state after reading it once.\n\n"
            f"Initial state:\n{state_str}\n\n"
            f"Operations (apply in order):\n{ops_str}\n\n"
            f"After all operations, how many points does {query_entity} have?\n\n"
            "Respond with ONLY the final number."
        )
    elif template == "chat":
        system = (
            "You are a precise arithmetic assistant. "
            "You track numerical state changes and report final values. "
            "Always respond with only the requested number, no explanation."
        )
        user = (
            f"Track the following point updates carefully.\n\n"
            f"Initial state:\n{state_str}\n\n"
            f"Operations (apply in order):\n{ops_str}\n\n"
            f"Question: After all operations, how many points does {query_entity} have?\n\n"
            "Answer with ONLY the number."
        )
    elif template == "cot":
        system = ""
        user = (
            "You will track a sequence of point updates. "
            "You cannot refer back to the initial state after reading it once.\n\n"
            f"Initial state:\n{state_str}\n\n"
            f"Operations (apply in order):\n{ops_str}\n\n"
            f"After all operations, how many points does {query_entity} have?\n\n"
            "Think step by step, then give your final answer as a single number on the last line."
        )
    else:
        raise ValueError(f"Unknown template: {template}")

    return system, user


def run_template_condition(model_name: str, template: str, seed: int) -> dict:
    """Run WMF-AM probes for one model, one template, one seed."""
    details = []
    by_depth = {k: [] for k in DEPTHS}

    for k in DEPTHS:
        for i in range(PROBES_PER_DEPTH):
            random.seed(seed + k * 1000 + i)
            initial_state, ops, correct, query_entity = build_wmf_am_problem(k)
            system, user = build_prompt(template, initial_state, ops, query_entity)

            try:
                if system:
                    response = call_model(model_name, user, system=system)
                else:
                    response = call_model(model_name, user)
            except Exception as exc:
                print(f"    ERROR: {exc}")
                response = ""

            # Extract number — for cot, take last number in response
            if template == "cot":
                nums = re.findall(r"\d+", response)
                predicted = int(nums[-1]) if nums else -1
            else:
                nums = re.findall(r"\d+", response)
                predicted = int(nums[0]) if nums else -1

            accurate = int(predicted == correct)
            by_depth[k].append(accurate)
            details.append({
                "k": k, "seed": seed, "probe_idx": i,
                "correct": correct, "predicted": predicted, "accurate": accurate,
            })

            if (i + 1) % 5 == 0:
                acc_so_far = sum(d["accurate"] for d in details) / len(details)
                print(f"    [{template}] K={k} {i+1}/{PROBES_PER_DEPTH} running_acc={acc_so_far:.3f}")

    all_acc = [d["accurate"] for d in details]
    mean_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
    depth_means = {str(k): round(sum(v)/len(v), 4) if v else 0.0 for k, v in by_depth.items()}

    return {
        "model": model_name,
        "template": template,
        "seed": seed,
        "mean": round(mean_acc, 4),
        "by_depth": depth_means,
        "n_probes": len(details),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="WMF-AM Template Harmonization")
    parser.add_argument("--models", nargs="+", default=["all-15"])
    parser.add_argument("--templates", nargs="+", default=TEMPLATES, choices=TEMPLATES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if "all-15" in args.models:
        models = ALL_15
    elif "all-7" in args.models:
        models = ORIGINAL_7
    elif "all-8" in args.models:
        models = EXPANSION_8
    else:
        models = args.models

    for m in models:
        if m not in MODELS:
            print(f"ERROR: {m} not in MODELS registry")
            sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"wmf_am_template_harmonization_{ts}.json"

    print(f"WMF-AM Template Harmonization")
    print(f"  Models: {len(models)} models")
    print(f"  Templates: {args.templates}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Probes per depth: {PROBES_PER_DEPTH}")
    print(f"  Total calls: {len(models) * len(args.templates) * len(args.seeds) * len(DEPTHS) * PROBES_PER_DEPTH}")
    print()

    all_results = []
    for model in models:
        for template in args.templates:
            for seed in args.seeds:
                print(f"\n=== {model} | template={template} | seed={seed} ===")
                result = run_template_condition(model, template, seed)
                all_results.append(result)
                print(f"  Mean accuracy: {result['mean']:.4f}")
                print(f"  By depth: {result['by_depth']}")

    # Save results
    summary = []
    for r in all_results:
        summary.append({k: v for k, v in r.items() if k != "details"})

    output = {
        "timestamp": ts,
        "n_models": len(models),
        "templates": args.templates,
        "seeds": args.seeds,
        "results": summary,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Compute rank stability
    print("\n" + "=" * 70)
    print("Template Stability Analysis")
    print("=" * 70)

    from scipy.stats import kendalltau
    for s in args.seeds:
        print(f"\nSeed {s}:")
        template_ranks = {}
        for t in args.templates:
            scores = []
            for m in models:
                match = [r for r in summary if r["model"] == m and r["template"] == t and r["seed"] == s]
                if match:
                    scores.append((m, match[0]["mean"]))
            scores.sort(key=lambda x: -x[1])
            template_ranks[t] = [m for m, _ in scores]
            print(f"  {t}: {[(m.split(':')[-1], s) for m, s in scores[:5]]}")

        # Pairwise tau between templates
        for i, t1 in enumerate(args.templates):
            for t2 in args.templates[i+1:]:
                r1 = template_ranks[t1]
                r2 = template_ranks[t2]
                rank1 = [r1.index(m) for m in models]
                rank2 = [r2.index(m) for m in models]
                tau, p = kendalltau(rank1, rank2)
                print(f"  tau({t1}, {t2}) = {tau:.3f} (p={p:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
