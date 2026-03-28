"""
MCC-CE-v2 Runner — Uses harder question bank to eliminate floor effect.

The original MCC-CE produced CE=0 for all 7 models because questions were
too easy (>90% initial accuracy → nothing to flag). This runner uses the
v2 question bank (mcc_ce_v2_questions.py) which targets ~40% error rate
via numerical traps, counter-intuitive facts, and multi-step reasoning.

Usage:
    python mcc_ce_v2_runner.py --model ollama:qwen2.5:7b --n-problems 30
    python mcc_ce_v2_runner.py --model all --n-problems 30
    python mcc_ce_v2_runner.py --model all --n-problems 30 --trap-only
"""

import argparse
import json
import time
from pathlib import Path

from config import MODELS, RESULTS_DIR, call_model
from metacognitive_calibration import run_mcc_ce, compute_control_efficacy
from mcc_ce_v2_questions import get_balanced_set, get_trap_analysis_set


def run_mcc_ce_v2(
    model_name: str,
    n_problems: int = 30,
    trap_only: bool = False,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """
    Run MCC-CE with v2 harder questions.

    Args:
        model_name: Model key from config.
        n_problems: Number of questions (default 30 for v2, was 15 for v1).
        trap_only: If True, use only trap questions (highest diagnostic value).
        seed: Random seed for question selection.

    Returns:
        (raw_results, metrics_dict)
    """
    if trap_only:
        problems = get_trap_analysis_set(n=n_problems, seed=seed)
    else:
        problems = get_balanced_set(n=n_problems, seed=seed)

    # Reuse existing run_mcc_ce with the new problems
    results = run_mcc_ce(model_name, problems, batch_size=n_problems)

    # Tag as v2
    for r in results:
        r["sub_dim"] = "MCC-CE-v2"
        r["question_bank"] = "v2_trap" if trap_only else "v2_balanced"

    # Add question metadata to results
    for i, r in enumerate(results):
        if i < len(problems):
            r["difficulty"] = problems[i].get("difficulty", "unknown")
            r["category"] = problems[i].get("category", "unknown")
            r["common_wrong"] = problems[i].get("common_wrong")

    metrics = compute_control_efficacy(results)
    metrics["version"] = "v2"
    metrics["n_problems"] = n_problems
    metrics["initial_error_rate"] = round(
        sum(r["was_wrong"] for r in results) / max(len(results), 1), 4
    )

    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="MCC-CE-v2 with harder questions")
    parser.add_argument("--model", required=True,
                        help="Model name from config, or 'all'")
    parser.add_argument("--n-problems", type=int, default=30,
                        help="Number of problems (default: 30)")
    parser.add_argument("--trap-only", action="store_true",
                        help="Use only trap questions (most diagnostic)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MCC-CE-v2: {model_name}")
        print(f"{'='*60}")

        results, metrics = run_mcc_ce_v2(
            model_name, args.n_problems, args.trap_only, args.seed
        )

        print(f"  Initial error rate: {metrics['initial_error_rate']:.1%}")
        print(f"  Flagging rate:      {metrics['flagging_rate']:.3f}")
        print(f"  Correction efficacy:{metrics['correction_efficacy']:.3f}")
        print(f"  False alarm rate:   {metrics['false_alarm_rate']:.3f}")
        print(f"  MCC-CE-v2 score:    {metrics['mcc_ce_score']:.3f}")

        all_results[model_name] = {
            "results": results,
            "metrics": metrics,
        }

    # Save all results
    suffix = "_trap" if args.trap_only else ""
    out_file = RESULTS_DIR / f"mcc_ce_v2{suffix}_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_file}")

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n{'Model':<25} {'ErrRate':>8} {'Flag':>8} {'Corr':>8} {'CE-v2':>8}")
        print("-" * 60)
        for model_name, data in sorted(all_results.items(),
                                        key=lambda x: -x[1]["metrics"]["mcc_ce_score"]):
            m = data["metrics"]
            print(f"{model_name:<25} {m['initial_error_rate']:>7.1%} "
                  f"{m['flagging_rate']:>7.3f} {m['correction_efficacy']:>7.3f} "
                  f"{m['mcc_ce_score']:>7.3f}")


if __name__ == "__main__":
    main()
