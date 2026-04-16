#!/usr/bin/env python3
"""
Block 3: Closed-Loop Drift Test for WMF-AM.

Converts WMF-AM from open-loop (K operations in one prompt, ask final state)
to closed-loop iterative (model reports state after each step, that reported
state is fed as input to the next step). Measures divergence rate from
ground truth.

Key insight: agent systems are closed-loop (each step's output affects next
step's input). This probe tests whether models accumulate errors when
self-conditioning, and whether drift rate predicts agent performance.
"""

import json
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path

import importlib.util
_config_path = Path(__file__).parent / "config.py"
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
call_model = _config.call_model
MODELS = _config.MODELS
PROJECT_ROOT = _config.PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"

# ── Surface form templates ──────────────────────────────────────────────────

TEMPLATES = {
    "points_scoring": {
        "entity_word": "points",
        "init": "{entity} starts with {value} points.",
        "gain": "{entity} gains {amount} points.",
        "lose": "{entity} loses {amount} points.",
        "query": "What is {entity}'s current score?",
        "state_update": "{entity}'s current score is {value}.",
    },
    "warehouse": {
        "entity_word": "items",
        "init": "The warehouse has {value} items for {entity}.",
        "gain": "{amount} items are added for {entity}.",
        "lose": "{amount} items are removed for {entity}.",
        "query": "How many items does {entity} have?",
        "state_update": "{entity} currently has {value} items.",
    },
    "bank_account": {
        "entity_word": "dollars",
        "init": "{entity}'s account balance is ${value}.",
        "gain": "${amount} is deposited into {entity}'s account.",
        "lose": "${amount} is withdrawn from {entity}'s account.",
        "query": "What is {entity}'s current balance?",
        "state_update": "{entity}'s current balance is ${value}.",
    },
}

ENTITIES = ["Alice", "Bob", "Charlie", "Diana", "Eve"]


def extract_number(text: str) -> int | None:
    """Extract first integer from model response."""
    # Remove thinking tags (for reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Try to find a number
    nums = re.findall(r"-?\d+", text.strip())
    if nums:
        return int(nums[-1])  # Take last number (usually the final answer)
    return None


def generate_operations(rng: random.Random, k: int) -> list[tuple[str, int]]:
    """Generate K operations as (type, amount) pairs. Types: 'gain' or 'lose'."""
    ops = []
    for _ in range(k):
        op_type = rng.choice(["gain", "lose"])
        amt = rng.randint(1, 10)
        ops.append((op_type, amt))
    return ops


def run_closed_loop_trial(
    model_name: str,
    entity: str,
    initial_value: int,
    operations: list[tuple[str, int]],
    template_name: str = "points_scoring",
) -> dict:
    """
    Run one closed-loop trial: feed model's reported state back as input.

    Returns per-step trajectory: ground_truth, model_state, absolute_error.
    """
    tmpl = TEMPLATES[template_name]
    ground_truth = initial_value
    model_state = initial_value  # Start aligned
    steps = []

    for step_idx, (op_type, amount) in enumerate(operations):
        # Update ground truth
        if op_type == "gain":
            ground_truth += amount
        else:
            ground_truth -= amount

        # Build prompt using MODEL'S reported state (closed-loop)
        state_line = tmpl["state_update"].format(entity=entity, value=model_state)
        op_line = tmpl[op_type].format(entity=entity, amount=amount)
        query_line = tmpl["query"].format(entity=entity)

        prompt = f"{state_line} {op_line} {query_line} Respond with ONLY the number."

        try:
            response = call_model(model_name, prompt)
            predicted = extract_number(response)
            if predicted is None:
                predicted = model_state  # If can't parse, assume no change
        except Exception as e:
            predicted = model_state
            response = f"ERROR: {e}"

        abs_error = abs(predicted - ground_truth)

        steps.append({
            "step": step_idx + 1,
            "op_type": op_type,
            "op_amount": amount,
            "ground_truth": ground_truth,
            "model_input_state": model_state,
            "model_output": predicted,
            "abs_error": abs_error,
            "raw_response": str(response)[:200],
        })

        # Closed-loop: feed model's output as next step's input
        model_state = predicted

    return {
        "initial_value": initial_value,
        "entity": entity,
        "template": template_name,
        "n_steps": len(operations),
        "steps": steps,
        "final_ground_truth": ground_truth,
        "final_model_state": model_state,
        "final_abs_error": abs(model_state - ground_truth),
    }


def compute_drift_metrics(trial: dict) -> dict:
    """Compute drift metrics from a single closed-loop trial."""
    steps = trial["steps"]
    errors = [s["abs_error"] for s in steps]
    n = len(errors)

    if n == 0:
        return {"drift_rate": 0, "total_divergence": 0, "max_error": 0}

    # Drift rate: slope of |error| vs step (linear regression)
    from scipy import stats
    xs = list(range(1, n + 1))
    if len(set(errors)) > 1:
        slope, _, _, _, _ = stats.linregress(xs, errors)
    else:
        slope = 0.0

    return {
        "drift_rate": slope,
        "total_divergence": sum(errors),
        "max_error": max(errors),
        "final_error": errors[-1],
        "mean_error": sum(errors) / n,
        "first_error_step": next((i + 1 for i, e in enumerate(errors) if e > 0), n + 1),
        "n_correct_steps": sum(1 for e in errors if e == 0),
        "pct_correct": sum(1 for e in errors if e == 0) / n,
    }


def run_model(model_name: str, k_values: list[int], n_seeds: int = 4,
              templates: list[str] | None = None) -> dict:
    """Run full closed-loop battery for one model."""
    if templates is None:
        templates = list(TEMPLATES.keys())

    base_seeds = [2026, 100, 200, 300][:n_seeds]
    all_trials = []

    for seed in base_seeds:
        rng = random.Random(seed)
        for k in k_values:
            for tmpl_name in templates:
                entity = rng.choice(ENTITIES)
                initial = rng.randint(5, 20)
                ops = generate_operations(rng, k)

                trial = run_closed_loop_trial(
                    model_name, entity, initial, ops, tmpl_name
                )
                trial["seed"] = seed
                trial["k"] = k

                metrics = compute_drift_metrics(trial)
                trial.update(metrics)
                all_trials.append(trial)

                print(f"  {model_name} | K={k} seed={seed} {tmpl_name}: "
                      f"drift={metrics['drift_rate']:.3f} "
                      f"final_err={metrics['final_error']} "
                      f"pct_correct={metrics['pct_correct']:.2f}")

    # Aggregate per model
    drift_rates = [t["drift_rate"] for t in all_trials]
    pct_corrects = [t["pct_correct"] for t in all_trials]

    return {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "k_values": k_values,
        "n_seeds": n_seeds,
        "n_trials": len(all_trials),
        "mean_drift_rate": sum(drift_rates) / len(drift_rates),
        "mean_pct_correct": sum(pct_corrects) / len(pct_corrects),
        "trials": all_trials,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WMF-AM Closed-Loop Drift Test")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names (default: all study + API models)")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 15, 20],
                        help="K values for closed-loop depth")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--templates", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.models is None:
        # Default: test with a small set first
        args.models = ["openrouter:gpt-4o-mini"]

    print("=" * 70)
    print("WMF-AM Closed-Loop Drift Test")
    print("=" * 70)
    print(f"Models: {args.models}")
    print(f"K values: {args.k_values}")
    print(f"Seeds: {args.seeds}")

    results = []
    for model in args.models:
        print(f"\n── {model} ──")
        result = run_model(model, args.k_values, args.seeds, args.templates)
        results.append(result)
        print(f"  → mean_drift_rate={result['mean_drift_rate']:.4f}, "
              f"mean_pct_correct={result['mean_pct_correct']:.3f}")

    # Save results
    out_path = args.output or str(
        DATA_DIR / f"closed_loop_drift_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
