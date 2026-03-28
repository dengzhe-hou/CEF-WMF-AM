"""
WMF-AM Non-Arithmetic State Tracking Ablation

PURPOSE:
  Tests whether WMF-AM rankings hold when state updates are non-numeric
  (color/category transitions instead of arithmetic). If rankings are
  preserved, WMF-AM measures state tracking, not arithmetic ability.
  If rankings reverse, arithmetic parsing is the primary driver.

DESIGN:
  - Same K-depth structure as WMF-AM (K=3,5,7)
  - Same 3-entity setup
  - Operations are COLOR transitions instead of numeric: "Alice changes
    from red to blue", "Bob changes from green to yellow"
  - Model must track cumulative state (final color of each entity)
  - No arithmetic involved — pure sequential state tracking

  Domain variants:
  1. COLOR: entities change colors (red→blue→green→...)
  2. LOCATION: entities move between rooms (kitchen→garden→library→...)
  3. STATUS: entities change status (active→paused→reviewing→...)

Usage:
    python wmf_am_nonarithmetic.py --models ollama:qwen2.5:7b ollama:llama3.1:8b
    python wmf_am_nonarithmetic.py --models ollama:qwen2.5:7b --seed 42
"""

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

from config import MODELS, RESULTS_DIR, call_model

# ── State domains ────────────────────────────────────────────────────────────

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink",
          "white", "black", "gray", "brown", "cyan"]

LOCATIONS = ["kitchen", "garden", "library", "garage", "bedroom", "office",
             "basement", "attic", "patio", "workshop", "hallway", "studio"]

STATUSES = ["active", "paused", "reviewing", "completed", "pending",
            "archived", "drafting", "approved", "rejected", "waiting",
            "processing", "finalized"]

DOMAINS = {
    "color": {
        "values": COLORS,
        "verb": "changes color to",
        "question": "What color is {entity} now?",
        "attribute": "color",
    },
    "location": {
        "values": LOCATIONS,
        "verb": "moves to the",
        "question": "Where is {entity} now?",
        "attribute": "location",
    },
    "status": {
        "values": STATUSES,
        "verb": "changes status to",
        "question": "What is {entity}'s current status?",
        "attribute": "status",
    },
}

ENTITIES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank",
            "Grace", "Henry", "Iris", "James", "Kate", "Leo"]


def build_nonarith_problem(k_ops: int, domain_name: str, rng: random.Random):
    """
    Build a non-arithmetic state tracking problem.

    Returns: (prompt, correct_answer, metadata)
    """
    domain = DOMAINS[domain_name]
    values = domain["values"]

    # Pick 3 entities
    entities = rng.sample(ENTITIES, 3)

    # Assign initial states (distinct values)
    initial_vals = rng.sample(values, 3)
    state = {e: v for e, v in zip(entities, initial_vals)}

    # Generate K operations — each changes one entity's state
    operations = []
    for _ in range(k_ops):
        entity = rng.choice(entities)
        # Pick a new value different from current
        current = state[entity]
        candidates = [v for v in values if v != current]
        new_val = rng.choice(candidates)
        state[entity] = new_val
        operations.append(f"{entity} {domain['verb']} {new_val}.")

    # Pick query entity
    query_entity = rng.choice(entities)
    correct = state[query_entity]

    # Build prompt
    if domain_name == "color":
        init_str = ", ".join(f"{e} is {v}" for e, v in zip(entities, initial_vals))
    elif domain_name == "location":
        init_str = ", ".join(f"{e} is in the {v}" for e, v in zip(entities, initial_vals))
    else:
        init_str = ", ".join(f"{e}'s status is {v}" for e, v in zip(entities, initial_vals))

    ops_str = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(operations))
    question = domain["question"].format(entity=query_entity)

    prompt = f"""You will track a sequence of {domain['attribute']} changes. Pay careful attention to each update.

Initial state:
{init_str}

Updates (apply in order):
{ops_str}

{question}

Respond with ONLY the final {domain['attribute']} (one word)."""

    metadata = {
        "domain": domain_name,
        "k_operations": k_ops,
        "entities": entities,
        "initial_state": dict(zip(entities, initial_vals)),
        "operations": operations,
        "query_entity": query_entity,
        "correct_answer": correct,
        "final_state": dict(state),
    }

    return prompt, correct, metadata


def evaluate_response(response: str, correct: str, domain_name: str) -> int:
    """Check if response contains the correct answer (case-insensitive)."""
    resp_clean = response.strip().lower()
    correct_lower = correct.lower()

    # Direct match
    if correct_lower in resp_clean:
        return 1

    # Check first word
    first_word = resp_clean.split()[0] if resp_clean else ""
    first_word = re.sub(r'[^a-z]', '', first_word)
    if first_word == correct_lower:
        return 1

    return 0


def run_nonarith_ablation(model_name: str, n_problems: int = 10,
                          seed: int = 42) -> list[dict]:
    """Run non-arithmetic state tracking across K=3,5,7 and 3 domains."""
    rng = random.Random(seed)
    results = []
    depths = [3, 5, 7]

    for domain_name in DOMAINS:
        for k in depths:
            for prob_idx in range(n_problems):
                prompt, correct, meta = build_nonarith_problem(k, domain_name, rng)

                try:
                    response = call_model(model_name, prompt)
                except Exception as e:
                    response = f"ERROR: {e}"

                accurate = evaluate_response(response, correct, domain_name)

                results.append({
                    "sub_dim": "WMF-AM-NONARITH",
                    "model": model_name,
                    "domain": domain_name,
                    "k_operations": k,
                    "correct_answer": correct,
                    "raw_response": response[:500],
                    "accurate": accurate,
                    "prob_idx": prob_idx,
                    **{k2: v for k2, v in meta.items()
                       if k2 not in ("operations", "correct_answer")},
                })
                time.sleep(0.3)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="WMF-AM Non-Arithmetic State Tracking Ablation"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model names from config registry"
    )
    parser.add_argument(
        "--n-problems", type=int, default=10,
        help="Problems per depth per domain (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    all_results = []
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    for model_name in args.models:
        if model_name not in MODELS:
            print(f"WARN: {model_name} not in config, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Non-Arithmetic Ablation: {model_name}")
        print(f"{'='*60}")

        results = run_nonarith_ablation(model_name, args.n_problems, args.seed)
        all_results.extend(results)

        # Summary per domain × depth
        for domain in DOMAINS:
            for k in [3, 5, 7]:
                subset = [r for r in results
                          if r["domain"] == domain and r["k_operations"] == k]
                acc = sum(r["accurate"] for r in subset) / max(len(subset), 1)
                print(f"  {domain:8s} K={k}: {acc:.3f} ({len(subset)} trials)")

        overall = sum(r["accurate"] for r in results) / max(len(results), 1)
        print(f"  Overall: {overall:.3f}")

    # Save
    out_file = RESULTS_DIR / f"wmf_am_nonarith_{ts}.json"
    output = {
        "experiment": "WMF-AM-NONARITH-ABLATION",
        "timestamp": ts,
        "n_problems_per_depth_per_domain": args.n_problems,
        "seed": args.seed,
        "depths": [3, 5, 7],
        "domains": list(DOMAINS.keys()),
        "n_models": len(set(r["model"] for r in all_results)),
        "results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()
