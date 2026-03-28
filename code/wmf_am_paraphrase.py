"""
WMF-AM Prompt Paraphrase Ablation

PURPOSE:
  Tests whether WMF-AM rankings are robust to prompt rewording.
  If model rankings are preserved under paraphrased instructions
  (same logic, different wording), WMF-AM measures a stable construct.
  If rankings reverse, prompt-specific artifacts dominate.

DESIGN:
  - Same K-depth structure as WMF-AM (K=3,5,7)
  - Same numeric state tracking tasks
  - 5 prompt paraphrase templates with identical logic but different wording
  - Compare rank correlations across templates

TEMPLATES:
  1. ORIGINAL: Matches the standard WMF-AM prompt from wm_fidelity.py
  2. FORMAL: Academic/formal register
  3. CASUAL: Conversational/casual register
  4. MINIMAL: Bare minimum instructions
  5. VERBOSE: Highly detailed step-by-step instructions

Usage:
    python wmf_am_paraphrase.py --models ollama:qwen2.5:7b ollama:llama3.1:8b
"""

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

from config import MODELS, RESULTS_DIR, call_model

ENTITIES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]


def build_problem_data(k_ops: int, rng: random.Random):
    """Generate the raw problem data (shared across all templates)."""
    entities = rng.sample(ENTITIES, 3)
    state = {e: rng.randint(5, 20) for e in entities}
    initial_state = dict(state)

    operations = []
    for _ in range(k_ops):
        entity = rng.choice(entities)
        op_type = rng.choice(["gain", "lose"])
        amount = rng.randint(1, 8)
        if op_type == "gain":
            state[entity] += amount
            operations.append((entity, "gains", amount))
        else:
            state[entity] -= amount
            operations.append((entity, "loses", amount))

    query_entity = rng.choice(entities)
    correct = state[query_entity]

    return {
        "entities": entities,
        "initial_state": initial_state,
        "operations": operations,
        "query_entity": query_entity,
        "correct_answer": correct,
        "final_state": dict(state),
    }


def format_prompt(template_name: str, data: dict) -> str:
    """Format the problem data into a specific prompt template."""
    entities = data["entities"]
    initial = data["initial_state"]
    ops = data["operations"]
    query = data["query_entity"]

    state_str = ", ".join(f"{e}: {initial[e]} points" for e in entities)
    ops_lines = [f"{e} {verb} {amt} points" for e, verb, amt in ops]

    if template_name == "original":
        ops_str = "\n".join(f"  {i+1}. {op}." for i, op in enumerate(ops_lines))
        return f"""You will track a sequence of point updates. You cannot refer back to the initial state after reading it once.

Initial state:
{state_str}

Operations (apply in order):
{ops_str}

After all operations, how many points does {query} have?

Respond with ONLY the final number."""

    elif template_name == "formal":
        ops_str = "\n".join(f"  Step {i+1}: {op}." for i, op in enumerate(ops_lines))
        return f"""The following is a sequential state-tracking exercise. Please process each modification in the order given and determine the resulting value.

Starting values:
{state_str}

Modifications to apply sequentially:
{ops_str}

Question: What is the final point total for {query} after all modifications have been applied?

Please provide only the numerical answer."""

    elif template_name == "casual":
        ops_str = "\n".join(f"- {op}" for op in ops_lines)
        return f"""Hey, can you help me keep track of some scores?

So we start with: {state_str}

Then these things happen one after another:
{ops_str}

So after all that, how many points does {query} end up with?

Just tell me the number."""

    elif template_name == "minimal":
        ops_str = "; ".join(ops_lines)
        return f"""Start: {state_str}
Changes: {ops_str}
{query}'s final points = ?"""

    elif template_name == "verbose":
        ops_str = "\n".join(
            f"  {i+1}. {op}. (After this step, update {e}'s running total accordingly.)"
            for i, (op, (e, _, _)) in enumerate(zip(ops_lines, ops))
        )
        return f"""In this task, you need to carefully track point totals for multiple people as they change over time. Read the initial state, then process each operation one by one in the exact order listed. Each operation either adds points to or subtracts points from one person's total. You must keep a mental running total for each person.

Here are the initial point totals for each person:
{state_str}

Now, apply the following operations one at a time, in order. After each operation, mentally update the running total for the affected person:
{ops_str}

Now that you have processed all {len(ops)} operations, please tell me: what is {query}'s final point total?

Important: respond with ONLY the final number, nothing else."""

    else:
        raise ValueError(f"Unknown template: {template_name}")


TEMPLATES = ["original", "formal", "casual", "minimal", "verbose"]


def run_paraphrase_ablation(model_name: str, n_problems: int = 10,
                            seed: int = 42) -> list[dict]:
    """Run paraphrase ablation across K=3,5,7 and 5 templates."""
    rng = random.Random(seed)
    results = []
    depths = [3, 5, 7]

    for k in depths:
        for prob_idx in range(n_problems):
            # Generate problem data ONCE, then present in all 5 templates
            data = build_problem_data(k, rng)

            for template in TEMPLATES:
                prompt = format_prompt(template, data)

                try:
                    response = call_model(model_name, prompt)
                except Exception as e:
                    response = f"ERROR: {e}"

                nums = re.findall(r"-?\d+", response)
                predicted = int(nums[0]) if nums else -9999
                accurate = int(predicted == data["correct_answer"])

                results.append({
                    "sub_dim": "WMF-AM-PARAPHRASE",
                    "model": model_name,
                    "template": template,
                    "k_operations": k,
                    "correct_answer": data["correct_answer"],
                    "predicted": predicted,
                    "accurate": accurate,
                    "prob_idx": prob_idx,
                    "raw_response": response[:300],
                })
                time.sleep(0.3)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="WMF-AM Prompt Paraphrase Ablation"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model names from config registry"
    )
    parser.add_argument(
        "--n-problems", type=int, default=10,
        help="Problems per depth (each tested in 5 templates; default: 10)"
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
        print(f"Paraphrase Ablation: {model_name}")
        print(f"{'='*60}")

        results = run_paraphrase_ablation(model_name, args.n_problems, args.seed)
        all_results.extend(results)

        # Summary per template × depth
        for template in TEMPLATES:
            for k in [3, 5, 7]:
                subset = [r for r in results
                          if r["template"] == template and r["k_operations"] == k]
                acc = sum(r["accurate"] for r in subset) / max(len(subset), 1)
                print(f"  {template:10s} K={k}: {acc:.3f} ({len(subset)} trials)")

        # Overall per template
        print(f"  {'---':10s}")
        for template in TEMPLATES:
            subset = [r for r in results if r["template"] == template]
            acc = sum(r["accurate"] for r in subset) / max(len(subset), 1)
            print(f"  {template:10s} overall: {acc:.3f}")

    # Save
    out_file = RESULTS_DIR / f"wmf_am_paraphrase_{ts}.json"
    output = {
        "experiment": "WMF-AM-PARAPHRASE-ABLATION",
        "timestamp": ts,
        "n_problems_per_depth": args.n_problems,
        "seed": args.seed,
        "depths": [3, 5, 7],
        "templates": TEMPLATES,
        "n_models": len(set(r["model"] for r in all_results)),
        "results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_file}")

    # Compute rank correlation across templates if multiple models
    models_tested = sorted(set(r["model"] for r in all_results))
    if len(models_tested) >= 5:
        print(f"\n{'='*60}")
        print("Cross-template rank analysis")
        print(f"{'='*60}")
        from scipy.stats import kendalltau
        # Per-template mean accuracy
        template_scores = {}
        for template in TEMPLATES:
            scores = []
            for m in models_tested:
                subset = [r for r in all_results
                          if r["model"] == m and r["template"] == template]
                acc = sum(r["accurate"] for r in subset) / max(len(subset), 1)
                scores.append(acc)
            template_scores[template] = scores

        # Pairwise rank correlations
        for i, t1 in enumerate(TEMPLATES):
            for t2 in TEMPLATES[i+1:]:
                tau, p = kendalltau(template_scores[t1], template_scores[t2])
                print(f"  {t1} ↔ {t2}: τ={tau:.3f}, p={p:.4f}")


if __name__ == "__main__":
    main()
