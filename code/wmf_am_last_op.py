"""
WMF-AM Last-Operation-Only Control (Standalone Arithmetic Probe)

PURPOSE:
  Separates arithmetic ability from arithmetic accumulation under load.

  If models score high on WMF-AM (cumulative K-step tracking) but
  WMF-AM is hard, is the bottleneck:
    (a) the arithmetic itself? (e.g., can't compute X+Y)
    (b) the cumulative accumulation under K steps of load?

  This probe tests (a): single-step arithmetic with SAME surface form
  as WMF-AM (points, inventory, accounts domains). K=1 only — no
  accumulation is required.

  If models score near-ceiling here but floor on WMF-AM K=7:
    → Difficulty is accumulation, NOT arithmetic. ✓ Strong evidence.

  If models fail even single-step arithmetic:
    → Arithmetic skill confounds WMF-AM. ✗ Weaker construct claim.

DESIGN:
  - Same 3 surface forms as WMF-AM: points, inventory, accounts
  - K=1 (single operation: one gain or loss)
  - Numeric ranges: small (1–20), medium (20–100), large (100–1000)
  - 10 problems per domain × range = 90 problems per model
  - Exact-match scoring (same as WMF-AM)

COMPARISON:
  - WMF-AM K=7 accuracy tells us accumulation performance
  - Last-op K=1 accuracy tells us standalone arithmetic performance
  - Gap (WMF-AM K=7 - Last-op K=1) = accumulation overhead

Usage:
    python wmf_am_last_op.py --models ollama:qwen2.5:7b ollama:llama3.1:8b
    python wmf_am_last_op.py  # uses all 15 main-study models
"""

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

from config import MODELS, RESULTS_DIR, call_model

# ── Surface forms matching WMF-AM ─────────────────────────────────────────────

DOMAINS = {
    "points": {
        "entities": ["Alice", "Bob", "Carol", "David", "Emma",
                     "Frank", "Grace", "Henry", "Iris", "James"],
        "unit": "points",
        "gain_verb": "gains",
        "lose_verb": "loses",
        "init_template": "{entity} starts with {val} points.",
        "op_gain": "{entity} gains {delta} points.",
        "op_lose": "{entity} loses {delta} points.",
        "question": "How many points does {entity} have now?",
        "parse_hint": "Respond with ONLY the final number.",
    },
    "inventory": {
        "entities": ["Alice", "Bob", "Carol", "David", "Emma",
                     "Frank", "Grace", "Henry", "Iris", "James"],
        "unit": "items",
        "gain_verb": "adds",
        "lose_verb": "removes",
        "init_template": "{entity} has {val} items in their warehouse.",
        "op_gain": "{entity} adds {delta} items to their warehouse.",
        "op_lose": "{entity} removes {delta} items from their warehouse.",
        "question": "How many items does {entity} have in their warehouse now?",
        "parse_hint": "Respond with ONLY the final number.",
    },
    "accounts": {
        "entities": ["Alice", "Bob", "Carol", "David", "Emma",
                     "Frank", "Grace", "Henry", "Iris", "James"],
        "unit": "dollars",
        "gain_verb": "deposits",
        "lose_verb": "withdraws",
        "init_template": "{entity}'s account balance is ${val}.",
        "op_gain": "{entity} deposits ${delta} into their account.",
        "op_lose": "{entity} withdraws ${delta} from their account.",
        "question": "What is {entity}'s account balance now?",
        "parse_hint": "Respond with ONLY the final number (no $ sign).",
    },
}

# Numeric ranges for arithmetic difficulty sweep
NUMERIC_RANGES = {
    "small":  (1, 20),
    "medium": (20, 100),
    "large":  (100, 1000),
}

# 15 main-study models (matches WMF-AM paper results)
MAIN_STUDY_MODELS = [
    "ollama:qwen2.5:7b",
    "ollama:qwen2.5:14b",
    "ollama:qwen2.5:32b",
    "ollama:llama3.1:8b",
    "ollama:gemma2:27b",
    "ollama:deepseek-r1:14b",
    "ollama:mistral:7b",
    "ollama:phi3:14b",
    "ollama:gemma2:9b",
    "ollama:qwen2.5:3b",
    "ollama:llama3.2:3b",
    "ollama:deepseek-r1:7b",
    "ollama:mixtral:8x7b",
    "ollama:command-r:35b",
    "ollama:yi:34b",
]


def build_last_op_problem(domain_name: str, numeric_range: str, rng: random.Random):
    """
    Build a single-step arithmetic problem (K=1).

    Returns: (prompt, correct_answer:int, metadata:dict)
    """
    domain = DOMAINS[domain_name]
    lo, hi = NUMERIC_RANGES[numeric_range]

    entity = rng.choice(domain["entities"])
    init_val = rng.randint(lo, hi)
    delta = rng.randint(lo // 2 + 1, hi)  # ensure delta > 0
    is_gain = rng.choice([True, False])

    if is_gain:
        correct = init_val + delta
        op_line = domain["op_gain"].format(entity=entity, delta=delta)
        op_type = "gain"
    else:
        # Avoid negative balances for cleanliness
        if delta > init_val:
            delta = rng.randint(1, init_val)
        correct = init_val - delta
        op_line = domain["op_lose"].format(entity=entity, delta=delta)
        op_type = "lose"

    init_line = domain["init_template"].format(entity=entity, val=init_val)
    question = domain["question"].format(entity=entity)
    parse_hint = domain["parse_hint"]

    prompt = f"""{init_line}
{op_line}
{question}
{parse_hint}"""

    metadata = {
        "domain": domain_name,
        "numeric_range": numeric_range,
        "entity": entity,
        "init_val": init_val,
        "delta": delta,
        "op_type": op_type,
        "k_operations": 1,
        "correct_answer": correct,
    }
    return prompt, correct, metadata


def parse_number(response: str) -> int | None:
    """Extract an integer from the model's response."""
    resp = response.strip()
    # Remove $ or other currency symbols
    resp = re.sub(r'[\$,]', '', resp)
    # Try exact integer match at start
    m = re.match(r'^-?\d+', resp)
    if m:
        return int(m.group())
    # Search for any integer in response
    nums = re.findall(r'-?\d+', resp)
    if nums:
        return int(nums[0])
    return None


def evaluate_response(response: str, correct: int) -> int:
    val = parse_number(response)
    return 1 if val == correct else 0


def run_last_op_probe(model_name: str, n_problems: int = 10,
                      seed: int = 42) -> list[dict]:
    """Run last-op arithmetic probe across all domain × range combinations."""
    rng = random.Random(seed)
    results = []

    for domain_name in DOMAINS:
        for range_name in NUMERIC_RANGES:
            for prob_idx in range(n_problems):
                prompt, correct, meta = build_last_op_problem(
                    domain_name, range_name, rng
                )

                try:
                    response = call_model(model_name, prompt)
                except Exception as e:
                    response = f"ERROR: {e}"

                accurate = evaluate_response(response, correct)

                results.append({
                    "sub_dim": "WMF-AM-LAST-OP",
                    "model": model_name,
                    "domain": domain_name,
                    "numeric_range": range_name,
                    "k_operations": 1,
                    "prob_idx": prob_idx,
                    "correct_answer": correct,
                    "raw_response": response[:300],
                    "parsed_answer": parse_number(response),
                    "accurate": accurate,
                    **{k: v for k, v in meta.items()
                       if k not in ("correct_answer",)},
                })
                time.sleep(0.2)

    return results


def print_model_summary(model_name: str, results: list[dict]):
    print(f"\n{'='*60}")
    print(f"LAST-OP RESULTS: {model_name}")
    print(f"{'='*60}")
    overall = sum(r["accurate"] for r in results) / max(len(results), 1)
    print(f"  Overall accuracy: {overall:.3f}  (N={len(results)})")

    for domain in DOMAINS:
        for rng_name in NUMERIC_RANGES:
            sub = [r for r in results
                   if r["domain"] == domain and r["numeric_range"] == rng_name]
            if sub:
                acc = sum(r["accurate"] for r in sub) / len(sub)
                print(f"  {domain:10s} × {rng_name:6s}: {acc:.3f}  ({len(sub)} trials)")


def main():
    parser = argparse.ArgumentParser(
        description="WMF-AM Last-Operation-Only Standalone Arithmetic Control"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=(
            "Model names (default: all 15 main-study models). "
            "Example: ollama:qwen2.5:7b ollama:llama3.1:8b"
        ),
    )
    parser.add_argument(
        "--n-problems", type=int, default=10,
        help="Problems per domain per numeric range (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    models = args.models if args.models else MAIN_STUDY_MODELS
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    print(f"WMF-AM Last-Op Standalone Arithmetic Control")
    print(f"Timestamp: {ts}")
    print(f"Models: {len(models)}")
    print(f"Design: K=1, 3 domains × 3 numeric ranges × {args.n_problems} problems")
    print(f"Total trials per model: {3 * 3 * args.n_problems}")

    all_results = []

    for model_name in models:
        if model_name not in MODELS:
            print(f"WARN: {model_name} not in config registry, skipping")
            continue

        results = run_last_op_probe(model_name, args.n_problems, args.seed)
        all_results.extend(results)
        print_model_summary(model_name, results)

    # ── Aggregate across-model summary ────────────────────────────────────────
    if all_results:
        print(f"\n{'='*60}")
        print("CROSS-MODEL SUMMARY")
        print(f"{'='*60}")
        overall = sum(r["accurate"] for r in all_results) / len(all_results)
        print(f"Grand mean accuracy (K=1): {overall:.3f}")
        per_model = {}
        for r in all_results:
            m = r["model"]
            per_model.setdefault(m, []).append(r["accurate"])
        accs = {m: sum(v) / len(v) for m, v in per_model.items()}
        sorted_models = sorted(accs.items(), key=lambda x: -x[1])
        for m, acc in sorted_models:
            print(f"  {m:35s}: {acc:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_file = RESULTS_DIR / f"wmf_am_last_op_{ts}.json"
    output = {
        "experiment": "WMF-AM-LAST-OP-STANDALONE-ARITHMETIC-CONTROL",
        "description": (
            "Single-step (K=1) arithmetic probe to isolate standalone arithmetic "
            "ability from accumulation-under-load (WMF-AM). "
            "Expected: near-ceiling if WMF-AM difficulty is due to accumulation, "
            "not arithmetic parsing."
        ),
        "timestamp": ts,
        "n_problems_per_domain_per_range": args.n_problems,
        "seed": args.seed,
        "k_operations": 1,
        "domains": list(DOMAINS.keys()),
        "numeric_ranges": list(NUMERIC_RANGES.keys()),
        "n_models": len(set(r["model"] for r in all_results)),
        "grand_mean_accuracy": (
            sum(r["accurate"] for r in all_results) / max(len(all_results), 1)
        ),
        "results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()
