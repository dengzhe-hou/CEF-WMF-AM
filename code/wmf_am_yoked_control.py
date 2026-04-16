"""
WMF-AM Yoked (Cancellation) Control Task

PURPOSE:
  A STRICTER control than wmf_am_control.py for isolating the WMF-AM construct.
  The existing control uses inert/observational operations where the answer is
  trivially the initial state. This yoked control uses CANCELLATION operations:
  each operation is immediately followed by its inverse, so the model must process
  the same token-level computation load (gains, losses, transfers with actual
  numbers) but the net result is always the initial state.

DESIGN RATIONALE:
  - wmf_am_control.py: inert ops ("Alice checks her balance") → easy to ignore
  - This script: cancelling ops ("Alice gains 3" then "Alice loses 3") → model
    must parse identical arithmetic syntax as WMF-AM, but net change = 0
  - If model succeeds here but fails WMF-AM → confirms WMF-AM measures active
    cumulative state tracking, not just per-operation parsing
  - If model fails here too → suggests per-operation arithmetic parsing is the
    bottleneck, not cumulative state maintenance

OPERATION TYPES:
  - Gain/loss pair: "Alice gains 3 points." → "Alice loses 3 points."
  - Loss/gain pair: "Bob loses 2 points." → "Bob gains 2 points."
  - Transfer pair: "Alice gives 3 points to Bob." → "Bob gives 3 points to Alice."

Usage:
    python wmf_am_yoked_control.py --model ollama:qwen2.5:7b --n-problems 20
    python wmf_am_yoked_control.py --model ollama:qwen2.5:7b --n-problems 20 --seed 123
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from config import (
    MODELS,
    RESULTS_DIR,
    WMF_OPERATION_DEPTHS,
    call_model,
)

# Same entity pool as WMF-AM (from wm_fidelity.py)
ENTITY_TEMPLATES = [
    ("Alice", "owns", "paintings"),
    ("Bob", "has", "coins"),
    ("Carol", "collected", "stamps"),
    ("David", "saved", "documents"),
    ("Emma", "scored", "points"),
    ("Frank", "planted", "trees"),
    ("Grace", "wrote", "poems"),
    ("Henry", "built", "models"),
    ("Iris", "caught", "fish"),
    ("James", "sold", "tickets"),
    ("Kate", "baked", "loaves"),
    ("Leo", "ran", "miles"),
    ("Mia", "read", "books"),
    ("Noah", "drew", "sketches"),
]


def _generate_cancelling_pair(entities: list[str], rng: random.Random) -> list[str]:
    """
    Generate a pair of operations that cancel each other out.
    Returns a list of exactly 2 operation strings.
    """
    op_type = rng.choice(["gain_loss", "loss_gain", "transfer"])

    if op_type == "gain_loss":
        e = rng.choice(entities)
        amount = rng.randint(1, 10)
        return [
            f"{e} gains {amount} points.",
            f"{e} loses {amount} points.",
        ]
    elif op_type == "loss_gain":
        e = rng.choice(entities)
        amount = rng.randint(1, 5)
        return [
            f"{e} loses {amount} points.",
            f"{e} gains {amount} points.",
        ]
    else:  # transfer
        giver, receiver = rng.sample(entities, 2)
        amount = rng.randint(1, 3)
        return [
            f"{giver} gives {amount} points to {receiver}.",
            f"{receiver} gives {amount} points to {giver}.",
        ]


def build_yoked_control_problem(
    k_operations: int, rng: random.Random
) -> tuple[dict, list[str], int, str]:
    """
    Build a cancellation-operation control problem matched to WMF-AM at depth K.

    For even K: generate K/2 cancelling pairs (total = K operations).
    For odd K: generate (K-1)/2 pairs, then add one extra operation that
    cancels with a duplicated inverse from the last pair.

    Returns (initial_state, operation_list, correct_answer, query_entity).
    The correct_answer is always the INITIAL value (all operations cancel out).
    """
    entities = rng.sample([e for e, _, _ in ENTITY_TEMPLATES], 3)
    state = {e: rng.randint(5, 20) for e in entities}

    # Number of complete cancelling pairs
    n_pairs = k_operations // 2
    remainder = k_operations % 2

    operations = []
    for _ in range(n_pairs):
        pair = _generate_cancelling_pair(entities, rng)
        operations.extend(pair)

    # Handle odd K: add one more operation and its inverse from a new pair
    if remainder > 0:
        # Generate a new cancelling pair, but only use the first operation
        # Then find a place to insert the inverse so everything still cancels.
        # Simplest approach: add the first op of a new pair, then immediately
        # add the second op too — but we only need 1 more, so instead we
        # generate a single gain and append its loss right after the last
        # existing pair (effectively the last pair becomes a triple that cancels).
        #
        # Actually for odd K, we do: (K-1)/2 full pairs + 1 extra op.
        # To keep net = 0, the extra op must also cancel. We achieve this by
        # picking an entity and adding "X gains 0 points" — but that's trivial.
        # Better: split one pair across the extra slot.
        # Approach: generate one more pair, use first op as extra. Then prepend
        # the second op (inverse) at the start of the operations list. Total = K+1?
        # No — we need exactly K ops total.
        #
        # Correct approach for odd K:
        # We have (K-1)/2 pairs = K-1 ops. We need 1 more op = K total.
        # Add one op (e.g., "Alice gains 3") and also add its inverse
        # ("Alice loses 3") but that would be K+1. So instead:
        # Use (K//2) pairs = K-1 ops when K is odd (since K//2 = (K-1)/2).
        # Then add a single self-cancelling op: "Alice gains 0 points" is too
        # obvious. Better: pick an entity and use a transfer to self — but
        # that's semantically odd.
        #
        # Simplest correct solution: for odd K, use (K-1)//2 pairs giving K-1
        # ops, then insert one additional operation that references an entity
        # but changes net by 0. We use a gain followed by... wait, we only
        # have room for 1 op.
        #
        # Final approach: for odd K, generate (K+1)//2 pairs = K+1 ops, then
        # remove the last operation. The removed op's pair partner is still
        # present, so one operation is unmatched — net != 0. Instead:
        # remove one op from the MIDDLE of a pair and also remove its partner.
        # That gives K+1 - 2 = K-1. Still wrong.
        #
        # Correct final approach: generate (K-1)//2 pairs = K-1 ops. Generate
        # one more pair. Take both ops from that pair but merge the second
        # into the operation text of the first: "Alice gains 3 points and
        # then loses 3 points." — this is 1 operation line containing both
        # the action and its cancellation. Total = K ops.
        e = rng.choice(entities)
        amount = rng.randint(1, 5)
        operations.append(
            f"{e} gains {amount} points and then immediately loses {amount} points."
        )

    query_entity = rng.choice(entities)
    correct = state[query_entity]  # unchanged — all operations cancel
    return state, operations, correct, query_entity


def run_wmf_am_yoked_control(
    model_name: str, n_problems: int = 20, seed: int = 42
) -> list[dict]:
    """Run yoked (cancellation) control across all WMF operation depth levels."""
    rng = random.Random(seed)
    results = []

    for k in WMF_OPERATION_DEPTHS:
        for prob_idx in range(n_problems):
            state, ops, correct, query_entity = build_yoked_control_problem(k, rng)

            state_str = ", ".join(f"{e}: {v} points" for e, v in state.items())
            ops_str = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(ops))

            # Identical prompt structure to WMF-AM (from wm_fidelity.py)
            prompt = f"""You will track a sequence of point updates. You cannot refer back to the initial state after reading it once.

Initial state:
{state_str}

Operations (apply in order):
{ops_str}

After all operations, how many points does {query_entity} have?

Respond with ONLY the final number."""

            response = call_model(model_name, prompt)
            clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            nums = re.findall(r"-?\d+", clean)
            predicted = int(nums[-1]) if nums else -1

            results.append({
                "sub_dim": "WMF-AM-YOKED-CONTROL",
                "model": model_name,
                "k_operations": k,
                "n_ops_actual": len(ops),
                "correct_answer": correct,
                "predicted": predicted,
                "accurate": int(predicted == correct),
                "query_entity": query_entity,
                "initial_state": state,
                "operations": ops,
                "prompt_length_chars": len(prompt),
                "raw_response": response,
            })
            time.sleep(0.5)  # rate limit courtesy

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WMF-AM Yoked (Cancellation) Control Task"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name from config registry (e.g., ollama:qwen2.5:7b)"
    )
    parser.add_argument(
        "--n-problems", type=int, default=20,
        help="Problems per depth level (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    if args.model not in MODELS:
        print(f"Error: model '{args.model}' not found in config.MODELS")
        print(f"Available: {', '.join(sorted(MODELS.keys()))}")
        return

    print(f"\n{'='*60}")
    print(f"WMF-AM Yoked (Cancellation) Control")
    print(f"Model: {args.model}")
    print(f"Problems per depth: {args.n_problems}")
    print(f"Seed: {args.seed}")
    print(f"Depths: {WMF_OPERATION_DEPTHS}")
    print(f"{'='*60}\n")

    results = run_wmf_am_yoked_control(args.model, args.n_problems, args.seed)

    # Per-depth accuracy
    depth_accs = {}
    for k in WMF_OPERATION_DEPTHS:
        k_results = [r for r in results if r["k_operations"] == k]
        acc = sum(r["accurate"] for r in k_results) / max(len(k_results), 1)
        depth_accs[k] = acc
        print(f"  K={k:2d}: accuracy={acc:.3f} (n={len(k_results)})")

    # Overall mean
    overall_acc = sum(r["accurate"] for r in results) / max(len(results), 1)
    print(f"\n  Overall: accuracy={overall_acc:.3f} (n={len(results)})")

    # Save results
    safe_name = args.model.replace(":", "_").replace("/", "_")
    out_file = RESULTS_DIR / f"wmf_am_yoked_control_{safe_name}.json"
    output = {
        "metadata": {
            "task": "WMF-AM-YOKED-CONTROL",
            "model": args.model,
            "n_problems_per_depth": args.n_problems,
            "seed": args.seed,
            "depths": WMF_OPERATION_DEPTHS,
        },
        "summary": {
            "per_depth_accuracy": {str(k): round(v, 4) for k, v in depth_accs.items()},
            "overall_accuracy": round(overall_acc, 4),
        },
        "trials": results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_file}")


if __name__ == "__main__":
    main()
