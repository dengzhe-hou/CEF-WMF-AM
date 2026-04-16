"""
WMF-AM Matched-Length Control Task

PURPOSE:
  Isolate whether WMF-AM depth degradation measures *active state transformation*
  or merely *long-context processing*. This control uses the same surface form,
  same entities, same prompt length, and same K operation lines — but operations
  are INERT (they do not change entity values). The model only needs to recall
  the initial state, not compute anything.

DESIGN RATIONALE (addresses reviewer MAJOR: "WMF-AM construct identification"):
  - If WMF-AM measures state transformation → control should be near-ceiling at all K
  - If WMF-AM merely measures long-context → control should degrade like WMF-AM
  - Difference (WMF-AM − control) at each K isolates the "active manipulation" component

SURFACE FORM:
  Identical to WMF-AM (same entity names, same "points" framing, same K operation
  lines, same query format). Only difference: operations are observational, not
  state-changing.

  Inert operation examples:
    "Alice checks her current balance."
    "Bob reviews his score."
    "Carol asks about her points total."
    "Alice mentions her balance to Bob."
    "The system logs Carol's current total."

Usage:
    python wmf_am_control.py --model ollama:qwen2.5:7b --n-problems 20
    python wmf_am_control.py --model all --n-problems 20
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

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

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

# Inert operation templates — same word count range as real operations,
# but NO state changes. Parameterized by entity name.
INERT_TEMPLATES_SINGLE = [
    "{e} checks their current balance.",
    "{e} reviews their score so far.",
    "{e} asks about their points total.",
    "{e} confirms their current standing.",
    "{e} looks up their accumulated total.",
    "{e} verifies their balance is correct.",
    "{e} notes their current point count.",
    "{e} glances at the current scoreboard.",
    "{e} reports their total to the judge.",
    "{e} double-checks the displayed score.",
]

INERT_TEMPLATES_PAIR = [
    "{e1} mentions their balance to {e2}.",
    "{e2} asks {e1} about the leaderboard.",
    "{e1} and {e2} compare their rankings.",
    "The system logs {e1}'s and {e2}'s totals.",
    "{e1} tells {e2} the scores are unchanged.",
    "{e2} confirms the standings with {e1}.",
]


def build_control_problem(k_operations: int) -> tuple[dict, list[str], int, str]:
    """
    Build an inert-operation control problem matched to WMF-AM at depth K.
    Returns (initial_state, operation_list, correct_answer, query_entity).
    The correct_answer is always the INITIAL value (nothing changes).
    """
    entities = random.sample([e for e, _, _ in ENTITY_TEMPLATES], 3)
    state = {e: random.randint(5, 20) for e in entities}
    operations = []

    for _ in range(k_operations):
        # ~70% single-entity, ~30% pair (matches WMF-AM's add/subtract vs transfer ratio)
        if random.random() < 0.7:
            e = random.choice(entities)
            tmpl = random.choice(INERT_TEMPLATES_SINGLE)
            operations.append(tmpl.format(e=e))
        else:
            e1, e2 = random.sample(entities, 2)
            tmpl = random.choice(INERT_TEMPLATES_PAIR)
            operations.append(tmpl.format(e1=e1, e2=e2))

    query_entity = random.choice(entities)
    correct = state[query_entity]  # unchanged — no operations modify state
    return state, operations, correct, query_entity


def run_wmf_am_control(model_name: str, n_problems: int = 20) -> list[dict]:
    """Run matched-length control across all WMF operation depth levels."""
    results = []
    for k in WMF_OPERATION_DEPTHS:
        for _ in range(n_problems):
            state, ops, correct, query_entity = build_control_problem(k)

            state_str = ", ".join(f"{e}: {v} points" for e, v in state.items())
            ops_str = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(ops))

            # Identical prompt structure to WMF-AM
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
                "sub_dim": "WMF-AM-CONTROL",
                "model": model_name,
                "k_operations": k,
                "correct_answer": correct,
                "predicted": predicted,
                "accurate": int(predicted == correct),
                "query_entity": query_entity,
                "prompt_length_chars": len(prompt),
            })
            time.sleep(0.5)
    return results


def analyze_control_vs_wmf(control_results: list[dict], wmf_results: list[dict]):
    """
    Compare control vs WMF-AM accuracy at each K.
    Returns per-K accuracy and the manipulation-specific component (Δ).
    """
    analysis = {}
    for k in WMF_OPERATION_DEPTHS:
        ctrl_k = [r for r in control_results if r["k_operations"] == k]
        wmf_k = [r for r in wmf_results if r["k_operations"] == k]

        ctrl_acc = sum(r["accurate"] for r in ctrl_k) / max(len(ctrl_k), 1)
        wmf_acc = sum(r["accurate"] for r in wmf_k) / max(len(wmf_k), 1)

        analysis[k] = {
            "control_accuracy": round(ctrl_acc, 3),
            "wmf_am_accuracy": round(wmf_acc, 3),
            "delta_manipulation": round(ctrl_acc - wmf_acc, 3),
            "n_control": len(ctrl_k),
            "n_wmf": len(wmf_k),
        }
    return analysis


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WMF-AM Matched-Length Control Task")
    parser.add_argument("--model", required=True,
                        help="Model name from config, or 'all'")
    parser.add_argument("--n-problems", type=int, default=20,
                        help="Problems per depth level (default: 20)")
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Running WMF-AM Control: {model_name}")
        print(f"{'='*60}")

        results = run_wmf_am_control(model_name, args.n_problems)

        # Per-K summary
        for k in WMF_OPERATION_DEPTHS:
            k_results = [r for r in results if r["k_operations"] == k]
            acc = sum(r["accurate"] for r in k_results) / max(len(k_results), 1)
            print(f"  K={k:2d}: accuracy={acc:.3f} (n={len(k_results)})")

        # Save
        safe_name = model_name.replace(":", "_").replace("/", "_")
        out_file = RESULTS_DIR / f"wmf_am_control_{safe_name}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {out_file}")


if __name__ == "__main__":
    main()
