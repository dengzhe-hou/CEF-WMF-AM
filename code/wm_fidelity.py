"""
Experiment E1: Working Memory Fidelity (WMF)

Three sub-dimensions:
  WMF-IM  Information Maintenance  (Brown-Peterson adapted)
  WMF-AM  Active Manipulation      (state-update sequence)
  WMF-IR  Interference Resistance  (proactive interference)

Usage:
    python wm_fidelity.py --model gpt-4o --sub-dim all --n-problems 30
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
    WMF_LOAD_LEVELS,
    WMF_OPERATION_DEPTHS,
    call_model,
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ── Fact generation helpers ──────────────────────────────────────────────────

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

DISTRACTOR_ENTITIES = [
    ("Olivia", "kept", "jars"),
    ("Paul", "found", "keys"),
    ("Quinn", "made", "baskets"),
    ("Rachel", "grew", "plants"),
    ("Sam", "fixed", "clocks"),
    ("Tara", "sewed", "quilts"),
    ("Uma", "knit", "scarves"),
    ("Victor", "carved", "statues"),
]


def _make_fact(entity, verb, obj, value: int) -> dict:
    return {
        "entity": entity,
        "verb": verb,
        "object": obj,
        "value": value,
        "sentence": f"{entity} {verb} {value} {obj}.",
    }


def _random_filler(n_words: int = 200) -> str:
    """Generate neutral filler text (irrelevant paragraphs)."""
    fillers = [
        "The weather today was pleasant with mild temperatures across the region. "
        "Residents enjoyed outdoor activities in the afternoon sunshine.",
        "Scientists recently announced new discoveries in the field of astronomy. "
        "The findings will be published next month in a peer-reviewed journal.",
        "Local markets reported steady trade volumes throughout the quarter. "
        "Analysts expect similar trends to continue into the following months.",
        "Engineers presented a prototype at the annual technology conference. "
        "The device showed promising results during demonstration sessions.",
        "Historians uncovered rare manuscripts in a library archive last spring. "
        "The documents shed new light on events from several centuries ago.",
    ]
    words = 0
    sentences = []
    while words < n_words:
        s = random.choice(fillers)
        sentences.append(s)
        words += len(s.split())
    return " ".join(sentences)


# ── WMF-IM: Information Maintenance ─────────────────────────────────────────

def build_wmf_im_prompt(
    n_targets: int,
    n_distractors: int,
    interference_level: str,  # "high" or "low"
    filler_tokens: int = 500,
) -> tuple[list[dict], str]:
    """
    Build an IM probe: embed N target facts in passage, then query one at random.
    Returns (target_facts, prompt_string).
    """
    templates = random.sample(ENTITY_TEMPLATES, n_targets)
    targets = [_make_fact(e, v, o, random.randint(10, 99)) for e, v, o in templates]

    if interference_level == "high":
        # Distractors: same objects, similar phrasing, DIFFERENT values
        distractors = [
            _make_fact(e, v, o, random.randint(10, 99))
            for e, v, o in random.sample(DISTRACTOR_ENTITIES, min(n_distractors, len(DISTRACTOR_ENTITIES)))
        ]
    else:
        # Low interference: completely different semantic domain
        distractors = [
            {"sentence": f"The temperature in sector {i} was {random.randint(10, 40)} degrees."}
            for i in range(n_distractors)
        ]

    # Interleave targets and distractors, then add filler
    all_sentences = [t["sentence"] for t in targets] + [d["sentence"] for d in distractors]
    random.shuffle(all_sentences)

    context = " ".join(all_sentences) + " " + _random_filler(filler_tokens // 5)

    # Pick one target to query
    query_fact = random.choice(targets)

    prompt = f"""Read the following passage carefully, then answer the question.

---
{context}
---

Question: How many {query_fact['object']} did {query_fact['entity']} {query_fact['verb'].replace('owns','own').replace('has','have').replace('collected','collect')}?

Respond with ONLY the number. Do not explain."""

    return targets, prompt, query_fact


def run_wmf_im(model_name: str, n_problems: int = 20) -> list[dict]:
    """Run WMF-IM across all load levels and interference conditions."""
    results = []
    for n in WMF_LOAD_LEVELS:
        for interference in ["high", "low"]:
            for _ in range(n_problems):
                targets, prompt, query_fact = build_wmf_im_prompt(
                    n_targets=n,
                    n_distractors=max(2, n // 2),
                    interference_level=interference,
                )
                response = call_model(model_name, prompt)
                # Extract number from response
                nums = re.findall(r"\d+", response)
                predicted = int(nums[0]) if nums else -1
                correct = query_fact["value"]
                results.append({
                    "sub_dim": "WMF-IM",
                    "model": model_name,
                    "n_targets": n,
                    "interference": interference,
                    "correct_answer": correct,
                    "predicted": predicted,
                    "accurate": int(predicted == correct),
                    "prompt_length_chars": len(prompt),
                })
                time.sleep(0.5)  # rate limit courtesy
    return results


# ── WMF-AM: Active Manipulation ─────────────────────────────────────────────

def build_wmf_am_problem(k_operations: int) -> tuple[dict, list[str], int]:
    """
    Build a state-update sequence with K operations.
    Returns (initial_state, operation_list, correct_final_value).
    """
    entities = random.sample([e for e, _, _ in ENTITY_TEMPLATES], 3)
    state = {e: random.randint(5, 20) for e in entities}
    initial_state = dict(state)
    operations = []

    for _ in range(k_operations):
        op_type = random.choice(["add", "subtract", "transfer"])
        if op_type == "add":
            e = random.choice(entities)
            amount = random.randint(1, 10)
            state[e] += amount
            operations.append(f"{e} gains {amount} points.")
        elif op_type == "subtract":
            e = random.choice(entities)
            amount = min(random.randint(1, 5), state[e] - 1)
            if amount > 0:
                state[e] -= amount
                operations.append(f"{e} loses {amount} points.")
            else:
                state[e] += 1
                operations.append(f"{e} gains 1 point.")
        else:  # transfer
            giver, receiver = random.sample(entities, 2)
            amount = min(random.randint(1, 3), state[giver] - 1)
            if amount > 0:
                state[giver] -= amount
                state[receiver] += amount
                operations.append(f"{giver} gives {amount} points to {receiver}.")
            else:
                operations.append(f"No transfer occurs this round.")

    query_entity = random.choice(entities)
    return initial_state, operations, state[query_entity], query_entity


def run_wmf_am(model_name: str, n_problems: int = 20) -> list[dict]:
    """Run WMF-AM across all operation depth levels."""
    results = []
    for k in WMF_OPERATION_DEPTHS:
        for _ in range(n_problems):
            initial_state, ops, correct, query_entity = build_wmf_am_problem(k)

            state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())
            ops_str = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(ops))

            prompt = f"""You will track a sequence of point updates. You cannot refer back to the initial state after reading it once.

Initial state:
{state_str}

Operations (apply in order):
{ops_str}

After all operations, how many points does {query_entity} have?

Respond with ONLY the final number."""

            response = call_model(model_name, prompt)
            nums = re.findall(r"\d+", response)
            predicted = int(nums[0]) if nums else -1

            results.append({
                "sub_dim": "WMF-AM",
                "model": model_name,
                "k_operations": k,
                "correct_answer": correct,
                "predicted": predicted,
                "accurate": int(predicted == correct),
                "query_entity": query_entity,
            })
            time.sleep(0.5)
    return results


# ── WMF-IR: Interference Resistance ─────────────────────────────────────────

def run_wmf_ir(model_name: str, n_problems: int = 20) -> list[dict]:
    """
    Run WMF-IR: Present List A, then List B (similar), then query List A.
    Compare accuracy with vs. without List B (latter from WMF-IM low-interference results).
    """
    results = []
    for _ in range(n_problems):
        n = random.choice(WMF_LOAD_LEVELS[:3])  # use smaller N for this harder test
        templates_a = random.sample(ENTITY_TEMPLATES, n)
        list_a = [_make_fact(e, v, o, random.randint(10, 49)) for e, v, o in templates_a]

        # List B: same structure (same verb/object), different entity names, different values
        list_b = [
            _make_fact(f"X-{e}", v, o, random.randint(50, 99))
            for e, v, o in templates_a  # same verb/object as A — high interference
        ]

        list_a_str = " ".join(f["sentence"] for f in list_a)
        list_b_str = " ".join(f["sentence"] for f in list_b)

        query_fact = random.choice(list_a)

        prompt = f"""Read both passages, then answer the question about the FIRST passage only.

First passage:
{list_a_str}

Second passage (different people, same type of activity):
{list_b_str}

Question: According to the FIRST passage only, how many {query_fact['object']} did {query_fact['entity']} {query_fact['verb']}?

Respond with ONLY the number."""

        response = call_model(model_name, prompt)
        nums = re.findall(r"\d+", response)
        predicted = int(nums[0]) if nums else -1

        results.append({
            "sub_dim": "WMF-IR",
            "model": model_name,
            "n_list_a": n,
            "correct_answer": query_fact["value"],
            "predicted": predicted,
            "accurate": int(predicted == query_fact["value"]),
        })
        time.sleep(0.5)
    return results


# ── Scoring ──────────────────────────────────────────────────────────────────

def compute_wmf_score(results: list[dict]) -> dict[str, float]:
    """Compute WMF composite score from sub-dimension results."""
    by_subdim: dict[str, list] = {}
    for r in results:
        sd = r["sub_dim"]
        by_subdim.setdefault(sd, []).append(r["accurate"])

    im = sum(by_subdim.get("WMF-IM", [0])) / max(len(by_subdim.get("WMF-IM", [1])), 1)
    am = sum(by_subdim.get("WMF-AM", [0])) / max(len(by_subdim.get("WMF-AM", [1])), 1)
    ir = sum(by_subdim.get("WMF-IR", [0])) / max(len(by_subdim.get("WMF-IR", [1])), 1)
    composite = 0.40 * im + 0.35 * am + 0.25 * ir

    # Load curve: accuracy by N for WMF-IM
    load_curve = {}
    for r in results:
        if r["sub_dim"] == "WMF-IM":
            n = r["n_targets"]
            load_curve.setdefault(n, []).append(r["accurate"])
    load_curve_means = {n: sum(v) / len(v) for n, v in load_curve.items()}

    return {
        "WMF-IM": round(im, 4),
        "WMF-AM": round(am, 4),
        "WMF-IR": round(ir, 4),
        "WMF_composite": round(composite, 4),
        "load_curve": load_curve_means,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run WMF experiments.")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--sub-dim", default="all", choices=["all", "im", "am", "ir"])
    parser.add_argument("--n-problems", type=int, default=20,
                        help="Problems per condition (total = n * load_levels * conditions)")
    args = parser.parse_args()

    all_results = []

    if args.sub_dim in ("all", "im"):
        print(f"Running WMF-IM for {args.model}...")
        all_results.extend(run_wmf_im(args.model, args.n_problems))

    if args.sub_dim in ("all", "am"):
        print(f"Running WMF-AM for {args.model}...")
        all_results.extend(run_wmf_am(args.model, args.n_problems))

    if args.sub_dim in ("all", "ir"):
        print(f"Running WMF-IR for {args.model}...")
        all_results.extend(run_wmf_ir(args.model, args.n_problems))

    # Save raw results
    out_dir = RESULTS_DIR / "wmf" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "raw_results.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Compute and save scores
    scores = compute_wmf_score(all_results)
    scores_path = out_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\nWMF Results for {args.model}:")
    print(f"  WMF-IM : {scores['WMF-IM']:.3f}")
    print(f"  WMF-AM : {scores['WMF-AM']:.3f}")
    print(f"  WMF-IR : {scores['WMF-IR']:.3f}")
    print(f"  COMPOSITE: {scores['WMF_composite']:.3f}")
    print(f"\nLoad curve (accuracy by N items):")
    for n, acc in sorted(scores["load_curve"].items()):
        print(f"  N={n}: {acc:.3f}")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
