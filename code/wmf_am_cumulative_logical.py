#!/usr/bin/env python3
"""
WMF-AM Cumulative Logical State Tracking Probe

PURPOSE:
  A non-arithmetic cumulative state probe to complement WMF-AM.
  Tests whether cumulative tracking difficulty generalizes beyond
  arithmetic to logical state domains.

DOMAINS:
  1. "permissions": Entity gains/loses access permissions cumulatively
     (read, write, execute, admin — binary flags tracked cumulatively)
  2. "schedule": Entity's appointments are added/removed from a schedule
     (track how many meetings remain)
  3. "inventory": Entity collects/drops items from a fixed set
     (track which items are currently held)

Each domain requires cumulative tracking of K sequential updates.
Unlike the non-arithmetic ceiling control (which uses direct assignment),
these require the model to maintain and update state through each step.

Usage:
    python wmf_am_cumulative_logical.py --models ollama:qwen2.5:7b
    python wmf_am_cumulative_logical.py --phase ollama --n-problems 15
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, call_model, RESULTS_DIR

# ── Domain definitions ──────────────────────────────────────────────────────

ENTITIES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]

# Domain 1: Permission tracking (cumulative binary flags)
PERMISSIONS = ["read", "write", "execute", "admin"]

# Domain 2: Schedule tracking (count of appointments)
MEETING_TYPES = ["morning standup", "design review", "client call",
                 "team lunch", "project sync", "budget review"]

# Domain 3: Inventory tracking (set membership)
ITEMS = ["key", "map", "torch", "rope", "compass", "shield",
         "potion", "scroll", "gem", "ring"]


def build_permission_problem(k_ops, rng):
    """Build a cumulative permission tracking problem."""
    entity = rng.choice(ENTITIES)
    start_perms = sorted(rng.sample(PERMISSIONS, rng.randint(0, 2)))
    current = set(start_perms)
    ops = []
    for _ in range(k_ops):
        action = rng.choice(["grant", "revoke"])
        perm = rng.choice(PERMISSIONS)
        if action == "grant":
            ops.append(f"{entity} is granted {perm} access.")
            current.add(perm)
        else:
            ops.append(f"{entity}'s {perm} access is revoked.")
            current.discard(perm)

    start_str = ", ".join(start_perms) if start_perms else "no permissions"
    correct = ", ".join(sorted(current)) if current else "no permissions"

    prompt = f"{entity} starts with {start_str}.\n"
    for op in ops:
        prompt += f"{op}\n"
    prompt += f"\nWhat permissions does {entity} currently have? List them in alphabetical order, separated by commas. If none, say 'no permissions'."
    prompt += "\n\nRespond with ONLY the answer."

    return prompt, correct, "permissions", entity


def build_schedule_problem(k_ops, rng):
    """Build a cumulative meeting count problem."""
    entity = rng.choice(ENTITIES)
    count = rng.randint(1, 3)  # Start with 1-3 meetings
    ops = []
    for _ in range(k_ops):
        action = rng.choice(["add", "cancel"])
        meeting = rng.choice(MEETING_TYPES)
        if action == "add":
            ops.append(f"A {meeting} is added to {entity}'s schedule.")
            count += 1
        else:
            if count > 0:
                ops.append(f"The {meeting} on {entity}'s schedule is cancelled.")
                count -= 1
            else:
                ops.append(f"A {meeting} is added to {entity}'s schedule.")
                count += 1

    prompt = f"{entity} starts the day with {count - sum(1 for o in ops if 'added' in o) + sum(1 for o in ops if 'cancelled' in o)} meetings on their schedule.\n"
    # Simpler: just track from a known start
    start_count = rng.randint(2, 5)
    current = start_count
    ops = []
    for _ in range(k_ops):
        action = rng.choice(["add", "cancel"])
        meeting = rng.choice(MEETING_TYPES)
        if action == "add":
            ops.append(f"A {meeting} is added to {entity}'s schedule.")
            current += 1
        else:
            if current > 0:
                ops.append(f"The {meeting} is cancelled from {entity}'s schedule.")
                current -= 1
            else:
                ops.append(f"A {meeting} is added to {entity}'s schedule.")
                current += 1

    prompt = f"{entity} starts the day with {start_count} meetings.\n"
    for op in ops:
        prompt += f"{op}\n"
    prompt += f"\nHow many meetings does {entity} have now?\n"
    prompt += "\nRespond with ONLY the final number."

    return prompt, str(current), "schedule", entity


def build_inventory_problem(k_ops, rng):
    """Build a cumulative inventory set tracking problem."""
    entity = rng.choice(ENTITIES)
    start_items = sorted(rng.sample(ITEMS, rng.randint(1, 3)))
    current = set(start_items)
    ops = []
    for _ in range(k_ops):
        action = rng.choice(["picks up", "drops"])
        item = rng.choice(ITEMS)
        if action == "picks up":
            ops.append(f"{entity} picks up the {item}.")
            current.add(item)
        else:
            ops.append(f"{entity} drops the {item}.")
            current.discard(item)

    correct = ", ".join(sorted(current)) if current else "nothing"

    prompt = f"{entity} starts with: {', '.join(start_items)}.\n"
    for op in ops:
        prompt += f"{op}\n"
    prompt += f"\nWhat items does {entity} currently have? List them in alphabetical order, separated by commas. If none, say 'nothing'."
    prompt += "\n\nRespond with ONLY the answer."

    return prompt, correct, "inventory", entity


DOMAIN_BUILDERS = {
    "permissions": build_permission_problem,
    "schedule": build_schedule_problem,
    "inventory": build_inventory_problem,
}


def score_response(response, correct, domain):
    """Score a response against the correct answer."""
    if response is None:
        return 0

    # Strip thinking tags
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip().lower()
    correct_lower = correct.lower()

    if domain == "schedule":
        # Numeric: extract number
        nums = re.findall(r"\d+", clean)
        if nums:
            return int(nums[-1] == correct_lower or nums[0] == correct_lower)
        return 0
    else:
        # Set-based: extract comma-separated items
        # Normalize both
        def normalize_set(s):
            items = [x.strip().lower() for x in s.split(",")]
            return set(x for x in items if x and x != "no permissions" and x != "nothing")

        resp_set = normalize_set(clean)
        corr_set = normalize_set(correct_lower)

        if correct_lower in ("no permissions", "nothing"):
            return int(len(resp_set) == 0 or "no " in clean or "nothing" in clean)

        return int(resp_set == corr_set)


def run_probe(model_name, n_problems, seed, depths):
    """Run the cumulative logical probe for one model."""
    results = []
    rng = random.Random(seed)

    for domain_name, builder in DOMAIN_BUILDERS.items():
        for k in depths:
            for trial in range(n_problems):
                trial_rng = random.Random(seed * 1000 + k * 100 + trial)
                prompt, correct, domain, entity = builder(k, trial_rng)

                for attempt in range(3):
                    response = call_model(model_name, prompt)
                    if response is not None:
                        break
                    time.sleep(2)
                if response is None:
                    response = ""

                accurate = score_response(response, correct, domain)

                results.append({
                    "model": model_name,
                    "domain": domain_name,
                    "k_operations": k,
                    "trial": trial,
                    "correct_answer": correct,
                    "raw_response": response[:500],
                    "accurate": accurate,
                })

                time.sleep(0.3)

    return results


def main():
    parser = argparse.ArgumentParser(description="WMF-AM Cumulative Logical Probe")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--phase", choices=["ollama", "api", "all"], default="ollama")
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depths", nargs="+", type=int, default=[3, 5, 7])
    args = parser.parse_args()

    if args.models:
        model_list = args.models
    elif args.phase == "ollama":
        model_list = [k for k in MODELS if k.startswith("ollama:")]
    elif args.phase == "api":
        model_list = [k for k in MODELS if k.startswith("openrouter:")]
    else:
        model_list = [k for k in MODELS if k.startswith("ollama:") or k.startswith("openrouter:")]

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"Cumulative Logical Probe")
    print(f"Models: {len(model_list)}, Depths: {args.depths}, Problems/depth/domain: {args.n_problems}")
    print(f"Domains: {list(DOMAIN_BUILDERS.keys())}")
    print("=" * 60)

    all_results = []
    for model in model_list:
        print(f"\n  Running {model}...")
        results = run_probe(model, args.n_problems, args.seed, args.depths)
        all_results.extend(results)

        # Per-domain summary
        per_domain = defaultdict(lambda: {"c": 0, "t": 0})
        for r in results:
            key = f"{r['domain']}_K{r['k_operations']}"
            per_domain[key]["t"] += 1
            per_domain[key]["c"] += r["accurate"]

        for key in sorted(per_domain):
            v = per_domain[key]
            print(f"    {key:20s}: {v['c']}/{v['t']} = {v['c']/v['t']:.3f}")

        total_c = sum(r["accurate"] for r in results)
        total_t = len(results)
        print(f"    {'OVERALL':20s}: {total_c}/{total_t} = {total_c/total_t:.3f}")

    # Save
    outpath = RESULTS_DIR / f"wmf_am_cumulative_logical_{ts}.json"
    output = {
        "experiment": "wmf_am_cumulative_logical",
        "timestamp": ts,
        "n_models": len(model_list),
        "seed": args.seed,
        "depths": args.depths,
        "n_problems_per_depth_per_domain": args.n_problems,
        "domains": list(DOMAIN_BUILDERS.keys()),
        "results": all_results,
    }
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {outpath}")


if __name__ == "__main__":
    main()
