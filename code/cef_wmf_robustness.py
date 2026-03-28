"""
WMF-AM Prompt-Template Robustness Check (CEF — NeurIPS 2026 supplement).

Goal: Verify that WMF-AM model rankings are stable across 3 structurally
different prompt templates (same underlying math, different surface form).
This rules out the reviewer concern that CEF scores are "template artifacts."

Three templates:
  A — Original  : people scoring points, numbered operation list (same as pilot)
  B — Inventory : warehouse item counts, restocked/sold/transferred vocabulary
  C — Financial : bank accounts, deposit/withdraw/transfer vocabulary

Per model, per template: 15 probes at depths K ∈ {3, 5, 7}, 5 trials each.
Total: 7 models × 3 templates × 15 probes = 315 calls (~45-90 min on RTX Pro 6000).

Output: JSON + console table showing per-template rankings and Kendall τ.
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, call_model
from wm_fidelity import build_wmf_am_problem

RANDOM_SEED = 2026
DEPTHS = [3, 5, 7]
TRIALS_PER_DEPTH = 5  # 15 probes per template per model


# ── Template A: Original (people + points) ───────────────────────────────────

def format_template_A(initial_state: dict, operations: list, query_entity: str) -> str:
    state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())
    ops_str = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(operations))
    return f"""You will track a sequence of point updates.

Initial state:
{state_str}

Operations (apply in order):
{ops_str}

After all operations, how many points does {query_entity} have?

Respond with ONLY the final number."""


# ── Template B: Inventory / Warehouse ────────────────────────────────────────

# Map person names → item names, and point ops → inventory ops
_ITEM_MAP = {
    "Alice": "widgets",
    "Bob": "gadgets",
    "Carol": "sprockets",
    "David": "brackets",
    "Emma": "bearings",
    "Frank": "bolts",
    "Grace": "hinges",
    "Henry": "latches",
    "Iris": "valves",
    "James": "gaskets",
    "Kate": "seals",
    "Leo": "clips",
    "Mia": "pins",
    "Noah": "rivets",
}

def _to_item(name: str) -> str:
    return _ITEM_MAP.get(name, name.lower() + "s")

def _rewrite_op_B(op: str, entities: list) -> str:
    """Rewrite a 'points' operation as a warehouse operation."""
    # "Alice gains 3 points." → "widgets inventory increases by 3 units."
    # "Bob loses 2 points."   → "gadgets inventory decreases by 2 units."
    # "Carol gives 1 point to Alice." → "sprockets: 1 unit transferred to widgets."
    m_gain = re.match(r"(\w+) gains (\d+) points?\.", op)
    m_lose = re.match(r"(\w+) loses (\d+) points?\.", op)
    m_give = re.match(r"(\w+) gives (\d+) points? to (\w+)\.", op)
    m_none = re.match(r"No transfer occurs", op)
    if m_gain:
        item = _to_item(m_gain.group(1))
        return f"Restock {item}: +{m_gain.group(2)} units received."
    if m_lose:
        item = _to_item(m_lose.group(1))
        return f"Shipment from {item}: -{m_lose.group(2)} units dispatched."
    if m_give:
        src = _to_item(m_give.group(1))
        dst = _to_item(m_give.group(3))
        return f"Transfer {m_give.group(2)} units from {src} to {dst}."
    if m_none:
        return "No movement this cycle."
    return op  # fallback


def format_template_B(initial_state: dict, operations: list, query_entity: str) -> str:
    entities = list(initial_state.keys())
    inv_lines = "\n".join(f"  {_to_item(e)}: {v} units" for e, v in initial_state.items())
    ops_str = "\n".join(f"  {i+1}. {_rewrite_op_B(op, entities)}" for i, op in enumerate(operations))
    query_item = _to_item(query_entity)
    return f"""You are tracking warehouse inventory levels.

Opening stock:
{inv_lines}

Warehouse log (apply each entry in order):
{ops_str}

After processing all log entries, what is the current unit count for {query_item}?

Respond with ONLY the final number."""


# ── Template C: Financial / Bank accounts ────────────────────────────────────

_ACCOUNT_MAP = {
    "Alice": "Account-A",
    "Bob": "Account-B",
    "Carol": "Account-C",
    "David": "Account-D",
    "Emma": "Account-E",
    "Frank": "Account-F",
    "Grace": "Account-G",
    "Henry": "Account-H",
    "Iris": "Account-I",
    "James": "Account-J",
    "Kate": "Account-K",
    "Leo": "Account-L",
    "Mia": "Account-M",
    "Noah": "Account-N",
}

def _to_account(name: str) -> str:
    return _ACCOUNT_MAP.get(name, f"Account-{name[0]}")

def _rewrite_op_C(op: str) -> str:
    """Rewrite a 'points' operation as a bank transaction."""
    m_gain = re.match(r"(\w+) gains (\d+) points?\.", op)
    m_lose = re.match(r"(\w+) loses (\d+) points?\.", op)
    m_give = re.match(r"(\w+) gives (\d+) points? to (\w+)\.", op)
    m_none = re.match(r"No transfer occurs", op)
    if m_gain:
        acc = _to_account(m_gain.group(1))
        return f"Deposit ${m_gain.group(2)} into {acc}."
    if m_lose:
        acc = _to_account(m_lose.group(1))
        return f"Withdraw ${m_lose.group(2)} from {acc}."
    if m_give:
        src = _to_account(m_give.group(1))
        dst = _to_account(m_give.group(3))
        return f"Wire transfer of ${m_give.group(2)} from {src} to {dst}."
    if m_none:
        return "No transactions processed."
    return op


def format_template_C(initial_state: dict, operations: list, query_entity: str) -> str:
    bal_lines = "\n".join(f"  {_to_account(e)}: ${v}" for e, v in initial_state.items())
    ops_str = "\n".join(f"  {i+1}. {_rewrite_op_C(op)}" for i, op in enumerate(operations))
    query_acc = _to_account(query_entity)
    return f"""You are reconciling bank account balances.

Opening balances:
{bal_lines}

Transaction log (process each in sequence):
{ops_str}

After all transactions, what is the balance of {query_acc}?

Respond with ONLY the final dollar amount (number only, no $ sign)."""


TEMPLATES = {
    "A_points": format_template_A,
    "B_inventory": format_template_B,
    "C_financial": format_template_C,
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_model_robustness(model_name: str) -> dict:
    """Run all 3 templates for one model. Returns per-template accuracy."""
    result = {"model": model_name, "timestamp": datetime.utcnow().isoformat(), "templates": {}}

    for tmpl_name, fmt_fn in TEMPLATES.items():
        per_depth = {}
        trials = []
        for depth in DEPTHS:
            correct_count = 0
            for trial in range(TRIALS_PER_DEPTH):
                # Fresh seed per (template, depth, trial) for reproducibility
                random.seed(RANDOM_SEED + hash(tmpl_name) % 1000 + depth * 100 + trial)
                initial_state, operations, correct_answer, query_entity = build_wmf_am_problem(depth)
                prompt = fmt_fn(initial_state, operations, query_entity)
                try:
                    response = call_model(model_name, prompt)
                    nums = re.findall(r"\d+", response)
                    predicted = int(nums[0]) if nums else -1
                    is_correct = (predicted == correct_answer)
                except Exception as e:
                    predicted = -1
                    is_correct = False
                    print(f"    ERROR [{tmpl_name} K={depth} t={trial}]: {e}", flush=True)

                trials.append({
                    "template": tmpl_name,
                    "depth": depth,
                    "trial": trial,
                    "correct": correct_answer,
                    "predicted": predicted,
                    "is_correct": is_correct,
                })
                if is_correct:
                    correct_count += 1

            acc = correct_count / TRIALS_PER_DEPTH
            per_depth[depth] = acc

        mean_acc = sum(per_depth.values()) / len(per_depth)
        result["templates"][tmpl_name] = {
            "mean_accuracy": mean_acc,
            "per_depth": per_depth,
            "trials": trials,
        }
        print(f"    {tmpl_name}: mean={mean_acc:.3f}  K3={per_depth[3]:.2f} K5={per_depth[5]:.2f} K7={per_depth[7]:.2f}", flush=True)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "ollama:qwen2.5:7b",
    "ollama:qwen2.5:14b",
    "ollama:qwen2.5:32b",
    "ollama:llama3.1:8b",
    "ollama:gemma2:27b",
    "ollama:deepseek-r1:14b",
    "ollama:mistral:7b",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", default=str(RESULTS_DIR / "cef_wmf_robustness.json"))
    args = parser.parse_args()

    print(f"WMF-AM Robustness Check — {len(args.models)} models × 3 templates × 15 probes", flush=True)
    print(f"Models: {args.models}", flush=True)
    print(f"Started: {datetime.utcnow().isoformat()}", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    for model in args.models:
        print(f"\nModel: {model}", flush=True)
        try:
            r = run_model_robustness(model)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR on {model}: {e}", flush=True)
            continue

    # ── Cross-template ranking analysis ──────────────────────────────────────
    print("\n\n" + "=" * 70, flush=True)
    print("ROBUSTNESS ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    tmpl_names = list(TEMPLATES.keys())
    # Build accuracy matrix: models × templates
    model_names = [r["model"].replace("ollama:", "") for r in all_results]
    acc_matrix = {}
    for tmpl in tmpl_names:
        acc_matrix[tmpl] = [r["templates"][tmpl]["mean_accuracy"] for r in all_results]

    # Print summary table
    header = f"{'Model':<22}" + "".join(f" {t:>12}" for t in tmpl_names)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i, name in enumerate(model_names):
        row = f"{name:<22}" + "".join(f" {acc_matrix[t][i]:>12.3f}" for t in tmpl_names)
        print(row, flush=True)

    # Kendall tau between template rankings
    print("\nKendall τ between template rankings:", flush=True)
    for i in range(len(tmpl_names)):
        for j in range(i + 1, len(tmpl_names)):
            t1, t2 = tmpl_names[i], tmpl_names[j]
            tau, p = kendalltau(acc_matrix[t1], acc_matrix[t2])
            print(f"  τ({t1}, {t2}) = {tau:.3f}  p={p:.3f}", flush=True)

    # Spearman rho for robustness
    print("\nSpearman ρ between template rankings:", flush=True)
    for i in range(len(tmpl_names)):
        for j in range(i + 1, len(tmpl_names)):
            t1, t2 = tmpl_names[i], tmpl_names[j]
            rho, p = spearmanr(acc_matrix[t1], acc_matrix[t2])
            print(f"  ρ({t1}, {t2}) = {rho:.3f}  p={p:.3f}", flush=True)

    # Rank correlation matrix
    print("\nRanking by template (best → worst):", flush=True)
    for tmpl in tmpl_names:
        ranked = sorted(zip(model_names, acc_matrix[tmpl]), key=lambda x: -x[1])
        print(f"  {tmpl}: " + " > ".join(f"{m}({a:.3f})" for m, a in ranked), flush=True)

    # Save output
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_models": len(all_results),
        "models": args.models,
        "templates": tmpl_names,
        "depths": DEPTHS,
        "trials_per_depth": TRIALS_PER_DEPTH,
        "per_model": all_results,
        "summary": {
            tmpl: {
                "accuracies": acc_matrix[tmpl],
                "mean": sum(acc_matrix[tmpl]) / len(acc_matrix[tmpl]),
            }
            for tmpl in tmpl_names
        },
        "kendall_tau": {
            f"{tmpl_names[i]}_vs_{tmpl_names[j]}": float(kendalltau(acc_matrix[tmpl_names[i]], acc_matrix[tmpl_names[j]])[0])
            for i in range(len(tmpl_names))
            for j in range(i + 1, len(tmpl_names))
            if len(acc_matrix[tmpl_names[i]]) > 1
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)
    print(f"Finished: {datetime.utcnow().isoformat()}", flush=True)
