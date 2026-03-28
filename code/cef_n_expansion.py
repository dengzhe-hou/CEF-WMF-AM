"""
CEF N-Expansion: Run core CEF probes on API models to expand from N=7 (Ollama) to N>=15.

Phases per model:
  1. Outcome Correctness  — 20 factual QA items, exact-match scoring
  2. WMF-AM               — 15 probes at depths K=3,5,7, 4 random seeds
  3. MCC-MA               — 20-problem self-monitoring probe
  4. WMF-AM Yoked Control — 20 trials per depth K=2,4,6,8,12

Usage:
    python cef_n_expansion.py --models gpt-4o gpt-4o-mini claude-sonnet-4 --phases all
    python cef_n_expansion.py --models all-api
    python cef_n_expansion.py --models gpt-4o --phases wmf-am mcc-ma
    python cef_n_expansion.py --models gpt-4o --phases all --resume
"""

import argparse
import json
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure this directory is on sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS, RESULTS_DIR, call_model
from wm_fidelity import build_wmf_am_problem
from metacognitive_calibration import (
    EXAMPLE_PROBLEMS,
    load_problems,
    run_mcc_ma,
    compute_monitoring_accuracy,
    _check_answer_correct,
)

# ── Target models for N-expansion ────────────────────────────────────────────
# Original pilot (7 models): qwen2.5:7b/14b/32b, llama3.1:8b, gemma2:27b,
#   deepseek-r1:14b, mistral:7b
# Expansion models (8 new, all Ollama):

EXPANSION_MODELS = [
    "ollama:phi3:14b",
    "ollama:gemma2:9b",
    "ollama:qwen2.5:3b",
    "ollama:llama3.2:3b",
    "ollama:deepseek-r1:7b",
    "ollama:mixtral:8x7b",
    "ollama:command-r:35b",
    "ollama:yi:34b",
]

# Also include llama3.1:70b (already pulled, not in original 7-model pilot)
ALL_EXPANSION_MODELS = EXPANSION_MODELS + ["ollama:llama3.1:70b"]

ALL_PHASES = ["outcome", "wmf-am", "mcc-ma", "yoked-control"]

# ── Rate-limit configuration per provider ─────────────────────────────────────

RATE_LIMIT_DELAY = {
    "openai": 1.0,
    "anthropic": 1.0,
    "google": 0.5,
    "together": 0.5,
    "ollama": 0.0,
}

MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # exponential backoff: BACKOFF_BASE ** attempt


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_model_name(model_name: str) -> str:
    """Sanitise model name for use in file paths."""
    return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")


def _provider_for(model_name: str) -> str:
    """Return the provider string for a registered model."""
    return MODELS[model_name]["provider"]


def _rate_sleep(model_name: str) -> None:
    """Sleep for the provider-appropriate delay."""
    provider = _provider_for(model_name)
    delay = RATE_LIMIT_DELAY.get(provider, 1.0)
    if delay > 0:
        time.sleep(delay)


def call_model_with_retry(model_name: str, prompt: str, **kwargs) -> str:
    """call_model with exponential-backoff retry (up to MAX_RETRIES)."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return call_model(model_name, prompt, **kwargs)
        except Exception as exc:
            last_err = exc
            wait = BACKOFF_BASE ** attempt
            print(f"    [retry {attempt+1}/{MAX_RETRIES}] {type(exc).__name__}: {exc} — waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_err}") from last_err


def _existing_result_path(model_name: str) -> Path | None:
    """Return the most-recent result file for this model, or None."""
    safe = _safe_model_name(model_name)
    pattern = f"cef_nexp_{safe}_*.json"
    matches = sorted(RESULTS_DIR.glob(pattern))
    return matches[-1] if matches else None


# ── Phase 1: Outcome Correctness ─────────────────────────────────────────────

def run_outcome_correctness(model_name: str) -> dict:
    """
    20 factual QA items from EXAMPLE_PROBLEMS; exact-match scoring.
    Returns {"score": float, "n_correct": int, "n_total": int, "details": [...]}.
    """
    problems = [
        {"question": q, "answer": a, "domain": d, "difficulty": diff}
        for q, a, d, diff in EXAMPLE_PROBLEMS
    ]
    details = []
    n_correct = 0

    for i, prob in enumerate(problems):
        prompt = f"Answer this question in one short phrase:\n{prob['question']}"
        response = call_model_with_retry(model_name, prompt)
        is_correct = _check_answer_correct(response, prob["answer"])
        n_correct += int(is_correct)
        details.append({
            "idx": i,
            "question": prob["question"],
            "correct_answer": prob["answer"],
            "model_answer": response,
            "correct": int(is_correct),
        })
        _rate_sleep(model_name)
        print(f"    outcome [{i+1}/20] correct={is_correct}")

    score = n_correct / len(problems)
    return {
        "score": round(score, 4),
        "n_correct": n_correct,
        "n_total": len(problems),
        "details": details,
    }


# ── Phase 2: WMF-AM ──────────────────────────────────────────────────────────

WMF_AM_DEPTHS = [3, 5, 7]
WMF_AM_SEEDS = [42, 137, 256, 999]
WMF_AM_PROBES_PER_SEED_DEPTH = 15  # total across seeds*depths: 4*3=12 combos → ~1.25 each; we do 15 per depth-seed pair isn't right
# Spec says "15 probes at depths K=3,5,7, 4 random seeds" → 15 total probes spread
# across 3 depths × 4 seeds.  That gives ≈1.25 per cell, which is odd.
# Interpretation: 15 probes per depth, across 4 seeds → ~4 probes per seed per depth.
# We'll do ceil(15/4)=4 probes per (seed, depth) → 4*4=16 per depth (close to 15).
# Simplest: 15 probes per depth, cycling through seeds.

WMF_AM_PROBES_PER_DEPTH = 15


def run_wmf_am_phase(model_name: str) -> dict:
    """
    Run WMF-AM probes at K=3,5,7 with 4 random seeds.
    Returns {"mean": float, "by_depth": {K: acc}, "by_seed": {seed: acc}, "details": [...]}.
    """
    details = []
    by_depth: dict[int, list[int]] = {k: [] for k in WMF_AM_DEPTHS}
    by_seed: dict[int, list[int]] = {s: [] for s in WMF_AM_SEEDS}

    probe_idx = 0
    for k in WMF_AM_DEPTHS:
        for i in range(WMF_AM_PROBES_PER_DEPTH):
            seed = WMF_AM_SEEDS[i % len(WMF_AM_SEEDS)]
            random.seed(seed + k * 1000 + i)

            initial_state, ops, correct, query_entity = build_wmf_am_problem(k)

            state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())
            ops_str = "\n".join(f"  {i2+1}. {op}" for i2, op in enumerate(ops))

            prompt = (
                "You will track a sequence of point updates. "
                "You cannot refer back to the initial state after reading it once.\n\n"
                f"Initial state:\n{state_str}\n\n"
                f"Operations (apply in order):\n{ops_str}\n\n"
                f"After all operations, how many points does {query_entity} have?\n\n"
                "Respond with ONLY the final number."
            )

            response = call_model_with_retry(model_name, prompt)
            nums = re.findall(r"\d+", response)
            predicted = int(nums[0]) if nums else -1
            accurate = int(predicted == correct)

            by_depth[k].append(accurate)
            by_seed[seed].append(accurate)
            details.append({
                "probe_idx": probe_idx,
                "k": k,
                "seed": seed,
                "correct": correct,
                "predicted": predicted,
                "accurate": accurate,
            })
            probe_idx += 1
            _rate_sleep(model_name)
            print(f"    wmf-am [K={k}, {i+1}/{WMF_AM_PROBES_PER_DEPTH}] pred={predicted} correct={correct} ok={accurate}")

    # Aggregate
    all_acc = [d["accurate"] for d in details]
    mean_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
    depth_means = {str(k): round(sum(v) / len(v), 4) if v else 0.0 for k, v in by_depth.items()}
    seed_means = {str(s): round(sum(v) / len(v), 4) if v else 0.0 for s, v in by_seed.items()}

    return {
        "mean": round(mean_acc, 4),
        "by_depth": depth_means,
        "by_seed": seed_means,
        "n_probes": len(details),
        "details": details,
    }


# ── Phase 3: MCC-MA ──────────────────────────────────────────────────────────

def run_mcc_ma_phase(model_name: str) -> dict:
    """
    20-problem self-monitoring probe using run_mcc_ma from metacognitive_calibration.py.
    Returns {"monitoring_r": float, "n_correct": int, "n_wrong": int, "predicted_wrong": [...], "details": [...]}.
    """
    problems = load_problems(20)
    ma_results = run_mcc_ma(model_name, problems, batch_size=20)
    monitoring_r = compute_monitoring_accuracy(ma_results)

    n_correct = sum(1 for r in ma_results if r["is_correct"])
    n_wrong = sum(1 for r in ma_results if r["actually_wrong"])
    predicted_wrong = [r["question_num"] for r in ma_results if r["predicted_wrong"]]

    return {
        "monitoring_r": round(monitoring_r, 4),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "predicted_wrong": predicted_wrong,
        "details": ma_results,
    }


# ── Phase 4: WMF-AM Yoked Control ────────────────────────────────────────────
# The yoked control presents the same state-tracking format but with *inert*
# operations (no actual value changes), testing whether the model can simply
# read back a stated value after a sequence of no-ops.  This isolates whether
# WMF-AM failures come from active manipulation vs. long-context processing.

YOKED_DEPTHS = [2, 4, 6, 8, 12]
YOKED_TRIALS_PER_DEPTH = 20


def _build_yoked_control_problem(k_operations: int) -> tuple[dict, list[str], int, str]:
    """
    Build a yoked-control problem: initial state + K *inert* operations
    (restatements, irrelevant colour changes, etc.) that do NOT change any
    numeric value.  The correct answer is simply the initial value.
    """
    entities = random.sample(
        ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"], 3
    )
    state = {e: random.randint(5, 30) for e in entities}
    initial_state = dict(state)

    inert_templates = [
        "{entity} checks their score.",
        "{entity} takes a break.",
        "The scoreboard flashes for {entity}.",
        "{entity} reviews the leaderboard.",
        "An observer notes {entity}'s position.",
        "{entity} stretches and waits.",
        "The display refreshes for {entity}.",
        "{entity} glances at the clock.",
    ]

    operations = []
    for _ in range(k_operations):
        entity = random.choice(entities)
        template = random.choice(inert_templates)
        operations.append(template.format(entity=entity))

    query_entity = random.choice(entities)
    correct_value = state[query_entity]  # unchanged
    return initial_state, operations, correct_value, query_entity


def run_yoked_control_phase(model_name: str) -> dict:
    """
    Run WMF-AM yoked control: 20 trials per depth K=2,4,6,8,12.
    Returns {"mean": float, "by_depth": {K: acc}, "details": [...]}.
    """
    details = []
    by_depth: dict[int, list[int]] = {k: [] for k in YOKED_DEPTHS}

    probe_idx = 0
    for k in YOKED_DEPTHS:
        for i in range(YOKED_TRIALS_PER_DEPTH):
            random.seed(42 + k * 1000 + i)

            initial_state, ops, correct, query_entity = _build_yoked_control_problem(k)

            state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())
            ops_str = "\n".join(f"  {j+1}. {op}" for j, op in enumerate(ops))

            prompt = (
                "You will track a sequence of events. "
                "You cannot refer back to the initial state after reading it once.\n\n"
                f"Initial state:\n{state_str}\n\n"
                f"Events (observe in order):\n{ops_str}\n\n"
                f"After all events, how many points does {query_entity} have?\n\n"
                "Respond with ONLY the final number."
            )

            response = call_model_with_retry(model_name, prompt)
            nums = re.findall(r"\d+", response)
            predicted = int(nums[0]) if nums else -1
            accurate = int(predicted == correct)

            by_depth[k].append(accurate)
            details.append({
                "probe_idx": probe_idx,
                "k": k,
                "correct": correct,
                "predicted": predicted,
                "accurate": accurate,
            })
            probe_idx += 1
            _rate_sleep(model_name)

            if (i + 1) % 5 == 0 or i == 0:
                print(f"    yoked [K={k}, {i+1}/{YOKED_TRIALS_PER_DEPTH}] pred={predicted} correct={correct} ok={accurate}")

    all_acc = [d["accurate"] for d in details]
    mean_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
    depth_means = {str(k): round(sum(v) / len(v), 4) if v else 0.0 for k, v in by_depth.items()}

    return {
        "mean": round(mean_acc, 4),
        "by_depth": depth_means,
        "n_trials": len(details),
        "details": details,
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all_phases(model_name: str, phases: list[str]) -> dict:
    """Run requested phases for a single model; return combined result dict."""
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "model": model_name,
        "timestamp": timestamp,
        "outcome_correctness": None,
        "wmf_am": None,
        "mcc_ma": None,
        "yoked_control": None,
    }

    if "outcome" in phases:
        print(f"  Phase 1: Outcome Correctness ({model_name})")
        result["outcome_correctness"] = run_outcome_correctness(model_name)

    if "wmf-am" in phases:
        print(f"  Phase 2: WMF-AM ({model_name})")
        result["wmf_am"] = run_wmf_am_phase(model_name)

    if "mcc-ma" in phases:
        print(f"  Phase 3: MCC-MA ({model_name})")
        result["mcc_ma"] = run_mcc_ma_phase(model_name)

    if "yoked-control" in phases:
        print(f"  Phase 4: WMF-AM Yoked Control ({model_name})")
        result["yoked_control"] = run_yoked_control_phase(model_name)

    return result


def save_result(result: dict) -> Path:
    """Save result to RESULTS_DIR and return the path."""
    safe = _safe_model_name(result["model"])
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = RESULTS_DIR / f"cef_nexp_{safe}_{ts}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Strip verbose details for the summary file; keep a separate details file
    summary = {}
    for key, val in result.items():
        if isinstance(val, dict):
            summary[key] = {k: v for k, v in val.items() if k != "details"}
        else:
            summary[key] = val

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also save full details
    details_path = path.with_suffix(".details.json")
    with open(details_path, "w") as f:
        json.dump(result, f, indent=2)

    return path


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(all_results: list[dict]) -> None:
    """Print a compact summary table of all models' scores."""
    print("\n" + "=" * 80)
    print("CEF N-Expansion Summary")
    print("=" * 80)
    header = f"{'Model':<22} {'Outcome':>8} {'WMF-AM':>8} {'MCC-MA(r)':>10} {'Yoked':>8}"
    print(header)
    print("-" * 80)

    for r in all_results:
        model = r["model"]
        oc = r.get("outcome_correctness")
        wm = r.get("wmf_am")
        mc = r.get("mcc_ma")
        yk = r.get("yoked_control")

        oc_str = f"{oc['score']:.2f}" if oc else "  --"
        wm_str = f"{wm['mean']:.2f}" if wm else "  --"
        mc_str = f"{mc['monitoring_r']:.3f}" if mc else "   --"
        yk_str = f"{yk['mean']:.2f}" if yk else "  --"

        print(f"{model:<22} {oc_str:>8} {wm_str:>8} {mc_str:>10} {yk_str:>8}")

    print("=" * 80)

    # Depth breakdown for WMF-AM
    wm_models = [(r["model"], r["wmf_am"]) for r in all_results if r.get("wmf_am")]
    if wm_models:
        print("\nWMF-AM Accuracy by Depth:")
        depths = sorted({d for _, wm in wm_models for d in wm["by_depth"]})
        depth_header = f"{'Model':<22}" + "".join(f" K={d:>3}" for d in depths)
        print(depth_header)
        for model, wm in wm_models:
            vals = "".join(f" {wm['by_depth'].get(d, '--'):>5}" for d in depths)
            print(f"{model:<22}{vals}")

    # Depth breakdown for yoked control
    yk_models = [(r["model"], r["yoked_control"]) for r in all_results if r.get("yoked_control")]
    if yk_models:
        print("\nYoked Control Accuracy by Depth:")
        depths = sorted({d for _, yk in yk_models for d in yk["by_depth"]})
        depth_header = f"{'Model':<22}" + "".join(f" K={d:>3}" for d in depths)
        print(depth_header)
        for model, yk in yk_models:
            vals = "".join(f" {yk['by_depth'].get(d, '--'):>5}" for d in depths)
            print(f"{model:<22}{vals}")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CEF N-Expansion: run core probes on API models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all-expansion"],
        help=(
            "Model names to run. Use 'all-expansion' for all 9 expansion models, "
            "or specify individual names (e.g., ollama:phi3:14b ollama:gemma2:9b)."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["all"],
        choices=["all"] + ALL_PHASES,
        help="Phases to run (default: all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models whose result file already exists in RESULTS_DIR.",
    )
    parser.add_argument(
        "--delay-override",
        type=float,
        default=None,
        help="Override per-call delay in seconds (ignores provider defaults).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve model list
    if "all-expansion" in args.models:
        models = list(ALL_EXPANSION_MODELS)
    elif "all-new" in args.models:
        models = list(EXPANSION_MODELS)
    else:
        models = list(args.models)

    # Validate models exist in registry
    for m in models:
        if m not in MODELS:
            print(f"ERROR: model '{m}' not found in config.MODELS registry. Available: {list(MODELS.keys())}")
            sys.exit(1)

    # Resolve phases
    if "all" in args.phases:
        phases = list(ALL_PHASES)
    else:
        phases = list(args.phases)

    # Optional global delay override
    if args.delay_override is not None:
        for provider in RATE_LIMIT_DELAY:
            RATE_LIMIT_DELAY[provider] = args.delay_override

    print(f"CEF N-Expansion")
    print(f"  Models : {models}")
    print(f"  Phases : {phases}")
    print(f"  Resume : {args.resume}")
    print(f"  Results: {RESULTS_DIR}")
    print()

    all_results: list[dict] = []
    errors: list[tuple[str, str]] = []

    for model_name in models:
        # Resume check
        if args.resume:
            existing = _existing_result_path(model_name)
            if existing:
                print(f"[SKIP] {model_name} — result already exists: {existing}")
                # Load existing result for summary table
                with open(existing) as f:
                    all_results.append(json.load(f))
                continue

        print(f"\n{'='*60}")
        print(f"Running: {model_name}")
        print(f"{'='*60}")

        try:
            result = run_all_phases(model_name, phases)
            path = save_result(result)
            print(f"  Saved: {path}")
            all_results.append(result)
        except Exception as exc:
            tb = traceback.format_exc()
            errors.append((model_name, tb))
            print(f"  ERROR for {model_name}: {exc}")
            print(f"  (continuing to next model)")

    # Summary
    if all_results:
        print_summary_table(all_results)

    if errors:
        print(f"\n{len(errors)} model(s) failed:")
        for model_name, tb in errors:
            print(f"  {model_name}:")
            for line in tb.strip().split("\n")[-3:]:
                print(f"    {line}")

    print("Done.")


if __name__ == "__main__":
    main()
