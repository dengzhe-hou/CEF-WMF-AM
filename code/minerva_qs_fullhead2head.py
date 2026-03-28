"""
minerva_qs_fullhead2head.py

Full Minerva Quantity State (K=200) head-to-head on all 20 study models.

Protocol:
- Follows Minerva ICML 2025 Quantity State specification exactly:
  * 200 sequential add/subtract operations
  * Initial integer in [10, 100]
  * Each operation: add or subtract a random integer in [1, 20]
  * Scoring: exact match on final value
- N_SAMPLES=3 per model (reduced from 10 for speed; K=200 prompts are long)
- Per-call timeout of 300 seconds (skip if model hangs)
- Checkpoint after each model (safe to kill and restart)
- Saves partial results progressively

Purpose: Empirical head-to-head demonstrating that Minerva QS at K=200
floors all open-weight models, making it non-discriminative for this population,
while WMF-AM at K=3/5/7 retains discriminability.
"""

import json
import random
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import call_model, RESULTS_DIR

# ── Config ───────────────────────────────────────────────────────────────────
N_OPS = 200          # Minerva Quantity State spec (K=200)
N_SAMPLES = 3        # per model (reduced for speed; floor effect visible at N=3)
SEED = 42
OP_RANGE = (1, 20)
INIT_RANGE = (10, 100)
CALL_TIMEOUT = 300   # seconds per model call before skipping

# All 20 study models (preregistered set)
ALL_20_MODELS = [
    # Original 7
    "ollama:deepseek-r1:14b",
    "ollama:qwen2.5:32b",
    "ollama:qwen2.5:14b",
    "ollama:gemma2:27b",
    "ollama:qwen2.5:7b",
    "ollama:mistral:7b",
    "ollama:llama3.1:8b",
    # Expansion 8
    "ollama:phi3:14b",
    "ollama:gemma2:9b",
    "ollama:qwen2.5:3b",
    "ollama:llama3.2:3b",
    "ollama:deepseek-r1:7b",
    "ollama:mixtral:8x7b",
    "ollama:command-r:35b",
    "ollama:yi:34b",
    # Small 5
    "ollama:gemma2:2b",
    "ollama:qwen2.5:1.5b",
    "ollama:tinyllama:1.1b",
    "ollama:llama3.2:1b",
    "ollama:qwen2.5:0.5b",
]

# WMF-AM scores from paper (pre-computed)
WMF_AM_SCORES = {
    "ollama:deepseek-r1:14b": 0.983,
    "ollama:qwen2.5:32b":     0.650,
    "ollama:qwen2.5:14b":     0.467,
    "ollama:gemma2:27b":      0.450,
    "ollama:qwen2.5:7b":      0.350,
    "ollama:mistral:7b":      0.350,
    "ollama:llama3.1:8b":     0.183,
    "ollama:phi3:14b":        0.267,
    "ollama:gemma2:9b":       0.400,
    "ollama:qwen2.5:3b":      0.200,
    "ollama:llama3.2:3b":     0.133,
    "ollama:deepseek-r1:7b":  0.150,
    "ollama:mixtral:8x7b":    0.300,
    "ollama:command-r:35b":   0.350,
    "ollama:yi:34b":          0.250,
    "ollama:gemma2:2b":       0.217,
    "ollama:qwen2.5:1.5b":    0.117,
    "ollama:tinyllama:1.1b":  0.117,
    "ollama:llama3.2:1b":     0.067,
    "ollama:qwen2.5:0.5b":    0.050,
}

AGENT_SCORES = {
    "ollama:deepseek-r1:14b": 0.70,
    "ollama:qwen2.5:32b":     0.90,
    "ollama:qwen2.5:14b":     0.90,
    "ollama:gemma2:27b":      0.80,
    "ollama:qwen2.5:7b":      0.90,
    "ollama:mistral:7b":      0.30,
    "ollama:llama3.1:8b":     0.60,
    "ollama:phi3:14b":        0.20,
    "ollama:gemma2:9b":       0.90,
    "ollama:qwen2.5:3b":      0.40,
    "ollama:llama3.2:3b":     0.30,
    "ollama:deepseek-r1:7b":  0.40,
    "ollama:mixtral:8x7b":    0.40,
    "ollama:command-r:35b":   0.70,
    "ollama:yi:34b":          0.30,
    "ollama:gemma2:2b":       0.40,
    "ollama:qwen2.5:1.5b":    0.30,
    "ollama:tinyllama:1.1b":  0.00,
    "ollama:llama3.2:1b":     0.20,
    "ollama:qwen2.5:0.5b":    0.00,
}

# Checkpoint file — saves partial results so we can resume
CHECKPOINT_PATH = RESULTS_DIR / "minerva_qs_h2h_checkpoint.json"


def gen_minerva_qs_problem(rng: random.Random) -> tuple[str, int]:
    """Generate a Minerva Quantity State problem with K=200 operations."""
    init_val = rng.randint(*INIT_RANGE)
    running = init_val
    operations = []
    for i in range(1, N_OPS + 1):
        mag = rng.randint(*OP_RANGE)
        if rng.random() < 0.5:
            running += mag
            operations.append(f"{i}. Add {mag}")
        else:
            running -= mag
            operations.append(f"{i}. Subtract {mag}")
    ops_text = "\n".join(operations)
    prompt = (
        f"Context: Begin with the number {init_val}. "
        f"Perform the following operations:\n{ops_text}\n\n"
        f"Instruction: In the context, you are given an initial number and a "
        f"series of operations to perform on that number. Your task is to "
        f"determine the final result of the operations. "
        f"Write your final answer after the text \"FINAL ANSWER: \". "
        f"For example, \"FINAL ANSWER: 42\".\n\nFINAL ANSWER:"
    )
    return prompt, running


def extract_answer(response: str) -> int | None:
    """Extract final numeric answer from model response."""
    m = re.search(r"FINAL ANSWER:\s*(-?\d+)", response, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", response)
    if nums:
        return int(nums[-1])
    return None


class TimeoutError(Exception):
    pass


def call_with_timeout(model_key: str, prompt: str, timeout: int) -> str | None:
    """Call model with a timeout. Returns None on timeout."""
    result = [None]
    error = [None]

    def _call():
        try:
            result[0] = call_model(model_key, prompt)
        except Exception as e:
            error[0] = e

    import threading
    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None  # timed out
    if error[0]:
        raise error[0]
    return result[0]


def load_checkpoint() -> dict:
    """Load partial results from checkpoint if exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict):
    """Save partial results to checkpoint."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run_minerva_qs_h2h():
    rng = random.Random(SEED)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    print(f"Minerva QS Full Head-to-Head (K={N_OPS}, N={N_SAMPLES}/model, N=20 models)")
    print("=" * 70)
    print(f"Timeout per call: {CALL_TIMEOUT}s | Checkpoint: {CHECKPOINT_PATH}")
    print()

    # Load checkpoint (allows resume after kill)
    results = load_checkpoint()
    if results:
        done_models = list(results.keys())
        print(f"Resuming from checkpoint: {len(done_models)} models already done")
        print(f"  Done: {', '.join(m.split(':',1)[1] for m in done_models)}")
        print()

    for model_key in ALL_20_MODELS:
        if model_key in results:
            print(f"  [SKIP] {model_key.split(':',1)[1]:22s} (already in checkpoint)")
            continue

        model_name = model_key.split(":", 1)[1]
        correct = 0
        trials = []
        timed_out = 0

        print(f"  Running {model_name:22s} ...", end="", flush=True)
        t_model_start = time.time()

        for s in range(N_SAMPLES):
            # Advance the RNG to correct position based on model index + sample
            # (re-create fresh for each to be deterministic regardless of order)
            sample_rng = random.Random(SEED + ALL_20_MODELS.index(model_key) * 1000 + s)
            prompt, expected = gen_minerva_qs_problem(sample_rng)

            try:
                response = call_with_timeout(model_key, prompt, CALL_TIMEOUT)
                if response is None:
                    # Timeout
                    print(f"T", end="", flush=True)
                    timed_out += 1
                    trials.append({"sample": s, "timeout": True, "correct": False})
                    continue

                predicted = extract_answer(response)
                is_correct = (predicted == expected)
                correct += int(is_correct)
                print("." if is_correct else "x", end="", flush=True)
                trials.append({
                    "sample": s,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct,
                })
            except Exception as e:
                print(f"E", end="", flush=True)
                trials.append({"sample": s, "error": str(e)[:200], "correct": False})

        t_elapsed = time.time() - t_model_start
        effective_n = N_SAMPLES - timed_out
        accuracy = correct / N_SAMPLES  # count timeouts as incorrect
        wmf = WMF_AM_SCORES.get(model_key, "N/A")
        agent = AGENT_SCORES.get(model_key, "N/A")

        print(f"  MinervaQS={accuracy:.3f}  WMF={wmf}  Agent={agent}  ({t_elapsed:.0f}s)")

        results[model_key] = {
            "accuracy": accuracy,
            "correct": correct,
            "n_samples": N_SAMPLES,
            "timed_out": timed_out,
            "trials": trials,
        }
        save_checkpoint(results)

    # Save final results
    out_path = RESULTS_DIR / f"minerva_qs_h2h_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": "Minerva Quantity State Full Head-to-Head",
            "K": N_OPS,
            "n_samples": N_SAMPLES,
            "n_models": len(ALL_20_MODELS),
            "seed": SEED,
            "timeout_per_call": CALL_TIMEOUT,
            "timestamp": timestamp,
            "results": results,
            "wmf_am_scores": WMF_AM_SCORES,
            "agent_scores": AGENT_SCORES,
        }, f, indent=2)

    print(f"\nSaved: {out_path}")

    # Summary table
    print("\n--- Summary Table ---")
    print(f"{'Model':<24} {'MinervaQS':>10} {'WMF-AM':>8} {'Agent':>7} {'Floor?':>7}")
    print("-" * 60)
    qs_scores = []
    for mk in ALL_20_MODELS:
        if mk not in results:
            continue
        mn = mk.split(":", 1)[1]
        qs = results[mk]["accuracy"]
        wmf = WMF_AM_SCORES.get(mk, 0)
        ag = AGENT_SCORES.get(mk, 0)
        floor = "YES" if qs == 0.0 else "no"
        qs_scores.append(qs)
        print(f"{mn:<24} {qs:>10.3f} {wmf:>8.3f} {ag:>7.2f} {floor:>7}")

    if qs_scores:
        import scipy.stats
        agent_scores_list = [AGENT_SCORES[mk] for mk in ALL_20_MODELS if mk in results]
        wmf_scores_list = [WMF_AM_SCORES[mk] for mk in ALL_20_MODELS if mk in results]
        qs_list = [results[mk]["accuracy"] for mk in ALL_20_MODELS if mk in results]

        tau_qs, p_qs = scipy.stats.kendalltau(qs_list, agent_scores_list)
        tau_wmf, p_wmf = scipy.stats.kendalltau(wmf_scores_list, agent_scores_list)

        floored = sum(1 for q in qs_list if q == 0.0)
        print(f"\nMinerva QS range: {min(qs_list):.3f}–{max(qs_list):.3f} | floored at 0: {floored}/{len(qs_list)}")
        print(f"WMF-AM range:     {min(wmf_scores_list):.3f}–{max(wmf_scores_list):.3f}")
        print(f"\nKendall τ(MinervaQS, Agent) = {tau_qs:.3f} (p={p_qs:.3f}) — N={len(qs_list)}")
        print(f"Kendall τ(WMF-AM,     Agent) = {tau_wmf:.3f} (p={p_wmf:.3f}) — N={len(wmf_scores_list)}")

        # Append stats to saved file
        with open(out_path) as f:
            saved = json.load(f)
        saved["tau_minerva_qs_agent"] = tau_qs
        saved["p_minerva_qs_agent"] = p_qs
        saved["tau_wmf_am_agent"] = tau_wmf
        saved["p_wmf_am_agent"] = p_wmf
        saved["n_floored"] = floored
        saved["n_evaluated"] = len(qs_list)
        with open(out_path, "w") as f:
            json.dump(saved, f, indent=2)

    # Clean up checkpoint on success
    if CHECKPOINT_PATH.exists() and len(results) == len(ALL_20_MODELS):
        CHECKPOINT_PATH.unlink()
        print("\nCheckpoint cleaned up (all models complete).")

    return out_path


if __name__ == "__main__":
    run_minerva_qs_h2h()
