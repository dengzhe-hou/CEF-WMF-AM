"""
minerva_qs_comparison.py

Empirical comparison: Minerva Quantity State (K=200) vs WMF-AM (K=3/5/7)
on the original 7 Ollama models.

Purpose: Demonstrate that Minerva QS at K=200 floors for open-weight models
(~0 accuracy), making it non-discriminative as a predictor for this population,
while WMF-AM at K=3/5/7 provides the necessary discriminability.

Protocol: Follows Minerva ICML 2025 Quantity State specification:
- 200 sequential add/subtract operations
- Initial integer in [10, 100]
- Each operation: add or subtract a random integer in [1, 20]
- Scoring: exact match on final value
- 10 samples per model
"""

import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import call_model, RESULTS_DIR

# ── Config ──────────────────────────────────────────────────────────────────
N_OPS = 200          # Minerva Quantity State spec
N_SAMPLES = 10       # per model
SEED = 42
OP_RANGE = (1, 20)   # operation magnitude range
INIT_RANGE = (10, 100)

MODELS_TO_TEST = [
    "ollama:deepseek-r1:14b",
    "ollama:qwen2.5:32b",
    "ollama:qwen2.5:14b",
    "ollama:gemma2:27b",
    "ollama:qwen2.5:7b",
    "ollama:mistral:7b",
    "ollama:llama3.1:8b",
]

# WMF-AM results for these 7 models (4-seed means from paper)
WMF_AM_SCORES = {
    "ollama:deepseek-r1:14b": 0.983,
    "ollama:qwen2.5:32b":     0.650,
    "ollama:qwen2.5:14b":     0.467,
    "ollama:gemma2:27b":      0.450,
    "ollama:qwen2.5:7b":      0.350,
    "ollama:mistral:7b":      0.350,
    "ollama:llama3.1:8b":     0.183,
}

AGENT_SCORES = {
    "ollama:deepseek-r1:14b": 0.70,
    "ollama:qwen2.5:32b":     0.90,
    "ollama:qwen2.5:14b":     0.90,
    "ollama:gemma2:27b":      0.80,
    "ollama:qwen2.5:7b":      0.90,
    "ollama:mistral:7b":      0.30,
    "ollama:llama3.1:8b":     0.60,
}


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
        f"Write your final answer after the text \"FINAL ANSWER:\". "
        f"For example, \"FINAL ANSWER: 42\".\n\nFINAL ANSWER:"
    )
    return prompt, running


def extract_answer(response: str) -> int | None:
    """Extract final numeric answer from model response."""
    # Look for FINAL ANSWER: <number>
    m = re.search(r"FINAL ANSWER:\s*(-?\d+)", response, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: last integer in response
    nums = re.findall(r"-?\d+", response)
    if nums:
        return int(nums[-1])
    return None


def run_minerva_qs():
    rng = random.Random(SEED)
    results = {}
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    print(f"Minerva QS (K={N_OPS}) — {N_SAMPLES} samples per model")
    print("=" * 60)

    for model_key in MODELS_TO_TEST:
        model_name = model_key.split(":", 1)[1]
        correct = 0
        trials = []

        for s in range(N_SAMPLES):
            prompt, expected = gen_minerva_qs_problem(rng)
            try:
                response = call_model(model_key, prompt)
                predicted = extract_answer(response)
                is_correct = (predicted == expected)
                correct += int(is_correct)
                trials.append({
                    "sample": s,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct,
                })
            except Exception as e:
                print(f"  Error on {model_name} sample {s}: {e}")
                trials.append({"sample": s, "error": str(e), "correct": False})

        accuracy = correct / N_SAMPLES
        results[model_key] = {
            "accuracy": accuracy,
            "correct": correct,
            "n_samples": N_SAMPLES,
            "trials": trials,
        }
        wmf = WMF_AM_SCORES.get(model_key, "N/A")
        agent = AGENT_SCORES.get(model_key, "N/A")
        print(f"  {model_name:20s}  MinervaQS={accuracy:.3f}  WMF-AM={wmf}  Agent={agent}")

    # Save results
    out_path = RESULTS_DIR / f"minerva_qs_comparison_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": "Minerva Quantity State",
            "K": N_OPS,
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "timestamp": timestamp,
            "results": results,
            "wmf_am_scores": WMF_AM_SCORES,
            "agent_scores": AGENT_SCORES,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Quick summary
    print("\n--- Summary ---")
    print(f"{'Model':<22} {'MinervaQS':>10} {'WMF-AM':>8} {'Agent':>7}")
    print("-" * 52)
    for mk in MODELS_TO_TEST:
        mn = mk.split(":", 1)[1]
        qs = results[mk]["accuracy"]
        wmf = WMF_AM_SCORES[mk]
        ag = AGENT_SCORES[mk]
        print(f"{mn:<22} {qs:>10.3f} {wmf:>8.3f} {ag:>7.2f}")

    # Compute Kendall tau for Minerva QS vs Agent
    import scipy.stats
    qs_scores = [results[mk]["accuracy"] for mk in MODELS_TO_TEST]
    agent_scores_list = [AGENT_SCORES[mk] for mk in MODELS_TO_TEST]
    tau_qs, p_qs = scipy.stats.kendalltau(qs_scores, agent_scores_list)

    wmf_scores_list = [WMF_AM_SCORES[mk] for mk in MODELS_TO_TEST]
    tau_wmf, p_wmf = scipy.stats.kendalltau(wmf_scores_list, agent_scores_list)

    print(f"\nKendall τ(MinervaQS, Agent) = {tau_qs:.3f} (p={p_qs:.3f})")
    print(f"Kendall τ(WMF-AM,     Agent) = {tau_wmf:.3f} (p={p_wmf:.3f})")
    print(f"\nMinerva QS score range: {min(qs_scores):.3f}–{max(qs_scores):.3f}")
    print(f"WMF-AM score range:     {min(wmf_scores_list):.3f}–{max(wmf_scores_list):.3f}")

    # Append tau to saved results
    with open(out_path) as f:
        saved = json.load(f)
    saved["tau_minerva_qs_agent"] = tau_qs
    saved["p_minerva_qs_agent"] = p_qs
    saved["tau_wmf_am_agent"] = tau_wmf
    saved["p_wmf_am_agent"] = p_wmf
    with open(out_path, "w") as f:
        json.dump(saved, f, indent=2)

    return out_path


if __name__ == "__main__":
    run_minerva_qs()
