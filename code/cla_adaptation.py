"""
Experiment CLA: Cognitive Load Adaptation

Three sub-dimensions:
  CLA-DC  Degradation Curve Shape   — cliff-edge vs. graceful degradation
  CLA-RA  Resource Allocation       — does effort scale with difficulty?
  CLA-CR  Capacity Recovery         — does performance recover after max load?

Prior art:
  Intelligence Degradation [2601.15300] — observes degradation (TIER 1 differentiation:
  we characterize *curve shape* as a cognitive-architecture diagnostic, not just magnitude)

Composite: CLA = 0.50 × CLA-DC + 0.30 × CLA-RA + 0.20 × CLA-CR

Usage:
    python cla_adaptation.py --model ollama:qwen2.5:7b --output-dir ../data/results/cla
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from config import call_model, RESULTS_DIR

# ── Shared problem sets ────────────────────────────────────────────────────────

# WMF-style load problems: tracking N values through operations
# Used for CLA-DC (WMF load gradient) and CLA-CR (recovery block)
def _make_wm_problem(n_items: int) -> tuple[str, str]:
    """Return (prompt, correct_answer) for an N-item state tracking problem."""
    import random
    entities = [f"item_{chr(65+i)}" for i in range(n_items)]
    values = {e: random.randint(10, 99) for e in entities}
    ops = []
    current = dict(values)
    for _ in range(min(n_items, 6)):
        e = random.choice(entities)
        delta = random.choice([-5, -3, -2, 2, 3, 5])
        current[e] += delta
        ops.append(f"  {e} changes by {delta:+d}")
    target = random.choice(entities)
    answer = str(current[target])
    init_str = ", ".join(f"{e}={v}" for e, v in values.items())
    prompt = (
        f"Initial values: {init_str}\n"
        f"Operations (apply in order):\n" + "\n".join(ops) + "\n\n"
        f"What is the final value of {target}? Reply with just the number."
    )
    return prompt, answer


# Difficulty-graded reasoning problems for CLA-RA
DIFFICULTY_PROBLEMS = [
    # Level 1: trivial arithmetic
    ("What is 7 + 5?", "12"),
    ("A box has 3 apples. You add 4. How many?", "7"),
    # Level 2: simple word problems
    ("Alice has twice as many books as Bob. Bob has 6. How many does Alice have?", "12"),
    ("A train travels 60 km/h for 2 hours. How far?", "120"),
    # Level 3: multi-step
    ("If 3x + 7 = 22, what is x?", "5"),
    ("A store gives 20% off an item priced at $45. What is the sale price?", "36"),
    # Level 4: harder multi-step
    (
        "A tank is 40% full. After adding 120 litres it is 70% full. "
        "What is the tank's total capacity in litres?",
        "400",
    ),
    (
        "A rectangle has perimeter 56 cm. Its length is 4 cm more than its width. "
        "What is the area?",
        "180",
    ),
    # Level 5: complex / multi-constraint
    (
        "Three pipes fill a tank. Pipe A fills it in 6 hours, B in 8 hours, C drains it in 12 hours. "
        "All three open simultaneously. How many hours to fill the tank? "
        "Give answer as a fraction (e.g. '24/5') or decimal rounded to 2 places.",
        "24/5",
    ),
    (
        "A sequence: 2, 6, 12, 20, 30, ... What is the 10th term? "
        "Hint: differences between terms are 4, 6, 8, 10, ...",
        "110",
    ),
]

DIFFICULTY_LEVELS = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]  # one label per problem

# ── CLA-DC: Degradation Curve Shape ──────────────────────────────────────────

def run_cla_dc(model_name: str, n_levels: list[int] = None, reps: int = 3) -> dict:
    """
    Run CLA-DC: measure accuracy at each WMF load level (n_items).
    Returns load curve and degradation signature metrics.

    n_levels: list of N-item counts to test (default [3, 5, 7, 10, 14])
    reps: repetitions per level (for reliability)
    """
    if n_levels is None:
        n_levels = [3, 5, 7, 10, 14]

    import random
    random.seed(42)

    print(f"[CLA-DC] Model: {model_name}  levels: {n_levels}  reps: {reps}")
    load_curve_raw: dict[int, list[bool]] = {n: [] for n in n_levels}

    for n in n_levels:
        for rep in range(reps):
            prompt, answer = _make_wm_problem(n)
            messages = [{"role": "user", "content": prompt}]
            try:
                response = call_model(model_name, messages)
                correct = answer.strip() in response.strip()
                load_curve_raw[n].append(correct)
                print(f"  N={n} rep={rep+1}: {'✓' if correct else '✗'}  (ans={answer}, got={response.strip()[:20]})")
            except Exception as e:
                print(f"  N={n} rep={rep+1}: ERROR — {e}")
                load_curve_raw[n].append(False)
            time.sleep(0.3)

    # Aggregate
    load_curve = {n: float(np.mean(v)) for n, v in load_curve_raw.items()}
    accs = [load_curve[n] for n in sorted(n_levels)]

    # Degradation metrics
    diffs = np.diff(accs)
    smoothness = float(np.std(diffs))          # low = graceful, high = cliff-edge
    slope = float(np.polyfit(n_levels, accs, 1)[0]) if len(n_levels) >= 2 else 0.0
    floor = float(min(accs))
    breakpoint_n = None
    for i, d in enumerate(diffs):
        if d < -0.3:                           # drop > 30% in one step = cliff
            breakpoint_n = n_levels[i + 1]
            break

    pattern = "graceful" if smoothness < 0.15 else "cliff-edge"

    result = {
        "sub_dim": "CLA-DC",
        "load_curve": load_curve,
        "smoothness_score": round(smoothness, 4),  # lower is better (more graceful)
        "slope": round(slope, 6),
        "performance_floor": round(floor, 4),
        "breakpoint_n": breakpoint_n,
        "degradation_pattern": pattern,
        "CLA_DC_score": round(1.0 - min(smoothness, 1.0), 4),  # higher = more graceful
    }
    print(f"  CLA-DC: pattern={pattern}, smoothness={smoothness:.4f}, slope={slope:.4f}")
    return result


# ── CLA-RA: Resource Allocation ───────────────────────────────────────────────

_HEDGING_PATTERNS = re.compile(
    r"\b(let me think|this is complex|carefully|hmm|wait|let's see|"
    r"consider|step[- ]by[- ]step|first|second|third|therefore|thus|"
    r"however|but|although)\b",
    re.IGNORECASE,
)

def _count_resource_proxies(response: str) -> dict:
    tokens = len(response.split())
    hedges = len(_HEDGING_PATTERNS.findall(response))
    # Simple revision count: look for self-correction markers
    revisions = len(re.findall(r"\b(actually|wait|correction|revised|no,)\b", response, re.I))
    return {"response_tokens": tokens, "hedging_markers": hedges, "revisions": revisions}


def run_cla_ra(model_name: str) -> dict:
    """
    Run CLA-RA: measure whether the model allocates more response resources
    (tokens, hedging, revisions) to harder problems.
    """
    print(f"[CLA-RA] Model: {model_name}")
    results = []

    for (problem, answer), level in zip(DIFFICULTY_PROBLEMS, DIFFICULTY_LEVELS):
        messages = [
            {
                "role": "user",
                "content": (
                    "Please solve the following problem. "
                    "Show your reasoning, then give the final answer.\n\n" + problem
                ),
            }
        ]
        try:
            response = call_model(model_name, messages)
            proxies = _count_resource_proxies(response)
            correct = answer.strip().lower() in response.lower()
            results.append({
                "level": level,
                "correct": correct,
                "tokens": proxies["response_tokens"],
                "hedges": proxies["hedging_markers"],
                "revisions": proxies["revisions"],
            })
            print(f"  level={level} tokens={proxies['response_tokens']} correct={correct}")
        except Exception as e:
            print(f"  level={level}: ERROR — {e}")
        time.sleep(0.3)

    if not results:
        return {"sub_dim": "CLA-RA", "error": "no results", "CLA_RA_score": float("nan")}

    levels = [r["level"] for r in results]
    tokens = [r["tokens"] for r in results]

    # Pearson r(difficulty, response length)
    if len(set(levels)) > 1:
        r_tokens = float(np.corrcoef(levels, tokens)[0, 1])
    else:
        r_tokens = float("nan")

    result = {
        "sub_dim": "CLA-RA",
        "n_problems": len(results),
        "accuracy_by_level": {
            str(lv): float(np.mean([r["correct"] for r in results if r["level"] == lv]))
            for lv in sorted(set(levels))
        },
        "r_difficulty_tokens": round(r_tokens, 4),
        "CLA_RA_score": round(max(r_tokens, 0.0), 4),  # clip to [0,1]; negative = bad allocation
    }
    print(f"  CLA-RA: r(difficulty, tokens)={r_tokens:.4f}")
    return result


# ── CLA-CR: Capacity Recovery ─────────────────────────────────────────────────

def run_cla_cr(model_name: str, reps: int = 5) -> dict:
    """
    Run CLA-CR: measure whether performance on easy problems recovers
    after a high-load block.

    Block 1: Easy (N=3 state tracking) × reps
    Block 2: Hard (N=14 state tracking) × reps
    Block 3: Easy (N=3) × reps  ← recovery measured here
    """
    import random
    random.seed(99)

    print(f"[CLA-CR] Model: {model_name}  reps={reps}")

    def run_block(n: int, label: str) -> float:
        hits = []
        for i in range(reps):
            prompt, answer = _make_wm_problem(n)
            messages = [{"role": "user", "content": prompt}]
            try:
                response = call_model(model_name, messages)
                correct = answer.strip() in response.strip()
                hits.append(correct)
                print(f"  {label} rep={i+1}: {'✓' if correct else '✗'}")
            except Exception as e:
                print(f"  {label} rep={i+1}: ERROR — {e}")
                hits.append(False)
            time.sleep(0.3)
        return float(np.mean(hits))

    acc_block1 = run_block(3, "B1(easy)")
    acc_block2 = run_block(14, "B2(hard)")
    acc_block3 = run_block(3, "B3(recovery)")

    recovery_rate = acc_block3 / acc_block1 if acc_block1 > 0 else float("nan")
    deficit = acc_block1 - acc_block3

    result = {
        "sub_dim": "CLA-CR",
        "block1_easy_acc": round(acc_block1, 4),
        "block2_hard_acc": round(acc_block2, 4),
        "block3_recovery_acc": round(acc_block3, 4),
        "recovery_rate": round(recovery_rate, 4) if not np.isnan(recovery_rate) else None,
        "performance_deficit": round(deficit, 4),
        "CLA_CR_score": round(min(recovery_rate, 1.0), 4) if not np.isnan(recovery_rate) else float("nan"),
    }
    print(f"  CLA-CR: B1={acc_block1:.3f} B2={acc_block2:.3f} B3={acc_block3:.3f} "
          f"recovery={recovery_rate:.3f}")
    return result


# ── Composite ─────────────────────────────────────────────────────────────────

def compute_cla_composite(dc: dict, ra: dict, cr: dict) -> float:
    """CLA = 0.50 × CLA-DC + 0.30 × CLA-RA + 0.20 × CLA-CR"""
    dc_score = dc.get("CLA_DC_score", float("nan"))
    ra_score = ra.get("CLA_RA_score", float("nan"))
    cr_score = cr.get("CLA_CR_score", float("nan"))
    scores = [(0.50, dc_score), (0.30, ra_score), (0.20, cr_score)]
    valid = [(w, s) for w, s in scores if not np.isnan(s)]
    if not valid:
        return float("nan")
    total_w = sum(w for w, _ in valid)
    return sum(w * s for w, s in valid) / total_w


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLA-DC, CLA-RA, CLA-CR experiments.")
    parser.add_argument("--model", required=True, help="Model name (e.g. ollama:qwen2.5:7b)")
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="Output directory (default: RESULTS_DIR/cla/<model>)")
    parser.add_argument("--skip-dc", action="store_true")
    parser.add_argument("--skip-ra", action="store_true")
    parser.add_argument("--skip-cr", action="store_true")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per load level")
    args = parser.parse_args()

    model_slug = args.model.replace(":", "_").replace("/", "_")
    out_dir = args.output_dir or Path(RESULTS_DIR) / "cla" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"model": args.model}

    if not args.skip_dc:
        dc = run_cla_dc(args.model, reps=args.reps)
        all_results["CLA-DC"] = dc
        (out_dir / "cla_dc.json").write_text(json.dumps(dc, indent=2))

    if not args.skip_ra:
        ra = run_cla_ra(args.model)
        all_results["CLA-RA"] = ra
        (out_dir / "cla_ra.json").write_text(json.dumps(ra, indent=2))

    if not args.skip_cr:
        cr = run_cla_cr(args.model, reps=args.reps)
        all_results["CLA-CR"] = cr
        (out_dir / "cla_cr.json").write_text(json.dumps(cr, indent=2))

    # Composite
    dc = all_results.get("CLA-DC", {})
    ra = all_results.get("CLA-RA", {})
    cr = all_results.get("CLA-CR", {})
    composite = compute_cla_composite(dc, ra, cr)
    all_results["CLA_composite"] = round(composite, 4) if not np.isnan(composite) else None

    scores_summary = {
        "model": args.model,
        "CLA_composite": all_results["CLA_composite"],
        "CLA-DC": dc.get("CLA_DC_score"),
        "CLA-RA": ra.get("CLA_RA_score"),
        "CLA-CR": cr.get("CLA_CR_score"),
        "degradation_pattern": dc.get("degradation_pattern"),
        "load_curve": dc.get("load_curve"),
        "recovery_rate": cr.get("recovery_rate"),
        "r_difficulty_tokens": ra.get("r_difficulty_tokens"),
    }
    (out_dir / "scores.json").write_text(json.dumps(scores_summary, indent=2))
    (out_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))

    print(f"\n=== CLA RESULTS: {args.model} ===")
    print(f"  CLA-DC score:  {dc.get('CLA_DC_score', 'N/A'):.4f}  "
          f"(pattern: {dc.get('degradation_pattern', 'N/A')})")
    print(f"  CLA-RA score:  {ra.get('CLA_RA_score', 'N/A'):.4f}  "
          f"(r_tokens: {ra.get('r_difficulty_tokens', 'N/A')})")
    print(f"  CLA-CR score:  {cr.get('CLA_CR_score', 'N/A'):.4f}  "
          f"(recovery_rate: {cr.get('recovery_rate', 'N/A')})")
    print(f"  CLA composite: {composite:.4f}" if not np.isnan(composite) else "  CLA composite: N/A")
    print(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
