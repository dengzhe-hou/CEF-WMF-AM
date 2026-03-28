"""
Experiment E2: Metacognitive Calibration (MCC)

Two novel sub-dimensions (CEF contributions):
  MCC-MA  Metacognitive Monitoring Accuracy  (error prediction)
  MCC-CE  Metacognitive Control Efficacy    (correction after flagging)

Background data only (NOT a novel contribution — saturated by prior work):
  MCC-CC  Confidence Calibration / ECE      (Kadavath et al. 2022 and 5+ replications)
          Collected for descriptive context; excluded from MCC composite score.

Usage:
    python metacognitive_calibration.py --model gpt-4o --n-problems 50
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from config import MODELS, RESULTS_DIR, call_model

# ── Problem banks ────────────────────────────────────────────────────────────
# Problems are loaded from data/tasks/; here we include inline examples for testing.
# In production, load from MMLU, MATH dataset, or OOD synthetic bank.

EXAMPLE_PROBLEMS = [
    # (question, correct_answer, domain, difficulty)
    ("What is the capital of France?", "Paris", "geography", "easy"),
    ("What is 17 × 13?", "221", "math", "easy"),
    ("Who wrote 'Crime and Punishment'?", "Dostoevsky", "literature", "medium"),
    ("What is the chemical symbol for Tungsten?", "W", "chemistry", "medium"),
    ("What is the integral of sin(x) dx?", "-cos(x) + C", "math", "hard"),
    ("In what year did the Byzantine Empire fall?", "1453", "history", "medium"),
    ("What is the airspeed velocity of an unladen swallow?", "unknown", "trick", "hard"),
    ("What is the square root of 144?", "12", "math", "easy"),
    ("Which planet has the most moons?", "Saturn", "astronomy", "medium"),
    ("What is Avogadro's number?", "6.022e23", "chemistry", "medium"),
    ("Who developed the general theory of relativity?", "Einstein", "physics", "easy"),
    ("What is the derivative of e^x?", "e^x", "math", "easy"),
    ("What is the boiling point of water at sea level in Celsius?", "100", "physics", "easy"),
    ("Which element has atomic number 79?", "Gold", "chemistry", "medium"),
    ("What is the sum of angles in a triangle?", "180", "math", "easy"),
    ("Who painted the Sistine Chapel ceiling?", "Michelangelo", "art", "easy"),
    ("What is the speed of light in m/s?", "3e8", "physics", "medium"),
    ("In which year was the Magna Carta signed?", "1215", "history", "medium"),
    ("What is the Pythagorean theorem?", "a^2+b^2=c^2", "math", "easy"),
    ("What is the largest planet in the solar system?", "Jupiter", "astronomy", "easy"),
]


def load_problems(n: int = 50, task_dir: Path = None) -> list[dict]:
    """Load problems from task bank or fall back to inline examples."""
    if task_dir and (task_dir / "mcc_problems.json").exists():
        with open(task_dir / "mcc_problems.json") as f:
            all_problems = json.load(f)
    else:
        all_problems = [
            {"question": q, "answer": a, "domain": d, "difficulty": diff}
            for q, a, d, diff in EXAMPLE_PROBLEMS
        ]
    # Cycle if n > available
    problems = []
    while len(problems) < n:
        problems.extend(all_problems)
    return problems[:n]


# ── MCC-CC: Confidence Calibration ──────────────────────────────────────────

CONFIDENCE_PROMPT = """\
Answer the following question, then provide your confidence that your answer is correct.

Question: {question}

Respond EXACTLY in this format (no other text):
Answer: [your answer]
Confidence: [0-100, where 0=completely uncertain, 100=completely certain]"""


def _parse_cc_response(response: str) -> tuple[str, int]:
    """Extract answer and confidence from model response."""
    answer = ""
    confidence = 50  # default if parsing fails

    for line in response.split("\n"):
        line = line.strip()
        if line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
        elif line.lower().startswith("confidence:"):
            nums = re.findall(r"\d+", line)
            if nums:
                confidence = min(100, max(0, int(nums[0])))

    return answer, confidence


def _check_answer_correct(predicted: str, correct: str) -> bool:
    """Flexible answer matching."""
    p = predicted.lower().strip().rstrip(".").strip()
    c = correct.lower().strip().rstrip(".").strip()
    # Exact match or containment
    return c in p or p in c or p == c


def run_mcc_cc(model_name: str, problems: list[dict]) -> list[dict]:
    """Run MCC-CC: Confidence calibration across problems."""
    results = []
    for prob in problems:
        prompt = CONFIDENCE_PROMPT.format(question=prob["question"])
        response = call_model(model_name, prompt)
        answer, confidence = _parse_cc_response(response)
        is_correct = _check_answer_correct(answer, prob["answer"])

        results.append({
            "sub_dim": "MCC-CC",
            "model": model_name,
            "question": prob["question"],
            "correct_answer": prob["answer"],
            "predicted_answer": answer,
            "is_correct": int(is_correct),
            "confidence": confidence,
            "domain": prob.get("domain", "unknown"),
            "difficulty": prob.get("difficulty", "unknown"),
        })
        time.sleep(0.4)
    return results


def compute_ece(results_cc: list[dict], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    confidences = np.array([r["confidence"] / 100.0 for r in results_cc])
    accuracies = np.array([r["is_correct"] for r in results_cc], dtype=float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidences >= lo) & (confidences < hi)
        if in_bin.sum() == 0:
            continue
        acc_in_bin = accuracies[in_bin].mean()
        conf_in_bin = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc_in_bin - conf_in_bin)

    return float(ece)


# ── MCC-MA: Metacognitive Monitoring Accuracy ────────────────────────────────

MONITORING_PROMPT = """\
You have just answered 20 questions. Here are the questions and your answers:

{qa_list}

Now, WITHOUT changing your answers, predict which ones you got wrong.
List the question numbers you believe you answered INCORRECTLY.

Respond in this format:
Wrong question numbers: [list the numbers, e.g., "3, 7, 12" — or "None" if you think all are correct]"""


def run_mcc_ma(model_name: str, problems: list[dict], batch_size: int = 20) -> list[dict]:
    """
    Run MCC-MA: After answering a batch, predict which were wrong.
    Returns per-problem monitoring signals.
    """
    results = []

    # First: get answers (reuse MCC-CC results if available)
    answered = []
    for prob in problems[:batch_size]:
        simple_prompt = f"Answer this question in one short phrase:\n{prob['question']}"
        response = call_model(model_name, simple_prompt).strip()
        is_correct = _check_answer_correct(response, prob["answer"])
        answered.append({
            "question": prob["question"],
            "answer": response,
            "is_correct": is_correct,
            "correct_answer": prob["answer"],
        })
        time.sleep(0.3)

    # Build Q&A list for monitoring prompt
    qa_lines = "\n".join(
        f"{i+1}. Q: {a['question']}\n   A: {a['answer']}"
        for i, a in enumerate(answered)
    )

    monitoring_prompt = MONITORING_PROMPT.format(qa_list=qa_lines)
    response = call_model(model_name, monitoring_prompt)

    # Parse predicted wrong items
    predicted_wrong_set = set()
    match = re.search(r"Wrong question numbers:\s*(.*)", response, re.IGNORECASE)
    if match:
        nums_str = match.group(1)
        if "none" not in nums_str.lower():
            predicted_wrong_set = {int(n) for n in re.findall(r"\d+", nums_str)}

    # Compute monitoring accuracy for each problem
    for i, a in enumerate(answered):
        q_num = i + 1
        predicted_wrong = q_num in predicted_wrong_set
        actually_wrong = not a["is_correct"]
        results.append({
            "sub_dim": "MCC-MA",
            "model": model_name,
            "question_num": q_num,
            "is_correct": int(a["is_correct"]),
            "predicted_wrong": int(predicted_wrong),
            "actually_wrong": int(actually_wrong),
            "tp": int(predicted_wrong and actually_wrong),
            "fp": int(predicted_wrong and not actually_wrong),
            "fn": int(not predicted_wrong and actually_wrong),
            "tn": int(not predicted_wrong and not actually_wrong),
        })

    return results


def compute_monitoring_accuracy(results_ma: list[dict]) -> float:
    """Compute Pearson r between predicted and actual errors."""
    predicted = np.array([r["predicted_wrong"] for r in results_ma], dtype=float)
    actual = np.array([r["actually_wrong"] for r in results_ma], dtype=float)
    if predicted.std() == 0 or actual.std() == 0:
        return 0.0
    r, _ = pearsonr(predicted, actual)
    return float(r)


# ── MCC-CE: Metacognitive Control Efficacy ──────────────────────────────────

REVISION_PROMPT = """\
You previously answered these questions:

{qa_list}

You may revise any answers you are UNCERTAIN about.
First, list the question numbers you want to reconsider (or "None").
Then, for each reconsidered question, provide your revised answer.

Format:
Questions to revise: [e.g., "2, 5" or "None"]
Revisions:
[number]. [revised answer]
..."""


def run_mcc_ce(model_name: str, problems: list[dict], batch_size: int = 15) -> list[dict]:
    """
    Run MCC-CE: Present answers; ask which to revise; measure improvement.
    """
    results = []

    # Get initial answers
    answered = []
    for prob in problems[:batch_size]:
        simple_prompt = f"Answer this question in one short phrase:\n{prob['question']}"
        response = call_model(model_name, simple_prompt).strip()
        is_correct = _check_answer_correct(response, prob["answer"])
        answered.append({
            "question": prob["question"],
            "initial_answer": response,
            "is_correct_initial": is_correct,
            "correct_answer": prob["answer"],
        })
        time.sleep(0.3)

    # Build Q&A list
    qa_lines = "\n".join(
        f"{i+1}. Q: {a['question']}\n   A: {a['initial_answer']}"
        for i, a in enumerate(answered)
    )

    revision_prompt = REVISION_PROMPT.format(qa_list=qa_lines)
    response = call_model(model_name, revision_prompt)
    time.sleep(0.5)

    # Parse which questions were flagged for revision
    flagged = set()
    match_flag = re.search(r"Questions to revise:\s*(.*)", response, re.IGNORECASE)
    if match_flag:
        flag_str = match_flag.group(1)
        if "none" not in flag_str.lower():
            flagged = {int(n) for n in re.findall(r"\d+", flag_str)}

    # Parse revised answers
    revised_answers = {}
    for m in re.finditer(r"(\d+)\.\s+(.+)", response.split("Revisions:")[-1]):
        q_num = int(m.group(1))
        revised_answers[q_num] = m.group(2).strip()

    for i, a in enumerate(answered):
        q_num = i + 1
        was_flagged = q_num in flagged
        was_wrong = not a["is_correct_initial"]

        if was_flagged and q_num in revised_answers:
            revised_ans = revised_answers[q_num]
            is_correct_revised = _check_answer_correct(revised_ans, a["correct_answer"])
        else:
            revised_ans = a["initial_answer"]
            is_correct_revised = a["is_correct_initial"]

        results.append({
            "sub_dim": "MCC-CE",
            "model": model_name,
            "question_num": q_num,
            "was_flagged": int(was_flagged),
            "was_wrong": int(was_wrong),
            "is_correct_initial": int(a["is_correct_initial"]),
            "is_correct_revised": int(is_correct_revised),
            "improved": int(was_wrong and is_correct_revised),
        })

    return results


def compute_control_efficacy(results_ce: list[dict]) -> dict:
    """Compute MCC-CE metrics."""
    flagged_wrong = [r for r in results_ce if r["was_flagged"] and r["was_wrong"]]
    flagged_any = [r for r in results_ce if r["was_flagged"]]
    wrong_any = [r for r in results_ce if r["was_wrong"]]

    flagging_rate = len([r for r in results_ce if r["was_flagged"] and r["was_wrong"]]) / max(len(wrong_any), 1)
    correction_efficacy = sum(r["improved"] for r in flagged_wrong) / max(len(flagged_wrong), 1)
    false_alarm_rate = len([r for r in flagged_any if not r["was_wrong"]]) / max(len([r for r in results_ce if not r["was_wrong"]]), 1)

    return {
        "flagging_rate": round(flagging_rate, 4),
        "correction_efficacy": round(correction_efficacy, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "mcc_ce_score": round(flagging_rate * correction_efficacy, 4),
    }


# ── Composite MCC Score ──────────────────────────────────────────────────────

def compute_mcc_score(
    monitoring_r: float,
    control_eff: float,
    ece: float = None,  # collected for background/descriptive purposes only
) -> float:
    """Compute MCC composite score (MA + CE only; ECE excluded as non-novel)."""
    # monitoring_r can be negative; clamp to [0,1]
    mcc_ma = max(0.0, monitoring_r)
    mcc_ce = max(0.0, control_eff)
    return round(0.55 * mcc_ma + 0.45 * mcc_ce, 4)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MCC experiments.")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--n-problems", type=int, default=50)
    args = parser.parse_args()

    from config import TASKS_DIR
    problems = load_problems(args.n_problems, TASKS_DIR)

    print(f"Running MCC-CC for {args.model} ({len(problems)} problems)...")
    print("  (MCC-CC/ECE is background data only; not included in composite score)")
    cc_results = run_mcc_cc(args.model, problems)
    ece = compute_ece(cc_results)

    print(f"Running MCC-MA for {args.model}...")
    ma_results = run_mcc_ma(args.model, problems[:20])
    monitoring_r = compute_monitoring_accuracy(ma_results)

    print(f"Running MCC-CE for {args.model}...")
    ce_results = run_mcc_ce(args.model, problems[:15])
    ce_metrics = compute_control_efficacy(ce_results)

    composite = compute_mcc_score(monitoring_r, ce_metrics["mcc_ce_score"], ece=ece)

    # Save results
    out_dir = RESULTS_DIR / "mcc" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "cc_results.jsonl", "w") as f:
        for r in cc_results:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "ma_results.jsonl", "w") as f:
        for r in ma_results:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "ce_results.jsonl", "w") as f:
        for r in ce_results:
            f.write(json.dumps(r) + "\n")

    scores = {
        "ECE_background": round(ece, 4),          # background only — saturated prior art
        "MCC-MA (monitoring_r)": round(monitoring_r, 4),
        "MCC-CE": ce_metrics,
        "MCC_composite": composite,               # MA + CE only
    }
    with open(out_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\nMCC Results for {args.model}:")
    print(f"  ECE (background): {ece:.4f}  (lower is better; cited as prior work only)")
    print(f"  MCC-MA (r)      : {monitoring_r:.4f}")
    print(f"  Flagging rate   : {ce_metrics['flagging_rate']:.4f}")
    print(f"  Correction eff  : {ce_metrics['correction_efficacy']:.4f}")
    print(f"  False alarm     : {ce_metrics['false_alarm_rate']:.4f}")
    print(f"  COMPOSITE       : {composite:.4f}  (MA + CE only)")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
