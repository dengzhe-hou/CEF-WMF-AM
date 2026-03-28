"""
CEF Expanded Completion Battery v2

Expands from 20 items (EXAMPLE_PROBLEMS in metacognitive_calibration.py)
to 100 items across 5 difficulty levels and 5 domains.

Scoring: exact-match (case-insensitive, strip whitespace).

Usage:
    python cef_completion_battery_v2.py --models all-15
    python cef_completion_battery_v2.py --models ollama:qwen2.5:7b --n-items 50
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS, RESULTS_DIR, call_model

ORIGINAL_7 = [
    "ollama:qwen2.5:7b", "ollama:qwen2.5:14b", "ollama:qwen2.5:32b",
    "ollama:llama3.1:8b", "ollama:gemma2:27b", "ollama:deepseek-r1:14b",
    "ollama:mistral:7b",
]
EXPANSION_8 = [
    "ollama:phi3:14b", "ollama:gemma2:9b", "ollama:qwen2.5:3b",
    "ollama:llama3.2:3b", "ollama:deepseek-r1:7b", "ollama:mixtral:8x7b",
    "ollama:command-r:35b", "ollama:yi:34b",
]
ALL_15 = ORIGINAL_7 + EXPANSION_8

# 100-item battery: (question, answer, domain, difficulty)
# Domains: factual, math, science, reasoning, language
# Difficulty: 1=easy, 2=medium, 3=hard, 4=expert, 5=extreme
COMPLETION_BATTERY = [
    # ── Factual (20 items) ──
    ("What is the capital of France?", "Paris", "factual", 1),
    ("What is the largest planet in our solar system?", "Jupiter", "factual", 1),
    ("Who wrote Romeo and Juliet?", "Shakespeare", "factual", 1),
    ("What is the chemical symbol for gold?", "Au", "factual", 1),
    ("What year did World War II end?", "1945", "factual", 1),
    ("What is the tallest mountain on Earth?", "Mount Everest", "factual", 2),
    ("What is the smallest country in the world by area?", "Vatican City", "factual", 2),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "factual", 2),
    ("What is the speed of light in km/s (approximate)?", "300000", "factual", 2),
    ("What element has atomic number 1?", "Hydrogen", "factual", 2),
    ("What is the capital of Mongolia?", "Ulaanbaatar", "factual", 3),
    ("Who discovered penicillin?", "Alexander Fleming", "factual", 3),
    ("What is the deepest ocean trench?", "Mariana Trench", "factual", 3),
    ("In what year was the Treaty of Westphalia signed?", "1648", "factual", 3),
    ("What is the longest river in Africa?", "Nile", "factual", 3),
    ("Who was the first woman to win a Nobel Prize?", "Marie Curie", "factual", 4),
    ("What is the capital of Bhutan?", "Thimphu", "factual", 4),
    ("Who wrote 'The Wealth of Nations'?", "Adam Smith", "factual", 4),
    ("What is the hardest natural mineral?", "Diamond", "factual", 2),
    ("What planet is closest to the Sun?", "Mercury", "factual", 1),

    # ── Math (20 items) ──
    ("What is 7 × 8?", "56", "math", 1),
    ("What is 144 ÷ 12?", "12", "math", 1),
    ("What is the square root of 169?", "13", "math", 1),
    ("What is 15% of 200?", "30", "math", 2),
    ("What is 2^10?", "1024", "math", 2),
    ("What is the sum of the first 10 positive integers?", "55", "math", 2),
    ("What is the value of pi to 2 decimal places?", "3.14", "math", 2),
    ("What is the GCD of 48 and 36?", "12", "math", 2),
    ("What is 17 × 23?", "391", "math", 3),
    ("What is the factorial of 6?", "720", "math", 3),
    ("What is log base 2 of 256?", "8", "math", 3),
    ("What is the 10th Fibonacci number?", "55", "math", 3),
    ("What is the derivative of x^3?", "3x^2", "math", 3),
    ("What is the integral of 2x dx?", "x^2", "math", 3),
    ("How many prime numbers are there between 1 and 50?", "15", "math", 4),
    ("What is the sum of interior angles of a hexagon in degrees?", "720", "math", 4),
    ("What is 3^5 - 2^8?", "-13", "math", 4),
    ("What is the LCM of 12, 15, and 20?", "60", "math", 4),
    ("What is the value of e to 2 decimal places?", "2.72", "math", 3),
    ("What is 999 × 999?", "998001", "math", 5),

    # ── Science (20 items) ──
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide", "science", 1),
    ("How many bones are in the adult human body?", "206", "science", 2),
    ("What is the pH of pure water?", "7", "science", 2),
    ("What is the chemical formula for table salt?", "NaCl", "science", 2),
    ("What is the powerhouse of the cell?", "Mitochondria", "science", 1),
    ("What is Newton's second law of motion?", "F=ma", "science", 2),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen", "science", 2),
    ("How many chromosomes do humans have?", "46", "science", 2),
    ("What particle has a positive charge in an atom?", "Proton", "science", 1),
    ("What is the boiling point of water in Celsius?", "100", "science", 1),
    ("What is the chemical formula for sulfuric acid?", "H2SO4", "science", 3),
    ("What is the Avogadro constant (order of magnitude)?", "10^23", "science", 3),
    ("What organelle is responsible for protein synthesis?", "Ribosome", "science", 3),
    ("What is Planck's constant in units of J·s (order of magnitude)?", "10^-34", "science", 4),
    ("What is the charge of an electron in Coulombs (order of magnitude)?", "10^-19", "science", 4),
    ("What is the half-life of Carbon-14 in years (approximately)?", "5730", "science", 4),
    ("What type of bond involves sharing electrons?", "Covalent", "science", 2),
    ("What is the SI unit of electric current?", "Ampere", "science", 2),
    ("What is the speed of sound in air at sea level in m/s (approximately)?", "343", "science", 3),
    ("What force keeps planets in orbit around the Sun?", "Gravity", "science", 1),

    # ── Reasoning (20 items) ──
    ("If all roses are flowers and all flowers are plants, are all roses plants?", "Yes", "reasoning", 1),
    ("What comes next: 2, 4, 8, 16, ...?", "32", "reasoning", 1),
    ("If a shirt costs $25 after a 50% discount, what was the original price?", "50", "reasoning", 2),
    ("What comes next: 1, 1, 2, 3, 5, 8, ...?", "13", "reasoning", 2),
    ("If 5 workers can build a wall in 10 days, how many days for 10 workers?", "5", "reasoning", 2),
    ("A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost in cents?", "5", "reasoning", 3),
    ("If you have 3 red balls and 2 blue balls in a bag, what is the probability of drawing a red ball?", "3/5", "reasoning", 3),
    ("What is the next prime number after 29?", "31", "reasoning", 2),
    ("If A is taller than B, and B is taller than C, who is the shortest?", "C", "reasoning", 1),
    ("How many squares are on a standard chessboard?", "64", "reasoning", 2),
    ("A clock shows 3:15. What is the angle between the hour and minute hands?", "7.5", "reasoning", 4),
    ("If you flip a fair coin 3 times, what is the probability of getting all heads?", "1/8", "reasoning", 3),
    ("What is the minimum number of moves to solve the Tower of Hanoi with 3 disks?", "7", "reasoning", 3),
    ("In a room of 23 people, what is the approximate probability that two share a birthday?", "50%", "reasoning", 4),
    ("What day of the week was January 1, 2000?", "Saturday", "reasoning", 4),
    ("If you fold a paper in half 10 times, how many layers thick is it?", "1024", "reasoning", 3),
    ("What comes next: 1, 4, 9, 16, 25, ...?", "36", "reasoning", 2),
    ("Three friends split a $30 bill evenly. Each pays $10. They get $5 back and each take $1, giving $2 to the waiter. Where did the missing dollar go?", "There is no missing dollar", "reasoning", 4),
    ("How many times does the digit 1 appear in all numbers from 1 to 100?", "21", "reasoning", 5),
    ("If today is Wednesday, what day will it be 100 days from now?", "Friday", "reasoning", 3),

    # ── Language (20 items) ──
    ("What is the past tense of 'go'?", "went", "language", 1),
    ("What is the plural of 'mouse'?", "mice", "language", 1),
    ("What is a synonym for 'happy'?", "joyful", "language", 1),
    ("What is the antonym of 'ancient'?", "modern", "language", 1),
    ("What is the comparative form of 'good'?", "better", "language", 1),
    ("What literary device compares two things using 'like' or 'as'?", "simile", "language", 2),
    ("What is the term for a word that sounds like what it describes (e.g., 'buzz')?", "onomatopoeia", "language", 2),
    ("In the sentence 'The dog chased the cat', what is the subject?", "dog", "language", 2),
    ("What is the superlative form of 'bad'?", "worst", "language", 2),
    ("What part of speech is the word 'quickly'?", "adverb", "language", 2),
    ("What is the term for words that have opposite meanings?", "antonyms", "language", 2),
    ("What figure of speech is 'The world is a stage'?", "metaphor", "language", 3),
    ("What is the grammatical term for a group of words containing a subject and predicate?", "clause", "language", 3),
    ("What tense is 'I will have been running'?", "future perfect continuous", "language", 4),
    ("What is the term for a word formed from the initial letters of other words?", "acronym", "language", 3),
    ("What language family does Japanese belong to?", "Japonic", "language", 4),
    ("What is the term for a new word entering a language?", "neologism", "language", 4),
    ("What is the study of word origins called?", "etymology", "language", 3),
    ("In phonetics, what type of consonant is 'p'?", "plosive", "language", 4),
    ("What is the most widely spoken language by total number of speakers?", "English", "language", 3),
]


def check_answer(model_response: str, correct_answer: str) -> bool:
    """Flexible exact-match with common normalizations."""
    resp = model_response.strip().lower()
    correct = correct_answer.strip().lower()

    # Direct match
    if correct in resp:
        return True

    # Numeric match
    resp_nums = re.findall(r"-?\d+\.?\d*", resp)
    correct_nums = re.findall(r"-?\d+\.?\d*", correct)
    if correct_nums and resp_nums:
        try:
            if abs(float(resp_nums[0]) - float(correct_nums[0])) < 0.01:
                return True
        except ValueError:
            pass

    # First word match for short answers
    if len(correct.split()) == 1 and correct in resp.split():
        return True

    # Handle yes/no
    if correct in ("yes", "no"):
        return correct in resp

    return False


def run_battery(model_name: str, items: list) -> dict:
    """Run completion battery for one model."""
    details = []
    n_correct = 0
    by_domain = {}
    by_difficulty = {}

    for i, (question, answer, domain, difficulty) in enumerate(items):
        prompt = f"Answer this question in one short phrase:\n{question}"

        try:
            response = call_model(model_name, prompt)
        except Exception as exc:
            print(f"    ERROR at item {i}: {exc}")
            response = ""

        is_correct = check_answer(response, answer)
        n_correct += int(is_correct)

        by_domain.setdefault(domain, []).append(int(is_correct))
        by_difficulty.setdefault(difficulty, []).append(int(is_correct))

        details.append({
            "idx": i,
            "question": question,
            "correct_answer": answer,
            "model_answer": response[:200],
            "correct": int(is_correct),
            "domain": domain,
            "difficulty": difficulty,
        })

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(items)}] running_acc={n_correct/(i+1):.3f}")

    score = n_correct / len(items) if items else 0.0
    domain_means = {d: round(sum(v)/len(v), 4) for d, v in by_domain.items()}
    diff_means = {str(d): round(sum(v)/len(v), 4) for d, v in by_difficulty.items()}

    return {
        "model": model_name,
        "score": round(score, 4),
        "n_correct": n_correct,
        "n_total": len(items),
        "by_domain": domain_means,
        "by_difficulty": diff_means,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="CEF Expanded Completion Battery")
    parser.add_argument("--models", nargs="+", default=["all-15"])
    parser.add_argument("--n-items", type=int, default=100, help="Number of items (max 100)")
    args = parser.parse_args()

    if "all-15" in args.models:
        models = ALL_15
    elif "all-7" in args.models:
        models = ORIGINAL_7
    elif "all-8" in args.models:
        models = EXPANSION_8
    else:
        models = args.models

    for m in models:
        if m not in MODELS:
            print(f"ERROR: {m} not in MODELS registry")
            sys.exit(1)

    items = COMPLETION_BATTERY[:args.n_items]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"cef_completion_battery_v2_{ts}.json"

    print(f"CEF Expanded Completion Battery v2")
    print(f"  Models: {len(models)}")
    print(f"  Items: {len(items)}")
    print(f"  Domains: {set(d for _,_,d,_ in items)}")
    print()

    all_results = []
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        result = run_battery(model, items)
        all_results.append(result)
        print(f"  Score: {result['score']:.4f} ({result['n_correct']}/{result['n_total']})")
        print(f"  By domain: {result['by_domain']}")
        print(f"  By difficulty: {result['by_difficulty']}")

    # Save
    summary = []
    for r in all_results:
        summary.append({k: v for k, v in r.items() if k != "details"})

    output = {
        "timestamp": ts,
        "n_models": len(models),
        "n_items": len(items),
        "battery_version": "v2",
        "results": summary,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Also save full details
    details_path = out_path.with_suffix(".details.json")
    full_output = dict(output)
    full_output["results"] = all_results
    with open(details_path, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nSaved: {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<28} {'Score':>8} {'Factual':>8} {'Math':>8} {'Science':>8} {'Reason':>8} {'Lang':>8}")
    print("-" * 70)
    for r in summary:
        d = r["by_domain"]
        print(f"{r['model']:<28} {r['score']:>8.3f} {d.get('factual',0):>8.3f} {d.get('math',0):>8.3f} {d.get('science',0):>8.3f} {d.get('reasoning',0):>8.3f} {d.get('language',0):>8.3f}")


if __name__ == "__main__":
    main()
