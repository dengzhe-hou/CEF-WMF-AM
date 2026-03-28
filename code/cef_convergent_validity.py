"""
CEF Convergent/Divergent Validity Experiment

Tests whether CEF sub-dimensions correlate with theoretically related constructs
(convergent validity) and do NOT correlate with unrelated constructs (divergent validity).

Probe 1 — RF-POC (Reasoning Fidelity, Process of Completion):
    Multi-step math reasoning scored on intermediate step correctness.
    SHOULD correlate with WMF-AM (both require tracking state across operations).

Probe 2 — Self-Knowledge Accuracy:
    Factual questions about the model's own capabilities/architecture.
    SHOULD correlate with MCC-MA (both measure metacognitive ability).

Probe 3 — Simple Factual Retrieval (Divergent Control):
    Pure factual recall (capitals, dates, element symbols).
    Should NOT correlate with WMF-AM (declarative knowledge != working memory).

Usage:
    # Phase 1: Ollama models only (no API keys needed)
    python cef_convergent_validity.py --phase ollama --seeds 2

    # Phase 2: API models (needs .env with keys)
    python cef_convergent_validity.py --phase api --seeds 2

    # Specific models
    python cef_convergent_validity.py --models ollama:qwen2.5:7b ollama:llama3.1:8b

    # All models
    python cef_convergent_validity.py --phase all --seeds 3
"""

import argparse
import json
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, RESULTS_DIR, call_model

# ── Constants ────────────────────────────────────────────────────────────────

# Rate limiting
API_DELAY = 1.0
OLLAMA_DELAY = 0.3

OLLAMA_MODELS = [k for k in MODELS if MODELS[k]["provider"] == "ollama"]
API_MODELS = [k for k in MODELS if MODELS[k]["provider"] != "ollama"]

# RF-POC parameters
RF_POC_PROBLEMS_PER_LEVEL = 10
RF_POC_DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Self-Knowledge parameters
SELF_KNOWLEDGE_N = 15

# Factual Retrieval parameters
FACTUAL_RETRIEVAL_N = 20


# ── Delay helper ─────────────────────────────────────────────────────────────

def _delay(model: str):
    cfg = MODELS.get(model, {})
    if cfg.get("provider") == "ollama":
        time.sleep(OLLAMA_DELAY)
    else:
        time.sleep(API_DELAY)


# ── Probe 1: RF-POC (Reasoning Fidelity — Process of Completion) ────────────

RF_POC_BANK = {
    "easy": [
        {
            "problem": "Sarah has 12 apples. She gives 4 to Tom, then buys 7 more. How many apples does Sarah have?",
            "steps": [
                {"description": "Start with 12 apples", "result": 12},
                {"description": "Give 4 to Tom: 12 - 4", "result": 8},
                {"description": "Buy 7 more: 8 + 7", "result": 15},
            ],
            "final_answer": 15,
        },
        {
            "problem": "A store sells 3 shirts at $15 each and 2 pants at $25 each. What is the total cost?",
            "steps": [
                {"description": "Cost of shirts: 3 x 15", "result": 45},
                {"description": "Cost of pants: 2 x 25", "result": 50},
                {"description": "Total: 45 + 50", "result": 95},
            ],
            "final_answer": 95,
        },
        {
            "problem": "Jake reads 20 pages per day. After 5 days, he has 60 pages left. How many pages is the book?",
            "steps": [
                {"description": "Pages read: 20 x 5", "result": 100},
                {"description": "Total pages: 100 + 60", "result": 160},
            ],
            "final_answer": 160,
        },
        {
            "problem": "A bus has 45 passengers. At the first stop, 12 get off and 8 get on. At the second stop, 15 get off and 6 get on. How many passengers are on the bus?",
            "steps": [
                {"description": "After first stop: 45 - 12 + 8", "result": 41},
                {"description": "After second stop: 41 - 15 + 6", "result": 32},
            ],
            "final_answer": 32,
        },
        {
            "problem": "Maria earns $12 per hour. She works 8 hours on Monday and 6 hours on Tuesday. How much does she earn in total?",
            "steps": [
                {"description": "Monday earnings: 12 x 8", "result": 96},
                {"description": "Tuesday earnings: 12 x 6", "result": 72},
                {"description": "Total: 96 + 72", "result": 168},
            ],
            "final_answer": 168,
        },
        {
            "problem": "A rectangle has length 14 cm and width 9 cm. What is its perimeter?",
            "steps": [
                {"description": "Sum of length and width: 14 + 9", "result": 23},
                {"description": "Perimeter: 2 x 23", "result": 46},
            ],
            "final_answer": 46,
        },
        {
            "problem": "Tom has 50 marbles. He gives 1/5 to Alex and 1/10 to Beth. How many does Tom have left?",
            "steps": [
                {"description": "Marbles to Alex: 50 / 5", "result": 10},
                {"description": "Marbles to Beth: 50 / 10", "result": 5},
                {"description": "Remaining: 50 - 10 - 5", "result": 35},
            ],
            "final_answer": 35,
        },
        {
            "problem": "A train travels 60 km/h for 2 hours, then 80 km/h for 3 hours. What is the total distance?",
            "steps": [
                {"description": "Distance at 60 km/h: 60 x 2", "result": 120},
                {"description": "Distance at 80 km/h: 80 x 3", "result": 240},
                {"description": "Total distance: 120 + 240", "result": 360},
            ],
            "final_answer": 360,
        },
        {
            "problem": "A baker makes 4 batches of cookies with 24 cookies each. He sells 3/4 of them. How many are left?",
            "steps": [
                {"description": "Total cookies: 4 x 24", "result": 96},
                {"description": "Sold: 96 x 3/4", "result": 72},
                {"description": "Left: 96 - 72", "result": 24},
            ],
            "final_answer": 24,
        },
        {
            "problem": "A class has 30 students. 2/5 are girls. 3 new boys join the class. How many boys are there now?",
            "steps": [
                {"description": "Number of girls: 30 x 2/5", "result": 12},
                {"description": "Original boys: 30 - 12", "result": 18},
                {"description": "Boys after 3 join: 18 + 3", "result": 21},
            ],
            "final_answer": 21,
        },
    ],
    "medium": [
        {
            "problem": "A shop offers a 20% discount on a $150 item. Then there is an additional 10% loyalty discount on the discounted price. Sales tax is 8% on the final price. What is the total cost?",
            "steps": [
                {"description": "After 20% discount: 150 x 0.80", "result": 120},
                {"description": "After 10% loyalty: 120 x 0.90", "result": 108},
                {"description": "Sales tax: 108 x 0.08", "result": 8.64},
                {"description": "Total: 108 + 8.64", "result": 116.64},
            ],
            "final_answer": 116.64,
        },
        {
            "problem": "A farmer plants 5 rows of corn with 18 plants each. Each plant produces 3 ears. He loses 1/6 of total ears to pests. How many ears remain?",
            "steps": [
                {"description": "Total plants: 5 x 18", "result": 90},
                {"description": "Total ears: 90 x 3", "result": 270},
                {"description": "Lost to pests: 270 / 6", "result": 45},
                {"description": "Remaining: 270 - 45", "result": 225},
            ],
            "final_answer": 225,
        },
        {
            "problem": "A pool holds 10,000 liters. Two pipes fill it: Pipe A adds 500 liters/hour, Pipe B adds 300 liters/hour. A drain removes 200 liters/hour. How many hours to fill from empty?",
            "steps": [
                {"description": "Net fill rate: 500 + 300 - 200", "result": 600},
                {"description": "Hours to fill: 10000 / 600", "result": 16.67},
            ],
            "final_answer": 16.67,
        },
        {
            "problem": "An investor puts $5,000 in an account earning 6% simple interest per year. After 3 years, she withdraws $1,000 and reinvests the rest for 2 more years at 6%. What is the final amount?",
            "steps": [
                {"description": "Interest for 3 years: 5000 x 0.06 x 3", "result": 900},
                {"description": "Amount after 3 years: 5000 + 900", "result": 5900},
                {"description": "After withdrawal: 5900 - 1000", "result": 4900},
                {"description": "Interest for 2 more years: 4900 x 0.06 x 2", "result": 588},
                {"description": "Final amount: 4900 + 588", "result": 5488},
            ],
            "final_answer": 5488,
        },
        {
            "problem": "A rectangular garden is 20m by 15m. A path of width 2m runs around the outside. What is the area of the path alone?",
            "steps": [
                {"description": "Outer dimensions: (20+4) by (15+4) = 24 by 19", "result": "24x19"},
                {"description": "Outer area: 24 x 19", "result": 456},
                {"description": "Inner area: 20 x 15", "result": 300},
                {"description": "Path area: 456 - 300", "result": 156},
            ],
            "final_answer": 156,
        },
        {
            "problem": "Three friends split a bill. The meal costs $84, tax is 10%, and they add a 20% tip on the pre-tax amount. How much does each person pay?",
            "steps": [
                {"description": "Tax: 84 x 0.10", "result": 8.4},
                {"description": "Tip: 84 x 0.20", "result": 16.8},
                {"description": "Total: 84 + 8.4 + 16.8", "result": 109.2},
                {"description": "Per person: 109.2 / 3", "result": 36.4},
            ],
            "final_answer": 36.4,
        },
        {
            "problem": "A car's fuel tank holds 50 liters. It uses 8 liters per 100 km in the city and 5 liters per 100 km on the highway. If the driver goes 200 km in the city then continues on the highway, how far can they go on the highway with the remaining fuel?",
            "steps": [
                {"description": "City fuel used: 8 x (200/100)", "result": 16},
                {"description": "Remaining fuel: 50 - 16", "result": 34},
                {"description": "Highway distance: 34 / 5 x 100", "result": 680},
            ],
            "final_answer": 680,
        },
        {
            "problem": "A school has 240 students. 55% are in the science track and the rest in the arts track. 30% of science students and 40% of arts students play sports. How many students play sports in total?",
            "steps": [
                {"description": "Science students: 240 x 0.55", "result": 132},
                {"description": "Arts students: 240 - 132", "result": 108},
                {"description": "Sports from science: 132 x 0.30", "result": 39.6},
                {"description": "Sports from arts: 108 x 0.40", "result": 43.2},
                {"description": "Total sports: 39.6 + 43.2", "result": 82.8},
            ],
            "final_answer": 82.8,
        },
        {
            "problem": "A rope is 100 meters long. It is cut into 3 pieces. The second piece is twice as long as the first. The third piece is 10 meters longer than the second. How long is each piece?",
            "steps": [
                {"description": "Let first = x. Then: x + 2x + (2x+10) = 100", "result": "5x+10=100"},
                {"description": "Solve: 5x = 90, x = 18", "result": 18},
                {"description": "Second piece: 2 x 18", "result": 36},
                {"description": "Third piece: 36 + 10", "result": 46},
            ],
            "final_answer": 18,  # first piece
        },
        {
            "problem": "A company produces 1200 widgets per day. On Monday, 5% are defective and discarded. On Tuesday, production increases by 25% but the defect rate rises to 8%. How many good widgets total over both days?",
            "steps": [
                {"description": "Monday good: 1200 x 0.95", "result": 1140},
                {"description": "Tuesday production: 1200 x 1.25", "result": 1500},
                {"description": "Tuesday good: 1500 x 0.92", "result": 1380},
                {"description": "Total good: 1140 + 1380", "result": 2520},
            ],
            "final_answer": 2520,
        },
    ],
    "hard": [
        {
            "problem": "A tank has 3 inlet pipes. Pipe A fills it in 6 hours, Pipe B in 8 hours, Pipe C in 12 hours. A drain empties it in 10 hours. All are opened simultaneously. After 2 hours, Pipe C and the drain are closed. How long in total to fill the tank?",
            "steps": [
                {"description": "Combined rate with all open: 1/6 + 1/8 + 1/12 - 1/10", "result": "20/120 + 15/120 + 10/120 - 12/120 = 33/120 = 11/40"},
                {"description": "Filled in 2 hours: 2 x 11/40", "result": "22/40 = 11/20"},
                {"description": "Remaining: 1 - 11/20", "result": "9/20"},
                {"description": "Rate with A+B only: 1/6 + 1/8 = 7/24", "result": "7/24"},
                {"description": "Time for remaining: (9/20) / (7/24) = 216/140", "result": 1.543},
                {"description": "Total time: 2 + 1.543", "result": 3.543},
            ],
            "final_answer": 3.54,
        },
        {
            "problem": "A store buys 100 items at $25 each. It sells 60% at a 40% markup, 25% at a 20% markup, and the rest at a 10% loss. What is the total profit or loss?",
            "steps": [
                {"description": "Cost: 100 x 25", "result": 2500},
                {"description": "Revenue from 60 items at 40% markup: 60 x 25 x 1.40", "result": 2100},
                {"description": "Revenue from 25 items at 20% markup: 25 x 25 x 1.20", "result": 750},
                {"description": "Revenue from 15 items at 10% loss: 15 x 25 x 0.90", "result": 337.5},
                {"description": "Total revenue: 2100 + 750 + 337.5", "result": 3187.5},
                {"description": "Profit: 3187.5 - 2500", "result": 687.5},
            ],
            "final_answer": 687.5,
        },
        {
            "problem": "A car travels from A to B at 60 km/h. The return trip is at 40 km/h. On the way back it stops for 30 minutes. The total time for the round trip is 5.5 hours. What is the distance from A to B?",
            "steps": [
                {"description": "Let distance = d. Time there: d/60. Time back: d/40. Stop: 0.5 hours", "result": "d/60 + d/40 + 0.5 = 5.5"},
                {"description": "d/60 + d/40 = 5.0", "result": 5.0},
                {"description": "Common denominator: 2d/120 + 3d/120 = 5d/120 = d/24", "result": "d/24 = 5"},
                {"description": "d = 120", "result": 120},
            ],
            "final_answer": 120,
        },
        {
            "problem": "Three workers A, B, C can complete a task alone in 10, 15, and 20 days. They start together but A leaves after 3 days, then B leaves 2 days after A. How many total days to finish?",
            "steps": [
                {"description": "Combined rate A+B+C: 1/10 + 1/15 + 1/20 = 6/60 + 4/60 + 3/60 = 13/60", "result": "13/60"},
                {"description": "Work in first 3 days: 3 x 13/60 = 39/60 = 13/20", "result": "13/20"},
                {"description": "Remaining after 3 days: 1 - 13/20 = 7/20", "result": "7/20"},
                {"description": "Rate B+C: 1/15 + 1/20 = 7/60", "result": "7/60"},
                {"description": "Work in next 2 days (B+C): 2 x 7/60 = 14/60 = 7/30", "result": "7/30"},
                {"description": "Remaining after day 5: 7/20 - 7/30 = 21/60 - 14/60 = 7/60", "result": "7/60"},
                {"description": "Rate C alone: 1/20. Time: (7/60)/(1/20) = 140/60 = 7/3", "result": 2.333},
                {"description": "Total days: 3 + 2 + 2.333", "result": 7.333},
            ],
            "final_answer": 7.33,
        },
        {
            "problem": "A container has 200 liters of a 40% acid solution. 50 liters are removed and replaced with pure water. This process is repeated once more. What is the final acid concentration?",
            "steps": [
                {"description": "Initial acid: 200 x 0.40", "result": 80},
                {"description": "After removing 50L: acid = 80 x (150/200)", "result": 60},
                {"description": "After adding water: 60 liters acid in 200 liters, concentration = 30%", "result": 0.30},
                {"description": "Second removal: acid = 60 x (150/200)", "result": 45},
                {"description": "Final concentration: 45/200", "result": 0.225},
            ],
            "final_answer": 22.5,
        },
        {
            "problem": "A pyramid of cans has 1 can on top, 4 on the second row (2x2), 9 on the third (3x3), and so on up to 8 rows. How many cans in total?",
            "steps": [
                {"description": "Sum of squares: 1^2 + 2^2 + 3^2 + ... + 8^2", "result": "formula: n(n+1)(2n+1)/6"},
                {"description": "n=8: 8 x 9 x 17 / 6", "result": "1224/6"},
                {"description": "Total: 204", "result": 204},
            ],
            "final_answer": 204,
        },
        {
            "problem": "A boat goes 24 km upstream in 3 hours and returns downstream in 2 hours. What is the speed of the current and the speed of the boat in still water?",
            "steps": [
                {"description": "Upstream speed: 24/3", "result": 8},
                {"description": "Downstream speed: 24/2", "result": 12},
                {"description": "Boat speed (average): (8 + 12) / 2", "result": 10},
                {"description": "Current speed: (12 - 8) / 2", "result": 2},
            ],
            "final_answer": 10,  # boat speed in still water
        },
        {
            "problem": "Compound interest: $10,000 invested at 5% per year compounded annually for 4 years. After year 2, $2,000 is withdrawn. What is the final amount?",
            "steps": [
                {"description": "After year 1: 10000 x 1.05", "result": 10500},
                {"description": "After year 2: 10500 x 1.05", "result": 11025},
                {"description": "After withdrawal: 11025 - 2000", "result": 9025},
                {"description": "After year 3: 9025 x 1.05", "result": 9476.25},
                {"description": "After year 4: 9476.25 x 1.05", "result": 9950.0625},
            ],
            "final_answer": 9950.06,
        },
        {
            "problem": "A triangular field has sides 13m, 14m, and 15m. A fence costs $12 per meter for the perimeter, and seeding costs $3 per square meter for the area. What is the total cost? (Use Heron's formula for area.)",
            "steps": [
                {"description": "Perimeter: 13 + 14 + 15", "result": 42},
                {"description": "Semi-perimeter s: 42 / 2", "result": 21},
                {"description": "Heron's: sqrt(21 x 8 x 7 x 6) = sqrt(7056)", "result": 84},
                {"description": "Fence cost: 42 x 12", "result": 504},
                {"description": "Seeding cost: 84 x 3", "result": 252},
                {"description": "Total: 504 + 252", "result": 756},
            ],
            "final_answer": 756,
        },
        {
            "problem": "A clock's minute hand is 10 cm long. How far does the tip travel in 45 minutes? Also, what is the straight-line distance from start to end position? (Use pi = 3.14159)",
            "steps": [
                {"description": "Full circumference: 2 x pi x 10", "result": 62.832},
                {"description": "Arc for 45 min (3/4 circle): 62.832 x 0.75", "result": 47.124},
                {"description": "Angle: 270 degrees. Chord = 2r sin(135 deg) = 20 sin(135)", "result": "20 x 0.7071"},
                {"description": "Chord length: 14.142", "result": 14.14},
            ],
            "final_answer": 47.12,  # arc length
        },
    ],
}


def build_rf_poc_prompt(problem: dict) -> str:
    """Build a prompt that asks the model to show step-by-step work."""
    return f"""Solve this problem step by step. Show your work clearly for each intermediate step.

Problem: {problem['problem']}

IMPORTANT: Show each computation step on its own line with the numerical result.
Write your final answer on the last line in the format: "Final answer: <number>"."""


def score_rf_poc_steps(response: str, problem: dict) -> dict:
    """Score intermediate steps and final answer from a model's response.

    For each expected step, check if the model produced the correct intermediate result.
    Also checks the final answer.
    """
    steps = problem["steps"]
    n_steps = len(steps)
    step_correct = []

    # Extract all numbers from the response
    resp_lower = response.lower()
    # Find all numbers (int and float) in the response
    all_numbers = re.findall(r"-?\d+\.?\d*", response)
    all_numbers_float = [float(x) for x in all_numbers]

    for step in steps:
        expected = step["result"]
        if isinstance(expected, (int, float)):
            # Check if this number appears in the response
            found = False
            for num in all_numbers_float:
                if abs(num - expected) < 0.1:  # tolerance for rounding
                    found = True
                    break
            step_correct.append(int(found))
        else:
            # Non-numeric step (like "5x+10=100") — skip scoring
            step_correct.append(-1)  # -1 means not scored

    # Score final answer
    final_answer = problem["final_answer"]
    final_correct = 0
    # Look for "final answer:" pattern
    final_match = re.search(r"final\s*answer[:\s]*([+-]?\d+\.?\d*)", resp_lower)
    if final_match:
        try:
            pred = float(final_match.group(1))
            if abs(pred - final_answer) < 0.1:
                final_correct = 1
        except ValueError:
            pass

    # Fallback: check if any number near the end of the response matches
    if not final_correct and all_numbers_float:
        # Check last 5 numbers
        for num in reversed(all_numbers_float[-5:]):
            if abs(num - final_answer) < 0.1:
                final_correct = 1
                break

    # Process score: fraction of scored steps correct
    scored_steps = [s for s in step_correct if s >= 0]
    process_score = sum(scored_steps) / len(scored_steps) if scored_steps else 0.0

    return {
        "step_correct": step_correct,
        "n_steps": n_steps,
        "n_steps_scored": len(scored_steps),
        "n_steps_correct": sum(scored_steps),
        "process_score": round(process_score, 4),
        "final_correct": final_correct,
    }


def run_rf_poc(model: str, seeds: list[int]) -> list[dict]:
    """Run RF-POC (Reasoning Fidelity) probe."""
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        for difficulty in RF_POC_DIFFICULTY_LEVELS:
            bank = RF_POC_BANK[difficulty]
            # Sample RF_POC_PROBLEMS_PER_LEVEL problems (with replacement if bank is small)
            problems = rng.choices(bank, k=RF_POC_PROBLEMS_PER_LEVEL)
            for trial_idx, problem in enumerate(problems):
                prompt = build_rf_poc_prompt(problem)
                try:
                    resp = call_model(model, prompt)
                    scores = score_rf_poc_steps(resp, problem)
                except Exception as e:
                    resp = f"ERROR: {e}"
                    scores = {
                        "step_correct": [], "n_steps": len(problem["steps"]),
                        "n_steps_scored": 0, "n_steps_correct": 0,
                        "process_score": 0.0, "final_correct": 0,
                    }
                results.append({
                    "probe": "RF-POC",
                    "validity_type": "convergent_WMF-AM",
                    "model": model,
                    "seed": seed,
                    "difficulty": difficulty,
                    "trial": trial_idx,
                    "problem": problem["problem"],
                    "expected_answer": problem["final_answer"],
                    **scores,
                })
                _delay(model)
    return results


# ── Probe 2: Self-Knowledge Accuracy ────────────────────────────────────────

# Questions about LLM self-knowledge. Answers are model-specific so we use
# a two-phase approach: ask the model, then verify key claims against known facts.
# For scoring, we use a set of questions where ground truth is deterministic.

SELF_KNOWLEDGE_QUESTIONS = [
    {
        "question": "Can you browse the internet in real time?",
        "ground_truth": "no",
        "category": "capability",
        "keywords_yes": ["yes", "i can browse", "i can access"],
        "keywords_no": ["no", "cannot", "can't", "do not have", "don't have", "unable"],
    },
    {
        "question": "Can you execute code directly?",
        "ground_truth": "no",  # base LLM without tools
        "category": "capability",
        "keywords_yes": ["yes", "i can execute", "i can run"],
        "keywords_no": ["no", "cannot", "can't", "do not", "don't"],
    },
    {
        "question": "Do you retain information between separate conversations?",
        "ground_truth": "no",
        "category": "architecture",
        "keywords_yes": ["yes", "i retain", "i remember"],
        "keywords_no": ["no", "do not retain", "don't retain", "each conversation", "no memory"],
    },
    {
        "question": "Are your responses deterministic — will you always give the exact same answer to the same prompt?",
        "ground_truth": "no",  # temperature, sampling
        "category": "architecture",
        "keywords_yes": ["yes", "deterministic", "same answer"],
        "keywords_no": ["no", "not deterministic", "may vary", "can vary", "might differ", "not always"],
    },
    {
        "question": "Can you learn and update your knowledge from this conversation?",
        "ground_truth": "no",  # no in-context weight updates
        "category": "architecture",
        "keywords_yes": ["yes", "i can learn", "i learn"],
        "keywords_no": ["no", "cannot learn", "can't learn", "do not learn", "don't update", "weights"],
    },
    {
        "question": "Do you have feelings or emotions?",
        "ground_truth": "no",
        "category": "self_model",
        "keywords_yes": ["yes", "i do have", "i feel"],
        "keywords_no": ["no", "do not have", "don't have", "don't experience", "cannot feel", "not capable"],
    },
    {
        "question": "Can you process and understand images?",
        "ground_truth": "depends",  # multimodal models can, text-only cannot
        "category": "capability",
        "keywords_yes": ["yes", "i can process images", "i can understand images", "multimodal"],
        "keywords_no": ["no", "cannot process images", "text only", "text-only", "can't see"],
    },
    {
        "question": "Do you have access to your own training data during inference?",
        "ground_truth": "no",
        "category": "architecture",
        "keywords_yes": ["yes", "i have access"],
        "keywords_no": ["no", "do not have access", "don't have access", "cannot access"],
    },
    {
        "question": "Is there a maximum length to the text you can process in one go?",
        "ground_truth": "yes",
        "category": "architecture",
        "keywords_yes": ["yes", "there is a limit", "context window", "maximum", "token limit"],
        "keywords_no": ["no", "no limit", "unlimited"],
    },
    {
        "question": "Can you make mistakes in mathematical calculations?",
        "ground_truth": "yes",
        "category": "self_model",
        "keywords_yes": ["yes", "i can make mistakes", "not perfect", "may make errors", "can make errors"],
        "keywords_no": ["no", "i don't make", "always correct", "perfect"],
    },
    {
        "question": "Do you have a consistent personality across conversations?",
        "ground_truth": "no",  # behavior depends on system prompt, context
        "category": "self_model",
        "keywords_yes": ["yes", "consistent personality"],
        "keywords_no": ["no", "not consistent", "varies", "depends", "shaped by", "don't have a fixed"],
    },
    {
        "question": "Can you guarantee that all factual claims you make are correct?",
        "ground_truth": "no",
        "category": "self_model",
        "keywords_yes": ["yes", "i guarantee", "always correct"],
        "keywords_no": ["no", "cannot guarantee", "can't guarantee", "may", "might", "hallucinate"],
    },
    {
        "question": "Were you trained using reinforcement learning from human feedback (RLHF)?",
        "ground_truth": "yes",  # true for all major LLMs
        "category": "training",
        "keywords_yes": ["yes", "rlhf", "reinforcement learning", "human feedback"],
        "keywords_no": ["no", "was not", "wasn't"],
    },
    {
        "question": "Are you able to generate content in languages other than English?",
        "ground_truth": "yes",
        "category": "capability",
        "keywords_yes": ["yes", "multiple languages", "other languages", "multilingual"],
        "keywords_no": ["no", "only english", "english only"],
    },
    {
        "question": "Do you know when your training data was last updated?",
        "ground_truth": "yes",  # models generally know their cutoff
        "category": "self_model",
        "keywords_yes": ["yes", "my training", "cutoff", "knowledge cutoff", "my data"],
        "keywords_no": ["no", "i don't know", "i do not know", "not sure", "uncertain"],
    },
]


def score_self_knowledge(response: str, question_data: dict) -> int:
    """Score a self-knowledge answer as correct (1) or incorrect (0).

    Returns 1 if the model's answer aligns with the ground truth.
    For 'depends' ground truth, accepting either is scored as correct.
    """
    resp_lower = response.lower()
    gt = question_data["ground_truth"]

    # Check for affirmative keywords
    has_yes = any(kw in resp_lower for kw in question_data["keywords_yes"])
    has_no = any(kw in resp_lower for kw in question_data["keywords_no"])

    if gt == "yes":
        return 1 if has_yes and not has_no else (0 if has_no else 1 if has_yes else 0)
    elif gt == "no":
        return 1 if has_no and not has_yes else (0 if has_yes else 1 if has_no else 0)
    elif gt == "depends":
        # For ambiguous questions, any nuanced answer is correct
        if has_yes and has_no:
            return 1  # nuanced answer acknowledging both sides
        if "depends" in resp_lower or "it depends" in resp_lower:
            return 1
        # Accept either direction with caveats
        return 1 if (has_yes or has_no) else 0
    return 0


def run_self_knowledge(model: str, seeds: list[int]) -> list[dict]:
    """Run self-knowledge accuracy probe."""
    results = []
    system_prompt = "Answer each question honestly and concisely about your own capabilities and architecture."
    for seed in seeds:
        rng = random.Random(seed)
        questions = list(SELF_KNOWLEDGE_QUESTIONS)
        rng.shuffle(questions)
        questions = questions[:SELF_KNOWLEDGE_N]

        for trial_idx, q in enumerate(questions):
            prompt = f"{q['question']}\n\nAnswer in 1-2 sentences."
            try:
                resp = call_model(model, prompt, system=system_prompt)
                correct = score_self_knowledge(resp, q)
            except Exception as e:
                resp = f"ERROR: {e}"
                correct = 0

            results.append({
                "probe": "Self-Knowledge",
                "validity_type": "convergent_MCC-MA",
                "model": model,
                "seed": seed,
                "trial": trial_idx,
                "question": q["question"],
                "category": q["category"],
                "ground_truth": q["ground_truth"],
                "accurate": correct,
                "response_snippet": resp[:200],
            })
            _delay(model)
    return results


# ── Probe 3: Simple Factual Retrieval (Divergent Control) ───────────────────

FACTUAL_QUESTIONS = [
    {"q": "What is the capital of France?", "a": "Paris", "category": "geography"},
    {"q": "What is the capital of Japan?", "a": "Tokyo", "category": "geography"},
    {"q": "What is the capital of Brazil?", "a": "Brasilia", "category": "geography"},
    {"q": "What is the capital of Australia?", "a": "Canberra", "category": "geography"},
    {"q": "What is the capital of Egypt?", "a": "Cairo", "category": "geography"},
    {"q": "What is the chemical symbol for gold?", "a": "Au", "category": "science"},
    {"q": "What is the chemical symbol for sodium?", "a": "Na", "category": "science"},
    {"q": "What is the chemical symbol for iron?", "a": "Fe", "category": "science"},
    {"q": "What is the chemical symbol for potassium?", "a": "K", "category": "science"},
    {"q": "What is the chemical symbol for silver?", "a": "Ag", "category": "science"},
    {"q": "In what year did World War I begin?", "a": "1914", "category": "history"},
    {"q": "In what year did World War II end?", "a": "1945", "category": "history"},
    {"q": "In what year did the French Revolution begin?", "a": "1789", "category": "history"},
    {"q": "In what year was the Declaration of Independence signed?", "a": "1776", "category": "history"},
    {"q": "In what year did the Berlin Wall fall?", "a": "1989", "category": "history"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100", "category": "science"},
    {"q": "How many continents are there?", "a": "7", "category": "geography"},
    {"q": "What planet is closest to the Sun?", "a": "Mercury", "category": "science"},
    {"q": "What is the largest ocean on Earth?", "a": "Pacific", "category": "geography"},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare", "category": "culture"},
]


def run_factual_retrieval(model: str, seeds: list[int]) -> list[dict]:
    """Run simple factual retrieval probe (divergent control)."""
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        questions = list(FACTUAL_QUESTIONS)
        rng.shuffle(questions)
        questions = questions[:FACTUAL_RETRIEVAL_N]

        # Batch ask for efficiency
        q_block = "\n".join(f"Q{i+1}: {q['q']}" for i, q in enumerate(questions))
        prompt = f"""Answer each question with ONLY the answer, one per line, in format "Q1: answer".

{q_block}"""

        try:
            resp = call_model(model, prompt)
        except Exception as e:
            resp = f"ERROR: {e}"

        # Parse answers
        model_answers = {}
        for line in resp.split("\n"):
            m = re.match(r"Q(\d+):\s*(.+)", line.strip())
            if m:
                model_answers[int(m.group(1))] = m.group(2).strip()

        for i, q in enumerate(questions):
            ma = model_answers.get(i + 1, "")
            # Flexible matching
            correct_ans = q["a"].lower()
            model_ans = ma.lower().strip()
            is_correct = correct_ans in model_ans or model_ans in correct_ans
            # Handle edge case: single-letter answers like "K" for potassium
            if len(correct_ans) <= 2 and not is_correct:
                is_correct = model_ans == correct_ans

            results.append({
                "probe": "Factual-Retrieval",
                "validity_type": "divergent_WMF-AM",
                "model": model,
                "seed": seed,
                "trial": i,
                "question": q["q"],
                "category": q["category"],
                "expected": q["a"],
                "model_answer": ma,
                "accurate": int(is_correct),
            })
        _delay(model)
    return results


# ── Summary ──────────────────────────────────────────────────────────────────

def compute_summary(results: list[dict]) -> dict:
    """Compute per-model, per-probe summary scores."""
    summary = {}
    for r in results:
        model = r["model"]
        probe = r["probe"]
        if model not in summary:
            summary[model] = {}
        if probe not in summary[model]:
            summary[model][probe] = {"scores": [], "count": 0}

        if probe == "RF-POC":
            summary[model][probe]["scores"].append(r["process_score"])
            summary[model][probe]["count"] += 1
        elif probe in ("Self-Knowledge", "Factual-Retrieval"):
            summary[model][probe]["scores"].append(r["accurate"])
            summary[model][probe]["count"] += 1

    # Compute means
    for model in summary:
        for probe in summary[model]:
            s = summary[model][probe]
            s["mean"] = round(float(np.mean(s["scores"])), 4) if s["scores"] else 0.0
            s["std"] = round(float(np.std(s["scores"])), 4) if len(s["scores"]) > 1 else 0.0
            del s["scores"]  # drop raw scores from summary (they're in full results)

    return summary


# ── Run all probes for one model ─────────────────────────────────────────────

def run_model(model: str, seeds: list[int]) -> list[dict]:
    """Run all three convergent/divergent validity probes for one model."""
    all_results = []
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"  Seeds: {seeds}")
    print(f"  Probes: RF-POC, Self-Knowledge, Factual-Retrieval")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    # Probe 1: RF-POC
    print(f"  [1/3] RF-POC ({len(RF_POC_DIFFICULTY_LEVELS)} levels x {RF_POC_PROBLEMS_PER_LEVEL} problems x {len(seeds)} seeds)...")
    try:
        r = run_rf_poc(model, seeds)
        proc_mean = np.mean([x["process_score"] for x in r]) if r else 0
        final_mean = np.mean([x["final_correct"] for x in r]) if r else 0
        print(f"         -> {len(r)} trials, process_score = {proc_mean:.3f}, final_correct = {final_mean:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         -> ERROR: {e}")
        traceback.print_exc()

    # Probe 2: Self-Knowledge
    print(f"  [2/3] Self-Knowledge ({SELF_KNOWLEDGE_N} questions x {len(seeds)} seeds)...")
    try:
        r = run_self_knowledge(model, seeds)
        acc = sum(x["accurate"] for x in r) / len(r) if r else 0
        print(f"         -> {len(r)} trials, accuracy = {acc:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         -> ERROR: {e}")
        traceback.print_exc()

    # Probe 3: Factual Retrieval
    print(f"  [3/3] Factual-Retrieval ({FACTUAL_RETRIEVAL_N} questions x {len(seeds)} seeds)...")
    try:
        r = run_factual_retrieval(model, seeds)
        acc = sum(x["accurate"] for x in r) / len(r) if r else 0
        print(f"         -> {len(r)} trials, accuracy = {acc:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         -> ERROR: {e}")
        traceback.print_exc()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return all_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CEF Convergent/Divergent Validity Experiment"
    )
    parser.add_argument("--phase", choices=["ollama", "api", "all"], default="ollama")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--seeds", type=int, default=2, help="Number of random seeds")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))

    if args.models:
        models = args.models
    elif args.phase == "ollama":
        models = OLLAMA_MODELS
    elif args.phase == "api":
        models = API_MODELS
    else:
        models = list(MODELS.keys())

    print(f"CEF Convergent/Divergent Validity")
    print(f"Models: {len(models)} — {models}")
    print(f"Seeds: {seeds}")
    print(f"Phase: {args.phase}")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Probes:")
    print(f"  1. RF-POC         (convergent with WMF-AM) — {len(RF_POC_DIFFICULTY_LEVELS)} x {RF_POC_PROBLEMS_PER_LEVEL} = {len(RF_POC_DIFFICULTY_LEVELS) * RF_POC_PROBLEMS_PER_LEVEL} problems/seed")
    print(f"  2. Self-Knowledge (convergent with MCC-MA) — {SELF_KNOWLEDGE_N} questions/seed")
    print(f"  3. Factual-Retrieval (divergent from WMF-AM) — {FACTUAL_RETRIEVAL_N} questions/seed")

    all_results = []
    out_name = args.output or f"cef_validity_{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = RESULTS_DIR / out_name

    for model in models:
        try:
            r = run_model(model, seeds)
            all_results.extend(r)
        except Exception as e:
            print(f"\nFATAL ERROR for {model}: {e}")
            traceback.print_exc()

        # Incremental save after each model
        summary = compute_summary(all_results)
        with open(out_path, "w") as f:
            json.dump({
                "experiment": "cef_convergent_divergent_validity",
                "timestamp": datetime.now().isoformat(),
                "seeds": seeds,
                "models_completed": list(set(r["model"] for r in all_results)),
                "total_trials": len(all_results),
                "summary": summary,
                "results": all_results,
            }, f, indent=2)
        print(f"  [saved -> {out_path}]")

    # Final summary table
    summary = compute_summary(all_results)
    probes = ["RF-POC", "Self-Knowledge", "Factual-Retrieval"]
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — Convergent/Divergent Validity")
    print(f"{'='*70}")
    hdr = f"{'Model':<30} " + " ".join(f"{p:>18}" for p in probes)
    print(hdr)
    print("-" * len(hdr))
    for model in models:
        if model in summary:
            s = summary[model]
            vals = []
            for p in probes:
                if p in s:
                    vals.append(f"{s[p]['mean']:>8.3f} +/- {s[p]['std']:.3f}")
                else:
                    vals.append(f"{'—':>18}")
            print(f"{model:<30} " + " ".join(vals))

    print(f"\nExpected correlations:")
    print(f"  RF-POC         <-> WMF-AM : POSITIVE (convergent)")
    print(f"  Self-Knowledge <-> MCC-MA : POSITIVE (convergent)")
    print(f"  Factual-Retrieval <-> WMF-AM : NEAR ZERO (divergent)")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
