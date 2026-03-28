"""
CEF Full Benchmark — Main Track Expansion (N≥20 models, 4 dimensions)

Runs WMF-AM, WMF-IM, WMF-IR, MCC-MA, EMC-lite, and CLA-DC across all models
in config.py. Includes convergent/divergent validity probes (MMLU-like, GSM8K-like).

Usage:
    # Phase 1: Ollama models only (no API keys needed)
    python cef_benchmark.py --phase ollama --seeds 4

    # Phase 2: API models (needs .env with keys)
    python cef_benchmark.py --phase api --seeds 2

    # Single model
    python cef_benchmark.py --models ollama:qwen2.5:7b --seeds 2

    # All models
    python cef_benchmark.py --phase all --seeds 4
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

WMF_AM_DEPTHS = [3, 5, 7]        # K values for WMF-AM
WMF_AM_PROBES_PER_DEPTH = 5      # trials per depth per seed
WMF_IM_LOAD_LEVELS = [3, 5, 7, 10]
WMF_IM_PROBES_PER_LEVEL = 5
WMF_IR_PROBES = 15
MCC_MA_PROBLEMS = 20
CLA_DC_LOAD_LEVELS = [3, 5, 7, 10, 14]
CLA_DC_PROBES_PER_LEVEL = 5

# Convergent/divergent validity
VALIDITY_MMLU_N = 20   # MMLU-like knowledge items
VALIDITY_GSM8K_N = 10  # GSM8K-like math items

# Rate limiting
API_DELAY = 1.0   # seconds between API calls
OLLAMA_DELAY = 0.3

OLLAMA_MODELS = [k for k in MODELS if MODELS[k]["provider"] == "ollama"]
API_MODELS = [k for k in MODELS if MODELS[k]["provider"] != "ollama"]

# ── Entity/Template banks ────────────────────────────────────────────────────

ENTITIES = [
    ("Alice", "owns", "paintings"), ("Bob", "has", "coins"),
    ("Carol", "collected", "stamps"), ("David", "saved", "documents"),
    ("Emma", "scored", "points"), ("Frank", "planted", "trees"),
    ("Grace", "wrote", "poems"), ("Henry", "built", "models"),
    ("Iris", "caught", "fish"), ("James", "sold", "tickets"),
    ("Kate", "baked", "loaves"), ("Leo", "ran", "miles"),
    ("Mia", "read", "books"), ("Noah", "drew", "sketches"),
]

TEMPLATES = {
    "points_scoring": {
        "entity_word": "points",
        "gain": "{entity} gains {amount} points.",
        "lose": "{entity} loses {amount} points.",
        "transfer": "{giver} gives {amount} points to {receiver}.",
    },
    "warehouse": {
        "entity_word": "units",
        "gain": "{entity} receives {amount} units.",
        "lose": "{entity} ships {amount} units.",
        "transfer": "{giver} transfers {amount} units to {receiver}.",
    },
    "bank_accounts": {
        "entity_word": "dollars",
        "gain": "${amount} deposited into {entity}'s account.",
        "lose": "${amount} withdrawn from {entity}'s account.",
        "transfer": "${amount} transferred from {giver} to {receiver}.",
    },
}


# ── WMF-AM: Active Manipulation ──────────────────────────────────────────────

def build_wmf_am(rng: random.Random, k: int, template_name: str = "points_scoring"):
    """Build one WMF-AM probe with K operations."""
    tmpl = TEMPLATES[template_name]
    entities = rng.sample([e for e, _, _ in ENTITIES], 3)
    state = {e: rng.randint(5, 20) for e in entities}
    initial = dict(state)
    ops = []

    for _ in range(k):
        op = rng.choice(["add", "sub", "transfer"])
        if op == "add":
            e = rng.choice(entities)
            amt = rng.randint(1, 10)
            state[e] += amt
            ops.append(tmpl["gain"].format(entity=e, amount=amt))
        elif op == "sub":
            e = rng.choice(entities)
            amt = min(rng.randint(1, 5), state[e] - 1)
            if amt > 0:
                state[e] -= amt
                ops.append(tmpl["lose"].format(entity=e, amount=amt))
            else:
                state[e] += 1
                ops.append(tmpl["gain"].format(entity=e, amount=1))
        else:
            g, r = rng.sample(entities, 2)
            amt = min(rng.randint(1, 3), state[g] - 1)
            if amt > 0:
                state[g] -= amt
                state[r] += amt
                ops.append(tmpl["transfer"].format(giver=g, amount=amt, receiver=r))
            else:
                ops.append(f"No change this round.")

    query_e = rng.choice(entities)
    state_str = ", ".join(f"{e}: {v} {tmpl['entity_word']}" for e, v in initial.items())
    ops_str = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(ops))

    prompt = f"""Track the sequence of updates. Do NOT re-read the initial state after this.

Initial state:
{state_str}

Operations (apply in order):
{ops_str}

After all operations, how many {tmpl['entity_word']} does {query_e} have?

Respond with ONLY the final number."""

    return prompt, state[query_e], query_e, template_name


def run_wmf_am(model: str, seeds: list[int]) -> list[dict]:
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        for k in WMF_AM_DEPTHS:
            for t_idx, tmpl in enumerate(list(TEMPLATES.keys())[:2]):  # 2 templates
                for _ in range(WMF_AM_PROBES_PER_DEPTH):
                    prompt, correct, qe, tname = build_wmf_am(rng, k, tmpl)
                    try:
                        resp = call_model(model, prompt)
                        nums = re.findall(r"-?\d+", resp)
                        pred = int(nums[0]) if nums else -1
                    except Exception as e:
                        pred = -999
                        resp = f"ERROR: {e}"
                    results.append({
                        "sub_dim": "WMF-AM", "model": model, "seed": seed,
                        "k": k, "template": tname, "correct": correct,
                        "predicted": pred, "accurate": int(pred == correct),
                    })
                    _delay(model)
    return results


# ── WMF-IM: Information Maintenance ──────────────────────────────────────────

def build_wmf_im(rng: random.Random, n_targets: int, interference: str):
    templates = rng.sample(ENTITIES, n_targets)
    targets = [{"entity": e, "verb": v, "obj": o, "value": rng.randint(10, 99)}
               for e, v, o in templates]

    if interference == "high":
        distractors = [f"Meanwhile, X-{e} also {v} {rng.randint(10,99)} {o}."
                       for e, v, o in rng.sample(ENTITIES, min(3, len(ENTITIES)))]
    else:
        distractors = [f"The temperature in zone {i} was {rng.randint(10,40)} degrees."
                       for i in range(3)]

    sentences = [f"{t['entity']} {t['verb']} {t['value']} {t['obj']}." for t in targets]
    sentences.extend(distractors)
    rng.shuffle(sentences)

    query = rng.choice(targets)
    prompt = f"""Read the passage, then answer the question.

---
{' '.join(sentences)}
---

How many {query['obj']} did {query['entity']} have? Reply with ONLY the number."""

    return prompt, query["value"]


def run_wmf_im(model: str, seeds: list[int]) -> list[dict]:
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        for n in WMF_IM_LOAD_LEVELS:
            for intf in ["high", "low"]:
                for _ in range(WMF_IM_PROBES_PER_LEVEL):
                    prompt, correct = build_wmf_im(rng, n, intf)
                    try:
                        resp = call_model(model, prompt)
                        nums = re.findall(r"\d+", resp)
                        pred = int(nums[0]) if nums else -1
                    except Exception as e:
                        pred = -999
                    results.append({
                        "sub_dim": "WMF-IM", "model": model, "seed": seed,
                        "n_targets": n, "interference": intf,
                        "correct": correct, "predicted": pred,
                        "accurate": int(pred == correct),
                    })
                    _delay(model)
    return results


# ── WMF-IR: Interference Resistance ──────────────────────────────────────────

def run_wmf_ir(model: str, seeds: list[int]) -> list[dict]:
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        for _ in range(WMF_IR_PROBES):
            n = rng.choice([3, 5, 7])
            templates_a = rng.sample(ENTITIES, min(n, len(ENTITIES)))
            list_a = [(e, v, o, rng.randint(10, 49)) for e, v, o in templates_a]
            list_b = [(f"X-{e}", v, o, rng.randint(50, 99)) for e, v, o in templates_a]

            a_str = " ".join(f"{e} {v} {val} {o}." for e, v, o, val in list_a)
            b_str = " ".join(f"{e} {v} {val} {o}." for e, v, o, val in list_b)

            query = rng.choice(list_a)
            prompt = f"""Read both passages, then answer about the FIRST passage only.

First passage: {a_str}

Second passage: {b_str}

According to the FIRST passage only, how many {query[2]} did {query[0]} have? Reply with ONLY the number."""

            try:
                resp = call_model(model, prompt)
                nums = re.findall(r"\d+", resp)
                pred = int(nums[0]) if nums else -1
            except Exception as e:
                pred = -999
            results.append({
                "sub_dim": "WMF-IR", "model": model, "seed": seed,
                "n_list": n, "correct": query[3], "predicted": pred,
                "accurate": int(pred == query[3]),
            })
            _delay(model)
    return results


# ── MCC-MA: Metacognitive Monitoring Accuracy ────────────────────────────────

MCC_PROBLEMS = [
    # 20 mixed-difficulty problems (factual + reasoning)
    {"q": "What is the capital of Australia?", "a": "Canberra", "domain": "factual"},
    {"q": "What is 17 × 23?", "a": "391", "domain": "math"},
    {"q": "In which year did the Berlin Wall fall?", "a": "1989", "domain": "factual"},
    {"q": "What is the derivative of x^3 + 2x?", "a": "3x^2 + 2", "domain": "math"},
    {"q": "Who wrote 'Crime and Punishment'?", "a": "Fyodor Dostoevsky", "domain": "factual"},
    {"q": "What is 256 ÷ 16?", "a": "16", "domain": "math"},
    {"q": "What element has atomic number 79?", "a": "Gold", "domain": "factual"},
    {"q": "What is the square root of 1764?", "a": "42", "domain": "math"},
    {"q": "Which planet has the most moons in our solar system?", "a": "Saturn", "domain": "factual"},
    {"q": "If f(x) = ln(x^2 + 1), what is f'(0)?", "a": "0", "domain": "math"},
    {"q": "What is the longest river in Africa?", "a": "Nile", "domain": "factual"},
    {"q": "What is 7! (7 factorial)?", "a": "5040", "domain": "math"},
    {"q": "Who discovered penicillin?", "a": "Alexander Fleming", "domain": "factual"},
    {"q": "What is the integral of 1/x?", "a": "ln|x| + C", "domain": "math"},
    {"q": "What is the chemical formula for sulfuric acid?", "a": "H2SO4", "domain": "factual"},
    {"q": "What is 13^2 + 14^2?", "a": "365", "domain": "math"},
    {"q": "In which country is Machu Picchu located?", "a": "Peru", "domain": "factual"},
    {"q": "What is the sum of the first 20 positive integers?", "a": "210", "domain": "math"},
    {"q": "What is the speed of light in m/s (approximate)?", "a": "3 × 10^8 or 300000000", "domain": "factual"},
    {"q": "If 3x + 7 = 22, what is x?", "a": "5", "domain": "math"},
]


def run_mcc_ma(model: str, seeds: list[int]) -> list[dict]:
    """Phase 1: answer 20 problems. Phase 2: predict which are wrong."""
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        # Shuffle problem order per seed
        problems = list(MCC_PROBLEMS)
        rng.shuffle(problems)

        # Phase 1: Answer all problems
        answers = []
        q_block = "\n".join(f"Q{i+1}: {p['q']}" for i, p in enumerate(problems))
        answer_prompt = f"""Answer each question. Give ONLY the answer for each, one per line, in format "Q1: answer".

{q_block}"""

        try:
            resp = call_model(model, answer_prompt)
        except Exception as e:
            resp = f"ERROR: {e}"

        # Parse answers
        model_answers = {}
        for line in resp.split("\n"):
            m = re.match(r"Q(\d+):\s*(.+)", line.strip())
            if m:
                model_answers[int(m.group(1))] = m.group(2).strip()

        # Score each answer
        scored = []
        for i, p in enumerate(problems):
            ma = model_answers.get(i+1, "")
            # Flexible matching
            correct_ans = p["a"].lower().replace(",", "")
            model_ans = ma.lower().replace(",", "").strip()
            is_correct = correct_ans in model_ans or model_ans in correct_ans
            scored.append({"idx": i+1, "correct": is_correct, "model_answer": ma})

        # Phase 2: Ask model to predict which it got wrong (showing Q+A pairs)
        qa_pairs = "\n".join(
            f"Q{i+1}: {p['q']} → Your answer: {model_answers.get(i+1, '(no answer)')}"
            for i, p in enumerate(problems)
        )
        monitor_prompt = f"""You just answered 20 questions. Here are your question-answer pairs:

{qa_pairs}

Now predict: which question numbers do you think you answered INCORRECTLY?
List ONLY the question numbers you think are wrong, comma-separated. If you think all are correct, say "none"."""

        try:
            monitor_resp = call_model(model, monitor_prompt)
        except Exception as e:
            monitor_resp = "ERROR"

        # Parse predicted errors
        predicted_wrong = set()
        if "none" not in monitor_resp.lower():
            for num in re.findall(r"\d+", monitor_resp):
                n = int(num)
                if 1 <= n <= 20:
                    predicted_wrong.add(n)

        actual_wrong = {s["idx"] for s in scored if not s["correct"]}

        # Compute monitoring accuracy
        if len(actual_wrong) == 0 and len(predicted_wrong) == 0:
            ma_score = 1.0  # correctly predicted no errors
        elif len(actual_wrong) == 0:
            ma_score = 0.0  # false alarms
        else:
            # Jaccard similarity between predicted and actual
            intersection = len(predicted_wrong & actual_wrong)
            union = len(predicted_wrong | actual_wrong)
            ma_score = intersection / union if union > 0 else 0.0

        results.append({
            "sub_dim": "MCC-MA", "model": model, "seed": seed,
            "n_correct": sum(1 for s in scored if s["correct"]),
            "n_wrong": len(actual_wrong),
            "n_predicted_wrong": len(predicted_wrong),
            "true_positives": len(predicted_wrong & actual_wrong),
            "false_positives": len(predicted_wrong - actual_wrong),
            "false_negatives": len(actual_wrong - predicted_wrong),
            "ma_jaccard": round(ma_score, 4),
            "actual_wrong": sorted(actual_wrong),
            "predicted_wrong": sorted(predicted_wrong),
        })
        _delay(model)
    return results


# ── MCC-CE: Calibration-Execution Link ───────────────────────────────────────

MCC_CE_PROBLEMS = [
    # 15 problems where models are likely to make some errors
    {"q": "What is 47 × 83?", "a": "3901", "domain": "math"},
    {"q": "What is the cube root of 2197?", "a": "13", "domain": "math"},
    {"q": "What year did the Suez Canal open?", "a": "1869", "domain": "factual"},
    {"q": "What is 19! / 18!?", "a": "19", "domain": "math"},
    {"q": "Who was the 23rd President of the United States?", "a": "Benjamin Harrison", "domain": "factual"},
    {"q": "What is the sum of all prime numbers less than 30?", "a": "129", "domain": "math"},
    {"q": "In what year was the Treaty of Westphalia signed?", "a": "1648", "domain": "factual"},
    {"q": "What is the determinant of [[3,1],[5,2]]?", "a": "1", "domain": "math"},
    {"q": "What element has the highest electronegativity?", "a": "Fluorine", "domain": "factual"},
    {"q": "What is 2^15?", "a": "32768", "domain": "math"},
    {"q": "Who discovered the structure of DNA alongside Watson?", "a": "Francis Crick", "domain": "factual"},
    {"q": "What is the 12th Fibonacci number?", "a": "144", "domain": "math"},
    {"q": "What is the capital of Myanmar?", "a": "Naypyidaw", "domain": "factual"},
    {"q": "What is log_2(1024)?", "a": "10", "domain": "math"},
    {"q": "Which country has the longest coastline?", "a": "Canada", "domain": "factual"},
]


def run_mcc_ce(model: str, seeds: list[int]) -> list[dict]:
    """Three-phase protocol: answer → self-evaluate → correct.
    Measures: P(flagged|wrong), P(improved|flagged∧wrong), P(flagged|correct)."""
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        problems = list(MCC_CE_PROBLEMS)
        rng.shuffle(problems)

        # Phase 1: Answer all problems
        q_block = "\n".join(f"Q{i+1}: {p['q']}" for i, p in enumerate(problems))
        answer_prompt = f"""Answer each question. Give ONLY the answer for each, one per line, in format "Q1: answer".

{q_block}"""
        try:
            resp1 = call_model(model, answer_prompt)
        except Exception as e:
            resp1 = f"ERROR: {e}"
        _delay(model)

        # Parse Phase 1 answers
        model_answers = {}
        for line in resp1.split("\n"):
            m = re.match(r"Q(\d+):\s*(.+)", line.strip())
            if m:
                model_answers[int(m.group(1))] = m.group(2).strip()

        # Score Phase 1
        scored = []
        for i, p in enumerate(problems):
            ma = model_answers.get(i+1, "")
            correct_ans = p["a"].lower().replace(",", "")
            model_ans = ma.lower().replace(",", "").strip()
            is_correct = correct_ans in model_ans or model_ans in correct_ans
            scored.append({"idx": i+1, "correct": is_correct, "model_answer": ma,
                           "gold": p["a"], "question": p["q"]})

        # Phase 2: Self-evaluate — flag uncertain answers + scalar confidence
        qa_pairs = "\n".join(
            f"Q{i+1}: {p['q']} → Your answer: {model_answers.get(i+1, '(no answer)')}"
            for i, p in enumerate(problems)
        )
        eval_prompt = f"""You answered 15 questions. Here are your question-answer pairs:

{qa_pairs}

Review each answer. For each, provide:
1. A confidence score from 0 to 100 (0 = certain it's wrong, 100 = certain it's correct)
2. Your decision: KEEP or REVISE

Format: "Q1: 85 KEEP" or "Q1: 30 REVISE", one per line."""
        try:
            resp2 = call_model(model, eval_prompt)
        except Exception as e:
            resp2 = "ERROR"
        _delay(model)

        flagged = set()
        confidence_scores = {}
        for line in resp2.split("\n"):
            m = re.match(r"Q(\d+):\s*(\d+)\s*(KEEP|REVISE|CONFIDENT|UNCERTAIN)", line.strip(), re.IGNORECASE)
            if m:
                qnum = int(m.group(1))
                conf = int(m.group(2))
                decision = m.group(3).upper()
                confidence_scores[qnum] = conf
                if decision in ("REVISE", "UNCERTAIN"):
                    flagged.add(qnum)
            else:
                # Fallback: try parsing just CONFIDENT/UNCERTAIN
                m2 = re.match(r"Q(\d+):\s*(UNCERTAIN|CONFIDENT|REVISE|KEEP)", line.strip(), re.IGNORECASE)
                if m2:
                    qnum = int(m2.group(1))
                    if m2.group(2).upper() in ("UNCERTAIN", "REVISE"):
                        flagged.add(qnum)

        # Phase 3: Correct flagged answers
        corrections = {}
        if flagged:
            flagged_qs = "\n".join(
                f"Q{s['idx']}: {s['question']} [Your answer: {s['model_answer']}]"
                for s in scored if s["idx"] in flagged
            )
            correct_prompt = f"""You flagged these answers as uncertain. Please provide corrected answers.

{flagged_qs}

For each, give your revised answer. Format: "Q1: revised_answer", one per line."""
            try:
                resp3 = call_model(model, correct_prompt)
            except Exception as e:
                resp3 = "ERROR"
            _delay(model)

            for line in resp3.split("\n"):
                m = re.match(r"Q(\d+):\s*(.+)", line.strip())
                if m:
                    corrections[int(m.group(1))] = m.group(2).strip()

        # Compute MCC-CE metrics
        actual_wrong = {s["idx"] for s in scored if not s["correct"]}
        actual_correct = {s["idx"] for s in scored if s["correct"]}

        # P(flagged | wrong) = sensitivity
        p_flag_wrong = (len(flagged & actual_wrong) / len(actual_wrong)) if actual_wrong else None
        # P(flagged | correct) = false alarm rate
        p_flag_correct = (len(flagged & actual_correct) / len(actual_correct)) if actual_correct else 0.0
        # P(improved | flagged ∧ wrong) = correction rate
        flagged_and_wrong = flagged & actual_wrong
        n_improved = 0
        for idx in flagged_and_wrong:
            if idx in corrections:
                gold = next(s["gold"] for s in scored if s["idx"] == idx)
                corr = corrections[idx].lower().replace(",", "").strip()
                if gold.lower() in corr or corr in gold.lower():
                    n_improved += 1
        p_improved = (n_improved / len(flagged_and_wrong)) if flagged_and_wrong else None

        # Compute calibration from confidence scores
        # ECE-like: |confidence - accuracy| across confidence bins
        conf_correct_pairs = []
        for s in scored:
            if s["idx"] in confidence_scores:
                conf_correct_pairs.append((confidence_scores[s["idx"]] / 100.0, int(s["correct"])))

        if conf_correct_pairs:
            confs = [c for c, _ in conf_correct_pairs]
            accs = [a for _, a in conf_correct_pairs]
            mean_conf = np.mean(confs)
            mean_acc = np.mean(accs)
            overconfidence = round(mean_conf - mean_acc, 4)
        else:
            mean_conf = None
            overconfidence = None

        results.append({
            "sub_dim": "MCC-CE", "model": model, "seed": seed,
            "n_problems": len(problems),
            "n_correct_phase1": len(actual_correct),
            "n_wrong_phase1": len(actual_wrong),
            "n_flagged": len(flagged),
            "n_flagged_wrong": len(flagged & actual_wrong),
            "n_flagged_correct": len(flagged & actual_correct),
            "n_improved": n_improved,
            # Monitoring metrics
            "p_flag_given_wrong": round(p_flag_wrong, 4) if p_flag_wrong is not None else None,
            "p_flag_given_correct": round(p_flag_correct, 4),
            # Control metric
            "p_improved_given_flagged_wrong": round(p_improved, 4) if p_improved is not None else None,
            # Calibration metrics
            "mean_confidence": round(mean_conf, 4) if mean_conf is not None else None,
            "overconfidence": overconfidence,
            "confidence_scores": {str(k): v for k, v in confidence_scores.items()},
        })
    return results


# ── EMC-TO: Temporal Ordering ────────────────────────────────────────────────

EMC_TO_EVENT_BANKS = [
    # Historical events (correct chronological order)
    [("The fall of Constantinople", 1453), ("Columbus reaches the Americas", 1492),
     ("Martin Luther posts 95 Theses", 1517), ("Spanish Armada defeated", 1588),
     ("Thirty Years' War begins", 1618), ("Newton publishes Principia", 1687),
     ("French Revolution begins", 1789), ("Battle of Waterloo", 1815)],
    # Scientific discoveries
    [("Galileo's telescope observations", 1610), ("Newton's Principia", 1687),
     ("Discovery of oxygen", 1774), ("Dalton's atomic theory", 1803),
     ("Darwin's Origin of Species", 1859), ("Mendeleev's periodic table", 1869),
     ("Discovery of X-rays", 1895), ("Einstein's special relativity", 1905)],
    # Technology milestones
    [("First telegraph message", 1844), ("Telephone invented", 1876),
     ("First powered flight", 1903), ("First TV broadcast", 1928),
     ("ENIAC computer operational", 1946), ("Sputnik launched", 1957),
     ("Moon landing", 1969), ("World Wide Web created", 1991)],
]


def run_emc_to(model: str, seeds: list[int]) -> list[dict]:
    """Present scrambled events, ask model to reorder chronologically.
    Score: Kendall's τ between model's ordering and correct ordering."""
    from scipy.stats import kendalltau
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        for bank_idx, bank in enumerate(EMC_TO_EVENT_BANKS):
            # Select 5-8 events from this bank
            for n_events in [5, 7]:
                events = rng.sample(bank, min(n_events, len(bank)))
                correct_order = sorted(events, key=lambda x: x[1])
                shuffled = list(events)
                rng.shuffle(shuffled)

                event_list = "\n".join(f"{i+1}. {e[0]}" for i, e in enumerate(shuffled))
                prompt = f"""Put these events in chronological order (earliest first).

{event_list}

Reply with ONLY the numbers in chronological order, comma-separated. Example: "3, 1, 5, 2, 4"."""

                try:
                    resp = call_model(model, prompt)
                    nums = [int(x.strip()) for x in re.findall(r"\d+", resp)]
                    # Map model's ordering back to events
                    if len(nums) >= len(shuffled):
                        model_order = []
                        for n in nums[:len(shuffled)]:
                            if 1 <= n <= len(shuffled):
                                model_order.append(shuffled[n-1])
                        # Compute Kendall's τ
                        if len(model_order) == len(shuffled):
                            correct_ranks = [correct_order.index(e) for e in model_order]
                            tau, p_val = kendalltau(range(len(correct_ranks)), correct_ranks)
                        else:
                            tau, p_val = 0.0, 1.0
                    else:
                        tau, p_val = 0.0, 1.0
                except Exception as e:
                    tau, p_val = 0.0, 1.0
                    resp = f"ERROR: {e}"

                results.append({
                    "sub_dim": "EMC-TO", "model": model, "seed": seed,
                    "bank": bank_idx, "n_events": n_events,
                    "tau": round(tau, 4), "p_value": round(p_val, 4),
                })
                _delay(model)
    return results


# ── CLA-RA: Resource Allocation ──────────────────────────────────────────────

CLA_RA_PROBLEMS = [
    # Easy
    {"q": "What is 5 + 3?", "a": "8", "difficulty": 1},
    {"q": "What color is the sky on a clear day?", "a": "blue", "difficulty": 1},
    {"q": "How many legs does a dog have?", "a": "4", "difficulty": 1},
    # Medium
    {"q": "What is the derivative of sin(x)?", "a": "cos(x)", "difficulty": 2},
    {"q": "What is the capital of Slovenia?", "a": "Ljubljana", "difficulty": 2},
    {"q": "What is 23 × 17?", "a": "391", "difficulty": 2},
    # Hard
    {"q": "What is the integral of sec^2(x) tan(x)?", "a": "sec^2(x)/2 + C", "difficulty": 3},
    {"q": "In what year was the Peace of Augsburg signed?", "a": "1555", "difficulty": 3},
    {"q": "What is the 15th prime number?", "a": "47", "difficulty": 3},
    # Very hard
    {"q": "What is the sum of the series 1/1! + 1/2! + 1/3! + ... + 1/10!?", "a": "1.7182818", "difficulty": 4},
    {"q": "Who was the Roman Emperor when Pompeii was destroyed?", "a": "Titus", "difficulty": 4},
    {"q": "What is the chromatic number of the Petersen graph?", "a": "3", "difficulty": 4},
]


def run_cla_ra(model: str, seeds: list[int]) -> list[dict]:
    """Measure resource allocation: do models invest more tokens/hedging on harder problems?
    Pearson r(difficulty, response_length) and r(difficulty, hedging_markers)."""
    results = []
    hedging_words = {"maybe", "perhaps", "possibly", "might", "could be", "i think",
                     "not sure", "approximately", "roughly", "around", "about",
                     "if i recall", "i believe", "it seems"}

    for seed in seeds:
        rng = random.Random(seed)
        problems = list(CLA_RA_PROBLEMS)
        rng.shuffle(problems)

        for p in problems:
            prompt = f"""Answer this question. Show your reasoning, then give the final answer.

Question: {p['q']}"""

            try:
                resp = call_model(model, prompt)
                resp_len = len(resp.split())
                resp_lower = resp.lower()
                hedge_count = sum(1 for h in hedging_words if h in resp_lower)
            except Exception as e:
                resp = f"ERROR: {e}"
                resp_len = 0
                hedge_count = 0

            # Check correctness (flexible)
            correct_ans = p["a"].lower().replace(",", "")
            model_ans = resp.lower().replace(",", "").strip()
            is_correct = correct_ans in model_ans

            results.append({
                "sub_dim": "CLA-RA", "model": model, "seed": seed,
                "difficulty": p["difficulty"],
                "question": p["q"],
                "response_length": resp_len,
                "hedge_count": hedge_count,
                "correct": int(is_correct),
            })
            _delay(model)
    return results


# ── CLA-CR: Cognitive Recovery ───────────────────────────────────────────────

CLA_CR_EASY_PROBLEMS = [
    {"q": "What is 7 + 8?", "a": "15"},
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "What is 12 × 3?", "a": "36"},
    {"q": "What color are bananas?", "a": "yellow"},
    {"q": "How many days in a week?", "a": "7"},
    {"q": "What is 100 - 37?", "a": "63"},
    {"q": "What planet is closest to the Sun?", "a": "Mercury"},
    {"q": "What is 9 × 9?", "a": "81"},
]

CLA_CR_HARD_PROBLEMS = [
    {"q": "What is 347 × 29?", "a": "10063"},
    {"q": "What is the 20th prime number?", "a": "71"},
    {"q": "What is the integral of x^2 * e^x?", "a": "e^x(x^2 - 2x + 2) + C"},
    {"q": "In what year was the Edict of Nantes issued?", "a": "1598"},
    {"q": "What is 17^3?", "a": "4913"},
    {"q": "What is the determinant of [[2,3,1],[4,1,3],[1,2,4]]?", "a": "-25"},
    {"q": "What is the sum of all divisors of 360?", "a": "1170"},
    {"q": "Who was the last Ptolemaic ruler of Egypt?", "a": "Cleopatra VII"},
]


def run_cla_cr(model: str, seeds: list[int]) -> list[dict]:
    """Three-block paradigm: Block1 (easy) → Block2 (hard, max load) → Block3 (easy recovery).
    Recovery = accuracy(Block3) / accuracy(Block1). If < 1.0, cognitive fatigue analog."""
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        easy1 = list(CLA_CR_EASY_PROBLEMS)
        hard = list(CLA_CR_HARD_PROBLEMS)
        easy3 = list(CLA_CR_EASY_PROBLEMS)
        rng.shuffle(easy1)
        rng.shuffle(hard)
        rng.shuffle(easy3)

        block_results = {"B1": [], "B2": [], "B3": []}

        # Run all three blocks in sequence via conversation
        history = []
        for block_name, problems in [("B1", easy1), ("B2", hard), ("B3", easy3)]:
            q_block = "\n".join(f"Q{i+1}: {p['q']}" for i, p in enumerate(problems))
            block_prompt = f"""Answer each question with ONLY the answer, one per line.

{q_block}

Format: "Q1: answer"."""

            try:
                resp = call_model(model, block_prompt, history=history)
                history.append({"role": "user", "content": block_prompt})
                history.append({"role": "assistant", "content": resp})
            except Exception as e:
                resp = f"ERROR: {e}"
            _delay(model)

            # Parse and score
            answers = {}
            for line in resp.split("\n"):
                m = re.match(r"Q(\d+):\s*(.+)", line.strip())
                if m:
                    answers[int(m.group(1))] = m.group(2).strip()

            for i, p in enumerate(problems):
                ma = answers.get(i+1, "")
                correct_ans = p["a"].lower().replace(",", "")
                model_ans = ma.lower().replace(",", "").strip()
                is_correct = correct_ans in model_ans or model_ans in correct_ans
                block_results[block_name].append(int(is_correct))

        # Compute recovery
        b1_acc = np.mean(block_results["B1"]) if block_results["B1"] else 0.0
        b2_acc = np.mean(block_results["B2"]) if block_results["B2"] else 0.0
        b3_acc = np.mean(block_results["B3"]) if block_results["B3"] else 0.0
        recovery = (b3_acc / b1_acc) if b1_acc > 0 else 0.0

        results.append({
            "sub_dim": "CLA-CR", "model": model, "seed": seed,
            "b1_accuracy": round(b1_acc, 4),
            "b2_accuracy": round(b2_acc, 4),
            "b3_accuracy": round(b3_acc, 4),
            "recovery": round(recovery, 4),
            "post_load_decline": int(recovery < 0.85),
        })
    return results


# ── CLA-DC: Degradation Curve Shape ──────────────────────────────────────────

def run_cla_dc(model: str, seeds: list[int]) -> list[dict]:
    """Measure WMF-AM accuracy across load levels to characterize degradation curve."""
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        for k in CLA_DC_LOAD_LEVELS:
            for _ in range(CLA_DC_PROBES_PER_LEVEL):
                prompt, correct, qe, tname = build_wmf_am(rng, k, "points_scoring")
                try:
                    resp = call_model(model, prompt)
                    nums = re.findall(r"-?\d+", resp)
                    pred = int(nums[0]) if nums else -1
                except Exception as e:
                    pred = -999
                results.append({
                    "sub_dim": "CLA-DC", "model": model, "seed": seed,
                    "k": k, "correct": correct, "predicted": pred,
                    "accurate": int(pred == correct),
                })
                _delay(model)
    return results


# ── Convergent/Divergent Validity Probes ─────────────────────────────────────

MMLU_ITEMS = [
    {"q": "Which of the following is NOT a type of RNA? (A) mRNA (B) tRNA (C) dRNA (D) rRNA", "a": "C"},
    {"q": "The Krebs cycle occurs in the: (A) cytoplasm (B) nucleus (C) mitochondrial matrix (D) ribosome", "a": "C"},
    {"q": "Which amendment to the US Constitution abolished slavery? (A) 12th (B) 13th (C) 14th (D) 15th", "a": "B"},
    {"q": "In economics, GDP stands for: (A) General Domestic Product (B) Gross Domestic Product (C) Gross Domestic Price (D) General Demand Product", "a": "B"},
    {"q": "The Pythagorean theorem applies to: (A) all triangles (B) right triangles (C) equilateral triangles (D) isosceles triangles", "a": "B"},
    {"q": "Which element is most abundant in Earth's atmosphere? (A) Oxygen (B) Carbon dioxide (C) Nitrogen (D) Argon", "a": "C"},
    {"q": "Who painted the Mona Lisa? (A) Michelangelo (B) Raphael (C) Leonardo da Vinci (D) Donatello", "a": "C"},
    {"q": "The French Revolution began in: (A) 1776 (B) 1789 (C) 1799 (D) 1804", "a": "B"},
    {"q": "Which planet is known as the Red Planet? (A) Venus (B) Jupiter (C) Mars (D) Saturn", "a": "C"},
    {"q": "DNA replication is: (A) conservative (B) dispersive (C) semi-conservative (D) random", "a": "C"},
    {"q": "The Treaty of Versailles ended: (A) World War I (B) World War II (C) Korean War (D) Vietnam War", "a": "A"},
    {"q": "Which organelle is responsible for photosynthesis? (A) Mitochondria (B) Chloroplast (C) Ribosome (D) Lysosome", "a": "B"},
    {"q": "The speed of sound in air is approximately: (A) 100 m/s (B) 343 m/s (C) 1000 m/s (D) 3000 m/s", "a": "B"},
    {"q": "Which philosopher wrote 'The Republic'? (A) Aristotle (B) Socrates (C) Plato (D) Epicurus", "a": "C"},
    {"q": "What is the pH of pure water at 25°C? (A) 0 (B) 7 (C) 10 (D) 14", "a": "B"},
    {"q": "The smallest prime number is: (A) 0 (B) 1 (C) 2 (D) 3", "a": "C"},
    {"q": "Ohm's law relates: (A) force and mass (B) voltage, current, resistance (C) energy and mass (D) pressure and volume", "a": "B"},
    {"q": "The human body has how many chromosomes? (A) 23 (B) 44 (C) 46 (D) 48", "a": "C"},
    {"q": "Which gas is produced during photosynthesis? (A) CO2 (B) N2 (C) O2 (D) H2", "a": "C"},
    {"q": "Newton's second law states F equals: (A) mv (B) ma (C) mgh (D) mv²/r", "a": "B"},
]

GSM8K_ITEMS = [
    {"q": "A store sells apples for $2 each. If Sarah buys 5 apples and pays with a $20 bill, how much change does she get?", "a": "10"},
    {"q": "A train travels at 60 mph. How far does it travel in 2.5 hours?", "a": "150"},
    {"q": "If a rectangle has length 8 and width 5, what is its perimeter?", "a": "26"},
    {"q": "A class has 30 students. If 2/5 are boys, how many girls are in the class?", "a": "18"},
    {"q": "A shirt costs $45 and is on sale for 20% off. What is the sale price?", "a": "36"},
    {"q": "If 3 workers can paint a house in 12 days, how many days would 4 workers take?", "a": "9"},
    {"q": "A car uses 8 liters of fuel per 100 km. How many liters for 350 km?", "a": "28"},
    {"q": "The average of 5 numbers is 12. If four numbers are 10, 8, 15, and 12, what is the fifth?", "a": "15"},
    {"q": "A book has 240 pages. If you read 30 pages per day, how many days to finish?", "a": "8"},
    {"q": "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices remain?", "a": "2"},
]


def run_validity_probes(model: str) -> list[dict]:
    """Run MMLU-like and GSM8K-like items for convergent/divergent validity."""
    results = []

    # MMLU-like
    q_block = "\n".join(f"Q{i+1}: {item['q']}" for i, item in enumerate(MMLU_ITEMS))
    prompt = f"""Answer each multiple-choice question with ONLY the letter (A, B, C, or D).

{q_block}

Format: one answer per line, e.g., "Q1: B"."""

    try:
        resp = call_model(model, prompt)
    except Exception as e:
        resp = f"ERROR: {e}"

    answers = {}
    for line in resp.split("\n"):
        m = re.match(r"Q(\d+):\s*([A-Da-d])", line.strip())
        if m:
            answers[int(m.group(1))] = m.group(2).upper()

    for i, item in enumerate(MMLU_ITEMS):
        pred = answers.get(i+1, "?")
        results.append({
            "sub_dim": "VALIDITY-MMLU", "model": model,
            "idx": i+1, "correct": item["a"], "predicted": pred,
            "accurate": int(pred == item["a"]),
        })

    _delay(model)

    # GSM8K-like
    q_block = "\n".join(f"Q{i+1}: {item['q']}" for i, item in enumerate(GSM8K_ITEMS))
    prompt = f"""Solve each math problem. Give ONLY the final numeric answer for each.

{q_block}

Format: one answer per line, e.g., "Q1: 42"."""

    try:
        resp = call_model(model, prompt)
    except Exception as e:
        resp = f"ERROR: {e}"

    answers = {}
    for line in resp.split("\n"):
        m = re.match(r"Q(\d+):\s*\$?([\d.]+)", line.strip())
        if m:
            answers[int(m.group(1))] = m.group(2)

    for i, item in enumerate(GSM8K_ITEMS):
        pred = answers.get(i+1, "?")
        results.append({
            "sub_dim": "VALIDITY-GSM8K", "model": model,
            "idx": i+1, "correct": item["a"], "predicted": str(pred),
            "accurate": int(str(pred).replace(".0", "") == item["a"]),
        })

    return results


# ── EMC-lite import ──────────────────────────────────────────────────────────

def run_emc_lite_for_model(model: str, seeds: list[int]) -> list[dict]:
    """Run the existing emc_lite protocol for this model."""
    from emc_lite import run_all_trials
    results = []
    for seed in seeds:
        trials = run_all_trials(model, seed=seed)
        for t in trials:
            t["seed"] = seed
            t["sub_dim"] = "EMC-lite"
        results.extend(trials)
    return results


# ── Utilities ────────────────────────────────────────────────────────────────

def _delay(model: str):
    cfg = MODELS.get(model, {})
    if cfg.get("provider") == "ollama":
        time.sleep(OLLAMA_DELAY)
    else:
        time.sleep(API_DELAY)


def compute_summary(results: list[dict]) -> dict:
    """Compute per-model, per-subdimension summary scores."""
    summary = {}
    for r in results:
        model = r["model"]
        sd = r["sub_dim"]
        if model not in summary:
            summary[model] = {}
        if sd not in summary[model]:
            summary[model][sd] = {"correct": 0, "total": 0, "details": []}

        if "accurate" in r:
            summary[model][sd]["correct"] += r["accurate"]
            summary[model][sd]["total"] += 1
        elif "ma_jaccard" in r:
            summary[model][sd]["details"].append(r["ma_jaccard"])
        elif "tau" in r:  # EMC-TO
            summary[model][sd]["details"].append(r["tau"])
        elif "p_flag_given_wrong" in r:  # MCC-CE
            val = r["p_flag_given_wrong"]
            if val is not None:
                summary[model][sd]["details"].append(val)
        elif "recovery" in r:  # CLA-CR
            summary[model][sd]["details"].append(r["recovery"])
        elif "response_length" in r:  # CLA-RA — store correlation later
            summary[model][sd]["details"].append((r["difficulty"], r["response_length"]))

    # Compute means
    for model in summary:
        for sd in summary[model]:
            s = summary[model][sd]
            if s["total"] > 0:
                s["accuracy"] = round(s["correct"] / s["total"], 4)
            elif s["details"]:
                if isinstance(s["details"][0], tuple):
                    # CLA-RA: compute Pearson r
                    from scipy.stats import pearsonr
                    diffs = [d[0] for d in s["details"]]
                    lens = [d[1] for d in s["details"]]
                    if len(set(diffs)) > 1 and len(set(lens)) > 1:
                        corr, _ = pearsonr(diffs, lens)
                        s["accuracy"] = round(corr, 4)
                    else:
                        s["accuracy"] = 0.0
                else:
                    s["accuracy"] = round(float(np.mean(s["details"])), 4)
            else:
                s["accuracy"] = 0.0

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def run_model(model: str, seeds: list[int], skip_emc: bool = False) -> list[dict]:
    """Run the full CEF benchmark (all 11 sub-dimensions) for one model."""
    all_results = []
    t0 = time.time()
    n_steps = 12 if not skip_emc else 11
    step = 0
    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"  Seeds: {seeds}")
    print(f"  Sub-dimensions: 11 (WMF×3 + MCC×2 + EMC×3 + CLA×3)")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    # WMF-AM
    step += 1
    print(f"  [{step}/{n_steps}] WMF-AM ({len(WMF_AM_DEPTHS)} depths × {WMF_AM_PROBES_PER_DEPTH} probes × 2 templates × {len(seeds)} seeds)...")
    try:
        r = run_wmf_am(model, seeds)
        acc = sum(x["accurate"] for x in r) / len(r) if r else 0
        print(f"         → {len(r)} trials, accuracy = {acc:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")
        traceback.print_exc()

    # WMF-IM
    step += 1
    print(f"  [{step}/{n_steps}] WMF-IM ({len(WMF_IM_LOAD_LEVELS)} loads × 2 interference × {WMF_IM_PROBES_PER_LEVEL} probes × {len(seeds)} seeds)...")
    try:
        r = run_wmf_im(model, seeds)
        acc = sum(x["accurate"] for x in r) / len(r) if r else 0
        print(f"         → {len(r)} trials, accuracy = {acc:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")

    # WMF-IR
    step += 1
    print(f"  [{step}/{n_steps}] WMF-IR ({WMF_IR_PROBES} probes × {len(seeds)} seeds)...")
    try:
        r = run_wmf_ir(model, seeds)
        acc = sum(x["accurate"] for x in r) / len(r) if r else 0
        print(f"         → {len(r)} trials, accuracy = {acc:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")

    # MCC-MA
    step += 1
    print(f"  [{step}/{n_steps}] MCC-MA (20 problems × {len(seeds)} seeds)...")
    try:
        r = run_mcc_ma(model, seeds)
        if r:
            avg_j = np.mean([x["ma_jaccard"] for x in r])
            print(f"         → {len(r)} rounds, mean Jaccard = {avg_j:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")

    # MCC-CE
    step += 1
    print(f"  [{step}/{n_steps}] MCC-CE (15 problems × 3 phases × {len(seeds)} seeds)...")
    try:
        r = run_mcc_ce(model, seeds)
        if r:
            avg_pflag = np.mean([x["p_flag_given_wrong"] for x in r if x["p_flag_given_wrong"] is not None])
            print(f"         → {len(r)} rounds, mean P(flag|wrong) = {avg_pflag:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")

    # EMC-lite (covers EI + SA)
    if not skip_emc:
        step += 1
        print(f"  [{step}/{n_steps}] EMC-lite/EI+SA ({len(seeds)} seeds)...")
        try:
            r = run_emc_lite_for_model(model, seeds)
            print(f"         → {len(r)} trials")
            all_results.extend(r)
        except Exception as e:
            print(f"         → ERROR (EMC-lite may need separate run): {e}")

    # EMC-TO
    step += 1
    print(f"  [{step}/{n_steps}] EMC-TO ({len(EMC_TO_EVENT_BANKS)} banks × 2 sizes × {len(seeds)} seeds)...")
    try:
        r = run_emc_to(model, seeds)
        if r:
            avg_tau = np.mean([x["tau"] for x in r])
            print(f"         → {len(r)} trials, mean τ = {avg_tau:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")

    # CLA-DC
    step += 1
    print(f"  [{step}/{n_steps}] CLA-DC ({len(CLA_DC_LOAD_LEVELS)} levels × {CLA_DC_PROBES_PER_LEVEL} probes × {len(seeds)} seeds)...")
    try:
        r = run_cla_dc(model, seeds)
        by_k = {}
        for x in r:
            by_k.setdefault(x["k"], []).append(x["accurate"])
        curve = {k: round(np.mean(v), 3) for k, v in sorted(by_k.items())}
        print(f"         → {len(r)} trials, curve: {curve}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")

    # CLA-RA
    step += 1
    print(f"  [{step}/{n_steps}] CLA-RA ({len(CLA_RA_PROBLEMS)} problems × {len(seeds)} seeds)...")
    try:
        r = run_cla_ra(model, seeds)
        if r:
            from scipy.stats import pearsonr
            diffs = [x["difficulty"] for x in r]
            lens = [x["response_length"] for x in r]
            if len(set(diffs)) > 1 and len(set(lens)) > 1:
                corr, _ = pearsonr(diffs, lens)
                print(f"         → {len(r)} trials, r(difficulty, length) = {corr:.3f}")
            else:
                print(f"         → {len(r)} trials, insufficient variance")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")

    # CLA-CR
    step += 1
    print(f"  [{step}/{n_steps}] CLA-CR (3-block paradigm × {len(seeds)} seeds)...")
    try:
        r = run_cla_cr(model, seeds)
        if r:
            avg_rec = np.mean([x["recovery"] for x in r])
            print(f"         → {len(r)} rounds, mean recovery = {avg_rec:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")

    # Validity probes
    step += 1
    print(f"  [{step}/{n_steps}] Validity probes (MMLU={VALIDITY_MMLU_N}, GSM8K={VALIDITY_GSM8K_N})...")
    try:
        r = run_validity_probes(model)
        mmlu_acc = np.mean([x["accurate"] for x in r if x["sub_dim"] == "VALIDITY-MMLU"])
        gsm_acc = np.mean([x["accurate"] for x in r if x["sub_dim"] == "VALIDITY-GSM8K"])
        print(f"         → MMLU: {mmlu_acc:.3f}, GSM8K: {gsm_acc:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\n  Total: {len(all_results)} trials in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="CEF Full Benchmark")
    parser.add_argument("--phase", choices=["ollama", "api", "all"], default="ollama")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--seeds", type=int, default=4, help="Number of random seeds")
    parser.add_argument("--skip-emc", action="store_true", help="Skip EMC-lite (run separately)")
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

    print(f"CEF Full Benchmark")
    print(f"Models: {len(models)}")
    print(f"Seeds: {seeds}")
    print(f"Phase: {args.phase}")
    print(f"Start: {datetime.now().isoformat()}")

    all_results = []
    for model in models:
        try:
            r = run_model(model, seeds, skip_emc=args.skip_emc)
            all_results.extend(r)
        except Exception as e:
            print(f"\nFATAL ERROR for {model}: {e}")
            traceback.print_exc()

        # Save incrementally
        out_name = args.output or f"cef_benchmark_{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path = RESULTS_DIR / out_name
        with open(out_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "seeds": seeds,
                "models_completed": list(set(r["model"] for r in all_results)),
                "total_trials": len(all_results),
                "results": all_results,
            }, f, indent=2)

    # Final summary
    summary = compute_summary(all_results)
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    dims = ["WMF-AM", "WMF-IM", "WMF-IR", "MCC-MA", "MCC-CE", "EMC-lite",
            "EMC-TO", "CLA-DC", "CLA-RA", "CLA-CR", "VALIDITY-MMLU", "VALIDITY-GSM8K"]
    hdr = f"{'Model':<25} " + " ".join(f"{d:>8}" for d in dims)
    print(hdr)
    print("-" * len(hdr))
    for model in models:
        if model in summary:
            s = summary[model]
            vals = []
            for d in dims:
                v = s.get(d, {}).get("accuracy", "-")
                vals.append(f"{v:>8}" if isinstance(v, str) else f"{v:>8.3f}")
            print(f"{model:<25} " + " ".join(vals))

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
