"""
CLA-CR-v3: Cognitive Load Adaptation — Recovery probe (v3, hardened)

Problem with v1/v2: recovery = 1.000 for ALL models because:
  - Multi-digit multiplication is still solvable by LLMs (they tokenize digits)
  - Medium blocks were trivial (factual recall, basic arithmetic)
  - Binary recovery metric masks partial degradation
  - Only 5 blocks — not enough to observe cumulative fatigue

v3 fixes:
  1. ACTUALLY impossible overload: working-memory-exceeding tasks (memorise
     30+ random words then answer positional queries), rapid interleaved
     context-switching across 5+ task types simultaneously.
  2. Genuinely medium problems: 2-step word problems, simple logic puzzles,
     syllogisms — require reasoning but are solvable.
  3. Recovery = medium_after / medium_before (ratio), not binary.
  4. 7 alternating blocks: M1-O1-M2-O2-M3-O3-M4 for a real recovery curve.
  5. Overload blocks get progressively harder (fatigue accumulation).

Usage:
    python cef_cla_cr_v3.py --seeds 2
    python cef_cla_cr_v3.py --models ollama:qwen2.5:7b ollama:llama3.1:8b
    python cef_cla_cr_v3.py --models ollama:deepseek-r1:14b --seeds 3 --output test.json
"""

import argparse
import json
import random
import re
import string
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, RESULTS_DIR, call_model

# ── Default models (7 Ollama, excluding 70b for timeout reasons) ────────────
DEFAULT_MODELS = [
    "ollama:qwen2.5:7b",
    "ollama:qwen2.5:14b",
    "ollama:qwen2.5:32b",
    "ollama:llama3.1:8b",
    "ollama:gemma2:27b",
    "ollama:deepseek-r1:14b",
    "ollama:mistral:7b",
]

API_DELAY = 1.0
OLLAMA_DELAY = 0.3


def _delay(model):
    cfg = MODELS.get(model, {})
    if cfg.get("provider") == "ollama":
        time.sleep(OLLAMA_DELAY)
    else:
        time.sleep(API_DELAY)


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDIUM BLOCK GENERATORS — genuinely medium difficulty
# ═══════════════════════════════════════════════════════════════════════════════

# Pool of 2-step word problems
_WORD_PROBLEM_TEMPLATES = [
    {
        "template": (
            "{name1} has {a} apples. {name2} gives {name1} {b} more apples, "
            "then {name1} gives away half of all the apples. "
            "How many apples does {name1} have now?"
        ),
        "answer_fn": lambda a, b: (a + b) // 2,
        "constraints": lambda: {"a": random.choice(range(10, 30, 2)), "b": random.choice(range(4, 20, 2))},
    },
    {
        "template": (
            "A store sells shirts for ${price} each. {name1} buys {n} shirts "
            "and gets a {disc}% discount on the total. How much does {name1} pay?"
        ),
        "answer_fn": lambda price, n, disc: round(price * n * (100 - disc) / 100, 2),
        "constraints": lambda: {"price": random.choice([15, 20, 25, 30]), "n": random.randint(2, 5),
                                "disc": random.choice([10, 20, 25])},
    },
    {
        "template": (
            "{name1} drives at {s1} km/h for {t1} hours, then at {s2} km/h for {t2} hours. "
            "What is the total distance traveled in km?"
        ),
        "answer_fn": lambda s1, t1, s2, t2: s1 * t1 + s2 * t2,
        "constraints": lambda: {"s1": random.choice([40, 50, 60, 80]), "t1": random.randint(1, 3),
                                "s2": random.choice([30, 45, 70, 90]), "t2": random.randint(1, 3)},
    },
    {
        "template": (
            "A tank holds {cap} liters. Water flows in at {rate_in} liters/min and "
            "drains at {rate_out} liters/min. Starting empty, how many minutes "
            "until the tank is full?"
        ),
        "answer_fn": lambda cap, rate_in, rate_out: cap // (rate_in - rate_out),
        "constraints": lambda: (lambda ri, ro: {"cap": (ri - ro) * random.choice([10, 15, 20, 30]),
                                                "rate_in": ri, "rate_out": ro})(
            random.choice([8, 10, 12, 15]), random.choice([2, 3, 4, 5])),
    },
]

# Pool of simple logic puzzles
_LOGIC_PUZZLES = [
    {
        "q": (
            "All bloops are razzles. All razzles are lazzles. "
            "If something is a bloop, is it necessarily a lazzle? "
            "Answer YES or NO, then explain in one sentence."
        ),
        "a_contains": "YES",
    },
    {
        "q": (
            "Some cats are fluffy. All fluffy things are soft. "
            "Can we conclude that some cats are soft? "
            "Answer YES or NO, then explain in one sentence."
        ),
        "a_contains": "YES",
    },
    {
        "q": (
            "No fish can fly. Some birds can fly. "
            "Can we conclude that no fish are birds? "
            "Answer YES or NO, then explain in one sentence."
        ),
        "a_contains": "YES",
    },
    {
        "q": (
            "All roses are flowers. Some flowers fade quickly. "
            "Can we conclude that some roses fade quickly? "
            "Answer YES or NO, then explain in one sentence."
        ),
        "a_contains": "NO",
    },
    {
        "q": (
            "If it rains, the ground is wet. The ground is wet. "
            "Can we conclude it rained? "
            "Answer YES or NO, then explain in one sentence."
        ),
        "a_contains": "NO",
    },
    {
        "q": (
            "Every dog in this park is wearing a collar. Rex is in this park. "
            "Rex is a dog. Is Rex wearing a collar? "
            "Answer YES or NO, then explain in one sentence."
        ),
        "a_contains": "YES",
    },
    {
        "q": (
            "No prime number greater than 2 is even. 15 is odd. "
            "Can we conclude that 15 is prime? "
            "Answer YES or NO, then explain in one sentence."
        ),
        "a_contains": "NO",
    },
    {
        "q": (
            "All squares are rectangles. All rectangles have four sides. "
            "Does a square have four sides? "
            "Answer YES or NO, then explain in one sentence."
        ),
        "a_contains": "YES",
    },
]

# 2-step sequence / pattern problems
_SEQUENCE_PROBLEMS = [
    {"q": "What is the next number in the sequence: 2, 6, 18, 54, ...?", "a": "162"},
    {"q": "What is the next number in the sequence: 3, 7, 15, 31, ...?", "a": "63"},
    {"q": "What is the next number in the sequence: 1, 4, 9, 16, 25, ...?", "a": "36"},
    {"q": "What is the next number in the sequence: 2, 3, 5, 8, 13, ...?", "a": "21"},
    {"q": "What is the next number in the sequence: 1, 1, 2, 3, 5, 8, ...?", "a": "13"},
    {"q": "What is the next number in the sequence: 100, 81, 64, 49, ...?", "a": "36"},
    {"q": "What is the next number in the sequence: 5, 11, 23, 47, ...?", "a": "95"},
    {"q": "What is the next number in the sequence: 2, 5, 10, 17, 26, ...?", "a": "37"},
]

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace",
         "Henry", "Iris", "James", "Karen", "Leo", "Mia", "Noah"]


def _generate_medium_problem(rng):
    """Generate one medium-difficulty problem. Returns (question_str, answer_str, problem_type)."""
    kind = rng.choice(["word_problem", "logic", "sequence"])

    if kind == "word_problem":
        tmpl = rng.choice(_WORD_PROBLEM_TEMPLATES)
        old_state = random.getstate()
        random.setstate(rng.getstate())
        params = tmpl["constraints"]()
        rng.setstate(random.getstate())
        random.setstate(old_state)
        names = rng.sample(NAMES, 2)
        params["name1"] = names[0]
        params["name2"] = names[1]
        q = tmpl["template"].format(**params)
        # Compute answer using only the numeric params
        numeric = {k: v for k, v in params.items() if isinstance(v, (int, float))}
        ans = tmpl["answer_fn"](**numeric)
        ans_str = str(ans)
        if ans_str.endswith(".0"):
            ans_str = ans_str[:-2]
        return (
            q + "\nGive ONLY the numerical answer.",
            ans_str,
            "word_problem",
        )

    elif kind == "logic":
        puzzle = rng.choice(_LOGIC_PUZZLES)
        return (puzzle["q"], puzzle["a_contains"], "logic")

    else:  # sequence
        prob = rng.choice(_SEQUENCE_PROBLEMS)
        return (prob["q"] + "\nGive ONLY the number.", prob["a"], "sequence")


def _build_medium_block(rng, n_problems=8):
    """Build a medium block of n_problems diverse problems."""
    problems = []
    for _ in range(n_problems):
        q, a, ptype = _generate_medium_problem(rng)
        problems.append({"q": q, "a": a, "type": ptype})
    return problems


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERLOAD BLOCK GENERATORS — actually impossible tasks
# ═══════════════════════════════════════════════════════════════════════════════

# Common nouns for word memorization (concrete, no obvious semantic grouping)
_WORD_BANK = [
    "anchor", "basket", "candle", "dolphin", "engine", "falcon", "glacier",
    "hammer", "insect", "jungle", "kettle", "lantern", "marble", "needle",
    "oyster", "penguin", "quarry", "ribbon", "saddle", "tunnel", "umbrella",
    "vessel", "walnut", "zipper", "badge", "carpet", "dagger", "elbow",
    "fossil", "gravel", "helmet", "ivory", "jacket", "kitten", "locket",
    "magnet", "napkin", "orchid", "pocket", "quartz", "rocket", "socket",
    "tablet", "urchin", "velvet", "warden", "yogurt", "beacon", "cobalt",
    "donkey", "faucet", "goblet", "hatchet", "igloo", "jigsaw", "kayak",
    "lizard", "muffin", "nutmeg", "offset", "parcel", "raisin", "sponge",
    "thimble", "violin", "waffle", "zenith", "apricot", "bonfire", "cactus",
    "drizzle", "emerald", "feather", "gazelle", "hammock", "icicle", "juniper",
]


def _build_overload_memory(rng, n_words, n_queries):
    """
    Overload type 1: Memorise a random word list, then answer positional queries.
    At n_words >= 25, this exceeds working memory for any model — the list is
    presented once and must be recalled without re-reading.
    """
    words = rng.sample(_WORD_BANK, n_words)

    # Build queries about positions
    queries = []
    indices = rng.sample(range(n_words), min(n_queries, n_words))
    for idx in indices:
        query_type = rng.choice(["position_of_word", "word_at_position", "neighbors"])
        if query_type == "position_of_word":
            queries.append({
                "q": f"What position number (1-indexed) is the word '{words[idx]}'?",
                "a": str(idx + 1),
            })
        elif query_type == "word_at_position":
            queries.append({
                "q": f"What word is at position {idx + 1}?",
                "a": words[idx],
            })
        else:  # neighbors
            if 0 < idx < n_words - 1:
                queries.append({
                    "q": f"What word comes immediately AFTER '{words[idx]}'?",
                    "a": words[idx + 1],
                })
            else:
                # Fall back to position query for edge cases
                queries.append({
                    "q": f"What word is at position {idx + 1}?",
                    "a": words[idx],
                })

    word_list_str = "\n".join(f"{i+1}. {w}" for i, w in enumerate(words))
    query_str = "\n".join(f"Q{i+1}: {q['q']}" for i, q in enumerate(queries))

    prompt = f"""MEMORIZATION TASK: Study the following list of {n_words} words carefully.
You will NOT be able to refer back to this list.

{word_list_str}

Now answer these questions about the list above from memory.
Give ONLY the answer for each question, one per line, format "Q1: answer".

{query_str}"""

    return prompt, queries


def _build_overload_interleaved(rng, n_tasks):
    """
    Overload type 2: Rapid context-switching between n_tasks interleaved mini-tasks.
    Each task is a different type (arithmetic, letter manipulation, counting,
    word reversal, alphabetical ordering) and they are interleaved so the model
    must maintain n_tasks separate working contexts simultaneously.
    """
    task_types = ["arithmetic", "letter_count", "word_reverse", "alpha_sort", "digit_sum"]
    selected_types = rng.sample(task_types * 2, n_tasks)  # allow repeats

    # Generate 3 steps per task, interleaved
    tasks = []
    for i, ttype in enumerate(selected_types):
        task_id = f"T{i+1}"
        if ttype == "arithmetic":
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            c = rng.randint(2, 9)
            tasks.append({
                "id": task_id, "type": ttype,
                "steps": [
                    f"[{task_id}] Compute {a} + {b}. Store the result as {task_id}_val.",
                    f"[{task_id}] Multiply {task_id}_val by {c}.",
                    f"[{task_id}] What is the final value of {task_id}_val?",
                ],
                "answer": str((a + b) * c),
            })
        elif ttype == "letter_count":
            word = rng.choice(["extraordinary", "philosophical", "communication",
                               "revolutionary", "determination", "constellation",
                               "uncomfortable", "understanding", "accomplishment"])
            letter = rng.choice(list(word))
            count = word.count(letter)
            tasks.append({
                "id": task_id, "type": ttype,
                "steps": [
                    f"[{task_id}] Consider the word '{word}'.",
                    f"[{task_id}] Count how many times the letter '{letter}' appears in '{word}'.",
                    f"[{task_id}] What is the count for {task_id}?",
                ],
                "answer": str(count),
            })
        elif ttype == "word_reverse":
            word = rng.choice(["elephant", "computer", "dinosaur", "notebook", "umbrella",
                               "sandwich", "mountain", "calendar", "birthday", "treasure"])
            tasks.append({
                "id": task_id, "type": ttype,
                "steps": [
                    f"[{task_id}] Take the word '{word}'.",
                    f"[{task_id}] Reverse the letters of '{word}'.",
                    f"[{task_id}] What is the reversed word for {task_id}?",
                ],
                "answer": word[::-1],
            })
        elif ttype == "alpha_sort":
            words = rng.sample(["cat", "dog", "ant", "bee", "fox", "owl",
                                "rat", "yak", "eel", "hen", "cow", "pig"], 5)
            sorted_words = sorted(words)
            tasks.append({
                "id": task_id, "type": ttype,
                "steps": [
                    f"[{task_id}] Here are words: {', '.join(words)}.",
                    f"[{task_id}] Sort these words alphabetically.",
                    f"[{task_id}] What is the 3rd word in alphabetical order for {task_id}?",
                ],
                "answer": sorted_words[2],
            })
        else:  # digit_sum
            num = rng.randint(10000, 99999)
            dsum = sum(int(d) for d in str(num))
            tasks.append({
                "id": task_id, "type": ttype,
                "steps": [
                    f"[{task_id}] Consider the number {num}.",
                    f"[{task_id}] Sum all the digits of {num}.",
                    f"[{task_id}] What is the digit sum for {task_id}?",
                ],
                "answer": str(dsum),
            })

    # Interleave: step 1 of all tasks, then step 2 of all tasks, then step 3
    interleaved_lines = []
    for step_idx in range(3):
        order = list(range(n_tasks))
        rng.shuffle(order)
        for task_idx in order:
            interleaved_lines.append(tasks[task_idx]["steps"][step_idx])

    prompt = f"""MULTI-TASK CHALLENGE: You must track {n_tasks} separate tasks simultaneously.
The tasks are INTERLEAVED — you must keep each task's state separate in your mind.

Instructions (presented in interleaved order):
{chr(10).join(interleaved_lines)}

Now give the FINAL answer for each task, one per line, format "T1: answer".
Give ONLY the answer value for each task."""

    queries = [{"q": f"Final answer for {t['id']}", "a": t["answer"]} for t in tasks]
    return prompt, queries


def _build_overload_working_memory_span(rng, n_items):
    """
    Overload type 3: Running memory update — track n_items changing variables
    through a sequence of operations. Similar to the n-back task.
    """
    var_names = [f"X{i+1}" for i in range(n_items)]
    values = {v: rng.randint(1, 20) for v in var_names}
    initial = dict(values)

    operations = []
    n_ops = n_items * 3  # 3 operations per variable on average
    for _ in range(n_ops):
        var = rng.choice(var_names)
        op = rng.choice(["add", "subtract", "swap", "double"])
        if op == "add":
            delta = rng.randint(1, 10)
            operations.append(f"Add {delta} to {var}.")
            values[var] += delta
        elif op == "subtract":
            delta = rng.randint(1, min(5, max(1, values[var] - 1)))
            operations.append(f"Subtract {delta} from {var}.")
            values[var] -= delta
        elif op == "swap":
            other = rng.choice([v for v in var_names if v != var])
            operations.append(f"Swap the values of {var} and {other}.")
            values[var], values[other] = values[other], values[var]
        elif op == "double":
            operations.append(f"Double the value of {var}.")
            values[var] *= 2

    init_str = ", ".join(f"{v} = {initial[v]}" for v in var_names)
    ops_str = "\n".join(f"Step {i+1}: {op}" for i, op in enumerate(operations))

    # Ask about a random subset of variables
    n_queries = min(n_items, 5)
    query_vars = rng.sample(var_names, n_queries)
    query_str = "\n".join(f"Q{i+1}: What is the current value of {v}?"
                          for i, v in enumerate(query_vars))

    prompt = f"""VARIABLE TRACKING TASK: Track {n_items} variables through a series of operations.

Initial values: {init_str}

Operations (apply in order):
{ops_str}

Now answer these questions. Give ONLY the number for each, format "Q1: number".

{query_str}"""

    queries = [{"q": f"Value of {v}", "a": str(values[v])} for v in query_vars]
    return prompt, queries


def build_overload_block(rng, difficulty_level):
    """
    Build an overload block at the given difficulty level (1-3).
    Higher levels = more items, more tasks, more operations.
    Mixes all three overload types for each block.

    Returns list of (prompt, queries_list) tuples.
    """
    overload_items = []

    if difficulty_level == 1:
        # Level 1: 25 words/8 queries, 5 interleaved tasks, 6 tracked variables
        overload_items.append(_build_overload_memory(rng, n_words=25, n_queries=8))
        overload_items.append(_build_overload_interleaved(rng, n_tasks=5))
        overload_items.append(_build_overload_working_memory_span(rng, n_items=6))
    elif difficulty_level == 2:
        # Level 2: 35 words/10 queries, 7 interleaved tasks, 8 tracked variables
        overload_items.append(_build_overload_memory(rng, n_words=35, n_queries=10))
        overload_items.append(_build_overload_interleaved(rng, n_tasks=7))
        overload_items.append(_build_overload_working_memory_span(rng, n_items=8))
    else:
        # Level 3: 45 words/12 queries, 9 interleaved tasks, 10 tracked variables
        overload_items.append(_build_overload_memory(rng, n_words=45, n_queries=12))
        overload_items.append(_build_overload_interleaved(rng, n_tasks=9))
        overload_items.append(_build_overload_working_memory_span(rng, n_items=10))

    return overload_items


# ═══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def _score_medium_block(model, problems, history):
    """
    Send a medium block to the model, score responses.
    Returns (accuracy, n_correct, n_total, updated_history).
    """
    q_block = "\n".join(f"Q{i+1}: {p['q']}" for i, p in enumerate(problems))
    prompt = f"""Answer each question below. Give ONLY the answer for each, one per line.
Format: "Q1: answer"

{q_block}"""

    try:
        resp = call_model(model, prompt, history=history)
        history = list(history) if history else []
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": resp})
    except Exception:
        resp = ""
        history = list(history) if history else []
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": "(error)"})

    # Parse answers
    answers = {}
    for line in resp.split("\n"):
        m = re.match(r"Q(\d+):\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            answers[int(m.group(1))] = m.group(2).strip()

    correct = 0
    per_problem = []
    for i, p in enumerate(problems):
        model_ans = answers.get(i + 1, "").strip()
        expected = p["a"]

        if p["type"] == "logic":
            # For logic: check if YES/NO matches
            got = "YES" if "YES" in model_ans.upper() else ("NO" if "NO" in model_ans.upper() else "")
            is_correct = (got == expected)
        else:
            # For numeric/word: normalize and compare
            expected_norm = expected.lower().replace(",", "").replace(" ", "").replace("$", "")
            model_norm = model_ans.lower().replace(",", "").replace(" ", "").replace("$", "")
            is_correct = (expected_norm in model_norm or model_norm == expected_norm)
            # Also try numeric comparison for float answers
            if not is_correct:
                try:
                    is_correct = abs(float(expected_norm) - float(model_norm)) < 0.01
                except (ValueError, TypeError):
                    pass

        if is_correct:
            correct += 1
        per_problem.append({"expected": expected, "got": model_ans, "correct": is_correct})

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, correct, len(problems), history, per_problem


def _score_overload_block(model, overload_items, history):
    """
    Send all overload sub-tasks to the model. Score but don't use accuracy for
    recovery — the point is to tax the model. Returns updated history and overload accuracy.
    """
    total_correct = 0
    total_queries = 0

    for prompt, queries in overload_items:
        try:
            resp = call_model(model, prompt, history=history)
            history = list(history) if history else []
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": resp})
        except Exception:
            resp = ""
            history = list(history) if history else []
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": "(error)"})
        _delay(model)

        # Parse answers — handle both Q1: and T1: formats
        answers = {}
        for line in resp.split("\n"):
            m = re.match(r"[QT](\d+):\s*(.+)", line.strip(), re.IGNORECASE)
            if m:
                answers[int(m.group(1))] = m.group(2).strip()

        for i, q in enumerate(queries):
            model_ans = answers.get(i + 1, "").strip().lower()
            expected = q["a"].strip().lower()
            if expected in model_ans or model_ans == expected:
                total_correct += 1
            total_queries += 1

    overload_acc = total_correct / total_queries if total_queries > 0 else 0.0
    return overload_acc, total_correct, total_queries, history


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PROBE: 7-block paradigm
# ═══════════════════════════════════════════════════════════════════════════════

N_MEDIUM_PROBLEMS = 8  # per medium block


def run_cla_cr_v3(model, seeds):
    """
    7-block paradigm: M1 - O1 - M2 - O2 - M3 - O3 - M4

    Recovery metrics:
      - recovery_i = M_{i+1}_acc / M1_acc   (each medium block vs baseline)
      - mean_recovery = mean(recovery_2, recovery_3, recovery_4)
      - cumulative_degradation = M4_acc / M1_acc (final vs first)
      - overload_acc_curve = [O1_acc, O2_acc, O3_acc] (shows overload is working)

    Overload blocks get progressively harder:
      O1 = difficulty 1, O2 = difficulty 2, O3 = difficulty 3
    """
    results = []

    for seed in seeds:
        rng = random.Random(seed)

        # Generate all blocks
        medium_blocks = [_build_medium_block(rng, N_MEDIUM_PROBLEMS) for _ in range(4)]
        overload_blocks = [build_overload_block(rng, level) for level in [1, 2, 3]]

        # 7-block sequence
        block_sequence = [
            ("M1", "medium", 0),
            ("O1", "overload", 0),
            ("M2", "medium", 1),
            ("O2", "overload", 1),
            ("M3", "medium", 2),
            ("O3", "overload", 2),
            ("M4", "medium", 3),
        ]

        history = []  # Cumulative conversation history (the model sees ALL prior blocks)
        block_data = {}

        for block_name, block_type, block_idx in block_sequence:
            print(f"    [{block_name}] ", end="", flush=True)

            if block_type == "medium":
                acc, n_correct, n_total, history, per_problem = _score_medium_block(
                    model, medium_blocks[block_idx], history
                )
                block_data[block_name] = {
                    "accuracy": acc,
                    "n_correct": n_correct,
                    "n_total": n_total,
                    "per_problem": per_problem,
                }
                print(f"acc={acc:.3f} ({n_correct}/{n_total})", flush=True)
            else:
                acc, n_correct, n_total, history = _score_overload_block(
                    model, overload_blocks[block_idx], history
                )
                block_data[block_name] = {
                    "accuracy": acc,
                    "n_correct": n_correct,
                    "n_total": n_total,
                }
                print(f"overload_acc={acc:.3f} ({n_correct}/{n_total})", flush=True)

            _delay(model)

        # Compute recovery metrics
        m1_acc = block_data["M1"]["accuracy"]
        m2_acc = block_data["M2"]["accuracy"]
        m3_acc = block_data["M3"]["accuracy"]
        m4_acc = block_data["M4"]["accuracy"]
        o1_acc = block_data["O1"]["accuracy"]
        o2_acc = block_data["O2"]["accuracy"]
        o3_acc = block_data["O3"]["accuracy"]

        # Recovery ratios (relative to M1 baseline)
        recovery_2 = m2_acc / m1_acc if m1_acc > 0 else 0.0
        recovery_3 = m3_acc / m1_acc if m1_acc > 0 else 0.0
        recovery_4 = m4_acc / m1_acc if m1_acc > 0 else 0.0
        mean_recovery = np.mean([recovery_2, recovery_3, recovery_4])
        cumulative_degradation = m4_acc / m1_acc if m1_acc > 0 else 0.0

        result = {
            "probe": "CLA-CR-v3",
            "sub_dim": "CLA-CR-v3",
            "model": model,
            "seed": seed,
            # Raw accuracies
            "m1_acc": round(m1_acc, 4),
            "m2_acc": round(m2_acc, 4),
            "m3_acc": round(m3_acc, 4),
            "m4_acc": round(m4_acc, 4),
            "o1_acc": round(o1_acc, 4),
            "o2_acc": round(o2_acc, 4),
            "o3_acc": round(o3_acc, 4),
            # Recovery ratios (the KEY metrics)
            "recovery_2": round(recovery_2, 4),
            "recovery_3": round(recovery_3, 4),
            "recovery_4": round(recovery_4, 4),
            "mean_recovery": round(mean_recovery, 4),
            "cumulative_degradation": round(cumulative_degradation, 4),
            # Overload curve (should decrease if overload is working)
            "overload_curve": [round(o1_acc, 4), round(o2_acc, 4), round(o3_acc, 4)],
            # Detailed block data (for debugging)
            "block_data": {k: {kk: vv for kk, vv in v.items() if kk != "per_problem"}
                           for k, v in block_data.items()},
        }
        results.append(result)

        print(f"    => recovery=[{recovery_2:.3f}, {recovery_3:.3f}, {recovery_4:.3f}], "
              f"mean={mean_recovery:.3f}, cumul_degrad={cumulative_degradation:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLA-CR-v3: Hardened Cognitive Recovery Probe (7-block paradigm)"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Models to test (default: 7 Ollama models)"
    )
    parser.add_argument(
        "--seeds", type=int, default=2,
        help="Number of seeds (default: 2)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output filename (default: cla_cr_v3.json)"
    )
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    models = [m for m in args.models if m in MODELS]

    if not models:
        print(f"ERROR: No valid models found. Available: {list(MODELS.keys())}")
        sys.exit(1)

    out_name = args.output or "cla_cr_v3.json"
    out_path = RESULTS_DIR / out_name

    print(f"CLA-CR-v3: Hardened Cognitive Recovery Probe")
    print(f"{'=' * 60}")
    print(f"Models:  {len(models)}: {models}")
    print(f"Seeds:   {seeds}")
    print(f"Blocks:  M1-O1-M2-O2-M3-O3-M4 (7 blocks, progressive overload)")
    print(f"Output:  {out_path}")
    print(f"Start:   {datetime.now().isoformat()}")
    print(f"{'=' * 60}")

    all_results = []

    for model in models:
        print(f"\n{'─' * 60}")
        print(f"  Model: {model}")
        print(f"{'─' * 60}")
        t0 = time.time()

        try:
            r = run_cla_cr_v3(model, seeds)
            all_results.extend(r)

            if r:
                mean_rec = np.mean([x["mean_recovery"] for x in r])
                mean_deg = np.mean([x["cumulative_degradation"] for x in r])
                mean_o = [np.mean([x["overload_curve"][i] for x in r]) for i in range(3)]
                print(f"  SUMMARY: mean_recovery={mean_rec:.3f}, "
                      f"cumul_degrad={mean_deg:.3f}, "
                      f"overload_curve={[round(x, 3) for x in mean_o]}")
        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            traceback.print_exc()

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        # Incremental save
        with open(out_path, "w") as f:
            json.dump({
                "probe": "CLA-CR-v3",
                "sub_dim": "CLA-CR-v3",
                "timestamp": datetime.now().isoformat(),
                "seeds": seeds,
                "n_medium_problems": N_MEDIUM_PROBLEMS,
                "block_structure": "M1-O1-M2-O2-M3-O3-M4",
                "overload_types": [
                    "word_position_memory (25/35/45 words)",
                    "interleaved_context_switch (5/7/9 tasks)",
                    "variable_tracking (6/8/10 vars)",
                ],
                "models_completed": list(set(x["model"] for x in all_results)),
                "total_rounds": len(all_results),
                "results": all_results,
            }, f, indent=2)
        print(f"  Saved → {out_path}")

    # Final summary table
    print(f"\n{'=' * 70}")
    print("CLA-CR-v3 SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<30} {'M1':>5} {'M2':>5} {'M3':>5} {'M4':>5} "
          f"{'Rec2':>5} {'Rec3':>5} {'Rec4':>5} {'MeanR':>6} {'O1':>5} {'O2':>5} {'O3':>5}")
    print(f"{'─' * 30} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} "
          f"{'─' * 5} {'─' * 5} {'─' * 5} {'─' * 6} {'─' * 5} {'─' * 5} {'─' * 5}")

    for model in models:
        items = [x for x in all_results if x["model"] == model]
        if not items:
            continue
        m1 = np.mean([x["m1_acc"] for x in items])
        m2 = np.mean([x["m2_acc"] for x in items])
        m3 = np.mean([x["m3_acc"] for x in items])
        m4 = np.mean([x["m4_acc"] for x in items])
        r2 = np.mean([x["recovery_2"] for x in items])
        r3 = np.mean([x["recovery_3"] for x in items])
        r4 = np.mean([x["recovery_4"] for x in items])
        mr = np.mean([x["mean_recovery"] for x in items])
        o1 = np.mean([x["o1_acc"] for x in items])
        o2 = np.mean([x["o2_acc"] for x in items])
        o3 = np.mean([x["o3_acc"] for x in items])
        print(f"{model:<30} {m1:5.3f} {m2:5.3f} {m3:5.3f} {m4:5.3f} "
              f"{r2:5.3f} {r3:5.3f} {r4:5.3f} {mr:6.3f} {o1:5.3f} {o2:5.3f} {o3:5.3f}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
