"""
CEF Agent-Level Validation — Does CEF predict actual agent behavior?

Fixed ReAct scaffold + LLM backbone swap. The scaffold is IDENTICAL for all
models; only the LLM backbone changes. This isolates LLM cognitive ability
from scaffold engineering.

10 deterministic tasks spanning working memory, metacognition, episodic memory,
and general agent competence. All tools return deterministic outputs (no real
API calls).

Usage:
    # Phase 1: Ollama models only
    python cef_agent_validation.py --phase ollama --max-steps 15

    # Specific models
    python cef_agent_validation.py --models ollama:qwen2.5:7b ollama:llama3.1:8b

    # All models
    python cef_agent_validation.py --phase all --max-steps 20
"""

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, RESULTS_DIR, call_model

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MAX_STEPS = 15
API_DELAY = 1.0
OLLAMA_DELAY = 0.3

OLLAMA_MODELS = [k for k in MODELS if MODELS[k]["provider"] == "ollama"]
API_MODELS = [k for k in MODELS if MODELS[k]["provider"] != "ollama"]

# CEF dimension tags for correlation analysis
TASK_CEF_MAPPING = {
    "multi_step_calc":       "WMF-AM",
    "entity_tracking":       "WMF-AM",
    "sequential_search":     "WMF-AM",
    "uncertain_lookup":      "MCC-MA",
    "multi_source_conflict": "MCC-MA",
    "conversation_recall":   "EMC",
    "source_attribution":    "EMC",
    "shopping_assistant":    "GENERAL",
    "schedule_coordination": "GENERAL",
    "data_pipeline":         "GENERAL",
}

# ── ReAct Scaffold ───────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are a ReAct agent. You solve tasks by interleaving Thought and Action steps.

Format — you MUST follow this EXACTLY on every turn:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>(<arguments>)

When you have the final answer, use:
THOUGHT: <your final reasoning>
ACTION: FINISH(<your final answer>)

Rules:
- Each response must contain exactly ONE THOUGHT and ONE ACTION.
- Do NOT include OBSERVATION — the system provides that after your ACTION.
- Available tools are listed in the task description.
- Use FINISH() only when you are confident in your answer.
"""


def parse_action(response: str) -> tuple[str, str]:
    """Extract tool name and arguments from model response.

    Returns (tool_name, args_string) or ("INVALID", raw_response).
    """
    # Try to find ACTION: line
    match = re.search(r"ACTION:\s*(\w+)\((.*)?\)", response, re.DOTALL)
    if match:
        tool_name = match.group(1).strip()
        args_str = (match.group(2) or "").strip()
        return tool_name, args_str

    # Fallback: look for tool_name(args) pattern anywhere
    match = re.search(r"(\w+)\(([^)]*)\)", response)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return "INVALID", response


def run_agent(
    model: str,
    task_prompt: str,
    tools: dict[str, callable],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict:
    """Run the ReAct agent loop for a single task.

    Returns dict with keys: steps (list of dicts), final_answer, n_steps, errors.
    """
    history: list[dict] = []
    steps: list[dict] = []
    errors = 0
    final_answer = None

    provider = MODELS[model]["provider"]
    delay = OLLAMA_DELAY if provider == "ollama" else API_DELAY

    for step_i in range(max_steps):
        # Build the user message (first turn includes task, later turns have observations)
        if step_i == 0:
            user_msg = task_prompt
        else:
            # The last observation was already appended to history
            user_msg = steps[-1]["observation"]

        try:
            response = call_model(
                model,
                prompt=user_msg,
                system=REACT_SYSTEM_PROMPT,
                history=history if step_i > 0 else None,
            )
        except Exception as e:
            errors += 1
            steps.append({"step": step_i, "error": str(e)})
            break

        tool_name, args_str = parse_action(response)

        # Record model turn in history
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": response})

        if tool_name == "FINISH":
            final_answer = args_str
            steps.append({
                "step": step_i,
                "response": response,
                "tool": "FINISH",
                "args": args_str,
                "observation": None,
            })
            break

        elif tool_name == "INVALID":
            errors += 1
            observation = "OBSERVATION: Invalid action format. Use ACTION: tool_name(args)"
            steps.append({
                "step": step_i,
                "response": response,
                "tool": "INVALID",
                "args": "",
                "observation": observation,
            })

        elif tool_name in tools:
            try:
                result = tools[tool_name](args_str)
                observation = f"OBSERVATION: {result}"
            except Exception as e:
                errors += 1
                observation = f"OBSERVATION: Error calling {tool_name}: {e}"
            steps.append({
                "step": step_i,
                "response": response,
                "tool": tool_name,
                "args": args_str,
                "observation": observation,
            })

        else:
            errors += 1
            available = ", ".join(tools.keys())
            observation = f"OBSERVATION: Unknown tool '{tool_name}'. Available: {available}"
            steps.append({
                "step": step_i,
                "response": response,
                "tool": tool_name,
                "args": args_str,
                "observation": observation,
            })

        time.sleep(delay)

    return {
        "steps": steps,
        "final_answer": final_answer,
        "n_steps": len(steps),
        "errors": errors,
    }


# ── Deterministic Tool Implementations ───────────────────────────────────────

def _make_calculator():
    """Calculator tool: evaluate simple arithmetic expressions."""
    def calculator(expr: str) -> str:
        expr = expr.strip().strip('"').strip("'")
        # Allow only digits, operators, parens, spaces
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expr):
            return f"Error: invalid expression '{expr}'"
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    return calculator


# Task 2: Entity tracking
_BANK_BALANCES = {"Alice": 100, "Bob": 250, "Carol": 175}
_TRANSFERS = [
    ("Alice", "Bob", 30),
    ("Bob", "Carol", 50),
    ("Carol", "Alice", 20),
    ("Alice", "Carol", 45),
    ("Bob", "Alice", 15),
]
# After transfers: Alice = 100-30+20-45+15 = 60, Bob = 250+30-50-15 = 215, Carol = 175+50-20+45 = 250
_FINAL_BALANCES = {"Alice": 60, "Bob": 215, "Carol": 250}


def _make_check_balance():
    """Returns current balance (starts at initial, but agent must track mentally)."""
    # The tool only returns the INITIAL balance — the agent must track transfers itself.
    def check_balance(person: str) -> str:
        person = person.strip().strip('"').strip("'")
        if person in _BANK_BALANCES:
            return f"{person}'s initial balance is ${_BANK_BALANCES[person]}"
        return f"Unknown person: {person}"
    return check_balance


# Task 3: Sequential search
_FILE_CONTENTS = {
    "report_q1.txt": "Revenue was $1.2M in Q1. Growth rate steady at 5%.",
    "report_q2.txt": "Revenue was $1.4M in Q2. New product launch successful.",
    "report_q3.txt": "Revenue was $1.1M in Q3. Supply chain issues noted. Target code: 7842.",
    "report_q4.txt": "Revenue was $1.5M in Q4. Strong holiday sales.",
    "summary.txt": "Annual revenue was $5.2M. See quarterly reports for details.",
}


def _make_read_file():
    def read_file(filename: str) -> str:
        filename = filename.strip().strip('"').strip("'")
        if filename in _FILE_CONTENTS:
            return _FILE_CONTENTS[filename]
        return f"File not found: {filename}"
    return read_file


def _make_list_files():
    def list_files(_: str = "") -> str:
        return "Files: " + ", ".join(_FILE_CONTENTS.keys())
    return list_files


# Task 4: Uncertain lookup
def _make_verify_fact():
    def verify_fact(claim: str) -> str:
        claim = claim.strip().strip('"').strip("'").lower()
        if "tallest building" in claim and "dubai" in claim:
            return "VERIFIED: The Burj Khalifa in Dubai is the tallest building in the world at 828m."
        if "capital" in claim and "australia" in claim:
            return "VERIFIED: The capital of Australia is Canberra (not Sydney)."
        if "boiling point" in claim and "mercury" in claim:
            return "VERIFIED: The boiling point of mercury is 356.7°C (629.88K)."
        return f"UNVERIFIED: Could not confirm claim. The fact database does not contain information about: {claim}"
    return verify_fact


def _make_answer():
    """Generic answer tool used in some tasks."""
    def answer(text: str) -> str:
        return f"Answer recorded: {text}"
    return answer


# Task 5: Multi-source conflict
_SOURCE_A = "According to the World Atlas (2024), Lake Baikal holds approximately 23,615 cubic kilometers of water, making it the largest freshwater lake by volume."
_SOURCE_B = "The Environmental Database (2023) states Lake Baikal contains approximately 23,000 cubic kilometers of water. Note: Some measurements suggest Lake Tanganyika may hold comparable volume."


def _make_query_source():
    def query_source(source: str) -> str:
        source = source.strip().strip('"').strip("'").lower()
        if "a" in source or "atlas" in source or "1" in source:
            return _SOURCE_A
        if "b" in source or "environmental" in source or "2" in source:
            return _SOURCE_B
        return "Available sources: source_A (World Atlas 2024), source_B (Environmental Database 2023)"
    return query_source


# Task 6: Conversation recall
_TOOL_SEQUENCE = [
    ("weather", "Current weather in Tokyo: 15°C, partly cloudy, humidity 65%"),
    ("news", "Top headline: 'Global summit on AI regulation concludes in Geneva'"),
    ("stock", "AAPL: $187.42 (+1.3%), GOOGL: $142.15 (-0.5%)"),
    ("translate", "Japanese for 'good morning': おはようございます (ohayou gozaimasu)"),
    ("calendar", "Next event: Team meeting at 3:00 PM, Conference Room B"),
]


def _make_tool_sequence():
    """Returns tools that produce a predetermined sequence of results."""
    call_count = {"n": 0}

    def get_info(query: str) -> str:
        idx = min(call_count["n"], len(_TOOL_SEQUENCE) - 1)
        topic, result = _TOOL_SEQUENCE[idx]
        call_count["n"] += 1
        return result

    return get_info, call_count


# Task 7: Source attribution
_DOCUMENTS = {
    "doc1": "Climate Report 2024: Global temperatures rose by 1.2°C above pre-industrial levels. Arctic ice reduced by 13% per decade.",
    "doc2": "Economic Outlook: GDP growth projected at 3.1% for 2025. Unemployment expected to remain below 4.5%.",
    "doc3": "Health Survey: 68% of adults report exercising at least twice weekly. Average sleep duration is 6.8 hours.",
}


def _make_read_document():
    def read_document(doc_id: str) -> str:
        doc_id = doc_id.strip().strip('"').strip("'").lower()
        if doc_id in _DOCUMENTS:
            return _DOCUMENTS[doc_id]
        return f"Unknown document. Available: doc1, doc2, doc3"
    return read_document


# Task 8: Shopping assistant
_ITEMS = {
    "search:headphones": [
        {"id": "H1", "name": "SoundMax Pro", "price": 79.99, "rating": 4.2},
        {"id": "H2", "name": "AudioBliss 300", "price": 45.99, "rating": 4.7},
        {"id": "H3", "name": "BassKing Ultra", "price": 120.00, "rating": 4.8},
        {"id": "H4", "name": "ClearTone Budget", "price": 29.99, "rating": 3.9},
    ],
}

_ITEM_DETAILS = {
    "H1": {"name": "SoundMax Pro", "price": 79.99, "rating": 4.2, "features": "Noise cancelling, 30hr battery, Bluetooth 5.2"},
    "H2": {"name": "AudioBliss 300", "price": 45.99, "rating": 4.7, "features": "Lightweight, 20hr battery, foldable design"},
    "H3": {"name": "BassKing Ultra", "price": 120.00, "rating": 4.8, "features": "Studio quality, 40hr battery, wired+wireless"},
    "H4": {"name": "ClearTone Budget", "price": 29.99, "rating": 3.9, "features": "Basic, 10hr battery, wired only"},
}


def _make_search_items():
    def search_items(query: str) -> str:
        query = query.strip().strip('"').strip("'").lower()
        key = f"search:{query}"
        if key in _ITEMS:
            items = _ITEMS[key]
            lines = [f"  {it['id']}: {it['name']} — ${it['price']}, rating {it['rating']}/5" for it in items]
            return "Found items:\n" + "\n".join(lines)
        return "No items found for query. Try: headphones"
    return search_items


def _make_get_details():
    def get_details(item_id: str) -> str:
        item_id = item_id.strip().strip('"').strip("'").upper()
        if item_id in _ITEM_DETAILS:
            d = _ITEM_DETAILS[item_id]
            return f"{d['name']}: ${d['price']}, rating {d['rating']}/5, features: {d['features']}"
        return f"Item not found: {item_id}"
    return get_details


# Task 9: Schedule coordination
_CALENDARS = {
    "alice": {
        "2024-03-15": ["9:00-10:00 Team standup", "11:00-12:00 Client call", "14:00-15:30 Workshop"],
    },
    "bob": {
        "2024-03-15": ["9:30-10:30 Design review", "13:00-14:00 Lunch meeting"],
    },
    "carol": {
        "2024-03-15": ["9:00-9:30 Quick sync", "10:30-11:30 Planning", "15:00-16:00 Retrospective"],
    },
}
# Free slots for all three on 2024-03-15:
# Alice free: 10:00-11:00, 12:00-14:00, 15:30+
# Bob free: before 9:30, 10:30-13:00, 14:00+
# Carol free: 9:30-10:30, 11:30-15:00, 16:00+
# Common 1-hour slot: 12:00-13:00


def _make_check_calendar():
    def check_calendar(args: str) -> str:
        # Parse "person, date" or "person date"
        parts = re.split(r'[,\s]+', args.strip().strip('"').strip("'"))
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            return "Usage: check_calendar(person, date). Example: check_calendar(alice, 2024-03-15)"
        person = parts[0].lower()
        date = parts[-1]
        if person in _CALENDARS and date in _CALENDARS[person]:
            events = _CALENDARS[person][date]
            return f"{person.title()}'s schedule for {date}:\n" + "\n".join(f"  - {e}" for e in events)
        if person in _CALENDARS:
            return f"{person.title()} has no events on {date} (fully free)."
        return f"Unknown person: {person}. Available: alice, bob, carol"
    return check_calendar


# Task 10: Data pipeline (ETL)
_RAW_DATA = '{"records": [{"id": 1, "value": "42", "status": "active"}, {"id": 2, "value": "17", "status": "inactive"}, {"id": 3, "value": "85", "status": "active"}, {"id": 4, "value": "33", "status": "active"}, {"id": 5, "value": "91", "status": "inactive"}]}'


def _make_extract_data():
    def extract_data(source: str) -> str:
        return _RAW_DATA
    return extract_data


def _make_transform_data():
    def transform_data(instruction: str) -> str:
        instruction = instruction.strip().lower()
        # The correct transform: filter active, convert values to int, sum
        if "active" in instruction or "filter" in instruction:
            return '{"filtered": [{"id": 1, "value": 42}, {"id": 3, "value": 85}, {"id": 4, "value": 33}], "count": 3}'
        if "sum" in instruction:
            return '{"total": 160, "count": 3}'
        return '{"result": "Applied transformation. Specify: filter active records, then sum values."}'
    return transform_data


def _make_load_data():
    def load_data(data: str) -> str:
        data = data.strip()
        if "160" in data:
            return "SUCCESS: Loaded total=160 into destination. Pipeline complete."
        return f"LOADED: Data written to destination. Content: {data[:100]}"
    return load_data


# ── Task Definitions ─────────────────────────────────────────────────────────

def build_tasks() -> list[dict]:
    """Build all 10 tasks with their tools, prompts, and expected answers."""

    tasks = []

    # Task 1: Multi-step calculation (WMF-AM)
    tasks.append({
        "id": "multi_step_calc",
        "name": "Multi-step Calculation",
        "cef_dim": "WMF-AM",
        "prompt": (
            "Calculate the result of ((17 × 3) + 29) × 2 - 15.\n"
            "You must use the calculator tool for each step. Show your work.\n\n"
            "Available tool: calculator(expression)\n"
            "Example: calculator(17 * 3)\n\n"
            "When done, use FINISH(number) with the final numeric answer."
        ),
        "tools": {"calculator": _make_calculator()},
        "expected": "145",
        "eval_fn": _eval_numeric,
    })

    # Task 2: Entity tracking (WMF-AM)
    transfer_desc = "\n".join(
        f"  {i+1}. {src} transfers ${amt} to {dst}"
        for i, (src, dst, amt) in enumerate(_TRANSFERS)
    )
    tasks.append({
        "id": "entity_tracking",
        "name": "Entity Tracking (Bank Transfers)",
        "cef_dim": "WMF-AM",
        "prompt": (
            "Three people have bank accounts:\n"
            "  - Alice: $100\n"
            "  - Bob: $250\n"
            "  - Carol: $175\n\n"
            "The following transfers occur in order:\n"
            f"{transfer_desc}\n\n"
            "You can verify initial balances with: check_balance(person_name)\n"
            "After ALL transfers, what is each person's final balance?\n\n"
            "Available tool: check_balance(person_name)\n\n"
            "Use FINISH(Alice: $X, Bob: $Y, Carol: $Z) with the final balances."
        ),
        "tools": {"check_balance": _make_check_balance()},
        "expected": _FINAL_BALANCES,
        "eval_fn": _eval_entity_tracking,
    })

    # Task 3: Sequential search (WMF-AM)
    tasks.append({
        "id": "sequential_search",
        "name": "Sequential File Search",
        "cef_dim": "WMF-AM",
        "prompt": (
            "Find which file contains the phrase 'Target code' and extract the number after it.\n\n"
            "Available tools:\n"
            "  list_files() — lists all available files\n"
            "  read_file(filename) — reads the content of a file\n\n"
            "Use FINISH(number) with the target code number."
        ),
        "tools": {"list_files": _make_list_files(), "read_file": _make_read_file()},
        "expected": "7842",
        "eval_fn": _eval_numeric,
    })

    # Task 4: Uncertain lookup (MCC-MA)
    tasks.append({
        "id": "uncertain_lookup",
        "name": "Uncertain Fact Lookup",
        "cef_dim": "MCC-MA",
        "prompt": (
            "What is the boiling point of mercury in degrees Celsius?\n\n"
            "IMPORTANT: If you are not fully certain of the answer, use the verify_fact tool.\n"
            "Do NOT guess — verify first.\n\n"
            "Available tool: verify_fact(claim)\n"
            "Example: verify_fact(The boiling point of mercury is around 357°C)\n\n"
            "Use FINISH(answer) with the verified temperature."
        ),
        "tools": {"verify_fact": _make_verify_fact()},
        "expected": "356.7",
        "eval_fn": _eval_uncertain_lookup,
    })

    # Task 5: Multi-source conflict (MCC-MA)
    tasks.append({
        "id": "multi_source_conflict",
        "name": "Multi-Source Conflict Detection",
        "cef_dim": "MCC-MA",
        "prompt": (
            "How much water does Lake Baikal hold? Query both available sources and report.\n\n"
            "Available tool: query_source(source_name)\n"
            "Sources: source_A, source_B\n\n"
            "If the sources disagree, you MUST note the discrepancy in your answer.\n"
            "Use FINISH(your answer including any discrepancies noted)."
        ),
        "tools": {"query_source": _make_query_source()},
        "expected": "conflict",  # sentinel — we check for discrepancy mention
        "eval_fn": _eval_conflict_detection,
    })

    # Task 6: Conversation recall (EMC)
    get_info, call_count = _make_tool_sequence()
    tasks.append({
        "id": "conversation_recall",
        "name": "Conversation Recall",
        "cef_dim": "EMC",
        "prompt": (
            "I need you to gather information using the get_info tool. Call it 5 times with these queries:\n"
            "  1. get_info(weather in Tokyo)\n"
            "  2. get_info(latest news)\n"
            "  3. get_info(stock prices)\n"
            "  4. get_info(translate good morning to Japanese)\n"
            "  5. get_info(my calendar)\n\n"
            "After all 5 calls, tell me: What was the FIRST piece of information you retrieved?\n"
            "(i.e., what did the weather query return?)\n\n"
            "Available tool: get_info(query)\n\n"
            "Use FINISH(the first result you got) — quote it as accurately as possible."
        ),
        "tools": {"get_info": get_info},
        "expected": _TOOL_SEQUENCE[0][1],
        "eval_fn": _eval_conversation_recall,
    })

    # Task 7: Source attribution (EMC)
    tasks.append({
        "id": "source_attribution",
        "name": "Source Attribution",
        "cef_dim": "EMC",
        "prompt": (
            "Read all three documents (doc1, doc2, doc3), then answer:\n"
            "Which document mentions GDP growth?\n\n"
            "Available tool: read_document(doc_id)\n\n"
            "Use FINISH(doc_id) with the document ID."
        ),
        "tools": {"read_document": _make_read_document()},
        "expected": "doc2",
        "eval_fn": _eval_source_attribution,
    })

    # Task 8: Shopping assistant (GENERAL)
    tasks.append({
        "id": "shopping_assistant",
        "name": "Shopping Assistant",
        "cef_dim": "GENERAL",
        "prompt": (
            "Find me the best-rated headphones under $80.\n\n"
            "Available tools:\n"
            "  search_items(query) — search for products\n"
            "  get_details(item_id) — get full details of an item\n\n"
            "Use FINISH(item_name) with the name of the best item."
        ),
        "tools": {
            "search_items": _make_search_items(),
            "get_details": _make_get_details(),
        },
        # Under $80: H1 ($79.99, 4.2), H2 ($45.99, 4.7), H4 ($29.99, 3.9)
        # Best rated under $80: H2 AudioBliss 300 (4.7)
        "expected": "AudioBliss 300",
        "eval_fn": _eval_shopping,
    })

    # Task 9: Schedule coordination (GENERAL)
    tasks.append({
        "id": "schedule_coordination",
        "name": "Schedule Coordination",
        "cef_dim": "GENERAL",
        "prompt": (
            "Find a 1-hour time slot on 2024-03-15 that works for Alice, Bob, and Carol.\n\n"
            "Available tool: check_calendar(person, date)\n"
            "Example: check_calendar(alice, 2024-03-15)\n\n"
            "Check each person's schedule, find the common free slot, and report it.\n"
            "Use FINISH(start_time - end_time) e.g. FINISH(12:00-13:00)."
        ),
        "tools": {"check_calendar": _make_check_calendar()},
        "expected": "12:00-13:00",
        "eval_fn": _eval_schedule,
    })

    # Task 10: Data pipeline (GENERAL)
    tasks.append({
        "id": "data_pipeline",
        "name": "ETL Data Pipeline",
        "cef_dim": "GENERAL",
        "prompt": (
            "Execute a data pipeline with these steps:\n"
            "1. Extract data from 'raw_source'\n"
            "2. Transform: filter only active records, then compute the sum of their values\n"
            "3. Load the final total into the destination\n\n"
            "Available tools:\n"
            "  extract_data(source_name) — extract raw data\n"
            "  transform_data(instruction) — apply a transformation\n"
            "  load_data(data) — load into destination\n\n"
            "Use FINISH(total_value) with the sum of active record values."
        ),
        "tools": {
            "extract_data": _make_extract_data(),
            "transform_data": _make_transform_data(),
            "load_data": _make_load_data(),
        },
        "expected": "160",
        "eval_fn": _eval_numeric,
    })

    return tasks


# ── Evaluation Functions ─────────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    """Strip whitespace, quotes, dollar signs for comparison."""
    if s is None:
        return ""
    return re.sub(r'[\s\$,]', '', str(s).lower().strip().strip('"').strip("'"))


def _eval_numeric(final_answer: str, expected: str, steps: list, **kw) -> dict:
    """Check if the final answer contains the expected number."""
    if final_answer is None:
        return {"task_completion": 0, "process_quality": 0.0}

    norm = _normalize_answer(final_answer)
    exp = _normalize_answer(expected)
    correct = exp in norm

    # Process quality: did they use calculator (not just guess)?
    tool_calls = [s for s in steps if s.get("tool") not in ("INVALID", "FINISH", None)]
    pq = 1.0 if (correct and len(tool_calls) >= 1) else (0.5 if correct else 0.0)

    return {"task_completion": int(correct), "process_quality": pq}


def _eval_entity_tracking(final_answer: str, expected: dict, steps: list, **kw) -> dict:
    """Check if all three balances are correct."""
    if final_answer is None:
        return {"task_completion": 0, "process_quality": 0.0}

    norm = _normalize_answer(final_answer)
    n_correct = 0
    for person, balance in expected.items():
        if str(balance) in final_answer:
            n_correct += 1

    correct = n_correct == 3
    partial = n_correct / 3.0

    # Process quality: checking initial balances is good but not required
    tool_calls = [s for s in steps if s.get("tool") == "check_balance"]
    used_tools = len(tool_calls) > 0
    pq = partial * (1.0 if used_tools else 0.8)

    return {"task_completion": int(correct), "process_quality": round(pq, 3)}


def _eval_uncertain_lookup(final_answer: str, expected: str, steps: list, **kw) -> dict:
    """Check if model verified the fact before answering."""
    if final_answer is None:
        return {"task_completion": 0, "process_quality": 0.0}

    # Did they use verify_fact?
    used_verify = any(s.get("tool") == "verify_fact" for s in steps)

    # Is the answer correct? (356.7 or close)
    correct = "356.7" in final_answer or "356.73" in final_answer

    # Process quality rewards verification behavior
    if correct and used_verify:
        pq = 1.0
    elif correct and not used_verify:
        pq = 0.3  # Right answer but didn't verify — poor metacognition
    elif not correct and used_verify:
        pq = 0.5  # Good process, wrong extraction
    else:
        pq = 0.0

    return {"task_completion": int(correct), "process_quality": pq}


def _eval_conflict_detection(final_answer: str, expected: str, steps: list, **kw) -> dict:
    """Check if model queried both sources AND flagged the discrepancy."""
    if final_answer is None:
        return {"task_completion": 0, "process_quality": 0.0}

    source_calls = [s for s in steps if s.get("tool") == "query_source"]
    queried_both = len(source_calls) >= 2

    # Check for discrepancy acknowledgment
    answer_lower = final_answer.lower()
    flags_conflict = any(w in answer_lower for w in [
        "discrepan", "differ", "conflict", "disagree", "inconsisten",
        "not match", "varying", "different", "approximately",
    ])

    # Mentions the actual numbers
    mentions_numbers = "23,615" in final_answer or "23615" in final_answer or "23,000" in final_answer or "23000" in final_answer

    correct = queried_both and flags_conflict
    pq = 0.0
    if queried_both:
        pq += 0.4
    if flags_conflict:
        pq += 0.4
    if mentions_numbers:
        pq += 0.2

    return {"task_completion": int(correct), "process_quality": round(pq, 3)}


def _eval_conversation_recall(final_answer: str, expected: str, steps: list, **kw) -> dict:
    """Check if model recalls the first tool result after 5 interactions."""
    if final_answer is None:
        return {"task_completion": 0, "process_quality": 0.0}

    # Did they make all 5 calls?
    info_calls = [s for s in steps if s.get("tool") == "get_info"]
    made_all_calls = len(info_calls) >= 5

    # Does the answer contain key info from the first result?
    answer_lower = final_answer.lower()
    recalls_weather = any(w in answer_lower for w in ["15°c", "15 c", "tokyo", "partly cloudy", "humidity"])

    correct = recalls_weather
    pq = 0.0
    if made_all_calls:
        pq += 0.5
    if recalls_weather:
        pq += 0.5

    return {"task_completion": int(correct), "process_quality": round(pq, 3)}


def _eval_source_attribution(final_answer: str, expected: str, steps: list, **kw) -> dict:
    """Check if model correctly attributes GDP growth to doc2."""
    if final_answer is None:
        return {"task_completion": 0, "process_quality": 0.0}

    correct = "doc2" in final_answer.lower() or "document 2" in final_answer.lower()

    # Did they read all docs?
    doc_calls = [s for s in steps if s.get("tool") == "read_document"]
    read_all = len(doc_calls) >= 3

    pq = 0.0
    if read_all:
        pq += 0.5
    if correct:
        pq += 0.5

    return {"task_completion": int(correct), "process_quality": round(pq, 3)}


def _eval_shopping(final_answer: str, expected: str, steps: list, **kw) -> dict:
    """Check if model found the best-rated headphones under $80."""
    if final_answer is None:
        return {"task_completion": 0, "process_quality": 0.0}

    answer_lower = final_answer.lower()
    # Correct answer: AudioBliss 300 (H2)
    correct = "audiobliss" in answer_lower or "h2" in answer_lower

    # Process: did they search AND check details?
    searched = any(s.get("tool") == "search_items" for s in steps)
    checked_details = any(s.get("tool") == "get_details" for s in steps)

    pq = 0.0
    if searched:
        pq += 0.3
    if checked_details:
        pq += 0.3
    if correct:
        pq += 0.4

    return {"task_completion": int(correct), "process_quality": round(pq, 3)}


def _eval_schedule(final_answer: str, expected: str, steps: list, **kw) -> dict:
    """Check if model found the 12:00-13:00 common slot."""
    if final_answer is None:
        return {"task_completion": 0, "process_quality": 0.0}

    # Accept 12:00-13:00 or similar
    correct = "12:00" in final_answer and "13:00" in final_answer

    # Did they check all three calendars?
    cal_calls = [s for s in steps if s.get("tool") == "check_calendar"]
    checked_all = len(cal_calls) >= 3

    pq = 0.0
    if checked_all:
        pq += 0.5
    if correct:
        pq += 0.5

    return {"task_completion": int(correct), "process_quality": round(pq, 3)}


# ── Run All Tasks for One Model ─────────────────────────────────────────────

def run_model(model: str, max_steps: int) -> list[dict]:
    """Run all 10 agent tasks for a single model, return per-task results."""
    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"{'='*60}")

    tasks = build_tasks()
    results = []

    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] {task['name']} ({task['cef_dim']})...", end=" ", flush=True)
        t0 = time.time()

        try:
            agent_result = run_agent(
                model=model,
                task_prompt=task["prompt"],
                tools=task["tools"],
                max_steps=max_steps,
            )

            eval_result = task["eval_fn"](
                final_answer=agent_result["final_answer"],
                expected=task["expected"],
                steps=agent_result["steps"],
            )

            elapsed = time.time() - t0
            result = {
                "model": model,
                "task_id": task["id"],
                "task_name": task["name"],
                "cef_dim": task["cef_dim"],
                "task_completion": eval_result["task_completion"],
                "process_quality": eval_result["process_quality"],
                "n_steps": agent_result["n_steps"],
                "errors": agent_result["errors"],
                "final_answer": agent_result["final_answer"],
                "elapsed_s": round(elapsed, 1),
                "steps_detail": agent_result["steps"],
            }

            status = "PASS" if eval_result["task_completion"] else "FAIL"
            print(f"{status} (pq={eval_result['process_quality']:.2f}, steps={agent_result['n_steps']}, "
                  f"err={agent_result['errors']}, {elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            result = {
                "model": model,
                "task_id": task["id"],
                "task_name": task["name"],
                "cef_dim": task["cef_dim"],
                "task_completion": 0,
                "process_quality": 0.0,
                "n_steps": 0,
                "errors": 1,
                "final_answer": None,
                "elapsed_s": round(elapsed, 1),
                "steps_detail": [],
                "error": str(e),
            }

        results.append(result)

    return results


# ── Summary and Correlation ──────────────────────────────────────────────────

def compute_summary(all_results: list[dict]) -> dict:
    """Compute per-model summary statistics."""
    import numpy as np

    summary = {}
    models = sorted(set(r["model"] for r in all_results))

    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        completion = np.mean([r["task_completion"] for r in model_results])
        process_q = np.mean([r["process_quality"] for r in model_results])
        avg_steps = np.mean([r["n_steps"] for r in model_results])
        total_errors = sum(r["errors"] for r in model_results)

        # Per-dimension breakdown
        dims = {}
        for dim in ["WMF-AM", "MCC-MA", "EMC", "GENERAL"]:
            dim_results = [r for r in model_results if r["cef_dim"] == dim]
            if dim_results:
                dims[dim] = {
                    "completion": round(np.mean([r["task_completion"] for r in dim_results]), 3),
                    "process_quality": round(np.mean([r["process_quality"] for r in dim_results]), 3),
                    "n_tasks": len(dim_results),
                }

        summary[model] = {
            "overall_completion": round(completion, 3),
            "overall_process_quality": round(process_q, 3),
            "avg_steps": round(avg_steps, 1),
            "total_errors": total_errors,
            "n_tasks": len(model_results),
            "dimensions": dims,
        }

    return summary


def print_summary_table(summary: dict, models: list[str]):
    """Print formatted summary table."""
    print(f"\n{'='*90}")
    print("AGENT VALIDATION SUMMARY")
    print(f"{'='*90}")

    hdr = f"{'Model':<28} {'Compl':>6} {'ProcQ':>6} {'Steps':>6} {'Errs':>5}  {'WMF-AM':>7} {'MCC-MA':>7} {'EMC':>7} {'GEN':>7}"
    print(hdr)
    print("-" * len(hdr))

    for model in models:
        if model not in summary:
            continue
        s = summary[model]
        d = s["dimensions"]

        def dim_val(dim_name):
            if dim_name in d:
                return f"{d[dim_name]['completion']:.2f}"
            return "   -  "

        print(f"{model:<28} {s['overall_completion']:>6.3f} {s['overall_process_quality']:>6.3f} "
              f"{s['avg_steps']:>6.1f} {s['total_errors']:>5d}  "
              f"{dim_val('WMF-AM'):>7} {dim_val('MCC-MA'):>7} {dim_val('EMC'):>7} {dim_val('GENERAL'):>7}")

    print(f"\nCompl = task completion rate, ProcQ = process quality, Steps = avg steps per task")
    print(f"Dimension columns show task completion rate for that CEF dimension")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CEF Agent-Level Validation: Fixed scaffold + LLM backbone swap"
    )
    parser.add_argument("--phase", choices=["ollama", "api", "all"], default="ollama",
                        help="Which model group to run (default: ollama)")
    parser.add_argument("--models", nargs="+", help="Specific model(s) to run")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max ReAct steps per task (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    parser.add_argument("--tasks", nargs="+", help="Run specific task IDs only")
    args = parser.parse_args()

    if args.models:
        models = args.models
    elif args.phase == "ollama":
        models = OLLAMA_MODELS
    elif args.phase == "api":
        models = API_MODELS
    else:
        models = list(MODELS.keys())

    # Validate model names
    for m in models:
        if m not in MODELS:
            print(f"ERROR: Unknown model '{m}'. Available: {', '.join(MODELS.keys())}")
            sys.exit(1)

    print(f"CEF Agent-Level Validation")
    print(f"Models: {len(models)} — {', '.join(models)}")
    print(f"Max steps: {args.max_steps}")
    print(f"Start: {datetime.now().isoformat()}")

    all_results = []
    for model in models:
        try:
            results = run_model(model, args.max_steps)

            # Filter to specific tasks if requested
            if args.tasks:
                results = [r for r in results if r["task_id"] in args.tasks]

            all_results.extend(results)

        except Exception as e:
            print(f"\nFATAL ERROR for {model}: {e}")
            traceback.print_exc()

        # Save incrementally
        out_name = args.output or f"cef_agent_validation_{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path = RESULTS_DIR / out_name
        with open(out_path, "w") as f:
            json.dump({
                "experiment": "cef_agent_validation",
                "timestamp": datetime.now().isoformat(),
                "max_steps": args.max_steps,
                "models_completed": sorted(set(r["model"] for r in all_results)),
                "total_tasks": len(all_results),
                "results": all_results,
            }, f, indent=2, default=str)

    # Summary
    summary = compute_summary(all_results)
    print_summary_table(summary, models)

    # Save summary separately (without step details, for quick loading)
    summary_path = RESULTS_DIR / out_name.replace(".json", "_summary.json")
    summary_results = [{k: v for k, v in r.items() if k != "steps_detail"} for r in all_results]
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": "cef_agent_validation",
            "timestamp": datetime.now().isoformat(),
            "max_steps": args.max_steps,
            "summary": summary,
            "results": summary_results,
        }, f, indent=2, default=str)

    print(f"\nFull results: {out_path}")
    print(f"Summary:      {summary_path}")


if __name__ == "__main__":
    main()
